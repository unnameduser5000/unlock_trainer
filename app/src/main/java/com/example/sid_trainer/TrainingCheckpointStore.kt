package com.example.sid_trainer

import android.util.Log
import org.pytorch.executorch.DType
import org.pytorch.executorch.Tensor
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

data class TrainingCheckpointRestoreResult(
    val restored: Boolean,
    val restoreFailed: Boolean = false,
    val step: Long = 0,
    val parameterCount: Int = 0,
    val committedRequestKeys: Set<String> = emptySet(),
    val checkpointPath: String = "",
    val message: String = ""
)

object TrainingCheckpointStore {
    private const val LOG_TAG = "TrainingCheckpoint"
    private const val MAGIC_V1 = "SID_TRAINING_CHECKPOINT_V1"
    private const val MAGIC_V2 = "SID_TRAINING_CHECKPOINT_V2"
    private const val MAGIC_V3 = "SID_TRAINING_CHECKPOINT_V3"
    private const val MAX_COMMITTED_REQUEST_KEYS = 1_000_000

    private val rawBufferMethod by lazy {
        Tensor::class.java.getDeclaredMethod("getRawDataBuffer").also { it.isAccessible = true }
    }

    fun saveLatest(modelPath: String, step: Long, parameters: Map<String, Tensor>): File {
        return saveLatest(modelPath, step, parameters, optimizerSnapshot = null)
    }

    fun saveLatest(
        modelPath: String,
        step: Long,
        parameters: Map<String, Tensor>,
        optimizerSnapshot: MobileAdamW.Snapshot?
    ): File {
        return saveLatest(modelPath, step, parameters, optimizerSnapshot, emptySet())
    }

    fun saveLatest(
        modelPath: String,
        step: Long,
        parameters: Map<String, Tensor>,
        optimizerSnapshot: MobileAdamW.Snapshot?,
        committedRequestKeys: Collection<String>
    ): File {
        require(parameters.isNotEmpty()) { "Cannot checkpoint an empty parameter map." }
        require(committedRequestKeys.size <= MAX_COMMITTED_REQUEST_KEYS) {
            "Commit journal contains ${committedRequestKeys.size} keys; maximum is $MAX_COMMITTED_REQUEST_KEYS."
        }
        val checkpointFile = checkpointFileFor(modelPath)
        checkpointFile.parentFile?.mkdirs()
        val tmpFile = File(checkpointFile.parentFile, "${checkpointFile.name}.tmp")
        val backupFile = File(checkpointFile.parentFile, "${checkpointFile.name}.bak")

        FileOutputStream(tmpFile).use { fileOutput ->
            DataOutputStream(BufferedOutputStream(fileOutput)).use { output ->
                output.writeUTF(MAGIC_V3)
                output.writeUTF(File(modelPath).name)
                output.writeLong(System.currentTimeMillis())
                output.writeLong(step)
                val sorted = parameters.toSortedMap()
                output.writeInt(sorted.size)
                for ((name, tensor) in sorted) {
                    val data = tensor.rawDataBytes()
                    output.writeUTF(name)
                    output.writeUTF(tensor.dtype().name)
                    val shape = tensor.shape()
                    output.writeInt(shape.size)
                    shape.forEach(output::writeLong)
                    output.writeInt(data.size)
                    output.write(data)
                }
                output.writeBoolean(optimizerSnapshot != null)
                if (optimizerSnapshot != null) {
                    output.writeUTF("adamw")
                    output.writeLong(optimizerSnapshot.stepCount)
                    val sortedStates = optimizerSnapshot.parameterStates.toSortedMap()
                    output.writeInt(sortedStates.size)
                    for ((name, state) in sortedStates) {
                        output.writeUTF(name)
                        output.writeInt(state.expAvg.size)
                        output.write(state.expAvg.toByteArray())
                        output.writeInt(state.expAvgSq.size)
                        output.write(state.expAvgSq.toByteArray())
                    }
                }
                TrainingCommitJournalCodec.write(output, step, committedRequestKeys)
                output.flush()
                fileOutput.fd.sync()
            }
        }

        replaceCheckpoint(checkpointFile, tmpFile, backupFile)
        Log.i(
            LOG_TAG,
            "Saved training checkpoint path=${checkpointFile.absolutePath} step=$step " +
                "parameters=${parameters.size} optimizerState=${optimizerSnapshot != null} " +
                "committedRequests=${committedRequestKeys.size}"
        )
        return checkpointFile
    }

    fun restoreLatest(
        modelPath: String,
        parameters: Map<String, Tensor>,
        restoreOptimizer: (MobileAdamW.Snapshot) -> Unit
    ): TrainingCheckpointRestoreResult {
        return restoreLatestInternal(modelPath, parameters, restoreOptimizer)
    }

    fun restoreLatest(modelPath: String, parameters: Map<String, Tensor>): TrainingCheckpointRestoreResult {
        return restoreLatestInternal(modelPath, parameters, restoreOptimizer = null)
    }

    private fun restoreLatestInternal(
        modelPath: String,
        parameters: Map<String, Tensor>,
        restoreOptimizer: ((MobileAdamW.Snapshot) -> Unit)?
    ): TrainingCheckpointRestoreResult {
        val checkpointFile = checkpointFileFor(modelPath)
        val candidateFiles = listOf(
            checkpointFile,
            File(checkpointFile.parentFile, "${checkpointFile.name}.tmp"),
            File(checkpointFile.parentFile, "${checkpointFile.name}.bak")
        ).filter(File::exists)
        if (candidateFiles.isEmpty()) {
            return TrainingCheckpointRestoreResult(
                restored = false,
                checkpointPath = checkpointFile.absolutePath,
                message = "No checkpoint exists."
            )
        }
        if (parameters.isEmpty()) {
            return TrainingCheckpointRestoreResult(
                restored = false,
                checkpointPath = checkpointFile.absolutePath,
                message = "Runtime has no trainable parameters."
            )
        }

        return try {
            val candidates = candidateFiles.mapNotNull { file ->
                runCatching { CheckpointCandidate(file, readCheckpoint(file)) }
                    .onFailure { error ->
                        Log.w(LOG_TAG, "Ignoring invalid checkpoint candidate ${file.absolutePath}: ${error.message}")
                    }
                    .getOrNull()
            }
            val selected = candidates.maxWithOrNull(
                compareBy<CheckpointCandidate> { it.checkpoint.step }
                    .thenBy { it.checkpoint.savedAtEpochMs }
                    .thenBy { if (it.file == checkpointFile) 1 else 0 }
            ) ?: error("No valid checkpoint candidate exists for ${checkpointFile.absolutePath}.")
            val checkpoint = selected.checkpoint
            require(checkpoint.modelFileName == File(modelPath).name) {
                "Checkpoint model '${checkpoint.modelFileName}' does not match '${File(modelPath).name}'."
            }
            for (record in checkpoint.parameters) {
                val target = parameters[record.name]
                    ?: error("Checkpoint parameter '${record.name}' is missing from runtime.")
                require(target.dtype().name == record.dtypeName) {
                    "Parameter '${record.name}' dtype mismatch: runtime=${target.dtype().name} checkpoint=${record.dtypeName}"
                }
                require(target.shape().contentEquals(record.shape)) {
                    "Parameter '${record.name}' shape mismatch: runtime=${target.shape().contentToString()} checkpoint=${record.shape.contentToString()}"
                }
            }
            val restoredFile = promoteRecoveredCandidate(selected.file, checkpointFile)
            var restoredCount = 0
            for (record in checkpoint.parameters) {
                val target = requireNotNull(parameters[record.name])
                target.writeRawDataBytes(record.data)
                restoredCount++
            }
            if (checkpoint.optimizerSnapshot != null && restoreOptimizer != null) {
                restoreOptimizer(checkpoint.optimizerSnapshot)
            }
            Log.i(
                LOG_TAG,
                "Restored training checkpoint path=${restoredFile.absolutePath} step=${checkpoint.step} " +
                    "parameters=$restoredCount optimizerState=${checkpoint.optimizerSnapshot != null} " +
                    "committedRequests=${checkpoint.committedRequestKeys.size}"
            )
            TrainingCheckpointRestoreResult(
                restored = true,
                step = checkpoint.step,
                parameterCount = restoredCount,
                committedRequestKeys = checkpoint.committedRequestKeys,
                checkpointPath = restoredFile.absolutePath,
                message = "Restored checkpoint."
            )
        } catch (t: Throwable) {
            Log.e(LOG_TAG, "Failed to restore checkpoint ${checkpointFile.absolutePath}: ${t.message}", t)
            TrainingCheckpointRestoreResult(
                restored = false,
                restoreFailed = true,
                checkpointPath = checkpointFile.absolutePath,
                message = "Restore failed: ${t.message}"
            )
        }
    }

    fun checkpointFileFor(modelPath: String): File {
        val modelFile = File(modelPath)
        val parent = modelFile.parentFile ?: File(".")
        val checkpointDir = File(parent, "checkpoints")
        return File(checkpointDir, "${modelFile.nameWithoutExtension}.latest.sidckpt")
    }

    private fun readCheckpoint(file: File): TrainingCheckpoint {
        DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
            val magic = input.readUTF()
            require(magic == MAGIC_V1 || magic == MAGIC_V2 || magic == MAGIC_V3) {
                "Unsupported checkpoint magic '$magic'."
            }
            val modelFileName = input.readUTF()
            val savedAtEpochMs = input.readLong()
            val step = input.readLong()
            val count = input.readInt()
            require(count >= 0) { "Invalid checkpoint parameter count $count." }
            val records = ArrayList<TrainingCheckpointParameter>(count)
            repeat(count) {
                val name = input.readUTF()
                val dtypeName = input.readUTF()
                val shapeCount = input.readInt()
                require(shapeCount >= 0) { "Invalid shape rank $shapeCount for '$name'." }
                val shape = LongArray(shapeCount) { input.readLong() }
                val dataSize = input.readInt()
                require(dataSize >= 0) { "Invalid data size $dataSize for '$name'." }
                val data = ByteArray(dataSize)
                input.readFully(data)
                records += TrainingCheckpointParameter(name, dtypeName, shape, data)
            }
            val optimizerSnapshot = if (magic != MAGIC_V1 && input.readBoolean()) {
                val optimizerName = input.readUTF()
                require(optimizerName == "adamw") { "Unsupported optimizer checkpoint '$optimizerName'." }
                val optimizerStep = input.readLong()
                val stateCount = input.readInt()
                require(stateCount >= 0) { "Invalid optimizer state count $stateCount." }
                val states = LinkedHashMap<String, MobileAdamW.ParameterSnapshot>(stateCount)
                repeat(stateCount) {
                    val name = input.readUTF()
                    val expAvgSize = input.readInt()
                    require(expAvgSize >= 0) { "Invalid AdamW exp_avg size $expAvgSize for '$name'." }
                    val expAvgBytes = ByteArray(expAvgSize * Float.SIZE_BYTES)
                    input.readFully(expAvgBytes)
                    val expAvgSqSize = input.readInt()
                    require(expAvgSqSize >= 0) { "Invalid AdamW exp_avg_sq size $expAvgSqSize for '$name'." }
                    val expAvgSqBytes = ByteArray(expAvgSqSize * Float.SIZE_BYTES)
                    input.readFully(expAvgSqBytes)
                    states[name] = MobileAdamW.ParameterSnapshot(
                        expAvg = expAvgBytes.toFloatArray(),
                        expAvgSq = expAvgSqBytes.toFloatArray()
                    )
                }
                MobileAdamW.Snapshot(optimizerStep, states)
            } else {
                null
            }
            val committedRequestKeys = if (magic == MAGIC_V3) {
                TrainingCommitJournalCodec.read(input, step, MAX_COMMITTED_REQUEST_KEYS)
            } else {
                emptySet()
            }
            return TrainingCheckpoint(
                modelFileName = modelFileName,
                savedAtEpochMs = savedAtEpochMs,
                step = step,
                parameters = records,
                optimizerSnapshot = optimizerSnapshot,
                committedRequestKeys = committedRequestKeys
            )
        }
    }

    private fun replaceCheckpoint(checkpointFile: File, tmpFile: File, backupFile: File) {
        if (backupFile.exists() && !backupFile.delete()) {
            error("Could not remove stale checkpoint backup ${backupFile.absolutePath}")
        }
        if (checkpointFile.exists() && !checkpointFile.renameTo(backupFile)) {
            error("Could not preserve checkpoint ${checkpointFile.absolutePath}")
        }
        if (!tmpFile.renameTo(checkpointFile)) {
            if (!checkpointFile.exists() && backupFile.exists()) {
                backupFile.renameTo(checkpointFile)
            }
            error("Could not move checkpoint ${tmpFile.absolutePath} to ${checkpointFile.absolutePath}")
        }
        if (backupFile.exists() && !backupFile.delete()) {
            Log.w(LOG_TAG, "Could not remove checkpoint backup ${backupFile.absolutePath}")
        }
    }

    private fun promoteRecoveredCandidate(selectedFile: File, checkpointFile: File): File {
        if (selectedFile == checkpointFile) return checkpointFile
        if (checkpointFile.exists() && !checkpointFile.delete()) {
            error("Could not discard older checkpoint ${checkpointFile.absolutePath}")
        }
        require(selectedFile.renameTo(checkpointFile)) {
            "Could not promote recovered checkpoint ${selectedFile.absolutePath} to ${checkpointFile.absolutePath}"
        }
        Log.w(LOG_TAG, "Promoted recovered checkpoint ${selectedFile.absolutePath} to ${checkpointFile.absolutePath}")
        return checkpointFile
    }

    private fun Tensor.rawDataBytes(): ByteArray {
        return when (dtype()) {
            DType.FLOAT -> getDataAsFloatArray().toByteArray()
            DType.HALF -> getDataAsShortArray().toByteArray()
            DType.INT32 -> getDataAsIntArray().toByteArray()
            DType.INT64 -> getDataAsLongArray().toByteArray()
            DType.DOUBLE -> getDataAsDoubleArray().toByteArray()
            DType.INT8 -> getDataAsByteArray()
            DType.UINT8 -> getDataAsUnsignedByteArray()
            else -> error("Unsupported checkpoint tensor dtype '${dtype()}'.")
        }
    }

    private fun Tensor.writeRawDataBytes(data: ByteArray) {
        val rawBuffer = rawBufferMethod.invoke(this) as Buffer
        when (rawBuffer) {
            is ByteBuffer -> {
                val target = rawBuffer.duplicate().order(ByteOrder.nativeOrder())
                target.clear()
                require(target.remaining() >= data.size) {
                    "ByteBuffer capacity ${target.remaining()} is smaller than checkpoint data ${data.size}."
                }
                target.put(data)
            }

            is FloatBuffer -> {
                val values = data.toFloatArray()
                val target = rawBuffer.duplicate()
                target.clear()
                require(target.remaining() >= values.size) {
                    "FloatBuffer capacity ${target.remaining()} is smaller than checkpoint values ${values.size}."
                }
                target.put(values)
            }

            is ShortBuffer -> {
                val values = data.toShortArray()
                val target = rawBuffer.duplicate()
                target.clear()
                require(target.remaining() >= values.size) {
                    "ShortBuffer capacity ${target.remaining()} is smaller than checkpoint values ${values.size}."
                }
                target.put(values)
            }

            is IntBuffer -> {
                val values = data.toIntArray()
                val target = rawBuffer.duplicate()
                target.clear()
                require(target.remaining() >= values.size) {
                    "IntBuffer capacity ${target.remaining()} is smaller than checkpoint values ${values.size}."
                }
                target.put(values)
            }

            is LongBuffer -> {
                val values = data.toLongArray()
                val target = rawBuffer.duplicate()
                target.clear()
                require(target.remaining() >= values.size) {
                    "LongBuffer capacity ${target.remaining()} is smaller than checkpoint values ${values.size}."
                }
                target.put(values)
            }

            is DoubleBuffer -> {
                val values = data.toDoubleArray()
                val target = rawBuffer.duplicate()
                target.clear()
                require(target.remaining() >= values.size) {
                    "DoubleBuffer capacity ${target.remaining()} is smaller than checkpoint values ${values.size}."
                }
                target.put(values)
            }

            else -> error("Unsupported raw parameter buffer type ${rawBuffer::class.java.name}.")
        }
    }

    private data class TrainingCheckpoint(
        val modelFileName: String,
        val savedAtEpochMs: Long,
        val step: Long,
        val parameters: List<TrainingCheckpointParameter>,
        val optimizerSnapshot: MobileAdamW.Snapshot?,
        val committedRequestKeys: Set<String>
    )

    private data class CheckpointCandidate(
        val file: File,
        val checkpoint: TrainingCheckpoint
    )

    private data class TrainingCheckpointParameter(
        val name: String,
        val dtypeName: String,
        val shape: LongArray,
        val data: ByteArray
    )
}

internal object TrainingCommitJournalCodec {
    fun write(output: DataOutputStream, checkpointStep: Long, committedRequestKeys: Collection<String>) {
        output.writeLong(checkpointStep)
        output.writeInt(committedRequestKeys.size)
        committedRequestKeys.forEach(output::writeUTF)
    }

    fun read(input: DataInputStream, expectedCheckpointStep: Long, maxKeys: Int): Set<String> {
        val journalCheckpointStep = input.readLong()
        require(journalCheckpointStep == expectedCheckpointStep) {
            "Commit journal step $journalCheckpointStep does not match checkpoint step $expectedCheckpointStep."
        }
        val count = input.readInt()
        require(count in 0..maxKeys) { "Invalid commit journal entry count $count." }
        val keys = LinkedHashSet<String>(count)
        repeat(count) {
            val key = input.readUTF()
            require(key.isNotBlank()) { "Commit journal contains a blank request key." }
            require(keys.add(key)) { "Commit journal contains duplicate request key '$key'." }
        }
        return keys
    }
}

private fun FloatArray.toByteArray(): ByteArray {
    val bytes = ByteArray(size * Float.SIZE_BYTES)
    ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer().put(this)
    return bytes
}

private fun ShortArray.toByteArray(): ByteArray {
    val bytes = ByteArray(size * Short.SIZE_BYTES)
    ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).asShortBuffer().put(this)
    return bytes
}

private fun IntArray.toByteArray(): ByteArray {
    val bytes = ByteArray(size * Int.SIZE_BYTES)
    ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).asIntBuffer().put(this)
    return bytes
}

private fun LongArray.toByteArray(): ByteArray {
    val bytes = ByteArray(size * Long.SIZE_BYTES)
    ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).asLongBuffer().put(this)
    return bytes
}

private fun DoubleArray.toByteArray(): ByteArray {
    val bytes = ByteArray(size * Double.SIZE_BYTES)
    ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).asDoubleBuffer().put(this)
    return bytes
}

private fun ByteArray.toFloatArray(): FloatArray {
    require(size % Float.SIZE_BYTES == 0) {
        "Expected float32 bytes to be a multiple of ${Float.SIZE_BYTES}, got $size."
    }
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asFloatBuffer()
    return FloatArray(buffer.remaining()).also(buffer::get)
}

private fun ByteArray.toShortArray(): ShortArray {
    require(size % Short.SIZE_BYTES == 0) {
        "Expected int16/float16 bytes to be a multiple of ${Short.SIZE_BYTES}, got $size."
    }
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asShortBuffer()
    return ShortArray(buffer.remaining()).also(buffer::get)
}

private fun ByteArray.toIntArray(): IntArray {
    require(size % Int.SIZE_BYTES == 0) {
        "Expected int32 bytes to be a multiple of ${Int.SIZE_BYTES}, got $size."
    }
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asIntBuffer()
    return IntArray(buffer.remaining()).also(buffer::get)
}

private fun ByteArray.toLongArray(): LongArray {
    require(size % Long.SIZE_BYTES == 0) {
        "Expected int64 bytes to be a multiple of ${Long.SIZE_BYTES}, got $size."
    }
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asLongBuffer()
    return LongArray(buffer.remaining()).also(buffer::get)
}

private fun ByteArray.toDoubleArray(): DoubleArray {
    require(size % Double.SIZE_BYTES == 0) {
        "Expected float64 bytes to be a multiple of ${Double.SIZE_BYTES}, got $size."
    }
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asDoubleBuffer()
    return DoubleArray(buffer.remaining()).also(buffer::get)
}
