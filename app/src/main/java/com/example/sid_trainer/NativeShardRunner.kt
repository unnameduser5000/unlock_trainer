package com.example.sid_trainer

import android.util.Log
import com.google.protobuf.ByteString
import org.pytorch.executorch.DType
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.training.TrainingModule
import sid.Sid
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class ShardExecutionResult(
    val outputHiddenStates: Sid.TensorData,
    val outputShiftLogP: Sid.TensorData,
    val localLoss: Float
)

object NativeShardRunner {
    private const val LOG_TAG = "ExecuTorchShardRunner"
    private const val DEFAULT_METHOD = "forward"

    private val cacheLock = ReentrantLock()
    private var cachedModelPath: String? = null
    private var cachedRuntime: LoadedRuntime? = null

    fun execute(modelPath: String, request: Sid.ForwardChunkRequest): ShardExecutionResult {
        val candidateInputs = buildCandidateInputs(request)
        val runtime = acquireRuntime(modelPath)
        val invocation = try {
            runtime.execute(candidateInputs)
        } catch (failure: Throwable) {
            if (runtime is TrainingLoadedRuntime) {
                Log.w(
                    LOG_TAG,
                    "Training runtime failed for $modelPath, retrying with inference Module: ${failure.message}"
                )
                val fallbackRuntime = replaceWithInferenceRuntime(modelPath, runtime)
                fallbackRuntime.execute(candidateInputs)
            } else {
                throw failure
            }
        }
        return invocation.toExecutionResult(request)
    }

    private fun acquireRuntime(modelPath: String): LoadedRuntime = cacheLock.withLock {
        val existing = cachedRuntime
        if (cachedModelPath == modelPath && existing != null) {
            return existing
        }

        existing?.closeQuietly()
        cachedRuntime = loadRuntime(modelPath)
        cachedModelPath = modelPath
        return requireNotNull(cachedRuntime)
    }

    private fun loadRuntime(modelPath: String): LoadedRuntime {
        try {
            return loadInferenceRuntime(modelPath)
        } catch (inferenceFailure: Throwable) {
            Log.w(
                LOG_TAG,
                "Module.load() failed for $modelPath, trying TrainingModule: ${inferenceFailure.message}"
            )
        }

        try {
            val module = TrainingModule.load(modelPath)
            Log.w(
                LOG_TAG,
                "Loaded ExecuTorch training module from $modelPath. " +
                    "Prefer forward-only Module artifacts for mobile pipeline tests."
            )
            return TrainingLoadedRuntime(modelPath, module)
        } catch (trainingFailure: Throwable) {
            throw IllegalStateException(
                "Could not load ExecuTorch artifact as Module or TrainingModule: $modelPath",
                trainingFailure
            )
        }
    }

    private fun loadInferenceRuntime(modelPath: String): LoadedRuntime {
        val module = Module.load(modelPath)
        val methods = module.getMethods().toList()
        val selectedMethod = when {
            methods.contains(DEFAULT_METHOD) -> DEFAULT_METHOD
            methods.isNotEmpty() -> methods.first()
            else -> DEFAULT_METHOD
        }
        Log.i(
            LOG_TAG,
            "Loaded ExecuTorch module from $modelPath with methods=${methods.joinToString()} selected=$selectedMethod"
        )
        return InferenceLoadedRuntime(modelPath, module, selectedMethod)
    }

    private fun replaceWithInferenceRuntime(
        modelPath: String,
        expectedCurrentRuntime: LoadedRuntime
    ): LoadedRuntime = cacheLock.withLock {
        if (cachedRuntime === expectedCurrentRuntime) {
            cachedRuntime?.closeQuietly()
            cachedRuntime = loadInferenceRuntime(modelPath)
            cachedModelPath = modelPath
        }
        return requireNotNull(cachedRuntime)
    }

    private fun buildCandidateInputs(request: Sid.ForwardChunkRequest): List<Array<EValue>> {
        val hiddenStates = request.hiddenStates.toRequiredEValue("hidden_states")
        val attentionMask = request.attentionMask.toOptionalEValue()
        val positionIds = request.positionIds.toOptionalEValue()
        val labels = request.labels.toOptionalEValue()
        val shiftLogPPrev = request.shiftLogPPrev.toOptionalEValue()

        val allInputs = listOf(hiddenStates, attentionMask, positionIds, labels, shiftLogPPrev)
        val candidates = mutableListOf<List<EValue>>()

        val compactInputs = buildList {
            add(hiddenStates)
            if (!request.attentionMask.isEmptyTensor()) add(attentionMask)
            if (!request.positionIds.isEmptyTensor()) add(positionIds)
            if (!request.labels.isEmptyTensor()) add(labels)
            if (!request.shiftLogPPrev.isEmptyTensor()) add(shiftLogPPrev)
        }
        candidates += compactInputs

        if (!request.labels.isEmptyTensor()) {
            candidates.add(listOf(hiddenStates, labels))
        }

        if (!request.attentionMask.isEmptyTensor() || !request.positionIds.isEmptyTensor()) {
            candidates.add(buildList {
                add(hiddenStates)
                if (!request.attentionMask.isEmptyTensor()) add(attentionMask)
                if (!request.positionIds.isEmptyTensor()) add(positionIds)
            })
        }

        candidates.add(listOf(hiddenStates))

        if (allInputs.none { it.isNone() }) {
            candidates += allInputs
        }

        val lastConcreteIndex = allInputs.indexOfLast { !it.isNone() }
        if (lastConcreteIndex >= 0) {
            val trimmedInputs = allInputs.take(lastConcreteIndex + 1)
            if (trimmedInputs.none { it.isNone() }) {
                candidates += trimmedInputs
            }
        }

        val seen = LinkedHashSet<String>()
        val deduplicated = candidates
            .filter { it.isNotEmpty() }
            .map { it.toTypedArray() }
            .filter { candidate ->
                val signature = candidate.joinToString("|") { eValue ->
                    when {
                        eValue.isTensor() -> "tensor"
                        eValue.isNone() -> "none"
                        eValue.isInt() -> "int"
                        eValue.isDouble() -> "double"
                        eValue.isBool() -> "bool"
                        eValue.isString() -> "string"
                        else -> "unknown"
                    }
                }
                seen.add("${candidate.size}:$signature")
            }

        Log.i(
            LOG_TAG,
            "Prepared ${deduplicated.size} ExecuTorch candidate signatures: ${
                deduplicated.joinToString { candidate ->
                    candidate.joinToString(prefix = "[", postfix = "]") { eValue ->
                        when {
                            eValue.isTensor() -> "tensor"
                            eValue.isNone() -> "none"
                            eValue.isInt() -> "int"
                            eValue.isDouble() -> "double"
                            eValue.isBool() -> "bool"
                            eValue.isString() -> "string"
                            else -> "unknown"
                        }
                    }
                }
            }"
        )
        return deduplicated
    }

    private fun InvocationResult.toExecutionResult(request: Sid.ForwardChunkRequest): ShardExecutionResult {
        var localLoss = 0f
        val tensors = mutableListOf<Tensor>()

        outputs.forEachIndexed { index, value ->
            when {
                value.isTensor() -> {
                    val tensor = value.toTensor()
                    tensors += tensor
                    Log.i(
                        LOG_TAG,
                        "Output[$index] tensor dtype=${tensor.dtype()} shape=${tensor.shape().joinToString(prefix = "[", postfix = "]")}"
                    )
                }

                value.isDouble() -> {
                    localLoss = value.toDouble().toFloat()
                    Log.i(LOG_TAG, "Output[$index] double=$localLoss")
                }

                value.isInt() -> {
                    Log.i(LOG_TAG, "Output[$index] int=${value.toInt()}")
                }

                value.isBool() -> {
                    Log.i(LOG_TAG, "Output[$index] bool=${value.toBool()}")
                }

                value.isString() -> {
                    Log.i(LOG_TAG, "Output[$index] string=${value.toStr()}")
                }

                value.isNone() -> {
                    Log.i(LOG_TAG, "Output[$index] none")
                }
            }
        }

        val remainingTensors = tensors.toMutableList()
        if (remainingTensors.size >= 2 && remainingTensors.first().isScalarFloatingTensor()) {
            localLoss = remainingTensors.removeAt(0).scalarFloatValue()
        }

        val hiddenStates = remainingTensors.getOrNull(0)?.toSidTensor()
            ?: copyTensorWithData(request.hiddenStates, request.hiddenStates.data.toByteArray())
        val shiftLogP = remainingTensors.getOrNull(1)?.toSidTensor()
            ?: copyTensorWithData(request.shiftLogPPrev, request.shiftLogPPrev.data.toByteArray())

        return ShardExecutionResult(
            outputHiddenStates = hiddenStates,
            outputShiftLogP = shiftLogP,
            localLoss = localLoss
        )
    }

    private fun Sid.TensorData.toRequiredEValue(name: String): EValue {
        if (isEmptyTensor()) {
            error("Required tensor '$name' is empty and cannot be fed into ExecuTorch.")
        }
        return EValue.from(toExecuTorchTensor())
    }

    private fun Sid.TensorData.toOptionalEValue(): EValue {
        return if (isEmptyTensor()) {
            EValue.optionalNone()
        } else {
            EValue.from(toExecuTorchTensor())
        }
    }

    private fun Sid.TensorData.isEmptyTensor(): Boolean {
        return data.isEmpty
    }

    private fun Sid.TensorData.toExecuTorchTensor(): Tensor {
        val shape = shapeList.map { it.toLong() }.toLongArray()
        val rawBytes = data.toByteArray()
        return when (dataType.normalizedDataType()) {
            "float32", "float" -> Tensor.fromBlob(rawBytes.toFloatArray(), shape)
            "float16", "half" -> Tensor.fromBlob(rawBytes.toShortArray(), shape)
            "int32", "int" -> Tensor.fromBlob(rawBytes.toIntArray(), shape)
            "int64", "long" -> Tensor.fromBlob(rawBytes.toLongArray(), shape)
            "float64", "double" -> Tensor.fromBlob(rawBytes.toDoubleArray(), shape)
            "int8" -> Tensor.fromBlob(rawBytes, shape)
            "uint8" -> Tensor.fromBlobUnsigned(rawBytes, shape)
            else -> error("Unsupported TensorData dtype '$dataType' for ExecuTorch input.")
        }
    }

    private fun Tensor.toSidTensor(): Sid.TensorData {
        val shape = shape().map { it.toInt() }
        val dtype = dtype()
        val rawBytes = when (dtype) {
            DType.FLOAT -> getDataAsFloatArray().toByteArray()
            DType.HALF -> getDataAsShortArray().toByteArray()
            DType.INT32 -> getDataAsIntArray().toByteArray()
            DType.INT64 -> getDataAsLongArray().toByteArray()
            DType.DOUBLE -> getDataAsDoubleArray().toByteArray()
            DType.INT8 -> getDataAsByteArray()
            DType.UINT8 -> getDataAsUnsignedByteArray()
            else -> error("Unsupported ExecuTorch tensor dtype '$dtype' for protobuf output.")
        }
        return Sid.TensorData.newBuilder()
            .setData(ByteString.copyFrom(rawBytes))
            .addAllShape(shape)
            .setDataType(dtype.toProtoDataType())
            .build()
    }

    private fun Tensor.isScalarFloatingTensor(): Boolean {
        return numel() == 1L && when (dtype()) {
            DType.FLOAT, DType.HALF, DType.DOUBLE -> true
            else -> false
        }
    }

    private fun Tensor.scalarFloatValue(): Float {
        return when (dtype()) {
            DType.FLOAT -> getDataAsFloatArray().first()
            DType.HALF -> getDataAsFloatArray().first()
            DType.DOUBLE -> getDataAsDoubleArray().first().toFloat()
            else -> error("Tensor $this is not a floating-point scalar.")
        }
    }

    private fun DType.toProtoDataType(): String = when (this) {
        DType.FLOAT -> "float32"
        DType.HALF -> "float16"
        DType.INT32 -> "int32"
        DType.INT64 -> "int64"
        DType.DOUBLE -> "float64"
        DType.INT8 -> "int8"
        DType.UINT8 -> "uint8"
        else -> error("Unsupported ExecuTorch dtype '$this'.")
    }

    private fun String.normalizedDataType(): String {
        return trim().lowercase()
    }

    private fun ByteArray.toFloatArray(): FloatArray {
        if (isEmpty()) {
            return FloatArray(0)
        }
        require(size % Float.SIZE_BYTES == 0) {
            "Expected float32 tensor bytes to be a multiple of ${Float.SIZE_BYTES}, got $size."
        }
        val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun ByteArray.toShortArray(): ShortArray {
        if (isEmpty()) {
            return ShortArray(0)
        }
        require(size % Short.SIZE_BYTES == 0) {
            "Expected float16 tensor bytes to be a multiple of ${Short.SIZE_BYTES}, got $size."
        }
        val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asShortBuffer()
        return ShortArray(buffer.remaining()).also(buffer::get)
    }

    private fun ByteArray.toIntArray(): IntArray {
        if (isEmpty()) {
            return IntArray(0)
        }
        require(size % Int.SIZE_BYTES == 0) {
            "Expected int32 tensor bytes to be a multiple of ${Int.SIZE_BYTES}, got $size."
        }
        val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asIntBuffer()
        return IntArray(buffer.remaining()).also(buffer::get)
    }

    private fun ByteArray.toLongArray(): LongArray {
        if (isEmpty()) {
            return LongArray(0)
        }
        require(size % Long.SIZE_BYTES == 0) {
            "Expected int64 tensor bytes to be a multiple of ${Long.SIZE_BYTES}, got $size."
        }
        val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asLongBuffer()
        return LongArray(buffer.remaining()).also(buffer::get)
    }

    private fun ByteArray.toDoubleArray(): DoubleArray {
        if (isEmpty()) {
            return DoubleArray(0)
        }
        require(size % Double.SIZE_BYTES == 0) {
            "Expected float64 tensor bytes to be a multiple of ${Double.SIZE_BYTES}, got $size."
        }
        val buffer = ByteBuffer.wrap(this).order(ByteOrder.nativeOrder()).asDoubleBuffer()
        return DoubleArray(buffer.remaining()).also(buffer::get)
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

    private fun copyTensorWithData(template: Sid.TensorData, data: ByteArray): Sid.TensorData {
        return Sid.TensorData.newBuilder()
            .setData(ByteString.copyFrom(data))
            .addAllShape(template.shapeList)
            .setDataType(template.dataType)
            .build()
    }

    private fun Closeable.closeQuietly() {
        runCatching { close() }
            .onFailure { error ->
                Log.w(LOG_TAG, "Failed to close ExecuTorch runtime cleanly: ${error.message}")
            }
    }

    private data class InvocationResult(
        val runtimeName: String,
        val methodName: String,
        val inputCount: Int,
        val outputs: Array<EValue>
    )

    private sealed interface LoadedRuntime : Closeable {
        fun execute(candidateInputs: List<Array<EValue>>): InvocationResult
    }

    private class TrainingLoadedRuntime(
        private val modelPath: String,
        private val module: TrainingModule
    ) : LoadedRuntime {
        override fun execute(candidateInputs: List<Array<EValue>>): InvocationResult {
            var lastFailure: Throwable? = null
            for (inputs in candidateInputs) {
                try {
                    val outputs = module.executeForwardBackward(DEFAULT_METHOD, *inputs)
                    Log.i(
                        LOG_TAG,
                        "TrainingModule.executeForwardBackward() succeeded for $modelPath with ${inputs.size} inputs"
                    )
                    return InvocationResult(
                        runtimeName = "TrainingModule",
                        methodName = DEFAULT_METHOD,
                        inputCount = inputs.size,
                        outputs = outputs
                    )
                } catch (t: Throwable) {
                    lastFailure = t
                    Log.w(
                        LOG_TAG,
                        "TrainingModule execution failed for $modelPath with ${inputs.size} inputs: ${t.message}"
                    )
                }
            }
            throw IllegalStateException(
                "TrainingModule could not execute any supported input signature for $modelPath.",
                lastFailure
            )
        }

        override fun close() {
            // executorch-android 1.2.0 does not expose a public destroy/close API for TrainingModule.
        }
    }

    private class InferenceLoadedRuntime(
        private val modelPath: String,
        private val module: Module,
        private val methodName: String
    ) : LoadedRuntime {
        override fun execute(candidateInputs: List<Array<EValue>>): InvocationResult {
            var lastFailure: Throwable? = null
            for (inputs in candidateInputs) {
                try {
                    val outputs = module.execute(methodName, *inputs)
                    Log.i(
                        LOG_TAG,
                        "Module.execute() succeeded for $modelPath method=$methodName with ${inputs.size} inputs"
                    )
                    return InvocationResult(
                        runtimeName = "Module",
                        methodName = methodName,
                        inputCount = inputs.size,
                        outputs = outputs
                    )
                } catch (t: Throwable) {
                    lastFailure = t
                    Log.w(
                        LOG_TAG,
                        "Module execution failed for $modelPath method=$methodName with ${inputs.size} inputs: ${t.message}"
                    )
                }
            }
            throw IllegalStateException(
                "ExecuTorch Module could not execute any supported input signature for $modelPath.",
                lastFailure
            )
        }

        override fun close() {
            module.destroy()
        }
    }
}
