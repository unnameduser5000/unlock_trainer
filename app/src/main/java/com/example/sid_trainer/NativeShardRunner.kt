package com.example.sid_trainer

import android.os.Trace
import android.util.Log
import com.google.protobuf.ByteString
import org.pytorch.executorch.DType
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.training.TrainingModule
import sid.Sid
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class ShardExecutionResult(
    val outputHiddenStates: Sid.TensorData,
    val outputShiftLogP: Sid.TensorData,
    val localLoss: Float,
    val runtimeName: String,
    val methodName: String,
    val inputCount: Int,
    val evalOnly: Boolean,
    val learningRate: Double,
    val optimizerStepApplied: Boolean,
    val checkpointSaved: Boolean,
    val checkpointIntervalSteps: Long,
    val timing: ShardExecutionTiming
)

data class ShardExecutionTiming(
    val runtimeAcquireMs: Long = 0,
    val checkpointRestoreMs: Long = 0,
    val inputBuildMs: Long = 0,
    val executeMs: Long = 0,
    val gradientsMs: Long = 0,
    val optimizerCreateMs: Long = 0,
    val optimizerStepMs: Long = 0,
    val outputConvertMs: Long = 0
) {
    val totalMeasuredMs: Long
        get() = runtimeAcquireMs + checkpointRestoreMs + inputBuildMs + executeMs +
            gradientsMs + optimizerCreateMs + optimizerStepMs + outputConvertMs

    fun describeForLog(): String {
        return "runtimeAcquireMs=$runtimeAcquireMs checkpointRestoreMs=$checkpointRestoreMs " +
            "inputBuildMs=$inputBuildMs executeMs=$executeMs gradientsMs=$gradientsMs " +
            "optimizerCreateMs=$optimizerCreateMs optimizerStepMs=$optimizerStepMs " +
            "outputConvertMs=$outputConvertMs totalMeasuredMs=$totalMeasuredMs"
    }
}

private inline fun <T> tracedSection(name: String, block: () -> T): T {
    Trace.beginSection(name)
    return try {
        block()
    } finally {
        Trace.endSection()
    }
}

object NativeShardRunner {
    private const val LOG_TAG = "ExecuTorchShardRunner"
    private const val DEFAULT_METHOD = "forward"
    private const val TRAINING_METADATA_SCAN_BYTES = 1024 * 1024
    private const val DEFAULT_TRAINING_LEARNING_RATE = 1e-5
    private const val TRAINING_CHECKPOINT_INTERVAL_STEPS = 16L

    private val cacheLock = ReentrantLock()
    private var cachedModelPath: String? = null
    private var cachedRuntime: LoadedRuntime? = null

    fun execute(modelPath: String, request: Sid.ForwardChunkRequest): ShardExecutionResult {
        val runtimeAcquireStartedAtNs = System.nanoTime()
        val runtime = acquireRuntime(modelPath)
        val runtimeAcquireMs = elapsedMsSince(runtimeAcquireStartedAtNs)
        val invocation = runtime.execute(request)
        return invocation.toExecutionResult(request, runtimeAcquireMs)
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
        if (modelPath.hasTrainingMetadata()) {
            val module = TrainingModule.load(modelPath)
            Log.w(
                LOG_TAG,
                "Loaded ExecuTorch training module from $modelPath. " +
                    "This path executes the exported joint graph; it does not create cross-stage backprop RPC."
            )
            return TrainingLoadedRuntime(modelPath, module)
        }

        try {
            return loadInferenceRuntime(modelPath)
        } catch (inferenceFailure: Throwable) {
            throw IllegalStateException(
                "ExecuTorch artifact has no __et_training marker, so it must load via Module. " +
                    "Refusing unsafe TrainingModule fallback for forward-only artifact: $modelPath",
                inferenceFailure
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

    private fun buildInferenceInputCandidates(request: Sid.ForwardChunkRequest): List<Array<EValue>> {
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

    private fun buildTrainingInputs(request: Sid.ForwardChunkRequest): Array<EValue> {
        val inputs = buildList {
            add(request.hiddenStates.toRequiredEValue("hidden_states"))
            add(request.attentionMask.toRequiredEValue("attention_mask"))
            add(request.positionIds.toRequiredEValue("position_ids"))
            add(request.labels.toRequiredEValue("labels"))
            if (!request.shiftLogPPrev.isEmptyTensor()) {
                add(request.shiftLogPPrev.toRequiredEValue("prev_log_probs"))
            }
        }.toTypedArray()

        Log.i(
            LOG_TAG,
            "Prepared ExecuTorch training signature: ${
                inputs.joinToString(prefix = "[", postfix = "]") { it.describeForLog() }
            }"
        )
        return inputs
    }

    private fun InvocationResult.toExecutionResult(
        request: Sid.ForwardChunkRequest,
        runtimeAcquireMs: Long
    ): ShardExecutionResult {
        val startedAtNs = System.nanoTime()
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
        val outputConvertMs = elapsedMsSince(startedAtNs)
        val finalTiming = timing.copy(
            runtimeAcquireMs = runtimeAcquireMs,
            outputConvertMs = outputConvertMs
        )

        Log.i(
            LOG_TAG,
            "Shard execution timing runtime=$runtimeName method=$methodName inputs=$inputCount " +
                finalTiming.describeForLog()
        )

        return ShardExecutionResult(
            outputHiddenStates = hiddenStates,
            outputShiftLogP = shiftLogP,
            localLoss = localLoss,
            runtimeName = runtimeName,
            methodName = methodName,
            inputCount = inputCount,
            evalOnly = evalOnly,
            learningRate = learningRate,
            optimizerStepApplied = optimizerStepApplied,
            checkpointSaved = checkpointSaved,
            checkpointIntervalSteps = checkpointIntervalSteps,
            timing = finalTiming
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
        BeliefTopKCodec.decodeToDenseFloatArray(this)?.let { dense ->
            val tensor = Tensor.fromBlob(dense, shape)
            Log.i(
                LOG_TAG,
                "Decoded top-k belief tensor protoDtype=$dataType shape=${
                    shape.joinToString(prefix = "[", postfix = "]")
                } compressedBytes=${data.size()} denseElements=${dense.size}"
            )
            return tensor
        }
        val rawBytes = data.toByteArray()
        val tensor = when (dataType.normalizedDataType()) {
            "float32", "float" -> Tensor.fromBlob(rawBytes.toFloatArray(), shape)
            "float16", "half" -> Tensor.fromBlob(rawBytes.toFloatArrayFromHalf(), shape)
            "int32", "int" -> Tensor.fromBlob(rawBytes.toIntArray(), shape)
            "int64", "long" -> Tensor.fromBlob(rawBytes.toLongArray(), shape)
            "float64", "double" -> Tensor.fromBlob(rawBytes.toDoubleArray(), shape)
            "int8" -> Tensor.fromBlob(rawBytes, shape)
            "uint8" -> Tensor.fromBlobUnsigned(rawBytes, shape)
            else -> error("Unsupported TensorData dtype '$dataType' for ExecuTorch input.")
        }
        Log.i(
            LOG_TAG,
            "Input tensor dtype=${tensor.dtype()} protoDtype=$dataType shape=${
                shape.joinToString(prefix = "[", postfix = "]")
            } bytes=${data.size()}"
        )
        return tensor
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

    private fun ByteArray.toFloatArrayFromHalf(): FloatArray {
        return toShortArray().map { halfBits -> halfToFloat(halfBits) }.toFloatArray()
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

    private fun halfToFloat(rawHalf: Short): Float {
        val half = rawHalf.toInt() and 0xffff
        val sign = (half ushr 15) and 0x1
        val exponent = (half ushr 10) and 0x1f
        val fraction = half and 0x03ff

        val floatBits = when (exponent) {
            0 -> {
                if (fraction == 0) {
                    sign shl 31
                } else {
                    var normalizedFraction = fraction
                    var normalizedExponent = -14
                    while ((normalizedFraction and 0x0400) == 0) {
                        normalizedFraction = normalizedFraction shl 1
                        normalizedExponent--
                    }
                    normalizedFraction = normalizedFraction and 0x03ff
                    (sign shl 31) or ((normalizedExponent + 127) shl 23) or (normalizedFraction shl 13)
                }
            }

            0x1f -> {
                (sign shl 31) or (0xff shl 23) or (fraction shl 13)
            }

            else -> {
                (sign shl 31) or ((exponent - 15 + 127) shl 23) or (fraction shl 13)
            }
        }
        return Float.fromBits(floatBits)
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

    private fun String.hasTrainingMetadata(): Boolean {
        val marker = "__et_training".encodeToByteArray()
        File(this).inputStream().buffered().use { input ->
            val buffer = ByteArray(TRAINING_METADATA_SCAN_BYTES)
            val read = input.read(buffer)
            return read > 0 && buffer.indexOf(marker, read) >= 0
        }
    }

    private fun ByteArray.indexOf(pattern: ByteArray, validLength: Int): Int {
        if (pattern.isEmpty() || validLength < pattern.size) return -1
        for (start in 0..(validLength - pattern.size)) {
            var matched = true
            for (offset in pattern.indices) {
                if (this[start + offset] != pattern[offset]) {
                    matched = false
                    break
                }
            }
            if (matched) return start
        }
        return -1
    }

    private fun EValue.describeForLog(): String {
        return when {
            isTensor() -> {
                val tensor = toTensor()
                "tensor(dtype=${tensor.dtype()},shape=${
                    tensor.shape().joinToString(prefix = "[", postfix = "]")
                })"
            }

            isNone() -> "none"
            isInt() -> "int"
            isDouble() -> "double"
            isBool() -> "bool"
            isString() -> "string"
            else -> "unknown"
        }
    }

    private data class InvocationResult(
        val runtimeName: String,
        val methodName: String,
        val inputCount: Int,
        val outputs: Array<EValue>,
        val evalOnly: Boolean,
        val learningRate: Double,
        val optimizerStepApplied: Boolean,
        val checkpointSaved: Boolean,
        val checkpointIntervalSteps: Long,
        val timing: ShardExecutionTiming
    )

    private sealed interface LoadedRuntime : Closeable {
        fun execute(request: Sid.ForwardChunkRequest): InvocationResult
    }

    private class TrainingLoadedRuntime(
        private val modelPath: String,
        private val module: TrainingModule
    ) : LoadedRuntime {
        private var optimizer: MobileAdamW? = null
        private var optimizerLearningRate: Double? = null
        private var parameters: Map<String, Tensor>? = null
        private var pendingOptimizerSnapshot: MobileAdamW.Snapshot? = null
        private var checkpointRestoreAttempted = false
        private var optimizerStepCount = 0L
        private val committedRequestKeys = LinkedHashSet<String>()

        override fun execute(request: Sid.ForwardChunkRequest): InvocationResult {
            val learningRate = request.trainingLearningRate()
            val checkpointRestoreStartedAtNs = System.nanoTime()
            ensureCheckpointRestored(learningRate)
            val checkpointRestoreMs = elapsedMsSince(checkpointRestoreStartedAtNs)
            val requestKey = request.commitKey()
            val effectiveRequest = if (!request.evalOnly && committedRequestKeys.contains(requestKey)) {
                Log.w(
                    LOG_TAG,
                    "Duplicate train request detected after local commit; rerunning forward without optimizer step key=$requestKey"
                )
                request.toBuilder().setEvalOnly(true).build()
            } else {
                request
            }
            val inputBuildStartedAtNs = System.nanoTime()
            val inputs = tracedSection("sid_build_training_inputs") {
                buildTrainingInputs(effectiveRequest)
            }
            val inputBuildMs = elapsedMsSince(inputBuildStartedAtNs)

            val executeStartedAtNs = System.nanoTime()
            val outputs = tracedSection("sid_execute_forward_backward") {
                module.executeForwardBackward(DEFAULT_METHOD, *inputs)
            }
            val executeMs = elapsedMsSince(executeStartedAtNs)

            var optimizerCreateMs = 0L
            var optimizerStepMs = 0L
            var gradientsMs = 0L
            var gradientCount = 0
            var optimizerStepApplied = false
            var checkpointSaved = false

            if (!effectiveRequest.evalOnly) {
                val gradientsStartedAtNs = System.nanoTime()
                val gradients = tracedSection("sid_named_gradients") {
                    module.namedGradients(DEFAULT_METHOD)
                }
                gradientsMs = elapsedMsSince(gradientsStartedAtNs)
                gradientCount = gradients.size
                require(gradients.isNotEmpty()) {
                    "TrainingModule.executeForwardBackward() produced no gradients for $modelPath."
                }
                val adamW = optimizer.takeIf { optimizerLearningRate == learningRate } ?: run {
                    val optimizerStartedAtNs = System.nanoTime()
                    tracedSection("sid_create_adamw") {
                        createOptimizer(learningRate)
                    }.also {
                        optimizer = it
                        optimizerLearningRate = learningRate
                        optimizerCreateMs = elapsedMsSince(optimizerStartedAtNs)
                    }
                }
                val stepStartedAtNs = System.nanoTime()
                tracedSection("sid_adamw_step") {
                    adamW.step(gradients)
                }
                optimizerStepMs = elapsedMsSince(stepStartedAtNs)
                optimizerStepApplied = true
                committedRequestKeys += requestKey
                optimizerStepCount += 1
                checkpointSaved = maybeSaveCheckpointAfterStep()
            } else {
                Log.i(
                    LOG_TAG,
                    "Eval-only request for $modelPath: skipped namedGradients(), AdamW.create(), and AdamW.step()."
                )
            }
            Log.i(
                LOG_TAG,
                "TrainingModule.executeForwardBackward() succeeded for $modelPath " +
                    "with ${inputs.size} inputs evalOnly=${effectiveRequest.evalOnly} " +
                    "optimizerStepApplied=$optimizerStepApplied gradients=$gradientCount " +
                    "optimizerStepCount=$optimizerStepCount lr=$learningRate " +
                    "checkpointSaved=$checkpointSaved checkpointIntervalSteps=$TRAINING_CHECKPOINT_INTERVAL_STEPS " +
                    "inputBuildMs=$inputBuildMs executeMs=$executeMs gradientsMs=$gradientsMs " +
                    "optimizerCreateMs=$optimizerCreateMs optimizerStepMs=$optimizerStepMs"
            )
            return InvocationResult(
                runtimeName = "TrainingModule",
                methodName = DEFAULT_METHOD,
                inputCount = inputs.size,
                outputs = outputs,
                evalOnly = effectiveRequest.evalOnly,
                learningRate = learningRate,
                optimizerStepApplied = optimizerStepApplied,
                checkpointSaved = checkpointSaved,
                checkpointIntervalSteps = TRAINING_CHECKPOINT_INTERVAL_STEPS,
                timing = ShardExecutionTiming(
                    checkpointRestoreMs = checkpointRestoreMs,
                    inputBuildMs = inputBuildMs,
                    executeMs = executeMs,
                    gradientsMs = gradientsMs,
                    optimizerCreateMs = optimizerCreateMs,
                    optimizerStepMs = optimizerStepMs
                )
            )
        }

        private fun ensureCheckpointRestored(learningRate: Double) {
            if (checkpointRestoreAttempted) {
                return
            }
            checkpointRestoreAttempted = true
            val params = parametersForRuntime()
            var checkpointHasOptimizerState = false
            val result = TrainingCheckpointStore.restoreLatest(modelPath, params) { snapshot ->
                checkpointHasOptimizerState = true
                val existingOptimizer = optimizer
                if (existingOptimizer != null && optimizerLearningRate == learningRate) {
                    existingOptimizer.restore(snapshot)
                } else {
                    pendingOptimizerSnapshot = snapshot
                }
            }
            if (result.restored) {
                optimizerStepCount = result.step
            }
            Log.i(
                LOG_TAG,
                "Checkpoint restore status for $modelPath restored=${result.restored} " +
                    "step=${result.step} parameters=${result.parameterCount} " +
                    "optimizerState=$checkpointHasOptimizerState path=${result.checkpointPath} message=${result.message}"
            )
        }

        private fun parametersForRuntime(): Map<String, Tensor> {
            val cached = parameters
            if (cached != null) {
                return cached
            }
            val loaded = module.namedParameters(DEFAULT_METHOD)
            require(loaded.isNotEmpty()) {
                "TrainingModule has no trainable parameters for $modelPath."
            }
            parameters = loaded
            return loaded
        }

        private fun maybeSaveCheckpointAfterStep(): Boolean {
            if (!shouldSaveCheckpointAfterStep()) {
                Log.i(
                    LOG_TAG,
                    "Skipping checkpoint for $modelPath at step=$optimizerStepCount " +
                        "interval=$TRAINING_CHECKPOINT_INTERVAL_STEPS"
                )
                return false
            }
            runCatching {
                TrainingCheckpointStore.saveLatest(
                    modelPath,
                    optimizerStepCount,
                    parametersForRuntime(),
                    optimizer?.snapshot()
                )
            }.onFailure { error ->
                Log.e(
                    LOG_TAG,
                    "Failed to save training checkpoint for $modelPath at step=$optimizerStepCount: ${error.message}",
                    error
                )
            }
            return true
        }

        private fun shouldSaveCheckpointAfterStep(): Boolean {
            if (TRAINING_CHECKPOINT_INTERVAL_STEPS <= 1L) {
                return true
            }
            return optimizerStepCount == 1L || optimizerStepCount % TRAINING_CHECKPOINT_INTERVAL_STEPS == 0L
        }

        private fun createOptimizer(learningRate: Double): MobileAdamW {
            val parameters = parametersForRuntime()
            Log.i(
                LOG_TAG,
                "Creating mobile AdamW optimizer for $modelPath " +
                    "parameters=${parameters.size} lr=$learningRate"
            )
            val adamW = MobileAdamW(parameters, learningRate)
            pendingOptimizerSnapshot?.let { snapshot ->
                adamW.restore(snapshot)
                pendingOptimizerSnapshot = null
                Log.i(
                    LOG_TAG,
                    "Restored pending AdamW optimizer state for $modelPath " +
                        "parameters=${snapshot.parameterStates.size} step=${snapshot.stepCount}"
                )
            }
            return adamW
        }

        private fun Sid.ForwardChunkRequest.commitKey(): String {
            val stableRequestId = requestId.ifBlank { "batch-$batchId" }
            return "$stableRequestId|batch=$batchId|chunk=$chunkIdx"
        }

        private fun Sid.ForwardChunkRequest.trainingLearningRate(): Double {
            return if (learningRate > 0f) {
                learningRate.toDouble()
            } else {
                DEFAULT_TRAINING_LEARNING_RATE
            }
        }

        override fun close() {
            // executorch-android 1.2.0 Maven AAR does not expose TrainingModule.close().
        }
    }

    private class InferenceLoadedRuntime(
        private val modelPath: String,
        private val module: Module,
        private val methodName: String
    ) : LoadedRuntime {
        override fun execute(request: Sid.ForwardChunkRequest): InvocationResult {
            val inputBuildStartedAtNs = System.nanoTime()
            val candidateInputs = tracedSection("sid_build_inference_inputs") {
                buildInferenceInputCandidates(request)
            }
            val inputBuildMs = elapsedMsSince(inputBuildStartedAtNs)
            var lastFailure: Throwable? = null
            for (inputs in candidateInputs) {
                try {
                    val executeStartedAtNs = System.nanoTime()
                    val outputs = tracedSection("sid_module_execute") {
                        module.execute(methodName, *inputs)
                    }
                    val executeMs = elapsedMsSince(executeStartedAtNs)
                    Log.i(
                        LOG_TAG,
                        "Module.execute() succeeded for $modelPath method=$methodName with ${inputs.size} inputs " +
                            "inputBuildMs=$inputBuildMs executeMs=$executeMs"
                    )
                    return InvocationResult(
                        runtimeName = "Module",
                        methodName = methodName,
                        inputCount = inputs.size,
                        outputs = outputs,
                        evalOnly = request.evalOnly,
                        learningRate = 0.0,
                        optimizerStepApplied = false,
                        checkpointSaved = false,
                        checkpointIntervalSteps = TRAINING_CHECKPOINT_INTERVAL_STEPS,
                        timing = ShardExecutionTiming(
                            inputBuildMs = inputBuildMs,
                            executeMs = executeMs
                        )
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

    private fun elapsedMsSince(startedAtNs: Long): Long {
        return (System.nanoTime() - startedAtNs) / 1_000_000
    }
}
