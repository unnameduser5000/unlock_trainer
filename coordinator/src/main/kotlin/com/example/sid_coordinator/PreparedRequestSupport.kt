package com.example.sid_coordinator

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.protobuf.ByteString
import sid.Sid
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path

data class PreparedTensorRecord(
    val path: String,
    val dtype: String,
    val shape: List<Int>
)

data class PreparedLabelChoice(
    val text: String? = null,
    val token_ids: List<Int> = emptyList()
)

data class PreparedRequestRecord(
    val request_id: String,
    val batch_id: Int = 1,
    val chunk_idx: Int = 0,
    val model_name: String? = null,
    val dataset: String? = null,
    val dataset_index: Int? = null,
    val seq_len: Int? = null,
    val prompt_token_count: Int? = null,
    val valid_label_count: Int? = null,
    val learning_rate: Float? = null,
    val label_choices: List<PreparedLabelChoice>? = null,
    val tensors: Map<String, PreparedTensorRecord>
)

data class IndexedPreparedRequestRecord(
    val index: Int,
    val record: PreparedRequestRecord
)

data class TokenPredictionMetrics(
    val correct: Int,
    val count: Int,
    val labelChoiceCorrect: Int = 0,
    val labelChoiceCount: Int = 0
) {
    val accuracy: Double
        get() = if (count == 0) 0.0 else correct.toDouble() / count.toDouble()

    val labelChoiceAccuracy: Double
        get() = if (labelChoiceCount == 0) 0.0 else labelChoiceCorrect.toDouble() / labelChoiceCount.toDouble()
}

const val DEFAULT_BELIEF_TRANSPORT_MODE = "full"

fun normalizeBeliefTransportMode(rawMode: String?): String {
    return when (rawMode.orEmpty().trim().lowercase()) {
        "", "full", "dense" -> "full"
        "terminal", "terminal_only", "final", "final_only" -> "terminal"
        "none", "off", "disabled", "false" -> "none"
        else -> error("Unsupported belief transport mode '$rawMode'. Use full, terminal, or none.")
    }
}

fun readManifestRecord(path: Path, index: Int): PreparedRequestRecord {
    require(index >= 0) { "record index must be non-negative" }
    val line = Files.newBufferedReader(path).useLines { lines ->
        lines.drop(index).firstOrNull()
    } ?: error("No record at index $index in $path")
    return parsePreparedRecord(line)
}

fun readManifestRecords(path: Path): List<IndexedPreparedRequestRecord> {
    return Files.newBufferedReader(path).useLines { lines ->
        lines.mapIndexedNotNull { index, line ->
            if (line.isBlank()) {
                null
            } else {
                IndexedPreparedRequestRecord(index, parsePreparedRecord(line))
            }
        }.toList()
    }
}

fun PreparedRequestRecord.toForwardChunkRequest(
    manifestDir: Path,
    requestIdOverride: String = "",
    evalOnly: Boolean = false,
    beliefTransportMode: String = DEFAULT_BELIEF_TRANSPORT_MODE
): Sid.ForwardChunkRequest {
    val requestId = requestIdOverride.ifBlank { request_id }
    val normalizedBeliefMode = normalizeBeliefTransportMode(beliefTransportMode)
    return Sid.ForwardChunkRequest.newBuilder()
        .setRequestId(requestId)
        .setBatchId(batch_id)
        .setChunkIdx(chunk_idx)
        .setEvalOnly(evalOnly)
        .setBeliefTransportMode(normalizedBeliefMode)
        .setLearningRate(learning_rate?.takeIf { it > 0f } ?: 0f)
        .setHiddenStates(requiredTensor("hidden_states", manifestDir))
        .setAttentionMask(requiredTensor("attention_mask", manifestDir))
        .setPositionIds(requiredTensor("position_ids", manifestDir))
        .setLabels(requiredTensor("labels", manifestDir))
        .setShiftLogPPrev(emptyPreparedTensor("float32"))
        .build()
}

fun PreparedRequestRecord.countValidLabels(manifestDir: Path): Int {
    valid_label_count?.let { return it }
    val labelRecord = tensors["labels"] ?: error("Prepared request is missing tensor 'labels'")
    val labelPath = manifestDir.resolve(labelRecord.path).normalize()
    require(Files.exists(labelPath)) { "Tensor file does not exist: $labelPath" }
    val bytes = Files.readAllBytes(labelPath)
    require(bytes.size % Long.SIZE_BYTES == 0) {
        "Expected int64 labels bytes to be a multiple of ${Long.SIZE_BYTES}, got ${bytes.size}."
    }
    val labels = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
    var count = 0
    while (labels.hasRemaining()) {
        if (labels.get() != -100L) {
            count++
        }
    }
    return count
}

fun computeShiftedTokenPredictionMetrics(
    logProbs: Sid.TensorData,
    labels: Sid.TensorData,
    labelChoiceTokenIds: List<Int> = emptyList()
): TokenPredictionMetrics {
    if (logProbs.data.isEmpty || labels.data.isEmpty) {
        return TokenPredictionMetrics(correct = 0, count = 0)
    }
    require(logProbs.shapeCount == 3) {
        "Expected log-probs shape [batch, seq_len, vocab], got ${logProbs.shapeList}"
    }
    require(labels.shapeCount == 2) {
        "Expected labels shape [batch, seq_len], got ${labels.shapeList}"
    }

    val batch = logProbs.shapeList[0]
    val seqLen = logProbs.shapeList[1]
    val vocab = logProbs.shapeList[2]
    require(labels.shapeList[0] == batch && labels.shapeList[1] == seqLen) {
        "Labels shape ${labels.shapeList} does not match log-probs batch/seq ${logProbs.shapeList}."
    }

    val labelValues = labels.toLongArray()
    var correct = 0
    var count = 0
    var labelChoiceCorrect = 0
    var labelChoiceCount = 0
    val constrainedChoices = labelChoiceTokenIds.distinct().filter { it in 0 until vocab }
    for (b in 0 until batch) {
        for (labelPos in 1 until seqLen) {
            val label = labelValues[b * seqLen + labelPos]
            if (label == -100L) {
                continue
            }
            val predicted = logProbs.argmaxAt(batchIndex = b, positionIndex = labelPos - 1, seqLen = seqLen, vocab = vocab)
            if (predicted == label.toInt()) {
                correct++
            }
            count++
            if (constrainedChoices.isNotEmpty() && label.toInt() in constrainedChoices) {
                val constrainedPredicted = logProbs.argmaxAmong(
                    batchIndex = b,
                    positionIndex = labelPos - 1,
                    seqLen = seqLen,
                    vocab = vocab,
                    choices = constrainedChoices
                )
                if (constrainedPredicted == label.toInt()) {
                    labelChoiceCorrect++
                }
                labelChoiceCount++
            }
        }
    }
    return TokenPredictionMetrics(
        correct = correct,
        count = count,
        labelChoiceCorrect = labelChoiceCorrect,
        labelChoiceCount = labelChoiceCount
    )
}

fun PreparedRequestRecord.singleTokenLabelChoices(): List<Int> {
    return label_choices
        .orEmpty()
        .mapNotNull { choice -> choice.token_ids.singleOrNull() }
        .distinct()
}

private fun emptyPreparedTensor(dataType: String): Sid.TensorData {
    return Sid.TensorData.newBuilder()
        .setData(ByteString.EMPTY)
        .setDataType(dataType)
        .build()
}

private fun Sid.TensorData.toLongArray(): LongArray {
    require(dataType.trim().lowercase() in setOf("int64", "long")) {
        "Expected int64 labels, got dtype=$dataType"
    }
    val bytes = data.toByteArray()
    require(bytes.size % Long.SIZE_BYTES == 0) {
        "Expected int64 bytes to be a multiple of ${Long.SIZE_BYTES}, got ${bytes.size}."
    }
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
    return LongArray(buffer.remaining()).also(buffer::get)
}

private fun Sid.TensorData.argmaxAt(
    batchIndex: Int,
    positionIndex: Int,
    seqLen: Int,
    vocab: Int
): Int {
    val bytes = data.toByteArray()
    val base = ((batchIndex * seqLen + positionIndex) * vocab)
    var bestIndex = 0
    var bestValue = Float.NEGATIVE_INFINITY

    when (dataType.trim().lowercase()) {
        "float32", "float" -> {
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (v in 0 until vocab) {
                val value = buffer.getFloat((base + v) * Float.SIZE_BYTES)
                if (value > bestValue) {
                    bestValue = value
                    bestIndex = v
                }
            }
        }

        "float16", "half" -> {
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (v in 0 until vocab) {
                val value = halfToFloat(buffer.getShort((base + v) * Short.SIZE_BYTES))
                if (value > bestValue) {
                    bestValue = value
                    bestIndex = v
                }
            }
        }

        else -> error("Unsupported log-probs dtype '$dataType' for token accuracy.")
    }
    return bestIndex
}

private fun Sid.TensorData.argmaxAmong(
    batchIndex: Int,
    positionIndex: Int,
    seqLen: Int,
    vocab: Int,
    choices: List<Int>
): Int {
    val bytes = data.toByteArray()
    val base = ((batchIndex * seqLen + positionIndex) * vocab)
    var bestIndex = choices.first()
    var bestValue = Float.NEGATIVE_INFINITY

    when (dataType.trim().lowercase()) {
        "float32", "float" -> {
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (choice in choices) {
                val value = buffer.getFloat((base + choice) * Float.SIZE_BYTES)
                if (value > bestValue) {
                    bestValue = value
                    bestIndex = choice
                }
            }
        }

        "float16", "half" -> {
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (choice in choices) {
                val value = halfToFloat(buffer.getShort((base + choice) * Short.SIZE_BYTES))
                if (value > bestValue) {
                    bestValue = value
                    bestIndex = choice
                }
            }
        }

        else -> error("Unsupported log-probs dtype '$dataType' for label-choice accuracy.")
    }
    return bestIndex
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

private fun parsePreparedRecord(line: String): PreparedRequestRecord {
    val type = object : TypeToken<PreparedRequestRecord>() {}.type
    return Gson().fromJson(line, type)
}

private fun PreparedRequestRecord.requiredTensor(
    name: String,
    manifestDir: Path
): Sid.TensorData {
    val record = tensors[name] ?: error("Prepared request is missing tensor '$name'")
    val tensorPath = manifestDir.resolve(record.path).normalize()
    require(Files.exists(tensorPath)) { "Tensor file does not exist: $tensorPath" }
    val bytes = Files.readAllBytes(tensorPath)
    return Sid.TensorData.newBuilder()
        .setData(ByteString.copyFrom(bytes))
        .addAllShape(record.shape)
        .setDataType(record.dtype)
        .build()
}
