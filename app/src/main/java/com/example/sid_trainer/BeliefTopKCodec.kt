package com.example.sid_trainer

import com.google.protobuf.ByteString
import sid.Sid
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln

object BeliefTopKCodec {
    const val DEFAULT_TOP_K = 16

    private const val DATA_TYPE_PREFIX = "topk_log_probs"
    private const val ENTRY_BYTES = Int.SIZE_BYTES + Float.SIZE_BYTES
    private val byteOrder = ByteOrder.LITTLE_ENDIAN
    private val topKPattern = Regex("""k=(\d+)""")

    fun encodeForTransport(tensor: Sid.TensorData, topK: Int = DEFAULT_TOP_K): Sid.TensorData {
        if (tensor.data.isEmpty || tensor.shapeCount != 3) {
            return tensor
        }
        val dtype = tensor.dataType.trim().lowercase()
        if (dtype !in setOf("float32", "float", "float16", "half")) {
            return tensor
        }

        val batch = tensor.shapeList[0]
        val seqLen = tensor.shapeList[1]
        val vocab = tensor.shapeList[2]
        if (batch <= 0 || seqLen <= 0 || vocab <= 0) {
            return tensor
        }
        val k = topK.coerceIn(1, vocab)
        val vectorCount = batch * seqLen
        val output = ByteBuffer.allocate(vectorCount * k * ENTRY_BYTES).order(byteOrder)
        val bytes = tensor.data.toByteArray()
        val input = ByteBuffer.wrap(bytes).order(byteOrder)

        for (vectorIndex in 0 until vectorCount) {
            val base = vectorIndex * vocab
            val topIndices = IntArray(k)
            val topValues = FloatArray(k) { Float.NEGATIVE_INFINITY }
            for (vocabIndex in 0 until vocab) {
                val elementIndex = base + vocabIndex
                val value = when (dtype) {
                    "float32", "float" -> input.getFloat(elementIndex * Float.SIZE_BYTES)
                    else -> halfToFloat(input.getShort(elementIndex * Short.SIZE_BYTES))
                }
                insertTopK(topIndices, topValues, vocabIndex, value)
            }
            for (slot in 0 until k) {
                output.putInt(topIndices[slot])
                output.putFloat(topValues[slot])
            }
        }

        return Sid.TensorData.newBuilder()
            .setData(ByteString.copyFrom(output.array()))
            .addAllShape(tensor.shapeList)
            .setDataType("$DATA_TYPE_PREFIX:k=$k;value=float32;index=int32")
            .build()
    }

    fun isEncoded(tensor: Sid.TensorData): Boolean {
        return tensor.dataType.trim().lowercase().startsWith(DATA_TYPE_PREFIX)
    }

    fun decodeToDenseFloatArray(tensor: Sid.TensorData): FloatArray? {
        if (!isEncoded(tensor)) {
            return null
        }
        require(tensor.shapeCount == 3) {
            "Top-k belief tensor must preserve dense shape [batch, seq_len, vocab], got ${tensor.shapeList}."
        }
        val batch = tensor.shapeList[0]
        val seqLen = tensor.shapeList[1]
        val vocab = tensor.shapeList[2]
        require(batch > 0 && seqLen > 0 && vocab > 0) {
            "Invalid top-k belief dense shape ${tensor.shapeList}."
        }
        val k = parseTopK(tensor.dataType).coerceIn(1, vocab)
        val vectorCount = batch * seqLen
        val expectedBytes = vectorCount * k * ENTRY_BYTES
        val bytes = tensor.data.toByteArray()
        require(bytes.size == expectedBytes) {
            "Top-k belief payload size mismatch: expected=$expectedBytes actual=${bytes.size} k=$k shape=${tensor.shapeList}."
        }

        val input = ByteBuffer.wrap(bytes).order(byteOrder)
        val dense = FloatArray(vectorCount * vocab)
        for (vectorIndex in 0 until vectorCount) {
            var topMass = 0.0
            val entryBase = vectorIndex * k * ENTRY_BYTES
            for (slot in 0 until k) {
                val value = input.getFloat(entryBase + slot * ENTRY_BYTES + Int.SIZE_BYTES)
                topMass += exp(value.toDouble())
            }

            val rowBase = vectorIndex * vocab
            val logTail = if (k < vocab) {
                val tailMass = (1.0 - topMass).coerceAtLeast(1e-8)
                ln(tailMass / (vocab - k).toDouble()).toFloat()
            } else {
                Float.NEGATIVE_INFINITY
            }
            dense.fill(logTail, rowBase, rowBase + vocab)

            for (slot in 0 until k) {
                val offset = entryBase + slot * ENTRY_BYTES
                val index = input.getInt(offset)
                val value = input.getFloat(offset + Int.SIZE_BYTES)
                if (index in 0 until vocab) {
                    dense[rowBase + index] = value
                }
            }
        }
        return dense
    }

    private fun parseTopK(dataType: String): Int {
        return topKPattern.find(dataType)?.groupValues?.getOrNull(1)?.toIntOrNull()
            ?: error("Top-k belief dtype is missing k: '$dataType'.")
    }

    private fun insertTopK(indices: IntArray, values: FloatArray, index: Int, value: Float) {
        val last = values.lastIndex
        if (value <= values[last]) {
            return
        }
        var slot = last
        while (slot > 0 && value > values[slot - 1]) {
            values[slot] = values[slot - 1]
            indices[slot] = indices[slot - 1]
            slot--
        }
        values[slot] = value
        indices[slot] = index
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
}
