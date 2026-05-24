package com.example.sid_coordinator

import com.google.protobuf.ByteString
import io.grpc.ManagedChannelBuilder
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.Instant
import kotlin.math.max

private data class ModelShape(
    val hiddenDim: Int,
    val vocabSize: Int
)

private val MODEL_SHAPES = mapOf(
    "tinyllama" to ModelShape(hiddenDim = 2048, vocabSize = 32000),
    "phi2" to ModelShape(hiddenDim = 2560, vocabSize = 51200),
    "smollm2_360m" to ModelShape(hiddenDim = 960, vocabSize = 49152),
)

fun main(args: Array<String>) {
    val host = args.getOrNull(0) ?: "127.0.0.1"
    val port = args.getOrNull(1)?.toIntOrNull() ?: 50051
    val requestId = args.getOrNull(2) ?: "demo-${Instant.now().toEpochMilli()}"
    val modelPreset = args.getOrNull(3)?.lowercase() ?: "tinyllama"
    val seqLen = max(2, args.getOrNull(4)?.toIntOrNull() ?: 64)
    val batchSize = max(1, args.getOrNull(5)?.toIntOrNull() ?: 1)
    val chunkIdx = max(0, args.getOrNull(6)?.toIntOrNull() ?: 0)
    val shapes = MODEL_SHAPES[modelPreset]
        ?: error("Unknown model preset '$modelPreset'. Supported: ${MODEL_SHAPES.keys.joinToString()}")
    val maxMessageSizeBytes = 50 * 1024 * 1024

    val channel = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .maxInboundMessageSize(maxMessageSizeBytes)
        .build()

    try {
        val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
        val hiddenStates = buildFloatTensor(
            shape = listOf(batchSize, seqLen, shapes.hiddenDim),
            generator = { index -> ((index % 97) * 0.01f) - 0.5f }
        )
        val attentionMask = buildFloatTensor(
            shape = listOf(batchSize, 1, seqLen, seqLen),
            generator = { 0f }
        )
        val positionIds = buildLongTensor(
            shape = listOf(batchSize, seqLen),
            generator = { index -> (index % seqLen).toLong() }
        )
        val labels = buildLongTensor(
            shape = listOf(batchSize, seqLen),
            generator = { index -> ((index % 16) + 1).toLong() }
        )
        val previousLogProbs = if (chunkIdx == 0) {
            emptyTensor("float32")
        } else {
            buildFloatTensor(
                shape = listOf(batchSize, seqLen, shapes.vocabSize),
                generator = { index -> -((index % 23) * 0.02f) }
            )
        }

        val request = Sid.ForwardChunkRequest.newBuilder()
            .setRequestId(requestId)
            .setBatchId(1)
            .setChunkIdx(chunkIdx)
            .setHiddenStates(hiddenStates)
            .setAttentionMask(attentionMask)
            .setPositionIds(positionIds)
            .setLabels(labels)
            .setShiftLogPPrev(previousLogProbs)
            .build()

        val response = kotlinx.coroutines.runBlocking {
            stub.submitRequest(request)
        }
        println("requestId=$requestId")
        println("modelPreset=$modelPreset")
        println("seqLen=$seqLen")
        println("batchSize=$batchSize")
        println("chunkIdx=$chunkIdx")
        println("success=${response.success}")
        println("message=${response.message}")
        println("processedStageId=${response.processedStageId}")
        println("processedChunkIdx=${response.processedChunkIdx}")
        println("terminal=${response.terminal}")
        println("outputHiddenBytes=${response.outputHiddenStates.data.size()}")
    } finally {
        channel.shutdownNow()
    }
}

private fun buildFloatTensor(
    shape: List<Int>,
    generator: (Int) -> Float
): Sid.TensorData {
    val elementCount = shape.fold(1L) { acc, dim -> acc * dim }
    require(elementCount <= Int.MAX_VALUE) {
        "Float tensor is too large for this submit demo: shape=$shape"
    }

    val bytes = ByteBuffer.allocate(elementCount.toInt() * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .apply {
            val buffer = asFloatBuffer()
            repeat(elementCount.toInt()) { index ->
                buffer.put(generator(index))
            }
        }
        .array()

    return Sid.TensorData.newBuilder()
        .setData(ByteString.copyFrom(bytes))
        .addAllShape(shape)
        .setDataType("float32")
        .build()
}

private fun buildLongTensor(
    shape: List<Int>,
    generator: (Int) -> Long
): Sid.TensorData {
    val elementCount = shape.fold(1L) { acc, dim -> acc * dim }
    require(elementCount <= Int.MAX_VALUE) {
        "Long tensor is too large for this submit demo: shape=$shape"
    }

    val bytes = ByteBuffer.allocate(elementCount.toInt() * Long.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .apply {
            val buffer = asLongBuffer()
            repeat(elementCount.toInt()) { index ->
                buffer.put(generator(index))
            }
        }
        .array()

    return Sid.TensorData.newBuilder()
        .setData(ByteString.copyFrom(bytes))
        .addAllShape(shape)
        .setDataType("int64")
        .build()
}

private fun emptyTensor(dataType: String): Sid.TensorData {
    return Sid.TensorData.newBuilder()
        .setData(ByteString.EMPTY)
        .setDataType(dataType)
        .build()
}
