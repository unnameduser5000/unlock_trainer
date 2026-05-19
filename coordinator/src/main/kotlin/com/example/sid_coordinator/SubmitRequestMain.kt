package com.example.sid_coordinator

import com.google.protobuf.ByteString
import io.grpc.ManagedChannelBuilder
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.Instant

fun main(args: Array<String>) {
    val host = args.getOrNull(0) ?: "127.0.0.1"
    val port = args.getOrNull(1)?.toIntOrNull() ?: 50051
    val requestId = args.getOrNull(2) ?: "demo-${Instant.now().toEpochMilli()}"

    val channel = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .build()

    try {
        val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
        val hiddenStateBytes = ByteBuffer.allocate(4 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .apply {
                asFloatBuffer().put(floatArrayOf(1f, 2f, 3f, 4f))
            }
            .array()
        val request = Sid.ForwardChunkRequest.newBuilder()
            .setRequestId(requestId)
            .setBatchId(1)
            .setChunkIdx(0)
            .setHiddenStates(
                Sid.TensorData.newBuilder()
                    .setData(ByteString.copyFrom(hiddenStateBytes))
                    .addShape(4)
                    .setDataType("float32")
                    .build()
            )
            .setAttentionMask(
                Sid.TensorData.newBuilder()
                    .setData(ByteString.EMPTY)
                    .setDataType("int32")
                    .build()
            )
            .setPositionIds(
                Sid.TensorData.newBuilder()
                    .setData(ByteString.EMPTY)
                    .setDataType("int32")
                    .build()
            )
            .setLabels(
                Sid.TensorData.newBuilder()
                    .setData(ByteString.EMPTY)
                    .setDataType("int32")
                    .build()
            )
            .setShiftLogPPrev(
                Sid.TensorData.newBuilder()
                    .setData(ByteString.EMPTY)
                    .setDataType("float32")
                    .build()
            )
            .build()

        val response = kotlinx.coroutines.runBlocking {
            stub.submitRequest(request)
        }
        println("requestId=$requestId")
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
