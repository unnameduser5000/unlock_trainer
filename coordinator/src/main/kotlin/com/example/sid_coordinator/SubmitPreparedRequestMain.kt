package com.example.sid_coordinator

import io.grpc.ManagedChannelBuilder
import sid.CoordinatingServiceGrpcKt
import java.nio.file.Paths

fun main(args: Array<String>) {
    val host = args.getOrNull(0) ?: "127.0.0.1"
    val port = args.getOrNull(1)?.toIntOrNull() ?: 50051
    val manifestPath = Paths.get(args.getOrNull(2) ?: "data/sft_requests/requests.jsonl")
    val recordIndex = args.getOrNull(3)?.toIntOrNull() ?: 0
    val requestIdOverride = args.getOrNull(4).orEmpty()
    val evalOnly = args.getOrNull(5)?.toBooleanLenient() ?: false
    val maxMessageSizeBytes = 50 * 1024 * 1024

    val record = readManifestRecord(manifestPath, recordIndex)
    val manifestDir = manifestPath.toAbsolutePath().normalize().parent
    val requestId = requestIdOverride.ifBlank { record.request_id }
    val validLabels = record.countValidLabels(manifestDir)
    val request = record.toForwardChunkRequest(manifestDir, requestId, evalOnly)

    val channel = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .maxInboundMessageSize(maxMessageSizeBytes)
        .build()

    try {
        val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
        val response = kotlinx.coroutines.runBlocking {
            stub.submitRequest(request)
        }
        val metrics = if (response.success && response.terminal) {
            computeShiftedTokenPredictionMetrics(response.outputShiftLogP, request.labels)
        } else {
            TokenPredictionMetrics(correct = 0, count = 0)
        }
        println("requestId=$requestId")
        println("manifest=$manifestPath")
        println("recordIndex=$recordIndex")
        println("validLabels=$validLabels")
        println("evalOnly=$evalOnly")
        println("success=${response.success}")
        println("message=${response.message}")
        println("processedStageId=${response.processedStageId}")
        println("processedChunkIdx=${response.processedChunkIdx}")
        println("terminal=${response.terminal}")
        println("outputHiddenBytes=${response.outputHiddenStates.data.size()}")
        println("outputShiftLogPBytes=${response.outputShiftLogP.data.size()}")
        println("localLoss=${response.localLoss}")
        println("tokenCorrect=${metrics.correct}")
        println("tokenCount=${metrics.count}")
        println("tokenAccuracy=${metrics.accuracy}")
    } finally {
        channel.shutdownNow()
    }
}

private fun String.toBooleanLenient(): Boolean {
    return equals("true", ignoreCase = true) ||
        equals("1") ||
        equals("yes", ignoreCase = true) ||
        equals("eval", ignoreCase = true) ||
        equals("eval_only", ignoreCase = true)
}
