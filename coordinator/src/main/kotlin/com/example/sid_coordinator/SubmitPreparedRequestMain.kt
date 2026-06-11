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
    val beliefTransportMode = normalizeBeliefTransportMode(args.getOrNull(6))
    val maxMessageSizeBytes = 50 * 1024 * 1024

    val record = readManifestRecord(manifestPath, recordIndex)
    val manifestDir = manifestPath.toAbsolutePath().normalize().parent
    val requestId = requestIdOverride.ifBlank { record.request_id }
    val validLabels = record.countValidLabels(manifestDir)
    val request = record.toForwardChunkRequest(
        manifestDir = manifestDir,
        requestIdOverride = requestId,
        evalOnly = evalOnly,
        beliefTransportMode = beliefTransportMode
    )

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
            computeShiftedTokenPredictionMetrics(
                response.outputShiftLogP,
                request.labels,
                record.singleTokenLabelChoices()
            )
        } else {
            TokenPredictionMetrics(correct = 0, count = 0)
        }
        val maxPssPeakKb = response.stageMetricsList.maxOfOrNull { it.pssPeakKb } ?: 0L
        val maxPrivateDirtyPeakKb = response.stageMetricsList.maxOfOrNull { it.privateDirtyPeakKb } ?: 0L
        val maxJavaHeapPeakKb = response.stageMetricsList.maxOfOrNull { it.javaHeapPeakKb } ?: 0L
        println("requestId=$requestId")
        println("manifest=$manifestPath")
        println("recordIndex=$recordIndex")
        println("validLabels=$validLabels")
        println("evalOnly=$evalOnly")
        println("beliefTransportMode=$beliefTransportMode")
        println("success=${response.success}")
        println("message=${response.message}")
        println("processedStageId=${response.processedStageId}")
        println("processedChunkIdx=${response.processedChunkIdx}")
        println("terminal=${response.terminal}")
        println("outputHiddenBytes=${response.outputHiddenStates.data.size()}")
        println("outputShiftLogPBytes=${response.outputShiftLogP.data.size()}")
        println("maxPssPeakKb=$maxPssPeakKb")
        println("maxPrivateDirtyPeakKb=$maxPrivateDirtyPeakKb")
        println("maxJavaHeapPeakKb=$maxJavaHeapPeakKb")
        response.stageMetricsList.forEach { metric ->
            println(
                "stageMemory stage=${metric.stageId} chunk=${metric.chunkIdx} " +
                    "node=${metric.nodeId} device=${metric.deviceId.ifBlank { "unknown" }} " +
                    "pssBeforeKb=${metric.pssBeforeKb} pssPeakKb=${metric.pssPeakKb} " +
                    "pssDeltaPeakKb=${metric.pssPeakKb - metric.pssBeforeKb} " +
                    "privateDirtyBeforeKb=${metric.privateDirtyBeforeKb} " +
                    "privateDirtyPeakKb=${metric.privateDirtyPeakKb} " +
                    "privateDirtyDeltaPeakKb=${metric.privateDirtyPeakKb - metric.privateDirtyBeforeKb} " +
                    "javaHeapBeforeKb=${metric.javaHeapBeforeKb} javaHeapPeakKb=${metric.javaHeapPeakKb} " +
                    "javaHeapDeltaPeakKb=${metric.javaHeapPeakKb - metric.javaHeapBeforeKb} " +
                    "samples=${metric.memorySampleCount} intervalMs=${metric.memorySampleIntervalMs}"
            )
        }
        println("localLoss=${response.localLoss}")
        println("tokenCorrect=${metrics.correct}")
        println("tokenCount=${metrics.count}")
        println("tokenAccuracy=${metrics.accuracy}")
        println("labelChoiceCorrect=${metrics.labelChoiceCorrect}")
        println("labelChoiceCount=${metrics.labelChoiceCount}")
        println("labelChoiceAccuracy=${metrics.labelChoiceAccuracy}")
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
