package com.example.sid_coordinator

data class PersistedRequestState(
    val requestId: String,
    val batchId: Int,
    val latestChunkIdx: Int,
    val latestStageId: Int,
    val latestNodeId: Int,
    val latestEventType: String,
    val latestSuccess: Boolean,
    val latestMessage: String,
    val firstSeenEpochMs: Long,
    val lastUpdatedEpochMs: Long,
    val terminal: Boolean,
    val storedPayload: Boolean,
    val submitAttempts: Int,
    val lastSubmitEpochMs: Long?
)

data class PersistedRequestEvent(
    val eventId: Long,
    val requestId: String,
    val batchId: Int,
    val chunkIdx: Int,
    val stageId: Int,
    val nodeId: Int,
    val eventType: String,
    val success: Boolean,
    val message: String,
    val eventEpochMs: Long,
    val terminal: Boolean
)

data class PersistedRequestPayload(
    val requestId: String,
    val payloadProto: ByteArray,
    val submitAttempts: Int,
    val lastSubmitEpochMs: Long,
    val createdEpochMs: Long
)

data class PersistedRequestMetric(
    val metricId: Long = 0L,
    val runId: String,
    val requestId: String,
    val requestIndex: Int?,
    val attempt: Int = 0,
    val batchId: Int,
    val evalOnly: Boolean,
    val submittedEpochMs: Long,
    val completedEpochMs: Long,
    val elapsedMs: Long,
    val success: Boolean,
    val terminal: Boolean,
    val processedStageId: Int,
    val processedChunkIdx: Int,
    val outputHiddenBytes: Int,
    val outputShiftLogPBytes: Int,
    val localLoss: Float?,
    val tokenCorrect: Int,
    val tokenCount: Int,
    val message: String
)

data class PersistedRunSummary(
    val runId: String,
    val pipelineName: String,
    val modelShards: String,
    val configHash: String,
    val firstSeenEpochMs: Long,
    val lastUpdatedEpochMs: Long,
    val requestRows: Int,
    val successRows: Int,
    val failedRows: Int,
    val evalOnlyRows: Int,
    val trainRows: Int,
    val avgElapsedMs: Double?,
    val avgLoss: Double?,
    val tokenCorrect: Long,
    val tokenCount: Long
) {
    val tokenAccuracy: Double
        get() = if (tokenCount == 0L) 0.0 else tokenCorrect.toDouble() / tokenCount.toDouble()
}

private val PREPARED_REQUEST_SUFFIX = Regex("""-(\d{6})$""")

fun inferRunId(requestId: String): String {
    val trimmed = requestId.trim()
    if (trimmed.isBlank()) {
        return "unknown"
    }
    return PREPARED_REQUEST_SUFFIX.replace(trimmed, "").ifBlank { trimmed }
}

fun inferRequestIndex(requestId: String): Int? {
    return PREPARED_REQUEST_SUFFIX.find(requestId.trim())
        ?.groupValues
        ?.getOrNull(1)
        ?.toIntOrNull()
}

data class PersistedStageTimingMetric(
    val stageMetricId: Long = 0L,
    val requestEventId: Long,
    val runId: String,
    val requestId: String,
    val batchId: Int,
    val chunkIdx: Int,
    val stageId: Int,
    val nodeId: Int,
    val eventType: String,
    val eventEpochMs: Long,
    val runtime: String?,
    val method: String?,
    val inputCount: Int?,
    val evalOnly: Boolean?,
    val optimizerStepApplied: Boolean?,
    val localLoss: Float?,
    val localMs: Long?,
    val inputBuildMs: Long?,
    val executeMs: Long?,
    val gradientsMs: Long?,
    val optimizerCreateMs: Long?,
    val optimizerStepMs: Long?,
    val outputConvertMs: Long?,
    val totalMeasuredMs: Long?,
    val forwardMs: Long?,
    val totalStageMs: Long?,
    val outputBytes: Int?,
    val message: String
)

data class PersistedWorkerTelemetry(
    val telemetryId: Long = 0L,
    val observedEpochMs: Long,
    val deviceId: String,
    val nodeId: Int,
    val stageId: Int,
    val isActive: Boolean,
    val batteryLevel: Float,
    val isCharging: Boolean,
    val powerSource: String,
    val batteryStatus: Int,
    val batteryTempC: Float?,
    val batteryVoltageMv: Int?,
    val batteryCurrentUa: Long?,
    val thermalStatus: String,
    val appPssKb: Long?,
    val appPrivateDirtyKb: Long?,
    val runtimeUsedMemoryKb: Long?,
    val workerState: String
)
