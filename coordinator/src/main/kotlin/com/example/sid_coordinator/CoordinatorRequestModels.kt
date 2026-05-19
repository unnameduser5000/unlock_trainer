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
