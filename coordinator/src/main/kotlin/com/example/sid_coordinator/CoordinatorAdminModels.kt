package com.example.sid_coordinator

data class AdminSummarySnapshot(
    val pipelineName: String,
    val routingEpoch: Long,
    val leaseDurationSeconds: Long,
    val stageCount: Int,
    val liveNodeCount: Int,
    val inactiveNodeCount: Int,
    val offlineStageCount: Int,
    val drainedStageCount: Int,
    val schedulerEnabled: Boolean,
    val schedulerPolicy: String,
    val schedulerAllowUnlistedDevices: Boolean,
    val schedulerPreferConfiguredDevices: Boolean,
    val schedulerRebalanceOnRegister: Boolean,
    val schedulerEventCount: Int
)

data class AdminNodeSnapshot(
    val nodeId: Int,
    val stageId: Int,
    val deviceId: String,
    val ipAddress: String,
    val grpcPort: Int,
    val computeCapacity: Float,
    val memoryGb: Float,
    val assignmentReason: String,
    val registeredAtEpochMs: Long,
    val lastHeartbeatAtEpochMs: Long,
    val isActive: Boolean,
    val isExpired: Boolean,
    val isLive: Boolean
)

data class AdminNextHopSnapshot(
    val nodeId: Int,
    val ipAddress: String,
    val grpcPort: Int
)

data class AdminStageSnapshot(
    val stageId: Int,
    val deviceId: String,
    val preferredDeviceId: String,
    val minMemoryGb: Float,
    val minComputeCapacity: Float,
    val schedulingWeight: Float,
    val modelShardId: String,
    val expectedHost: String,
    val expectedPort: Int,
    val drained: Boolean,
    val terminal: Boolean,
    val routeReady: Boolean,
    val assignedNode: AdminNodeSnapshot?,
    val nextHop: AdminNextHopSnapshot?
)

data class AdminSchedulerEventSnapshot(
    val eventId: Long,
    val eventEpochMs: Long,
    val action: String,
    val stageId: Int?,
    val nodeId: Int?,
    val deviceId: String?,
    val reason: String,
    val message: String
)

data class AdminStatusSnapshot(
    val summary: AdminSummarySnapshot,
    val stages: List<AdminStageSnapshot>,
    val nodes: List<AdminNodeSnapshot>,
    val schedulerEvents: List<AdminSchedulerEventSnapshot>
)

data class AdminRequestStateSnapshot(
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
    val lifecycleState: String,
    val stalled: Boolean,
    val lastUpdatedAgeSeconds: Long,
    val storedPayload: Boolean,
    val submitAttempts: Int,
    val lastSubmitEpochMs: Long?
)

data class AdminRequestEventSnapshot(
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

data class AdminRequestDetailSnapshot(
    val state: AdminRequestStateSnapshot?,
    val events: List<AdminRequestEventSnapshot>
)
