package com.example.sid_coordinator

data class StageConfig(
    val stageId: Int,
    val deviceId: String,
    val modelShardId: String,
    val modelArtifactPath: String = "",
    val modelDownloadUrl: String = "",
    val modelSha256: String = "",
    val modelBytes: Long = 0,
    val minMemoryGb: Float = 0f,
    val minComputeCapacity: Float = 0f,
    val schedulingWeight: Float = 1f,
    val expectedHost: String,
    val expectedPort: Int
)

data class SchedulerConfig(
    val enabled: Boolean = false,
    val policy: String = "fill-first",
    val allowUnlistedDevices: Boolean = true,
    val preferConfiguredDevices: Boolean = true,
    val rebalanceOnRegister: Boolean = false,
    val minMemoryGb: Float = 0f,
    val minComputeCapacity: Float = 0f
) {
    fun normalized(): SchedulerConfig {
        val normalizedPolicy = policy.trim().lowercase().ifBlank { "fill-first" }
        require(normalizedPolicy in setOf("fill-first", "memory-first", "compute-first")) {
            "scheduler.policy must be one of fill-first, memory-first, compute-first"
        }
        require(minMemoryGb >= 0f) { "scheduler.minMemoryGb must be non-negative" }
        require(minComputeCapacity >= 0f) { "scheduler.minComputeCapacity must be non-negative" }
        return copy(policy = normalizedPolicy)
    }
}

data class CoordinatorConfig(
    val bindPort: Int = 50051,
    val adminBindPort: Int = 18080,
    val heartbeatLeaseSeconds: Long = 15,
    val cleanupIntervalSeconds: Long = 5,
    val requestStallTimeoutSeconds: Long = 30,
    val resolvedRequestRetentionSeconds: Long = 86_400,
    val artifactBaseUrl: String = "",
    val pipelineName: String = "default",
    val stateDbPath: String = "coordinator/data/coordinator.db",
    val scheduler: SchedulerConfig = SchedulerConfig(),
    val stages: List<StageConfig> = emptyList()
) {
    fun validated(): CoordinatorConfig {
        val normalizedScheduler = scheduler.normalized()
        require(bindPort in 1..65535) { "bindPort must be a valid TCP port" }
        require(adminBindPort in 1..65535) { "adminBindPort must be a valid TCP port" }
        require(adminBindPort != bindPort) { "adminBindPort must be different from bindPort" }
        require(heartbeatLeaseSeconds > 0) { "heartbeatLeaseSeconds must be positive" }
        require(cleanupIntervalSeconds > 0) { "cleanupIntervalSeconds must be positive" }
        require(requestStallTimeoutSeconds > 0) { "requestStallTimeoutSeconds must be positive" }
        require(resolvedRequestRetentionSeconds > 0) {
            "resolvedRequestRetentionSeconds must be positive"
        }
        if (artifactBaseUrl.isNotBlank()) {
            require(
                artifactBaseUrl.startsWith("http://") || artifactBaseUrl.startsWith("https://")
            ) {
                "artifactBaseUrl must start with http:// or https://"
            }
        }
        require(stateDbPath.isNotBlank()) { "stateDbPath must not be blank" }
        require(stages.isNotEmpty()) { "At least one stage must be configured" }

        val sortedStages = stages
            .sortedBy(StageConfig::stageId)
            .map { stage ->
                if (stage.schedulingWeight <= 0f) {
                    stage.copy(schedulingWeight = 1f)
                } else {
                    stage
                }
            }
        val expectedStageIds = sortedStages.indices.toList()
        val actualStageIds = sortedStages.map(StageConfig::stageId)
        require(actualStageIds == expectedStageIds) {
            "Stage ids must be contiguous and start at 0, found $actualStageIds"
        }

        if (!normalizedScheduler.enabled || !normalizedScheduler.allowUnlistedDevices) {
            sortedStages.forEach { stage ->
                require(stage.deviceId.isNotBlank()) {
                    "Stage ${stage.stageId} must declare deviceId when scheduler is disabled or unlisted devices are disallowed"
                }
            }
        }

        val duplicateDeviceIds = sortedStages.mapNotNull { stage ->
            stage.deviceId.trim().takeIf { it.isNotBlank() }?.let { stage.stageId to it }
        }
            .groupBy { it.second }
            .filterValues { it.size > 1 }
            .keys
        require(duplicateDeviceIds.isEmpty()) {
            "Each non-blank stage deviceId must be unique, duplicates: $duplicateDeviceIds"
        }

        sortedStages.forEach { stage ->
            require(stage.minMemoryGb >= 0f) {
                "Stage ${stage.stageId} minMemoryGb must be non-negative"
            }
            require(stage.minComputeCapacity >= 0f) {
                "Stage ${stage.stageId} minComputeCapacity must be non-negative"
            }
            require(stage.schedulingWeight > 0f) {
                "Stage ${stage.stageId} schedulingWeight must be positive"
            }
            require(stage.expectedPort in 1..65535) {
                "Stage ${stage.stageId} has invalid expectedPort ${stage.expectedPort}"
            }
            require(stage.expectedHost.isNotBlank()) {
                "Stage ${stage.stageId} must declare expectedHost"
            }
            require(stage.modelShardId.isNotBlank()) {
                "Stage ${stage.stageId} must declare modelShardId"
            }
            if (artifactBaseUrl.isNotBlank()) {
                require(stage.modelArtifactPath.isNotBlank()) {
                    "Stage ${stage.stageId} must declare modelArtifactPath when artifactBaseUrl is set"
                }
                require(stage.modelDownloadUrl.isNotBlank()) {
                    "Stage ${stage.stageId} must declare modelDownloadUrl when artifactBaseUrl is set"
                }
                require(stage.modelSha256.isNotBlank()) {
                    "Stage ${stage.stageId} must declare modelSha256 when artifactBaseUrl is set"
                }
                require(stage.modelBytes > 0) {
                    "Stage ${stage.stageId} must declare a positive modelBytes when artifactBaseUrl is set"
                }
            }
        }

        return copy(scheduler = normalizedScheduler, stages = sortedStages)
    }
}
