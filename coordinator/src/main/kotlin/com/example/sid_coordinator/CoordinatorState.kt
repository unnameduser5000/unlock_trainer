package com.example.sid_coordinator

import org.slf4j.LoggerFactory
import sid.Sid
import java.time.Duration
import java.time.Instant
import kotlin.math.max
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class RegisteredNode(
    val nodeId: Int,
    val stageId: Int,
    val deviceId: String,
    var ipAddress: String,
    var grpcPort: Int,
    var computeCapacity: Float,
    var memoryGb: Float,
    var registeredAt: Instant,
    var lastHeartbeatAt: Instant,
    var isActive: Boolean
) {
    fun isExpired(now: Instant, leaseDuration: Duration): Boolean {
        return now.isAfter(lastHeartbeatAt.plus(leaseDuration))
    }

    fun isLive(now: Instant, leaseDuration: Duration): Boolean {
        return isActive && !isExpired(now, leaseDuration)
    }
}

data class StageArtifactHandle(
    val stageId: Int,
    val modelShardId: String,
    val filePath: String,
    val sha256: String,
    val bytes: Long
)

class CoordinatorState(
    initialConfig: CoordinatorConfig,
    private val persistence: CoordinatorPersistence
) {
    private val logger = LoggerFactory.getLogger(CoordinatorState::class.java)
    private var config: CoordinatorConfig = initialConfig
    private var leaseDuration = Duration.ofSeconds(initialConfig.heartbeatLeaseSeconds)
    private val nextNodeId: AtomicInteger
    private val routingEpoch: AtomicLong
    private val lock = ReentrantLock()
    private var stagesByDeviceId = initialConfig.stages.associateBy(StageConfig::deviceId)
    private val nodesById = mutableMapOf<Int, RegisteredNode>()
    private val nodeIdByStageId = mutableMapOf<Int, Int>()
    private val drainedStageIds = mutableSetOf<Int>()

    init {
        val snapshot = persistence.loadSnapshot()
        routingEpoch = AtomicLong(snapshot.routingEpoch)
        nextNodeId = AtomicInteger(snapshot.nextNodeId)
        drainedStageIds.addAll(snapshot.drainedStageIds)

        snapshot.nodes.forEach { restored ->
            val offlineNode = restored.copy(isActive = false)
            nodesById[offlineNode.nodeId] = offlineNode
            nodeIdByStageId[offlineNode.stageId] = offlineNode.nodeId
        }
        if (snapshot.nodes.isNotEmpty()) {
            persistence.markAllNodesInactive()
            logger.info(
                "Restored {} node records from SQLite. They will remain offline until heartbeat or re-registration.",
                snapshot.nodes.size
            )
        }
    }

    fun registerNode(request: Sid.NodeInfo): Sid.RegistrationResponse = lock.withLock {
        val stage = stagesByDeviceId[request.deviceId]
            ?: return failureResponse(
                "device_id ${request.deviceId} is not present in pipeline ${config.pipelineName}"
            )

        val now = Instant.now()
        val existingNodeId = nodeIdByStageId[stage.stageId]
        val topologyChanged: Boolean
        val node = if (existingNodeId != null) {
            val current = requireNotNull(nodesById[existingNodeId])
            topologyChanged =
                current.ipAddress != request.ipAddress ||
                    current.grpcPort != request.grpcPort ||
                    !current.isActive
            current.ipAddress = request.ipAddress
            current.grpcPort = request.grpcPort
            current.computeCapacity = request.computeCapacity
            current.memoryGb = request.memoryGb
            current.registeredAt = now
            current.lastHeartbeatAt = now
            current.isActive = true
            current
        } else {
            val newNode = RegisteredNode(
                nodeId = nextNodeId.getAndIncrement(),
                stageId = stage.stageId,
                deviceId = request.deviceId,
                ipAddress = request.ipAddress,
                grpcPort = request.grpcPort,
                computeCapacity = request.computeCapacity,
                memoryGb = request.memoryGb,
                registeredAt = now,
                lastHeartbeatAt = now,
                isActive = true
            )
            nodesById[newNode.nodeId] = newNode
            nodeIdByStageId[stage.stageId] = newNode.nodeId
            topologyChanged = true
            newNode
        }

        if (topologyChanged) {
            routingEpoch.incrementAndGet()
        }
        persistNode(node)

        val route = resolveRoute(stage.stageId)
        logger.info(
            "Registered device={} stage={} nodeId={} runtime={}:{} nextHop={} routeReady={} epoch={}",
            node.deviceId,
            node.stageId,
            node.nodeId,
            node.ipAddress,
            node.grpcPort,
            route.nextHop?.let { "${it.ipAddress}:${it.grpcPort}" } ?: "terminal",
            route.routeReady,
            route.routingEpoch
        )

        Sid.RegistrationResponse.newBuilder()
            .setSuccess(true)
            .setMessage("registered")
            .setNodeId(node.nodeId)
            .setStageId(route.stageId)
            .setTerminal(route.terminal)
            .setModelShardId(route.modelShardId)
            .setRoutingEpoch(route.routingEpoch)
            .setRouteReady(route.routeReady)
            .setModelDownloadUrl(route.modelDownloadUrl)
            .setModelSha256(route.modelSha256)
            .setModelBytes(route.modelBytes)
            .apply {
                route.nextHop?.let { setNextHop(it) }
            }
            .build()
    }

    fun heartbeat(request: Sid.HeartbeatRequest): Sid.HeartbeatResponse = lock.withLock {
        val node = nodesById[request.nodeId]
        if (node == null || node.deviceId != request.deviceId) {
            return Sid.HeartbeatResponse.newBuilder()
                .setAck(false)
                .setCommand("REREGISTER")
                .setRoutingEpoch(routingEpoch.get())
                .setRouteReady(false)
                .setStageId(-1)
                .build()
        }

        val previousActive = node.isActive
        node.lastHeartbeatAt = Instant.now()
        node.isActive = request.isActive
        if (previousActive != node.isActive) {
            routingEpoch.incrementAndGet()
        }
        persistNode(node)

        val route = resolveRoute(node.stageId)
        val command = when {
            drainedStageIds.contains(node.stageId) -> "DRAIN"
            !request.isActive -> "PAUSED"
            route.terminal -> "RESUME"
            route.routeReady -> "RESUME"
            else -> "WAIT_DOWNSTREAM"
        }

        Sid.HeartbeatResponse.newBuilder()
            .setAck(true)
            .setCommand(command)
            .setTerminal(route.terminal)
            .setModelShardId(route.modelShardId)
            .setRoutingEpoch(route.routingEpoch)
            .setRouteReady(route.routeReady)
            .setStageId(route.stageId)
            .setModelDownloadUrl(route.modelDownloadUrl)
            .setModelSha256(route.modelSha256)
            .setModelBytes(route.modelBytes)
            .apply {
                route.nextHop?.let { setNextHop(it) }
            }
            .build()
    }

    fun evictExpiredNodes(): Int = lock.withLock {
        val now = Instant.now()
        val expiredNodeIds = nodesById.values
            .filter { it.isExpired(now, leaseDuration) }
            .map { it.nodeId }

        expiredNodeIds.forEach { nodeId ->
            val node = nodesById.remove(nodeId) ?: return@forEach
            nodeIdByStageId.remove(node.stageId)
            logger.warn(
                "Lease expired for device={} stage={} nodeId={} lastHeartbeatAt={}",
                node.deviceId,
                node.stageId,
                node.nodeId,
                node.lastHeartbeatAt
            )
            persistence.deleteNode(nodeId, routingEpoch.get(), nextNodeId.get())
        }

        if (expiredNodeIds.isNotEmpty()) {
            routingEpoch.incrementAndGet()
            persistence.saveMetaOnly(routingEpoch.get(), nextNodeId.get())
        }

        expiredNodeIds.size
    }

    fun drainStage(stageId: Int): AdminMutationResult = lock.withLock {
        val stage = config.stages.getOrNull(stageId)
            ?: return mutationFailure("drain_stage", "Unknown stage $stageId")
        if (drainedStageIds.add(stageId)) {
            routingEpoch.incrementAndGet()
            persistence.setStageDrained(stageId, true)
            persistence.saveMetaOnly(routingEpoch.get(), nextNodeId.get())
            logger.info("Stage {} ({}) drained manually", stageId, stage.deviceId)
        }
        mutationSuccess("drain_stage", "Stage $stageId drained")
    }

    fun resumeStage(stageId: Int): AdminMutationResult = lock.withLock {
        val stage = config.stages.getOrNull(stageId)
            ?: return mutationFailure("resume_stage", "Unknown stage $stageId")
        if (drainedStageIds.remove(stageId)) {
            routingEpoch.incrementAndGet()
            persistence.setStageDrained(stageId, false)
            persistence.saveMetaOnly(routingEpoch.get(), nextNodeId.get())
            logger.info("Stage {} ({}) resumed manually", stageId, stage.deviceId)
        }
        mutationSuccess("resume_stage", "Stage $stageId resumed")
    }

    fun evictNode(nodeId: Int): AdminMutationResult = lock.withLock {
        val node = nodesById.remove(nodeId)
            ?: return mutationFailure("evict_node", "Unknown node $nodeId")
        nodeIdByStageId.remove(node.stageId)
        routingEpoch.incrementAndGet()
        persistence.deleteNode(nodeId, routingEpoch.get(), nextNodeId.get())
        logger.warn(
            "Node {} evicted manually for stage {} device {}",
            node.nodeId,
            node.stageId,
            node.deviceId
        )
        return mutationSuccess("evict_node", "Node $nodeId evicted")
    }

    fun reloadConfig(newConfig: CoordinatorConfig): AdminMutationResult = lock.withLock {
        config = newConfig
        leaseDuration = Duration.ofSeconds(newConfig.heartbeatLeaseSeconds)
        stagesByDeviceId = newConfig.stages.associateBy(StageConfig::deviceId)

        val validStageIds = newConfig.stages.mapTo(mutableSetOf()) { it.stageId }
        val validDeviceIdsByStageId = newConfig.stages.associate { it.stageId to it.deviceId }

        val invalidNodes = nodesById.values.filter { node ->
            val expectedDeviceId = validDeviceIdsByStageId[node.stageId]
            expectedDeviceId == null || expectedDeviceId != node.deviceId
        }
        invalidNodes.forEach { node ->
            nodesById.remove(node.nodeId)
            nodeIdByStageId.remove(node.stageId)
            persistence.deleteNode(node.nodeId, routingEpoch.get(), nextNodeId.get())
            logger.warn(
                "Dropping node {} for stage {} after config reload because its device_id no longer matches",
                node.nodeId,
                node.stageId
            )
        }

        drainedStageIds.retainAll(validStageIds)
        persistence.replaceDrainedStages(drainedStageIds)
        routingEpoch.incrementAndGet()
        persistence.saveMetaOnly(routingEpoch.get(), nextNodeId.get())

        logger.info(
            "Config reloaded. pipeline={} stages={} drainedStages={}",
            newConfig.pipelineName,
            newConfig.stages.joinToString { "${it.stageId}:${it.deviceId}" },
            drainedStageIds.sorted()
        )
        return mutationSuccess("reload_config", "Configuration reloaded")
    }

    fun snapshot(): String = lock.withLock {
        val now = Instant.now()
        buildString {
            append("pipeline=")
            append(config.pipelineName)
            append(" stages=[")
            config.stages.forEachIndexed { index, stage ->
                if (index > 0) append("; ")
                append(stage.stageId)
                append(':')
                append(stage.deviceId)
                append(" -> ")
                val nodeId = nodeIdByStageId[stage.stageId]
                val node = nodeId?.let(nodesById::get)
                if (node == null) {
                    append("offline")
                } else if (!node.isLive(now, leaseDuration)) {
                    append("inactive(")
                    append(node.ipAddress)
                    append(':')
                    append(node.grpcPort)
                    append(')')
                } else {
                    append(node.ipAddress)
                    append(':')
                    append(node.grpcPort)
                }
            }
            append(']')
        }
    }

    fun adminStatus(): AdminStatusSnapshot = lock.withLock {
        val now = Instant.now()
        val nodeSnapshots = nodesById.values
            .sortedWith(compareBy<RegisteredNode> { it.stageId }.thenBy { it.nodeId })
            .map { it.toAdminNodeSnapshot(now) }

        val nodeByStageId = nodeSnapshots.associateBy(AdminNodeSnapshot::stageId)
        val stageSnapshots = config.stages.map { stage ->
            val route = resolveRoute(stage.stageId, now)
            AdminStageSnapshot(
                stageId = stage.stageId,
                deviceId = stage.deviceId,
                modelShardId = stage.modelShardId,
                expectedHost = stage.expectedHost,
                expectedPort = stage.expectedPort,
                drained = drainedStageIds.contains(stage.stageId),
                terminal = route.terminal,
                routeReady = route.routeReady,
                assignedNode = nodeByStageId[stage.stageId],
                nextHop = route.nextHop?.let {
                    AdminNextHopSnapshot(
                        nodeId = it.nodeId,
                        ipAddress = it.ipAddress,
                        grpcPort = it.grpcPort
                    )
                }
            )
        }

        val liveNodeCount = nodeSnapshots.count(AdminNodeSnapshot::isLive)
        val inactiveNodeCount = nodeSnapshots.count { !it.isLive }
        val offlineStageCount = stageSnapshots.count { !it.terminal && !it.routeReady }
        val drainedStageCount = stageSnapshots.count(AdminStageSnapshot::drained)

        AdminStatusSnapshot(
            summary = AdminSummarySnapshot(
                pipelineName = config.pipelineName,
                routingEpoch = routingEpoch.get(),
                leaseDurationSeconds = leaseDuration.seconds,
                stageCount = config.stages.size,
                liveNodeCount = liveNodeCount,
                inactiveNodeCount = inactiveNodeCount,
                offlineStageCount = offlineStageCount,
                drainedStageCount = drainedStageCount
            ),
            stages = stageSnapshots,
            nodes = nodeSnapshots
        )
    }

    fun reportRequestEvent(event: Sid.RequestEvent): Sid.RequestEventAck = lock.withLock {
        if (event.requestId.isBlank()) {
            return Sid.RequestEventAck.newBuilder()
                .setAck(false)
                .setMessage("request_id must not be blank")
                .build()
        }

        persistence.appendRequestEvent(
            PersistedRequestEvent(
                eventId = 0L,
                requestId = event.requestId,
                batchId = event.batchId,
                chunkIdx = event.chunkIdx,
                stageId = event.stageId,
                nodeId = event.nodeId,
                eventType = event.eventType.name,
                success = event.success,
                message = event.message,
                eventEpochMs = event.eventEpochMs,
                terminal = event.terminal
            )
        )

        return Sid.RequestEventAck.newBuilder()
            .setAck(true)
            .setMessage("recorded")
            .build()
    }

    fun listRecentRequests(limit: Int, lifecycleState: String?): List<AdminRequestStateSnapshot> {
        val normalizedFilter = lifecycleState
            ?.trim()
            ?.uppercase()
            ?.takeIf { it.isNotBlank() }
        return persistence.listRecentRequestStates(limit.coerceIn(1, 500))
            .map(::toAdminRequestStateSnapshot)
            .filter { snapshot ->
                normalizedFilter == null || snapshot.lifecycleState == normalizedFilter
            }
    }

    fun loadRequestDetail(requestId: String, eventLimit: Int): AdminRequestDetailSnapshot {
        val detail = persistence.loadRequestDetail(
            requestId = requestId,
            eventLimit = eventLimit.coerceIn(1, 1000)
        )
        return detail.copy(
            state = detail.state?.let { snapshot ->
                toAdminRequestStateSnapshot(
                    PersistedRequestState(
                        requestId = snapshot.requestId,
                        batchId = snapshot.batchId,
                        latestChunkIdx = snapshot.latestChunkIdx,
                        latestStageId = snapshot.latestStageId,
                        latestNodeId = snapshot.latestNodeId,
                        latestEventType = snapshot.latestEventType,
                        latestSuccess = snapshot.latestSuccess,
                        latestMessage = snapshot.latestMessage,
                        firstSeenEpochMs = snapshot.firstSeenEpochMs,
                        lastUpdatedEpochMs = snapshot.lastUpdatedEpochMs,
                        terminal = snapshot.terminal,
                        storedPayload = snapshot.storedPayload,
                        submitAttempts = snapshot.submitAttempts,
                        lastSubmitEpochMs = snapshot.lastSubmitEpochMs
                    )
                )
            }
        )
    }

    fun purgeRequest(requestId: String): AdminMutationResult = lock.withLock {
        if (requestId.isBlank()) {
            return mutationFailure("purge_request", "requestId must not be blank")
        }
        val deleted = persistence.deleteRequest(requestId)
        if (!deleted) {
            return mutationFailure("purge_request", "Unknown request $requestId")
        }
        logger.info("Purged request history for requestId={}", requestId)
        return mutationSuccess("purge_request", "Purged request $requestId")
    }

    fun purgeExpiredResolvedRequests(): Int = lock.withLock {
        val olderThanEpochMs = Instant.now().toEpochMilli() - (config.resolvedRequestRetentionSeconds * 1_000)
        return persistence.purgeResolvedRequests(olderThanEpochMs)
    }

    fun purgeResolvedRequests(olderThanSeconds: Long?): AdminMutationResult = lock.withLock {
        val effectiveAgeSeconds = olderThanSeconds ?: config.resolvedRequestRetentionSeconds
        if (effectiveAgeSeconds <= 0) {
            return mutationFailure(
                "purge_resolved_requests",
                "olderThanSeconds must be positive"
            )
        }
        val olderThanEpochMs = Instant.now().toEpochMilli() - (effectiveAgeSeconds * 1_000)
        val purgedCount = persistence.purgeResolvedRequests(olderThanEpochMs)
        logger.info(
            "Purged {} resolved request records older than {} seconds",
            purgedCount,
            effectiveAgeSeconds
        )
        return mutationSuccess(
            "purge_resolved_requests",
            "Purged $purgedCount resolved requests older than $effectiveAgeSeconds seconds"
        )
    }

    fun loadStageArtifact(stageId: Int): StageArtifactHandle? = lock.withLock {
        config.stages.getOrNull(stageId)
            ?.takeIf { it.modelArtifactPath.isNotBlank() }
            ?.let { stage ->
                StageArtifactHandle(
                    stageId = stage.stageId,
                    modelShardId = stage.modelShardId,
                    filePath = stage.modelArtifactPath,
                    sha256 = stage.modelSha256,
                    bytes = stage.modelBytes
                )
            }
    }

    fun loadRequestPayload(requestId: String): PersistedRequestPayload? = lock.withLock {
        persistence.loadRequestPayload(requestId)
    }

    fun storeRequestPayload(requestId: String, payloadProto: ByteArray, submittedAtEpochMs: Long) = lock.withLock {
        persistence.upsertRequestPayload(requestId, payloadProto, submittedAtEpochMs)
    }

    fun planRequestSubmission(): RequestSubmissionPlan = lock.withLock {
        val now = Instant.now()
        var firstStageNode: RegisteredNode? = null
        for (stage in config.stages) {
            if (drainedStageIds.contains(stage.stageId)) {
                return RequestSubmissionPlan(
                    accepted = false,
                    stageId = stage.stageId,
                    nodeId = -1,
                    host = null,
                    port = null,
                    message = "Stage ${stage.stageId} is drained."
                )
            }
            val node = nodeIdByStageId[stage.stageId]
                ?.let(nodesById::get)
                ?.takeIf { it.isLive(now, leaseDuration) }
                ?: return RequestSubmissionPlan(
                    accepted = false,
                    stageId = stage.stageId,
                    nodeId = -1,
                    host = null,
                    port = null,
                    message = "Stage ${stage.stageId} has no live worker."
                )
            if (stage.stageId == 0) {
                firstStageNode = node
            }
        }
        val node = firstStageNode
            ?: return RequestSubmissionPlan(
                accepted = false,
                stageId = -1,
                nodeId = -1,
                host = null,
                port = null,
                message = "No stage 0 configured."
            )
        RequestSubmissionPlan(
            accepted = true,
            stageId = 0,
            nodeId = node.nodeId,
            host = node.ipAddress,
            port = node.grpcPort,
            message = "Dispatching to stage 0 worker ${node.nodeId}"
        )
    }

    fun recordCoordinatorRequestEvent(
        requestId: String,
        batchId: Int,
        chunkIdx: Int,
        stageId: Int,
        nodeId: Int,
        eventType: Sid.RequestEventType,
        success: Boolean,
        message: String,
        terminal: Boolean
    ) {
        reportRequestEvent(
            Sid.RequestEvent.newBuilder()
                .setRequestId(requestId)
                .setBatchId(batchId)
                .setChunkIdx(chunkIdx)
                .setStageId(stageId)
                .setNodeId(nodeId)
                .setEventType(eventType)
                .setSuccess(success)
                .setMessage(message)
                .setEventEpochMs(Instant.now().toEpochMilli())
                .setTerminal(terminal)
                .build()
        )
    }

    private fun resolveRoute(stageId: Int): RouteSnapshot {
        return resolveRoute(stageId, Instant.now())
    }

    private fun resolveRoute(stageId: Int, now: Instant): RouteSnapshot {
        val stage = config.stages.first { it.stageId == stageId }
        val nextStage = config.stages.getOrNull(stageId + 1)
        if (nextStage == null) {
            return RouteSnapshot(
                stageId = stageId,
                terminal = true,
                modelShardId = stage.modelShardId,
                modelDownloadUrl = stage.modelDownloadUrl,
                modelSha256 = stage.modelSha256,
                modelBytes = stage.modelBytes,
                nextHop = null,
                routeReady = true,
                routingEpoch = routingEpoch.get()
            )
        }

        if (drainedStageIds.contains(nextStage.stageId)) {
            return RouteSnapshot(
                stageId = stageId,
                terminal = false,
                modelShardId = stage.modelShardId,
                modelDownloadUrl = stage.modelDownloadUrl,
                modelSha256 = stage.modelSha256,
                modelBytes = stage.modelBytes,
                nextHop = Sid.NextHop.newBuilder()
                    .setNodeId(-1)
                    .setIpAddress(nextStage.expectedHost)
                    .setGrpcPort(nextStage.expectedPort)
                    .build(),
                routeReady = false,
                routingEpoch = routingEpoch.get()
            )
        }

        val liveNode = nodeIdByStageId[nextStage.stageId]
            ?.let(nodesById::get)
            ?.takeIf { it.isLive(now, leaseDuration) }

        val host = liveNode?.ipAddress ?: nextStage.expectedHost
        val port = liveNode?.grpcPort ?: nextStage.expectedPort

        return RouteSnapshot(
            stageId = stageId,
            terminal = false,
            modelShardId = stage.modelShardId,
            modelDownloadUrl = stage.modelDownloadUrl,
            modelSha256 = stage.modelSha256,
            modelBytes = stage.modelBytes,
            nextHop = Sid.NextHop.newBuilder()
                .setNodeId(liveNode?.nodeId ?: -1)
                .setIpAddress(host)
                .setGrpcPort(port)
                .build(),
            routeReady = liveNode != null,
            routingEpoch = routingEpoch.get()
        )
    }

    private fun persistNode(node: RegisteredNode) {
        persistence.upsertNode(
            node = node,
            routingEpoch = routingEpoch.get(),
            nextNodeId = nextNodeId.get()
        )
    }

    private fun toAdminRequestStateSnapshot(state: PersistedRequestState): AdminRequestStateSnapshot {
        val nowEpochMs = Instant.now().toEpochMilli()
        val lifecycleState = deriveRequestLifecycleState(state, nowEpochMs)
        val ageSeconds = max(0L, (nowEpochMs - state.lastUpdatedEpochMs) / 1_000)
        return AdminRequestStateSnapshot(
            requestId = state.requestId,
            batchId = state.batchId,
            latestChunkIdx = state.latestChunkIdx,
            latestStageId = state.latestStageId,
            latestNodeId = state.latestNodeId,
            latestEventType = state.latestEventType,
            latestSuccess = state.latestSuccess,
            latestMessage = state.latestMessage,
            firstSeenEpochMs = state.firstSeenEpochMs,
            lastUpdatedEpochMs = state.lastUpdatedEpochMs,
            terminal = state.terminal,
            lifecycleState = lifecycleState,
            stalled = lifecycleState == "STALLED",
            lastUpdatedAgeSeconds = ageSeconds,
            storedPayload = state.storedPayload,
            submitAttempts = state.submitAttempts,
            lastSubmitEpochMs = state.lastSubmitEpochMs
        )
    }

    private fun deriveRequestLifecycleState(
        state: PersistedRequestState,
        nowEpochMs: Long
    ): String {
        if (state.latestEventType == Sid.RequestEventType.FAILED.name || !state.latestSuccess) {
            return "FAILED"
        }
        if (state.latestEventType == Sid.RequestEventType.COMPLETED.name && state.terminal) {
            return "COMPLETED"
        }
        val staleAfterMs = config.requestStallTimeoutSeconds * 1_000
        if (nowEpochMs - state.lastUpdatedEpochMs >= staleAfterMs) {
            return "STALLED"
        }
        return "IN_FLIGHT"
    }

    private fun RegisteredNode.toAdminNodeSnapshot(now: Instant): AdminNodeSnapshot {
        val expired = isExpired(now, leaseDuration)
        return AdminNodeSnapshot(
            nodeId = nodeId,
            stageId = stageId,
            deviceId = deviceId,
            ipAddress = ipAddress,
            grpcPort = grpcPort,
            computeCapacity = computeCapacity,
            memoryGb = memoryGb,
            registeredAtEpochMs = registeredAt.toEpochMilli(),
            lastHeartbeatAtEpochMs = lastHeartbeatAt.toEpochMilli(),
            isActive = isActive,
            isExpired = expired,
            isLive = isActive && !expired
        )
    }

    private fun failureResponse(message: String): Sid.RegistrationResponse {
        logger.warn("Registration rejected: {}", message)
        return Sid.RegistrationResponse.newBuilder()
            .setSuccess(false)
            .setMessage(message)
            .setNodeId(-1)
            .setStageId(-1)
            .setTerminal(false)
            .setModelShardId("")
            .setRoutingEpoch(routingEpoch.get())
            .setRouteReady(false)
            .setModelDownloadUrl("")
            .setModelSha256("")
            .setModelBytes(0)
            .build()
    }

    private fun mutationSuccess(action: String, message: String): AdminMutationResult {
        return AdminMutationResult(
            action = action,
            success = true,
            message = message,
            status = adminStatus()
        )
    }

    private fun mutationFailure(action: String, message: String): AdminMutationResult {
        return AdminMutationResult(
            action = action,
            success = false,
            message = message,
            status = adminStatus()
        )
    }

    private data class RouteSnapshot(
        val stageId: Int,
        val terminal: Boolean,
        val modelShardId: String,
        val modelDownloadUrl: String,
        val modelSha256: String,
        val modelBytes: Long,
        val nextHop: Sid.NextHop?,
        val routeReady: Boolean,
        val routingEpoch: Long
    )

    data class RequestSubmissionPlan(
        val accepted: Boolean,
        val stageId: Int,
        val nodeId: Int,
        val host: String?,
        val port: Int?,
        val message: String
    )
}
