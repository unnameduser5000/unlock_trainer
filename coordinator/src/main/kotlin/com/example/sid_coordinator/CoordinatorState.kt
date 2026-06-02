package com.example.sid_coordinator

import org.slf4j.LoggerFactory
import sid.Sid
import java.security.MessageDigest
import java.time.Duration
import java.time.Instant
import java.util.ArrayDeque
import kotlin.math.max
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class RegisteredNode(
    val nodeId: Int,
    var stageId: Int,
    val deviceId: String,
    var ipAddress: String,
    var grpcPort: Int,
    var computeCapacity: Float,
    var memoryGb: Float,
    var registeredAt: Instant,
    var lastHeartbeatAt: Instant,
    var isActive: Boolean,
    var assignmentReason: String = "unknown"
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
    private var preferredStageByDeviceId = preferredStageByDeviceId(initialConfig)
    private val nodesById = mutableMapOf<Int, RegisteredNode>()
    private val nodeIdByStageId = mutableMapOf<Int, Int>()
    private val drainedStageIds = mutableSetOf<Int>()
    private val schedulerEvents = ArrayDeque<AdminSchedulerEventSnapshot>()
    private var nextSchedulerEventId = 1L

    init {
        val snapshot = persistence.loadSnapshot()
        routingEpoch = AtomicLong(snapshot.routingEpoch)
        nextNodeId = AtomicInteger(snapshot.nextNodeId)
        drainedStageIds.addAll(snapshot.drainedStageIds)

        snapshot.nodes.forEach { restored ->
            val offlineNode = restored.copy(isActive = false, assignmentReason = "restored-from-state")
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
        persistence.listRecentSchedulerEvents(MAX_SCHEDULER_EVENTS).asReversed().forEach { event ->
            schedulerEvents.addFirst(event)
            nextSchedulerEventId = maxOf(nextSchedulerEventId, event.eventId + 1)
        }
    }

    fun registerNode(request: Sid.NodeInfo): Sid.RegistrationResponse = lock.withLock {
        val now = Instant.now()
        val existingDeviceNode = nodesById.values.firstOrNull { it.deviceId == request.deviceId }
        val assignment = chooseStageForRegistration(request, now, existingDeviceNode)
        if (assignment == null) {
            recordSchedulerEvent(
                action = "register_rejected",
                stageId = null,
                nodeId = existingDeviceNode?.nodeId,
                deviceId = request.deviceId,
                reason = "no-schedulable-stage",
                message = "No schedulable stage for device_id ${request.deviceId} in pipeline ${config.pipelineName}"
            )
            return failureResponse(
                "No schedulable stage for device_id ${request.deviceId} in pipeline ${config.pipelineName}"
            )
        }
        val stage = assignment.stage
        var topologyChanged = false
        val nodesToPersist = linkedMapOf<Int, RegisteredNode>()

        if (existingDeviceNode != null && existingDeviceNode.stageId != stage.stageId) {
            if (nodeIdByStageId[existingDeviceNode.stageId] == existingDeviceNode.nodeId) {
                nodeIdByStageId.remove(existingDeviceNode.stageId)
            }
            topologyChanged = true
        }

        val displacedNode = assignment.displacedNodeId?.let(nodesById::get)
        if (displacedNode != null && displacedNode.nodeId != existingDeviceNode?.nodeId) {
            val previousStageId = displacedNode.stageId
            if (nodeIdByStageId[previousStageId] == displacedNode.nodeId) {
                nodeIdByStageId.remove(previousStageId)
            }
            val relocationStage = assignment.displacedStage
            if (relocationStage != null) {
                displacedNode.stageId = relocationStage.stageId
                displacedNode.assignmentReason = assignment.displacedReason ?: "relocated-by-scheduler"
                nodeIdByStageId[relocationStage.stageId] = displacedNode.nodeId
                nodesToPersist[displacedNode.nodeId] = displacedNode
                recordSchedulerEvent(
                    action = "relocate_node",
                    stageId = relocationStage.stageId,
                    nodeId = displacedNode.nodeId,
                    deviceId = displacedNode.deviceId,
                    reason = displacedNode.assignmentReason,
                    message = "Moved node ${displacedNode.nodeId} from stage $previousStageId to stage ${relocationStage.stageId}"
                )
            } else {
                nodesById.remove(displacedNode.nodeId)
                persistence.deleteNode(displacedNode.nodeId, routingEpoch.get(), nextNodeId.get())
                recordSchedulerEvent(
                    action = "replace_node",
                    stageId = previousStageId,
                    nodeId = displacedNode.nodeId,
                    deviceId = displacedNode.deviceId,
                    reason = assignment.displacedReason ?: "replaced-by-scheduler",
                    message = "Removed node ${displacedNode.nodeId} from stage $previousStageId for registering device ${request.deviceId}"
                )
            }
            topologyChanged = true
        }

        val existingStageNode = nodeIdByStageId[stage.stageId]?.let(nodesById::get)
        if (existingStageNode != null && existingStageNode.nodeId != existingDeviceNode?.nodeId) {
            nodesById.remove(existingStageNode.nodeId)
            nodeIdByStageId.remove(stage.stageId)
            persistence.deleteNode(existingStageNode.nodeId, routingEpoch.get(), nextNodeId.get())
            recordSchedulerEvent(
                action = "replace_node",
                stageId = stage.stageId,
                nodeId = existingStageNode.nodeId,
                deviceId = existingStageNode.deviceId,
                reason = "stage-occupied",
                message = "Replaced node ${existingStageNode.nodeId} on stage ${stage.stageId} with registering device ${request.deviceId}"
            )
            logger.warn(
                "Scheduler replaced node {} device={} on stage={} with registering device={}",
                existingStageNode.nodeId,
                existingStageNode.deviceId,
                stage.stageId,
                request.deviceId
            )
            topologyChanged = true
        }

        val node = existingDeviceNode ?: newRegisteredNode(request, stage.stageId, now, assignment.reason).also {
            nodesById[it.nodeId] = it
            topologyChanged = true
        }

        if (existingDeviceNode != null) {
            topologyChanged =
                topologyChanged ||
                    existingDeviceNode.stageId != stage.stageId ||
                    existingDeviceNode.ipAddress != request.ipAddress ||
                    existingDeviceNode.grpcPort != request.grpcPort ||
                    !existingDeviceNode.isActive ||
                    existingDeviceNode.assignmentReason != assignment.reason
            existingDeviceNode.stageId = stage.stageId
            updateNodeFromRegistration(existingDeviceNode, request, now, assignment.reason)
        }
        nodeIdByStageId[stage.stageId] = node.nodeId

        if (topologyChanged) {
            routingEpoch.incrementAndGet()
            persistReassignedNodes(nodesToPersist.values + node)
        } else {
            persistNode(node)
        }
        recordSchedulerEvent(
            action = "register_node",
            stageId = node.stageId,
            nodeId = node.nodeId,
            deviceId = node.deviceId,
            reason = node.assignmentReason,
            message = "Registered device ${node.deviceId} as node ${node.nodeId} on stage ${node.stageId}"
        )

        val route = resolveRoute(stage.stageId)
        logger.info(
            "Registered device={} stage={} nodeId={} reason={} runtime={}:{} nextHop={} routeReady={} epoch={}",
            node.deviceId,
            node.stageId,
            node.nodeId,
            node.assignmentReason,
            node.ipAddress,
            node.grpcPort,
            route.nextHop?.let { "${it.ipAddress}:${it.grpcPort}" } ?: "terminal",
            route.routeReady,
            route.routingEpoch
        )

        Sid.RegistrationResponse.newBuilder()
            .setSuccess(true)
            .setMessage("registered; stage=${stage.stageId}; reason=${node.assignmentReason}")
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

    private fun updateNodeFromRegistration(
        node: RegisteredNode,
        request: Sid.NodeInfo,
        now: Instant,
        assignmentReason: String
    ) {
        node.ipAddress = request.ipAddress
        node.grpcPort = request.grpcPort
        node.computeCapacity = request.computeCapacity
        node.memoryGb = request.memoryGb
        node.registeredAt = now
        node.lastHeartbeatAt = now
        node.isActive = true
        node.assignmentReason = assignmentReason
    }

    private fun newRegisteredNode(
        request: Sid.NodeInfo,
        stageId: Int,
        now: Instant,
        assignmentReason: String
    ): RegisteredNode {
        return RegisteredNode(
            nodeId = nextNodeId.getAndIncrement(),
            stageId = stageId,
            deviceId = request.deviceId,
            ipAddress = request.ipAddress,
            grpcPort = request.grpcPort,
            computeCapacity = request.computeCapacity,
            memoryGb = request.memoryGb,
            registeredAt = now,
            lastHeartbeatAt = now,
            isActive = true,
            assignmentReason = assignmentReason
        )
    }

    private fun chooseStageForRegistration(
        request: Sid.NodeInfo,
        now: Instant,
        existingDeviceNode: RegisteredNode?
    ): StageAssignment? {
        val scheduler = config.scheduler
        val preferredStage = preferredStageByDeviceId[request.deviceId]
        if (!scheduler.enabled) {
            return preferredStage?.let { StageAssignment(it, "static-device-match") }
        }

        if (!scheduler.allowUnlistedDevices && preferredStage == null) {
            return null
        }
        if (request.memoryGb < scheduler.minMemoryGb || request.computeCapacity < scheduler.minComputeCapacity) {
            logger.warn(
                "Scheduler rejected device={} memoryGb={} computeCapacity={} minMemoryGb={} minComputeCapacity={}",
                request.deviceId,
                request.memoryGb,
                request.computeCapacity,
                scheduler.minMemoryGb,
                scheduler.minComputeCapacity
            )
            return null
        }

        if (
            scheduler.preferConfiguredDevices &&
            preferredStage != null &&
            !drainedStageIds.contains(preferredStage.stageId) &&
            nodeMeetsStageRequirements(request.computeCapacity, request.memoryGb, preferredStage)
        ) {
            val assigned = liveNodeForStage(preferredStage.stageId, now)
            if (assigned == null || assigned.deviceId == request.deviceId) {
                return StageAssignment(preferredStage, "preferred-device-match")
            }
            if (!isPreferredDeviceForStage(assigned.deviceId, preferredStage.stageId)) {
                val relocationStage = chooseRelocationStageForNode(
                    node = assigned,
                    now = now,
                    excludedStageIds = setOf(preferredStage.stageId),
                    vacatedNodeId = existingDeviceNode?.nodeId
                )
                if (relocationStage != null) {
                    return StageAssignment(
                        stage = preferredStage,
                        reason = "preferred-device-relocate-dynamic-worker",
                        displacedNodeId = assigned.nodeId,
                        displacedStage = relocationStage,
                        displacedReason = "relocated-for-preferred-device"
                    )
                }
            }
            if (scheduler.rebalanceOnRegister && !isPreferredDeviceForStage(assigned.deviceId, preferredStage.stageId)) {
                return StageAssignment(
                    stage = preferredStage,
                    reason = "preferred-device-replace-dynamic-worker",
                    displacedNodeId = assigned.nodeId,
                    displacedStage = null,
                    displacedReason = "evicted-for-preferred-device"
                )
            }
        }

        if (existingDeviceNode != null) {
            val existingStage = config.stages.getOrNull(existingDeviceNode.stageId)
            if (
                existingStage != null &&
                !drainedStageIds.contains(existingStage.stageId) &&
                nodeMeetsStageRequirements(request.computeCapacity, request.memoryGb, existingStage)
            ) {
                return StageAssignment(existingStage, "sticky-existing-assignment")
            }
        }

        val offlineStages = config.stages
            .filterNot { drainedStageIds.contains(it.stageId) }
            .filter { liveNodeForStage(it.stageId, now) == null }
            .filter { nodeMeetsStageRequirements(request.computeCapacity, request.memoryGb, it) }

        if (offlineStages.isNotEmpty()) {
            return StageAssignment(
                stage = chooseStageByPolicy(offlineStages, scheduler.policy),
                reason = "dynamic-fill-offline-stage"
            )
        }

        if (scheduler.rebalanceOnRegister) {
            val replaceableStage = config.stages
                .filterNot { drainedStageIds.contains(it.stageId) }
                .filter { nodeMeetsStageRequirements(request.computeCapacity, request.memoryGb, it) }
                .filter { stage ->
                    !isPreferredDeviceForStage(liveNodeForStage(stage.stageId, now)?.deviceId, stage.stageId)
                }
                .maxByOrNull { stage -> replacementGain(stage, request, now) }
                ?.takeIf { stage -> replacementGain(stage, request, now) > 0f }
            if (replaceableStage != null) {
                return StageAssignment(
                    stage = replaceableStage,
                    reason = "dynamic-rebalance-better-capacity",
                    displacedNodeId = liveNodeForStage(replaceableStage.stageId, now)?.nodeId,
                    displacedStage = null,
                    displacedReason = "evicted-for-better-capacity"
                )
            }
        }

        return null
    }

    private fun chooseStageByPolicy(
        stages: List<StageConfig>,
        policy: String
    ): StageConfig {
        return when (policy) {
            "memory-first" -> stages.maxWith(
                compareBy<StageConfig> { stageMemoryDemand(it) }
                    .thenBy { it.schedulingWeight }
                    .thenBy { it.stageId }
            )

            "compute-first" -> stages.maxWith(
                compareBy<StageConfig> { stageComputeDemand(it) }
                    .thenBy { stageMemoryDemand(it) }
                    .thenBy { it.stageId }
            )

            else -> stages.minBy(StageConfig::stageId)
        }
    }

    private fun chooseRelocationStageForNode(
        node: RegisteredNode,
        now: Instant,
        excludedStageIds: Set<Int>,
        vacatedNodeId: Int?
    ): StageConfig? {
        val candidates = config.stages
            .filterNot { drainedStageIds.contains(it.stageId) }
            .filterNot { excludedStageIds.contains(it.stageId) }
            .filter { stage ->
                val liveNode = liveNodeForStage(stage.stageId, now)
                liveNode == null || liveNode.nodeId == vacatedNodeId
            }
            .filter { stage ->
                nodeMeetsStageRequirements(node.computeCapacity, node.memoryGb, stage)
            }
        return candidates.takeIf { it.isNotEmpty() }
            ?.let { chooseStageByPolicy(it, config.scheduler.policy) }
    }

    private fun nodeMeetsStageRequirements(
        computeCapacity: Float,
        memoryGb: Float,
        stage: StageConfig
    ): Boolean {
        return memoryGb >= maxOf(config.scheduler.minMemoryGb, stage.minMemoryGb) &&
            computeCapacity >= maxOf(config.scheduler.minComputeCapacity, stage.minComputeCapacity)
    }

    private fun stageMemoryDemand(stage: StageConfig): Float {
        val modelGib = stage.modelBytes.toFloat() / (1024f * 1024f * 1024f)
        return stage.minMemoryGb + modelGib + stage.schedulingWeight
    }

    private fun stageComputeDemand(stage: StageConfig): Float {
        return stage.minComputeCapacity + stage.schedulingWeight
    }

    private fun replacementGain(stage: StageConfig, request: Sid.NodeInfo, now: Instant): Float {
        val current = liveNodeForStage(stage.stageId, now) ?: return Float.MAX_VALUE
        return nodeScore(request.computeCapacity, request.memoryGb) -
            nodeScore(current.computeCapacity, current.memoryGb)
    }

    private fun nodeScore(computeCapacity: Float, memoryGb: Float): Float {
        return computeCapacity + memoryGb * 10f
    }

    private fun liveNodeForStage(stageId: Int, now: Instant): RegisteredNode? {
        return nodeIdByStageId[stageId]
            ?.let(nodesById::get)
            ?.takeIf { it.isLive(now, leaseDuration) }
    }

    private fun isPreferredDeviceForStage(deviceId: String?, stageId: Int): Boolean {
        if (deviceId.isNullOrBlank()) {
            return false
        }
        return preferredStageByDeviceId[deviceId]?.stageId == stageId
    }

    private fun reconcilePreferredAssignmentsLocked(): Int {
        val now = Instant.now()
        val nodesToPersist = linkedMapOf<Int, RegisteredNode>()
        var moveCount = 0

        config.stages.forEach { preferredStage ->
            val preferredDeviceId = preferredStage.deviceId.takeIf { it.isNotBlank() }
                ?: return@forEach
            if (drainedStageIds.contains(preferredStage.stageId)) {
                return@forEach
            }
            val preferredNode = nodesById.values.firstOrNull { node ->
                node.deviceId == preferredDeviceId && node.isLive(now, leaseDuration)
            } ?: return@forEach
            if (preferredNode.stageId == preferredStage.stageId) {
                return@forEach
            }
            if (!nodeMeetsStageRequirements(preferredNode.computeCapacity, preferredNode.memoryGb, preferredStage)) {
                return@forEach
            }

            val occupant = liveNodeForStage(preferredStage.stageId, now)
            if (occupant == null || occupant.nodeId == preferredNode.nodeId) {
                if (moveNodeToStageLocked(preferredNode, preferredStage, "reconciled-preferred-device")) {
                    nodesToPersist[preferredNode.nodeId] = preferredNode
                    moveCount += 1
                }
                return@forEach
            }
            if (isPreferredDeviceForStage(occupant.deviceId, preferredStage.stageId)) {
                return@forEach
            }

            val relocationStage = chooseRelocationStageForNode(
                node = occupant,
                now = now,
                excludedStageIds = setOf(preferredStage.stageId),
                vacatedNodeId = preferredNode.nodeId
            )
            if (relocationStage != null) {
                if (moveNodeToStageLocked(occupant, relocationStage, "relocated-for-preferred-device")) {
                    nodesToPersist[occupant.nodeId] = occupant
                    moveCount += 1
                }
                if (moveNodeToStageLocked(preferredNode, preferredStage, "reconciled-preferred-device")) {
                    nodesToPersist[preferredNode.nodeId] = preferredNode
                    moveCount += 1
                }
            } else if (config.scheduler.rebalanceOnRegister) {
                nodesById.remove(occupant.nodeId)
                nodeIdByStageId.remove(occupant.stageId)
                persistence.deleteNode(occupant.nodeId, routingEpoch.get(), nextNodeId.get())
                recordSchedulerEvent(
                    action = "replace_node",
                    stageId = occupant.stageId,
                    nodeId = occupant.nodeId,
                    deviceId = occupant.deviceId,
                    reason = "evicted-for-preferred-device",
                    message = "Scheduler reconcile evicted node ${occupant.nodeId} so ${preferredNode.deviceId} can return to preferred stage ${preferredStage.stageId}"
                )
                if (moveNodeToStageLocked(preferredNode, preferredStage, "reconciled-preferred-device")) {
                    nodesToPersist[preferredNode.nodeId] = preferredNode
                    moveCount += 1
                }
            }
        }

        if (nodesToPersist.isNotEmpty()) {
            routingEpoch.incrementAndGet()
            persistReassignedNodes(nodesToPersist.values)
        }
        return moveCount
    }

    private fun moveNodeToStageLocked(
        node: RegisteredNode,
        stage: StageConfig,
        reason: String
    ): Boolean {
        if (node.stageId == stage.stageId && node.assignmentReason == reason) {
            return false
        }
        val previousStageId = node.stageId
        if (nodeIdByStageId[previousStageId] == node.nodeId) {
            nodeIdByStageId.remove(previousStageId)
        }
        node.stageId = stage.stageId
        node.assignmentReason = reason
        nodeIdByStageId[stage.stageId] = node.nodeId
        recordSchedulerEvent(
            action = "move_node",
            stageId = stage.stageId,
            nodeId = node.nodeId,
            deviceId = node.deviceId,
            reason = reason,
            message = "Moved node ${node.nodeId} from stage $previousStageId to stage ${stage.stageId}"
        )
        return true
    }

    private fun preferredStageByDeviceId(config: CoordinatorConfig): Map<String, StageConfig> {
        return config.stages.mapNotNull { stage ->
            stage.deviceId.trim().takeIf { it.isNotBlank() }?.let { it to stage }
        }.toMap()
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
        val observedAt = Instant.now()
        node.lastHeartbeatAt = observedAt
        node.isActive = request.isActive
        if (previousActive != node.isActive) {
            routingEpoch.incrementAndGet()
        }
        persistNode(node)
        persistence.appendWorkerTelemetry(
            PersistedWorkerTelemetry(
                observedEpochMs = observedAt.toEpochMilli(),
                deviceId = request.deviceId,
                nodeId = request.nodeId,
                stageId = node.stageId,
                isActive = request.isActive,
                batteryLevel = request.batteryLevel,
                isCharging = request.isCharging,
                powerSource = request.powerSource,
                batteryStatus = request.batteryStatus,
                batteryTempC = request.batteryTempC.takeIf { it != 0f },
                batteryVoltageMv = request.batteryVoltageMv.takeIf { it != 0 },
                batteryCurrentUa = request.batteryCurrentUa.takeIf { it != 0L },
                thermalStatus = request.thermalStatus,
                appPssKb = request.appPssKb.takeIf { it != 0L },
                appPrivateDirtyKb = request.appPrivateDirtyKb.takeIf { it != 0L },
                runtimeUsedMemoryKb = request.runtimeUsedMemoryKb.takeIf { it != 0L },
                workerState = request.workerState
            )
        )

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
        evictExpiredNodesLocked(Instant.now(), "lease-expired")
    }

    private fun evictExpiredNodesLocked(now: Instant, reason: String): Int {
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
            recordSchedulerEvent(
                action = "evict_expired_node",
                stageId = node.stageId,
                nodeId = node.nodeId,
                deviceId = node.deviceId,
                reason = reason,
                message = "Lease expired for node ${node.nodeId} on stage ${node.stageId}"
            )
        }

        if (expiredNodeIds.isNotEmpty()) {
            routingEpoch.incrementAndGet()
            persistence.saveMetaOnly(routingEpoch.get(), nextNodeId.get())
        }

        return expiredNodeIds.size
    }

    fun drainStage(stageId: Int): AdminMutationResult = lock.withLock {
        val stage = config.stages.getOrNull(stageId)
            ?: return mutationFailure("drain_stage", "Unknown stage $stageId")
        if (drainedStageIds.add(stageId)) {
            routingEpoch.incrementAndGet()
            persistence.setStageDrained(stageId, true)
            persistence.saveMetaOnly(routingEpoch.get(), nextNodeId.get())
            logger.info("Stage {} ({}) drained manually", stageId, stage.deviceId)
            recordSchedulerEvent(
                action = "drain_stage",
                stageId = stageId,
                nodeId = nodeIdByStageId[stageId],
                deviceId = stage.deviceId,
                reason = "manual-drain",
                message = "Stage $stageId drained manually"
            )
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
            recordSchedulerEvent(
                action = "resume_stage",
                stageId = stageId,
                nodeId = nodeIdByStageId[stageId],
                deviceId = stage.deviceId,
                reason = "manual-resume",
                message = "Stage $stageId resumed manually"
            )
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
        recordSchedulerEvent(
            action = "evict_node",
            stageId = node.stageId,
            nodeId = node.nodeId,
            deviceId = node.deviceId,
            reason = "manual-evict",
            message = "Node ${node.nodeId} evicted manually from stage ${node.stageId}"
        )
        return mutationSuccess("evict_node", "Node $nodeId evicted")
    }

    fun reconcileScheduler(): AdminMutationResult = lock.withLock {
        val expiredCount = evictExpiredNodesLocked(Instant.now(), "scheduler-reconcile")
        val reassignedCount = if (config.scheduler.enabled && config.scheduler.preferConfiguredDevices) {
            reconcilePreferredAssignmentsLocked()
        } else {
            0
        }
        if (expiredCount == 0 && reassignedCount == 0) {
            recordSchedulerEvent(
                action = "scheduler_reconcile",
                stageId = null,
                nodeId = null,
                deviceId = null,
                reason = "no-op",
                message = "Scheduler reconcile found no expired leases or preferred-device moves"
            )
        }
        mutationSuccess(
            "scheduler_reconcile",
            "Scheduler reconcile complete: expired=$expiredCount reassigned=$reassignedCount"
        )
    }

    fun reloadConfig(newConfig: CoordinatorConfig): AdminMutationResult = lock.withLock {
        config = newConfig
        leaseDuration = Duration.ofSeconds(newConfig.heartbeatLeaseSeconds)
        preferredStageByDeviceId = preferredStageByDeviceId(newConfig)

        val validStageIds = newConfig.stages.mapTo(mutableSetOf()) { it.stageId }
        val validDeviceIdsByStageId = newConfig.stages.associate { it.stageId to it.deviceId }

        val invalidNodes = nodesById.values.filter { node ->
            val expectedDeviceId = validDeviceIdsByStageId[node.stageId]
            expectedDeviceId == null ||
                (!newConfig.scheduler.enabled && expectedDeviceId != node.deviceId) ||
                (newConfig.scheduler.enabled && !newConfig.scheduler.allowUnlistedDevices && expectedDeviceId != node.deviceId)
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
            "Config reloaded. pipeline={} scheduler={} stages={} drainedStages={}",
            newConfig.pipelineName,
            "${newConfig.scheduler.enabled}:${newConfig.scheduler.policy}",
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
                preferredDeviceId = stage.deviceId,
                minMemoryGb = stage.minMemoryGb,
                minComputeCapacity = stage.minComputeCapacity,
                schedulingWeight = stage.schedulingWeight,
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
        val offlineStageCount = stageSnapshots.count { stage ->
            !stage.drained && stage.assignedNode?.isLive != true
        }
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
                drainedStageCount = drainedStageCount,
                schedulerEnabled = config.scheduler.enabled,
                schedulerPolicy = config.scheduler.policy,
                schedulerAllowUnlistedDevices = config.scheduler.allowUnlistedDevices,
                schedulerPreferConfiguredDevices = config.scheduler.preferConfiguredDevices,
                schedulerRebalanceOnRegister = config.scheduler.rebalanceOnRegister,
                schedulerEventCount = schedulerEvents.size
            ),
            stages = stageSnapshots,
            nodes = nodeSnapshots,
            schedulerEvents = schedulerEvents.toList()
        )
    }

    fun reportRequestEvent(event: Sid.RequestEvent): Sid.RequestEventAck = lock.withLock {
        if (event.requestId.isBlank()) {
            return Sid.RequestEventAck.newBuilder()
                .setAck(false)
                .setMessage("request_id must not be blank")
                .build()
        }

        val observedEpochMs = Instant.now().toEpochMilli()
        val message = if (event.eventEpochMs > 0L) {
            "${event.message} workerEventEpochMs=${event.eventEpochMs}"
        } else {
            event.message
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
                message = message,
                eventEpochMs = observedEpochMs,
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

    fun listRecentRuns(limit: Int): List<AdminRunSummarySnapshot> {
        return persistence.listRecentRuns(limit.coerceIn(1, 500))
            .map { summary ->
                AdminRunSummarySnapshot(
                    runId = summary.runId,
                    pipelineName = summary.pipelineName,
                    modelShards = summary.modelShards,
                    configHash = summary.configHash,
                    firstSeenEpochMs = summary.firstSeenEpochMs,
                    lastUpdatedEpochMs = summary.lastUpdatedEpochMs,
                    requestRows = summary.requestRows,
                    successRows = summary.successRows,
                    failedRows = summary.failedRows,
                    evalOnlyRows = summary.evalOnlyRows,
                    trainRows = summary.trainRows,
                    avgElapsedMs = summary.avgElapsedMs,
                    avgLoss = summary.avgLoss,
                    tokenCorrect = summary.tokenCorrect,
                    tokenCount = summary.tokenCount,
                    tokenAccuracy = summary.tokenAccuracy
                )
            }
    }

    fun loadRunDetail(runId: String, metricLimit: Int): AdminRunDetailSnapshot {
        return persistence.loadRunDetail(runId, metricLimit.coerceIn(1, 10_000))
    }

    fun exportRunMetricsCsv(runId: String, metricLimit: Int): String {
        val header = listOf(
            "metric_id",
            "run_id",
            "request_id",
            "request_index",
            "attempt",
            "batch_id",
            "eval_only",
            "submitted_epoch_ms",
            "completed_epoch_ms",
            "elapsed_ms",
            "success",
            "terminal",
            "processed_stage_id",
            "processed_chunk_idx",
            "output_hidden_bytes",
            "output_shift_log_p_bytes",
            "local_loss",
            "token_correct",
            "token_count",
            "token_accuracy",
            "message"
        )
        val lines = mutableListOf(header.joinToString(","))
        persistence.listRunMetrics(runId, metricLimit.coerceIn(1, 100_000)).forEach { metric ->
            lines += listOf(
                metric.metricId.toString(),
                metric.runId,
                metric.requestId,
                metric.requestIndex?.toString().orEmpty(),
                metric.attempt.toString(),
                metric.batchId.toString(),
                metric.evalOnly.toString(),
                metric.submittedEpochMs.toString(),
                metric.completedEpochMs.toString(),
                metric.elapsedMs.toString(),
                metric.success.toString(),
                metric.terminal.toString(),
                metric.processedStageId.toString(),
                metric.processedChunkIdx.toString(),
                metric.outputHiddenBytes.toString(),
                metric.outputShiftLogPBytes.toString(),
                metric.localLoss?.toString().orEmpty(),
                metric.tokenCorrect.toString(),
                metric.tokenCount.toString(),
                (if (metric.tokenCount == 0) 0.0 else metric.tokenCorrect.toDouble() / metric.tokenCount.toDouble()).toString(),
                metric.message
            ).joinToString(",") { it.csvEscape() }
        }
        return lines.joinToString("\n") + "\n"
    }

    fun exportRunStageTimingsCsv(runId: String, metricLimit: Int): String {
        val header = listOf(
            "stage_metric_id",
            "request_event_id",
            "run_id",
            "request_id",
            "batch_id",
            "chunk_idx",
            "stage_id",
            "node_id",
            "event_type",
            "event_epoch_ms",
            "runtime",
            "method",
            "input_count",
            "eval_only",
            "optimizer_step_applied",
            "local_loss",
            "local_ms",
            "input_build_ms",
            "execute_ms",
            "gradients_ms",
            "optimizer_create_ms",
            "optimizer_step_ms",
            "output_convert_ms",
            "total_measured_ms",
            "forward_ms",
            "total_stage_ms",
            "output_bytes",
            "message"
        )
        val lines = mutableListOf(header.joinToString(","))
        persistence.listStageTimingMetrics(runId, metricLimit.coerceIn(1, 100_000)).forEach { metric ->
            lines += listOf(
                metric.stageMetricId.toString(),
                metric.requestEventId.toString(),
                metric.runId,
                metric.requestId,
                metric.batchId.toString(),
                metric.chunkIdx.toString(),
                metric.stageId.toString(),
                metric.nodeId.toString(),
                metric.eventType,
                metric.eventEpochMs.toString(),
                metric.runtime.orEmpty(),
                metric.method.orEmpty(),
                metric.inputCount?.toString().orEmpty(),
                metric.evalOnly?.toString().orEmpty(),
                metric.optimizerStepApplied?.toString().orEmpty(),
                metric.localLoss?.toString().orEmpty(),
                metric.localMs?.toString().orEmpty(),
                metric.inputBuildMs?.toString().orEmpty(),
                metric.executeMs?.toString().orEmpty(),
                metric.gradientsMs?.toString().orEmpty(),
                metric.optimizerCreateMs?.toString().orEmpty(),
                metric.optimizerStepMs?.toString().orEmpty(),
                metric.outputConvertMs?.toString().orEmpty(),
                metric.totalMeasuredMs?.toString().orEmpty(),
                metric.forwardMs?.toString().orEmpty(),
                metric.totalStageMs?.toString().orEmpty(),
                metric.outputBytes?.toString().orEmpty(),
                metric.message
            ).joinToString(",") { it.csvEscape() }
        }
        return lines.joinToString("\n") + "\n"
    }

    fun listRecentWorkerTelemetry(limit: Int, deviceId: String?): List<AdminWorkerTelemetrySnapshot> {
        return persistence.listRecentWorkerTelemetry(limit.coerceIn(1, 100_000), deviceId)
            .map { it.toAdminSnapshot() }
    }

    fun exportWorkerTelemetryCsv(limit: Int, deviceId: String?): String {
        val header = listOf(
            "telemetry_id",
            "observed_epoch_ms",
            "device_id",
            "node_id",
            "stage_id",
            "is_active",
            "battery_level",
            "is_charging",
            "power_source",
            "battery_status",
            "battery_temp_c",
            "battery_voltage_mv",
            "battery_current_ua",
            "thermal_status",
            "app_pss_kb",
            "app_private_dirty_kb",
            "runtime_used_memory_kb",
            "worker_state"
        )
        val lines = mutableListOf(header.joinToString(","))
        persistence.listRecentWorkerTelemetry(limit.coerceIn(1, 100_000), deviceId)
            .asReversed()
            .forEach { telemetry ->
                lines += listOf(
                    telemetry.telemetryId.toString(),
                    telemetry.observedEpochMs.toString(),
                    telemetry.deviceId,
                    telemetry.nodeId.toString(),
                    telemetry.stageId.toString(),
                    telemetry.isActive.toString(),
                    telemetry.batteryLevel.toString(),
                    telemetry.isCharging.toString(),
                    telemetry.powerSource,
                    telemetry.batteryStatus.toString(),
                    telemetry.batteryTempC?.toString().orEmpty(),
                    telemetry.batteryVoltageMv?.toString().orEmpty(),
                    telemetry.batteryCurrentUa?.toString().orEmpty(),
                    telemetry.thermalStatus,
                    telemetry.appPssKb?.toString().orEmpty(),
                    telemetry.appPrivateDirtyKb?.toString().orEmpty(),
                    telemetry.runtimeUsedMemoryKb?.toString().orEmpty(),
                    telemetry.workerState
                ).joinToString(",") { it.csvEscape() }
            }
        return lines.joinToString("\n") + "\n"
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

    fun planStageRequestSubmission(stageId: Int): RequestSubmissionPlan = lock.withLock {
        val stage = config.stages.firstOrNull { it.stageId == stageId }
            ?: return RequestSubmissionPlan(
                accepted = false,
                stageId = stageId,
                nodeId = -1,
                host = null,
                port = null,
                message = "Stage $stageId is not configured."
            )
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
        val now = Instant.now()
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
        RequestSubmissionPlan(
            accepted = true,
            stageId = stage.stageId,
            nodeId = node.nodeId,
            host = node.ipAddress,
            port = node.grpcPort,
            message = "Dispatching to stage ${stage.stageId} worker ${node.nodeId}"
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

    fun recordRequestMetric(metric: PersistedRequestMetric) {
        lock.withLock {
            persistence.appendRequestMetric(
                metric = metric,
                pipelineName = config.pipelineName,
                modelShards = config.stages.joinToString("|") { stage ->
                    "${stage.stageId}:${stage.modelShardId}:${stage.modelSha256.ifBlank { "sha256-unset" }}"
                },
                configHash = configHash()
            )
        }
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

    private fun persistReassignedNodes(nodes: Collection<RegisteredNode>) {
        val uniqueNodes = nodes.associateBy(RegisteredNode::nodeId).values
        uniqueNodes.forEach { node ->
            persistence.deleteNode(node.nodeId, routingEpoch.get(), nextNodeId.get())
        }
        uniqueNodes.forEach(::persistNode)
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
            assignmentReason = assignmentReason,
            registeredAtEpochMs = registeredAt.toEpochMilli(),
            lastHeartbeatAtEpochMs = lastHeartbeatAt.toEpochMilli(),
            isActive = isActive,
            isExpired = expired,
            isLive = isActive && !expired
        )
    }

    private fun recordSchedulerEvent(
        action: String,
        stageId: Int?,
        nodeId: Int?,
        deviceId: String?,
        reason: String,
        message: String
    ) {
        val persisted = persistence.appendSchedulerEvent(
            AdminSchedulerEventSnapshot(
                eventId = 0L,
                eventEpochMs = Instant.now().toEpochMilli(),
                action = action,
                stageId = stageId,
                nodeId = nodeId,
                deviceId = deviceId,
                reason = reason,
                message = message
            )
        )
        nextSchedulerEventId = maxOf(nextSchedulerEventId, persisted.eventId + 1)
        schedulerEvents.addFirst(persisted)
        while (schedulerEvents.size > MAX_SCHEDULER_EVENTS) {
            schedulerEvents.removeLast()
        }
    }

    private fun configHash(): String {
        val material = buildString {
            append(config.pipelineName)
            append('|')
            append(config.scheduler)
            append('|')
            config.stages.forEach { stage ->
                append(stage.stageId)
                append(':')
                append(stage.deviceId)
                append(':')
                append(stage.modelShardId)
                append(':')
                append(stage.modelSha256)
                append(':')
                append(stage.modelBytes)
                append(';')
            }
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(material.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }
    }

    private fun String.csvEscape(): String {
        val escaped = replace("\"", "\"\"")
        return if (escaped.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"$escaped\""
        } else {
            escaped
        }
    }

    private fun PersistedWorkerTelemetry.toAdminSnapshot(): AdminWorkerTelemetrySnapshot {
        return AdminWorkerTelemetrySnapshot(
            telemetryId = telemetryId,
            observedEpochMs = observedEpochMs,
            deviceId = deviceId,
            nodeId = nodeId,
            stageId = stageId,
            isActive = isActive,
            batteryLevel = batteryLevel,
            isCharging = isCharging,
            powerSource = powerSource,
            batteryStatus = batteryStatus,
            batteryTempC = batteryTempC,
            batteryVoltageMv = batteryVoltageMv,
            batteryCurrentUa = batteryCurrentUa,
            thermalStatus = thermalStatus,
            appPssKb = appPssKb,
            appPrivateDirtyKb = appPrivateDirtyKb,
            runtimeUsedMemoryKb = runtimeUsedMemoryKb,
            workerState = workerState
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

    private data class StageAssignment(
        val stage: StageConfig,
        val reason: String,
        val displacedNodeId: Int? = null,
        val displacedStage: StageConfig? = null,
        val displacedReason: String? = null
    )

    data class RequestSubmissionPlan(
        val accepted: Boolean,
        val stageId: Int,
        val nodeId: Int,
        val host: String?,
        val port: Int?,
        val message: String
    )

    private companion object {
        private const val MAX_SCHEDULER_EVENTS = 100
    }
}
