package com.example.sid_trainer

import android.content.Context
import android.util.Log
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.io.File
import java.util.concurrent.TimeUnit

data class NextHopInfo(
    val nodeId: Int,
    val host: String,
    val port: Int
)

data class WorkerRegistration(
    val nodeId: Int,
    val deviceId: String,
    val stageId: Int,
    val isTerminal: Boolean,
    val nextHop: NextHopInfo?,
    val modelShardId: String,
    val modelDownloadUrl: String,
    val modelSha256: String,
    val modelBytes: Long,
    val routingEpoch: Long,
    val routeReady: Boolean
) {
    fun describe(): String {
        return buildString {
            append("node=")
            append(nodeId)
            append(", device=")
            append(deviceId)
            append(", stage=")
            append(stageId)
            append(", terminal=")
            append(isTerminal)
            append(", shard=")
            append(modelShardId.ifBlank { "unknown" })
            if (modelBytes > 0) {
                append(", bytes=")
                append(modelBytes)
            }
            append(", ready=")
            append(routeReady)
            append(", epoch=")
            append(routingEpoch)
            nextHop?.let {
                append(", next=")
                append(it.host)
                append(':')
                append(it.port)
            }
        }
    }
}

class GrpcManager(
    private val appContext: Context,
    private val coordinatorHost: String,
    private val coordinatorPort: Int,
    private val requestedLocalServerPort: Int,
    private val onChunkReceived: suspend (WorkerRegistration, Sid.ForwardChunkRequest) -> Sid.ForwardChunkResponse,
    private val onRoutingUpdated: (WorkerRegistration) -> Unit = {},
    private val onCoordinatorCommand: (String) -> Unit = {}
) {
    private val maxMessageSizeBytes = 50 * 1024 * 1024
    private val heartbeatIntervalMs = 5_000L
    private val heartbeatRpcDeadlineMs = 3_000L
    private val heartbeatCoroutineTimeoutMs = 4_000L
    private val registrationRpcDeadlineMs = 5_000L
    private val durableForwardRetryDelayMs = 2_000L
    private val scope = CoroutineScope(Dispatchers.IO)
    private val routingLock = Any()
    private val durableBoundaryOutbox = DurableBoundaryOutbox(
        File(appContext.filesDir, "durable_boundary_outbox")
    )

    private var coordinatorChannel: ManagedChannel? = null
    private var coordinatorStub: CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub? = null
    @Volatile
    private var nextNodeTarget: NextHopInfo? = null
    private var heartbeatJob: Job? = null
    private var durableForwardJob: Job? = null
    private var localServer: HttpForwardChunkServer? = null
    @Volatile
    private var actualLocalServerPort: Int = requestedLocalServerPort
    @Volatile
    private var workerRegistration: WorkerRegistration? = null
    @Volatile
    private var workerActiveForScheduling = false

    init {
        coordinatorChannel = ManagedChannelBuilder.forAddress(coordinatorHost, coordinatorPort)
            .usePlaintext()
            .maxInboundMessageSize(maxMessageSizeBytes)
            .build()
        coordinatorStub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(
            requireNotNull(coordinatorChannel)
        )
    }

    suspend fun registerNode(deviceId: String): WorkerRegistration? = withContext(Dispatchers.IO) {
        performRegistration(deviceId, startHeartbeat = true)
    }

    private suspend fun performRegistration(
        deviceId: String,
        startHeartbeat: Boolean
    ): WorkerRegistration? = withContext(Dispatchers.IO) {
        try {
            val localIpAddress = NetworkUtils.findSiteLocalIpv4Address() ?: "127.0.0.1"
            val request = Sid.NodeInfo.newBuilder()
                .setDeviceId(deviceId)
                .setNodeType("Android_Worker")
                .setComputeCapacity(100.0f)
                .setMemoryGb(4.0f)
                .setIpAddress(localIpAddress)
                .setGrpcPort(actualLocalServerPort)
                .build()

            val response = requireNotNull(coordinatorStub)
                .withDeadlineAfter(registrationRpcDeadlineMs, TimeUnit.MILLISECONDS)
                .registerNode(request)
            if (!response.success) {
                Log.e("GrpcManager", "Registration rejected: ${response.message}")
                return@withContext null
            }

            val registration = WorkerRegistration(
                nodeId = response.nodeId,
                deviceId = deviceId,
                stageId = response.stageId,
                isTerminal = response.terminal,
                nextHop = response.toNextHopInfo(),
                modelShardId = response.modelShardId,
                modelDownloadUrl = response.modelDownloadUrl,
                modelSha256 = response.modelSha256,
                modelBytes = response.modelBytes,
                routingEpoch = response.routingEpoch,
                routeReady = response.routeReady
            )
            applyRoutingUpdate(registration)
            durableBoundaryOutbox.activateStage(registration.stageId)
            startDurableForwarding()

            if (startHeartbeat) {
                startHeartbeat(deviceId)
            }

            registration
        } catch (t: Throwable) {
            Log.e("GrpcManager", "Registration failed", t)
            null
        }
    }

    suspend fun sendDataToNextNode(request: Sid.ForwardChunkRequest): HttpForwardChunkResult =
        withContext(Dispatchers.IO) {
            val target = nextNodeTarget
            if (target == null) {
                return@withContext HttpForwardChunkResult(
                    response = Sid.ForwardChunkResponse.newBuilder()
                        .setSuccess(false)
                        .setMessage("No downstream node connected.")
                        .setProcessedChunkIdx(request.chunkIdx)
                        .setProcessedStageId(-1)
                        .setTerminal(false)
                        .setFirstUnprocessedStageId(request.chunkIdx)
                        .setFailureKind(
                            Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DOWNSTREAM_UNAVAILABLE
                        )
                        .build(),
                    transport = ForwardChunkTransportMetrics(),
                    deliveryStatus = ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED
                )
            }

            try {
                HttpForwardChunkClient.forwardChunk(
                    host = target.host,
                    port = target.port,
                    request = request
                )
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                val deliveryStatus = if (t is ForwardConnectFailure) {
                    ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED
                } else {
                    ForwardDeliveryStatus.DELIVERY_UNKNOWN
                }
                val failureKind = if (deliveryStatus == ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED) {
                    Sid.RequestFailureKind.REQUEST_FAILURE_KIND_WORKER_UNAVAILABLE
                } else {
                    Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DOWNSTREAM_UNAVAILABLE
                }
                Log.e("GrpcManager", "Downstream forwarding failed", t)
                HttpForwardChunkResult(
                    response = Sid.ForwardChunkResponse.newBuilder()
                        .setSuccess(false)
                        .setMessage(
                            "Downstream forwarding failed: ${t.message}; " +
                                "deliveryStatus=$deliveryStatus"
                        )
                        .setProcessedChunkIdx(request.chunkIdx)
                        .setProcessedStageId(-1)
                        .setTerminal(false)
                        .setFirstUnprocessedStageId(request.chunkIdx)
                        .setFailureKind(failureKind)
                        .build(),
                    transport = ForwardChunkTransportMetrics(),
                    deliveryStatus = deliveryStatus
                )
            }
        }

    internal fun prepareDurableBoundary(
        request: Sid.ForwardChunkRequest,
        capacity: Int
    ): DurableBoundaryPrepareResult = durableBoundaryOutbox.prepare(request, capacity)

    internal fun publishDurableBoundary(requestId: String): DurableBoundaryEntry =
        durableBoundaryOutbox.markReady(requestId)

    internal fun resolveDurableBoundaryReplay(
        request: Sid.ForwardChunkRequest
    ): DurableBoundaryReplayResolution = durableBoundaryOutbox.resolveCommittedReplay(request)

    internal fun abortPreparedDurableBoundary(requestId: String): Boolean =
        durableBoundaryOutbox.abortPrepared(requestId)

    internal fun durableBoundaryPendingCount(): Int = durableBoundaryOutbox.pendingCount()

    suspend fun reportRequestEvent(
        registration: WorkerRegistration,
        requestId: String,
        batchId: Int,
        chunkIdx: Int,
        eventType: Sid.RequestEventType,
        success: Boolean,
        message: String,
        terminal: Boolean
    ) = withContext(Dispatchers.IO) {
        try {
            requireNotNull(coordinatorStub).reportRequestEvent(
                Sid.RequestEvent.newBuilder()
                    .setRequestId(requestId)
                    .setBatchId(batchId)
                    .setChunkIdx(chunkIdx)
                    .setNodeId(registration.nodeId)
                    .setStageId(registration.stageId)
                    .setEventType(eventType)
                    .setSuccess(success)
                    .setMessage(message)
                    .setEventEpochMs(System.currentTimeMillis())
                    .setTerminal(terminal)
                    .build()
            )
        } catch (t: Throwable) {
            Log.e("GrpcManager", "Failed to report request event $eventType for $requestId", t)
        }
    }

    fun shutdown() {
        workerActiveForScheduling = false
        heartbeatJob?.cancel()
        durableForwardJob?.cancel()
        scope.cancel()
        disconnectNextNode()
        coordinatorChannel?.shutdown()?.awaitTermination(2, TimeUnit.SECONDS)
        localServer?.stop()
        localServer = null
    }

    fun startServing(): Int {
        return startLocalServer()
    }

    fun setWorkerActive(isActive: Boolean) {
        workerActiveForScheduling = isActive
    }

    private fun connectToNextNode(host: String, port: Int) = synchronized(routingLock) {
        nextNodeTarget = NextHopInfo(
            nodeId = workerRegistration?.nextHop?.nodeId ?: -1,
            host = host,
            port = port
        )
    }

    private fun disconnectNextNode() = synchronized(routingLock) {
        nextNodeTarget = null
    }

    private fun startHeartbeat(deviceId: String) {
        heartbeatJob?.cancel()
        heartbeatJob = scope.launch {
            var consecutiveFailures = 0
            while (isActive) {
                try {
                    val currentNodeId = workerRegistration?.nodeId
                    if (currentNodeId == null) {
                        delay(heartbeatIntervalMs)
                        continue
                    }
                    val request = WorkerTelemetryReader.read(appContext).applyTo(
                        Sid.HeartbeatRequest.newBuilder()
                            .setDeviceId(deviceId)
                            .setNodeId(currentNodeId)
                            .setIsActive(workerActiveForScheduling),
                        workerState = if (workerActiveForScheduling) "ACTIVE" else "PAUSED"
                    ).build()
                    val startedAtNs = System.nanoTime()
                    val response = withTimeout(heartbeatCoroutineTimeoutMs) {
                        requireNotNull(coordinatorStub)
                            .withDeadlineAfter(heartbeatRpcDeadlineMs, TimeUnit.MILLISECONDS)
                            .heartbeat(request)
                    }
                    val heartbeatMs = ((System.nanoTime() - startedAtNs) / 1_000_000L).coerceAtLeast(0L)
                    if (consecutiveFailures > 0) {
                        Log.i(
                            "GrpcManager",
                            "Heartbeat recovered after $consecutiveFailures failure(s); latencyMs=$heartbeatMs"
                        )
                    }
                    consecutiveFailures = 0
                    if (!response.ack && response.command.equals("REREGISTER", ignoreCase = true)) {
                        Log.w("GrpcManager", "Coordinator requested re-registration for deviceId=$deviceId")
                        performRegistration(deviceId, startHeartbeat = false)
                        onCoordinatorCommand("REREGISTER")
                        delay(heartbeatIntervalMs)
                        continue
                    }
                    if (response.ack && workerRegistration != null) {
                        val current = requireNotNull(workerRegistration)
                        val updated = current.copy(
                            stageId = response.stageId,
                            isTerminal = response.terminal,
                            nextHop = response.toNextHopInfo(),
                            modelShardId = response.modelShardId.ifBlank { current.modelShardId },
                            modelDownloadUrl = response.modelDownloadUrl.ifBlank { current.modelDownloadUrl },
                            modelSha256 = response.modelSha256.ifBlank { current.modelSha256 },
                            modelBytes = response.modelBytes.takeIf { it > 0 } ?: current.modelBytes,
                            routingEpoch = response.routingEpoch,
                            routeReady = response.routeReady
                        )
                        applyRoutingUpdate(updated)
                    }
                    if (response.command.isNotBlank()) {
                        onCoordinatorCommand(response.command)
                    }
                } catch (t: Throwable) {
                    consecutiveFailures++
                    Log.w(
                        "GrpcManager",
                        "Heartbeat failed count=$consecutiveFailures; will retry in ${heartbeatIntervalMs}ms",
                        t
                    )
                }
                delay(heartbeatIntervalMs)
            }
        }
    }

    private fun startDurableForwarding() {
        if (durableForwardJob?.isActive == true) return
        durableForwardJob = scope.launch {
            val attemptsBySequence = mutableMapOf<Long, Int>()
            while (isActive) {
                val entry = durableBoundaryOutbox.peekReady()
                val registration = workerRegistration
                val target = nextNodeTarget
                if (entry == null || registration == null || registration.isTerminal || target == null) {
                    delay(durableForwardRetryDelayMs)
                    continue
                }

                val attempt = (attemptsBySequence[entry.outboxSequence] ?: 0) + 1
                attemptsBySequence[entry.outboxSequence] = attempt
                try {
                    reportRequestEvent(
                        registration = registration,
                        requestId = entry.request.requestId,
                        batchId = entry.request.batchId,
                        chunkIdx = registration.stageId,
                        eventType = Sid.RequestEventType.FORWARDING,
                        success = true,
                        message = "Durable boundary forward attempt=$attempt outboxSeq=${entry.outboxSequence} " +
                            "windowSeq=${entry.request.pipelineWindowSeq} pending=${durableBoundaryOutbox.pendingCount()} " +
                            "target=${target.host}:${target.port}",
                        terminal = false
                    )
                    val call = sendDataToNextNode(entry.request)
                    if (call.response.success) {
                        val pending = durableBoundaryOutbox.acknowledge(entry)
                        attemptsBySequence.remove(entry.outboxSequence)
                        Log.i(
                            "GrpcManager",
                            "Durable boundary acknowledged request=${entry.request.requestId} " +
                                "outboxSeq=${entry.outboxSequence} windowSeq=${entry.request.pipelineWindowSeq} " +
                                "attempt=$attempt pending=$pending downstreamStage=${call.response.processedStageId} " +
                                "terminal=${call.response.terminal}"
                        )
                        reportRequestEvent(
                            registration = registration,
                            requestId = entry.request.requestId,
                            batchId = entry.request.batchId,
                            chunkIdx = registration.stageId,
                            eventType = Sid.RequestEventType.BOUNDARY_ACKNOWLEDGED,
                            success = true,
                            message = "Durable boundary acknowledged outboxSeq=${entry.outboxSequence} " +
                                "windowSeq=${entry.request.pipelineWindowSeq} attempt=$attempt pending=$pending " +
                                "downstreamStage=${call.response.processedStageId}",
                            terminal = call.response.terminal
                        )
                        continue
                    }

                    Log.w(
                        "GrpcManager",
                        "Durable boundary retained request=${entry.request.requestId} " +
                            "outboxSeq=${entry.outboxSequence} attempt=$attempt message=${call.response.message}"
                    )
                    reportRequestEvent(
                        registration = registration,
                        requestId = entry.request.requestId,
                        batchId = entry.request.batchId,
                        chunkIdx = registration.stageId,
                        eventType = Sid.RequestEventType.BOUNDARY_FORWARD_RETRY,
                        success = false,
                        message = "Durable boundary retained for retry outboxSeq=${entry.outboxSequence} " +
                            "windowSeq=${entry.request.pipelineWindowSeq} attempt=$attempt: ${call.response.message}",
                        terminal = false
                    )
                } catch (t: Throwable) {
                    Log.e(
                        "GrpcManager",
                        "Durable boundary forwarding loop failed request=${entry.request.requestId} " +
                            "outboxSeq=${entry.outboxSequence}; retaining entry",
                        t
                    )
                }
                delay(durableForwardRetryDelayMs)
            }
        }
    }

    private fun startLocalServer(): Int {
        if (localServer != null) {
            return actualLocalServerPort
        }

        localServer = HttpForwardChunkServer(bindPort = requestedLocalServerPort) handler@{ request ->
            val registration = workerRegistration
            if (registration == null) {
                return@handler Sid.ForwardChunkResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Worker received a chunk before registration finished.")
                    .setProcessedChunkIdx(request.chunkIdx)
                    .setProcessedStageId(-1)
                    .setTerminal(false)
                    .build()
            }
            onChunkReceived(registration, request)
        }
        actualLocalServerPort = requireNotNull(localServer).start()
        return actualLocalServerPort
    }

    private fun applyRoutingUpdate(updated: WorkerRegistration) {
        synchronized(routingLock) {
            val previous = workerRegistration
            workerRegistration = updated

            if (updated.isTerminal || !updated.routeReady || updated.nextHop == null) {
                disconnectNextNode()
            } else if (previous?.nextHop != updated.nextHop) {
                connectToNextNode(updated.nextHop.host, updated.nextHop.port)
            } else if (nextNodeTarget == null) {
                connectToNextNode(updated.nextHop.host, updated.nextHop.port)
            }

            if (previous == null || previous != updated) {
                onRoutingUpdated(updated)
            }
        }
    }

    private fun Sid.RegistrationResponse.toNextHopInfo(): NextHopInfo? {
        if (!hasNextHop()) {
            return null
        }
        return NextHopInfo(
            nodeId = nextHop.nodeId,
            host = nextHop.ipAddress,
            port = nextHop.grpcPort
        )
    }

    private fun Sid.HeartbeatResponse.toNextHopInfo(): NextHopInfo? {
        if (!hasNextHop()) {
            return null
        }
        return NextHopInfo(
            nodeId = nextHop.nodeId,
            host = nextHop.ipAddress,
            port = nextHop.grpcPort
        )
    }
}
