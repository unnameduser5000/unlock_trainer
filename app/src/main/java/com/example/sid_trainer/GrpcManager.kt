package com.example.sid_trainer

import android.util.Log
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.util.concurrent.TimeUnit

data class NextHopInfo(
    val nodeId: Int,
    val host: String,
    val port: Int
)

data class WorkerRegistration(
    val nodeId: Int,
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
    private val coordinatorHost: String,
    private val coordinatorPort: Int,
    private val requestedLocalServerPort: Int,
    private val onChunkReceived: suspend (WorkerRegistration, Sid.ForwardChunkRequest) -> Sid.ForwardChunkResponse,
    private val onRoutingUpdated: (WorkerRegistration) -> Unit = {},
    private val onCoordinatorCommand: (String) -> Unit = {}
) {
    private val maxMessageSizeBytes = 50 * 1024 * 1024
    private val scope = CoroutineScope(Dispatchers.IO)
    private val routingLock = Any()

    private var coordinatorChannel: ManagedChannel? = null
    private var coordinatorStub: CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub? = null
    @Volatile
    private var nextNodeTarget: NextHopInfo? = null
    private var heartbeatJob: Job? = null
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

            val response = requireNotNull(coordinatorStub).registerNode(request)
            if (!response.success) {
                Log.e("GrpcManager", "Registration rejected: ${response.message}")
                return@withContext null
            }

            val registration = WorkerRegistration(
                nodeId = response.nodeId,
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

            if (startHeartbeat) {
                startHeartbeat(deviceId)
            }

            registration
        } catch (t: Throwable) {
            Log.e("GrpcManager", "Registration failed", t)
            null
        }
    }

    suspend fun sendDataToNextNode(request: Sid.ForwardChunkRequest): Sid.ForwardChunkResponse =
        withContext(Dispatchers.IO) {
            val target = nextNodeTarget
            if (target == null) {
                return@withContext Sid.ForwardChunkResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("No downstream node connected.")
                    .setProcessedChunkIdx(request.chunkIdx)
                    .setProcessedStageId(-1)
                    .setTerminal(false)
                    .build()
            }

            try {
                HttpForwardChunkClient.forwardChunk(
                    host = target.host,
                    port = target.port,
                    request = request
                )
            } catch (t: Throwable) {
                Log.e("GrpcManager", "Downstream forwarding failed", t)
                Sid.ForwardChunkResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Downstream forwarding failed: ${t.message}")
                    .setProcessedChunkIdx(request.chunkIdx)
                    .setProcessedStageId(-1)
                    .setTerminal(false)
                    .build()
            }
        }

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
            while (isActive) {
                try {
                    val currentNodeId = workerRegistration?.nodeId
                    if (currentNodeId == null) {
                        delay(5_000)
                        continue
                    }
                    val response = requireNotNull(coordinatorStub).heartbeat(
                        Sid.HeartbeatRequest.newBuilder()
                            .setDeviceId(deviceId)
                            .setNodeId(currentNodeId)
                            .setBatteryLevel(100.0f)
                            .setIsActive(workerActiveForScheduling)
                            .build()
                    )
                    if (!response.ack && response.command.equals("REREGISTER", ignoreCase = true)) {
                        Log.w("GrpcManager", "Coordinator requested re-registration for deviceId=$deviceId")
                        performRegistration(deviceId, startHeartbeat = false)
                        onCoordinatorCommand("REREGISTER")
                        delay(5_000)
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
                    Log.e("GrpcManager", "Heartbeat failed", t)
                }
                delay(5_000)
            }
        }
    }

    private fun startLocalServer(): Int {
        if (localServer != null) {
            return actualLocalServerPort
        }

        Log.i("GrpcManager", "Starting local data server with preferred port $requestedLocalServerPort")
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
        Log.i("GrpcManager", "Local data server is listening on port $actualLocalServerPort")
        return actualLocalServerPort
    }

    private fun applyRoutingUpdate(updated: WorkerRegistration) {
        synchronized(routingLock) {
            val previous = workerRegistration
            workerRegistration = updated

            if (updated.isTerminal || updated.nextHop == null) {
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
