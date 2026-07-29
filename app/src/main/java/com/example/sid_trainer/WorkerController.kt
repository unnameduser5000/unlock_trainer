package com.example.sid_trainer

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

internal data class WorkerStartConfig(
    val coordinatorHost: String,
    val coordinatorPort: Int,
    val deviceId: String,
    val localServerPort: Int
)

internal class WorkerController(
    context: Context,
    private val onRunningChanged: (Boolean) -> Unit,
    private val onModelPathChanged: (String) -> Unit,
    private val onModelCacheChanged: (String) -> Unit,
    private val onRoutingChanged: (String) -> Unit,
    private val localLog: (String, Throwable?) -> Unit
) {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val forwardChunkProcessor = ForwardChunkProcessor(
        context = appContext,
        acceptsNewChunks = { acceptsNewChunks },
        localLog = localLog
    )

    @Volatile
    private var acceptsNewChunks = true

    @Volatile
    private var activeModelPath: String? = null

    @Volatile
    private var selectedModelPath: String? = null

    private var workerJob: Job? = null
    private var activeGrpcManager: GrpcManager? = null

    fun importModel(uri: Uri) {
        log("Copying shard into app storage...")
        val destination = File(appContext.filesDir, "chunk_model.pte")
        appContext.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(destination).use { output ->
                input.copyTo(output)
            }
        }
        setModelPath(destination.absolutePath)
        log("Model imported: ${destination.absolutePath}")
        updateModelCacheSummary(
            message = "Manual shard imported",
            source = "manual",
            file = destination
        )
    }

    fun start(config: WorkerStartConfig) {
        setRunning(true)
        acceptsNewChunks = true
        activeModelPath = null
        setRoutingSummary("Registering...")
        updateModelCacheSummary(
            message = "Waiting for coordinator shard assignment...",
            source = "pending",
            file = selectedModelPath?.let(::File)
        )
        log(
            "Starting worker mainline with coordinator=${config.coordinatorHost}:${config.coordinatorPort}, " +
                "requestedLocalDataPort=${config.localServerPort}"
        )
        log("Starting worker mainline...")

        workerJob = CoroutineScope(Dispatchers.IO).launch {
            var grpcManager: GrpcManager? = null
            try {
                grpcManager = GrpcManager(
                    appContext = appContext,
                    coordinatorHost = config.coordinatorHost,
                    coordinatorPort = config.coordinatorPort,
                    requestedLocalServerPort = config.localServerPort,
                    onChunkReceived = { registration, request ->
                        forwardChunkProcessor.process(
                            grpcManager = requireNotNull(activeGrpcManager),
                            registration = registration,
                            modelPath = requireNotNull(activeModelPath) {
                                "No active model path available for stage ${registration.stageId}."
                            },
                            request = request
                        )
                    },
                    onRoutingUpdated = ::handleRoutingUpdate,
                    onCoordinatorCommand = ::handleCoordinatorCommand
                )

                activeGrpcManager = grpcManager
                val actualLocalPort = grpcManager.startServing()
                log("Local data server is listening on port $actualLocalPort")
                val registration = grpcManager.registerNode(config.deviceId)
                if (registration == null) {
                    log("Registration failed.")
                    setRoutingSummary("Registration failed")
                    return@launch
                }
                grpcManager.setWorkerActive(false)

                log(
                    "Registered node=${registration.nodeId}, stage=${registration.stageId}, " +
                        "terminal=${registration.isTerminal}, routeReady=${registration.routeReady}"
                )
                updateModelCacheSummary(
                    message = "Coordinator assigned shard ${registration.modelShardId}",
                    source = if (registration.modelDownloadUrl.isNotBlank()) "auto-download" else "manual",
                    shardId = registration.modelShardId,
                    expectedBytes = registration.modelBytes,
                    file = selectedModelPath?.let(::File)?.takeIf(File::exists)
                )
                registration.nextHop?.let { nextHop ->
                    log("Connected next hop ${nextHop.host}:${nextHop.port} (node ${nextHop.nodeId})")
                }
                val preparedModelPath = prepareModelPath(registration)
                if (preparedModelPath == null) {
                    log("Worker could not prepare a shard file.")
                    setRoutingSummary("Model preparation failed")
                    return@launch
                }
                activeModelPath = preparedModelPath
                grpcManager.setWorkerActive(true)
                log("Worker is active with shard ${registration.modelShardId}.")

                while (isActive) {
                    delay(1_000)
                }
            } catch (cancelled: CancellationException) {
                log("Worker cancelled.")
            } catch (t: Throwable) {
                localLog("Worker crashed: ${t.message}", t)
            } finally {
                activeModelPath = null
                grpcManager?.shutdown()
                activeGrpcManager = null
                setRunning(false)
                setRoutingSummary("Stopped")
                log("Worker stopped.")
            }
        }
    }

    fun stop() {
        log("Stopping worker...")
        acceptsNewChunks = false
        activeGrpcManager?.setWorkerActive(false)
        workerJob?.cancel()
        workerJob = null
    }

    private fun handleRoutingUpdate(registration: WorkerRegistration) {
        setRoutingSummary(registration.describe())
        log(
            "Route update: stage=${registration.stageId}, ready=${registration.routeReady}, " +
                "terminal=${registration.isTerminal}, epoch=${registration.routingEpoch}"
        )
        registration.nextHop?.let { nextHop ->
            log("Route next hop ${nextHop.host}:${nextHop.port} (node ${nextHop.nodeId})")
        }
    }

    private fun handleCoordinatorCommand(command: String) {
        if (command.isNotBlank()) {
            log("Coordinator command: $command")
        }
        when {
            command.equals("DRAIN", ignoreCase = true) -> {
                acceptsNewChunks = false
                activeGrpcManager?.setWorkerActive(false)
            }

            command.equals("PAUSED", ignoreCase = true) -> {
                // PAUSED is the coordinator echo for a worker-reported inactive heartbeat.
            }

            command.equals("RESUME", ignoreCase = true) ||
                command.equals("WAIT_DOWNSTREAM", ignoreCase = true) ||
                command.equals("REREGISTER", ignoreCase = true) -> {
                acceptsNewChunks = true
                activeGrpcManager?.setWorkerActive(true)
            }

            command.equals("SHUTDOWN", ignoreCase = true) -> stop()
        }
    }

    private suspend fun prepareModelPath(registration: WorkerRegistration): String? {
        if (registration.modelDownloadUrl.isBlank()) {
            return prepareLocalModelPath(registration)
        }

        log("Downloading shard ${registration.modelShardId} from ${registration.modelDownloadUrl}")
        updateModelCacheSummary(
            message = "Downloading shard from coordinator",
            source = "auto-download",
            shardId = registration.modelShardId,
            expectedBytes = registration.modelBytes
        )
        return try {
            val preparedArtifact = ModelArtifactManager.ensureModel(
                filesDir = appContext.filesDir,
                registration = registration,
                onLog = { message -> log(message) }
            )
            setModelPath(preparedArtifact.absolutePath)
            log("Shard ready at ${preparedArtifact.absolutePath}")
            updateModelCacheSummary(
                message = if (preparedArtifact.cacheHit) "Using cached shard" else "Shard downloaded and verified",
                source = "auto-download",
                shardId = registration.modelShardId,
                file = File(preparedArtifact.absolutePath),
                expectedBytes = registration.modelBytes,
                cacheHit = preparedArtifact.cacheHit
            )
            preparedArtifact.absolutePath
        } catch (t: Throwable) {
            log("Shard download failed: ${t.message}")
            updateModelCacheSummary(
                message = "Shard download failed: ${t.message}",
                source = "auto-download",
                shardId = registration.modelShardId,
                expectedBytes = registration.modelBytes
            )
            null
        }
    }

    private fun prepareLocalModelPath(registration: WorkerRegistration): String? {
        val cachedShard = File(appContext.filesDir, "shards/${registration.modelShardId}.pte")
            .takeIf(File::isFile)
        val localPath = selectedModelPath ?: cachedShard?.absolutePath
        if (localPath.isNullOrBlank()) {
            log("No auto-download URL provided and no local shard imported.")
            updateModelCacheSummary(
                message = "No local shard available",
                source = "manual",
                shardId = registration.modelShardId,
                expectedBytes = registration.modelBytes
            )
            return null
        }

        setModelPath(localPath)
        val source = if (cachedShard?.absolutePath == localPath) "device-cache" else "manual"
        log("Using $source shard $localPath")
        updateModelCacheSummary(
            message = "Using $source shard",
            source = source,
            shardId = registration.modelShardId,
            file = File(localPath).takeIf(File::exists),
            expectedBytes = registration.modelBytes
        )
        return localPath
    }

    private fun updateModelCacheSummary(
        message: String,
        source: String,
        shardId: String? = null,
        file: File? = null,
        expectedBytes: Long = 0L,
        cacheHit: Boolean? = null
    ) {
        val lines = mutableListOf(message, "source=$source")
        shardId?.takeIf(String::isNotBlank)?.let { lines += "shard=$it" }
        file?.let {
            lines += "path=${it.absolutePath}"
            if (it.exists()) {
                lines += "size=${humanReadableBytes(it.length())}"
            }
        }
        if (expectedBytes > 0) {
            lines += "expected=${humanReadableBytes(expectedBytes)}"
        }
        cacheHit?.let { lines += "cache=${if (it) "hit" else "miss"}" }
        publishUi { onModelCacheChanged(lines.joinToString("\n")) }
    }

    private fun setRunning(running: Boolean) = publishUi { onRunningChanged(running) }

    private fun setModelPath(path: String) {
        selectedModelPath = path
        publishUi { onModelPathChanged(path) }
    }

    private fun setRoutingSummary(summary: String) = publishUi { onRoutingChanged(summary) }

    private fun publishUi(action: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            action()
        } else {
            mainHandler.post { action() }
        }
    }

    private fun log(message: String) = localLog(message, null)

    private fun humanReadableBytes(bytes: Long): String {
        if (bytes <= 0) {
            return "unknown"
        }
        val kib = 1024L
        val mib = kib * 1024L
        val gib = mib * 1024L
        return when {
            bytes >= gib -> String.format("%.2f GiB", bytes.toDouble() / gib.toDouble())
            bytes >= mib -> String.format("%.2f MiB", bytes.toDouble() / mib.toDouble())
            bytes >= kib -> String.format("%.2f KiB", bytes.toDouble() / kib.toDouble())
            else -> "$bytes B"
        }
    }
}
