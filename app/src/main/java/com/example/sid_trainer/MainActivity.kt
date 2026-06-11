package com.example.sid_trainer

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Trace
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.google.protobuf.ByteString
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import sid.Sid
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    companion object {
        private const val UI_LOG_TAG = "SidWorkerUi"
        private const val ENABLE_TOPK_BELIEF_TRANSPORT = false
        private const val EXTRA_COORDINATOR_HOST = "sid.coordinator_host"
        private const val EXTRA_COORDINATOR_PORT = "sid.coordinator_port"
        private const val EXTRA_DEVICE_ID = "sid.device_id"
        private const val EXTRA_LOCAL_PORT = "sid.local_port"
        private const val EXTRA_AUTO_START = "sid.auto_start"
    }

    private val logMessages = mutableStateListOf<String>()
    private val isWorkerRunning = mutableStateOf(false)
    private val modelFilePath = mutableStateOf<String?>(null)
    private val modelCacheSummary = mutableStateOf("No shard prepared")
    private val coordinatorHost = mutableStateOf("192.168.1.10")
    private val coordinatorPort = mutableStateOf("50051")
    private val deviceId = mutableStateOf(defaultDeviceId())
    private val localServerPort = mutableStateOf("26052")
    private val routingSummary = mutableStateOf("Not registered")
    @Volatile
    private var acceptsNewChunks = true
    @Volatile
    private var activeModelPath: String? = null
    private val localExecutionMutex = Mutex()

    private var workerJob: Job? = null
    private var activeGrpcManager: GrpcManager? = null
    private var autoStartRequested = false
    private var autoStartOnLaunch = false

    private val pickModelLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { importModel(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyLaunchOverrides(intent)
        setContent {
            MaterialTheme {
                MainConsoleScreen()
            }
        }
        appendLog("SID worker ready.")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        applyLaunchOverrides(intent)
        appendLog("Launch options updated from adb intent.")
        if (autoStartOnLaunch && !isWorkerRunning.value) {
            startWorker()
        }
    }

    @Composable
    private fun MainConsoleScreen() {
        LaunchedEffect(Unit) {
            if (!autoStartRequested) {
                autoStartRequested = true
                if (autoStartOnLaunch) {
                    appendLog(
                        "Auto-starting worker with coordinator=${coordinatorHost.value}:${coordinatorPort.value}, deviceId=${deviceId.value}, localPort=${localServerPort.value}"
                    )
                    startWorker()
                } else {
                    appendLog("Auto-start disabled. Configure the fields and press Start Worker, or launch with sid.auto_start=true.")
                }
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text("SID Worker", style = MaterialTheme.typography.headlineSmall)
            Spacer(modifier = Modifier.height(12.dp))

            OutlinedTextField(
                value = coordinatorHost.value,
                onValueChange = { coordinatorHost.value = it },
                label = { Text("Coordinator Host") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = coordinatorPort.value,
                onValueChange = { coordinatorPort.value = it.filter(Char::isDigit) },
                label = { Text("Coordinator Port") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = deviceId.value,
                onValueChange = { deviceId.value = it },
                label = { Text("Device ID") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = localServerPort.value,
                onValueChange = { localServerPort.value = it.filter(Char::isDigit) },
                label = { Text("Local Data Port") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text("Model: ${modelFilePath.value ?: "Auto-download if coordinator provides one"}")
            Spacer(modifier = Modifier.height(4.dp))
            Text("Model Cache:\n${modelCacheSummary.value}", style = MaterialTheme.typography.bodySmall)
            Spacer(modifier = Modifier.height(4.dp))
            Text("Route: ${routingSummary.value}")

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Button(onClick = { pickModelLauncher.launch("*/*") }) {
                    Text("Import .pte")
                }

                Button(
                    onClick = { startWorker() },
                    enabled = !isWorkerRunning.value,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Text("Start Worker")
                }

                Button(
                    onClick = { stopWorker() },
                    enabled = isWorkerRunning.value,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828))
                ) {
                    Text("Stop Worker")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Text("Logs", style = MaterialTheme.typography.titleMedium)
            Spacer(modifier = Modifier.height(8.dp))

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
                    .padding(8.dp)
            ) {
                LazyColumn {
                    items(logMessages) { message ->
                        Text(
                            text = message,
                            color = Color(0xFF7CFF7C),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
    }

    private fun importModel(uri: Uri) {
        appendLog("Copying shard into app storage...")
        val destFile = File(filesDir, "chunk_model.pte")
        contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(destFile).use { output ->
                input.copyTo(output)
            }
        }
        modelFilePath.value = destFile.absolutePath
        appendLog("Model imported: ${destFile.absolutePath}")
        updateModelCacheSummary(
            message = "Manual shard imported",
            source = "manual",
            file = destFile
        )
    }

    private fun startWorker() {
        val parsedCoordinatorPort = coordinatorPort.value.toIntOrNull()
        val parsedLocalPort = localServerPort.value.toIntOrNull()
        if (parsedCoordinatorPort == null || parsedLocalPort == null) {
            appendLog("Coordinator port and local data port must be valid numbers.")
            return
        }

        isWorkerRunning.value = true
        acceptsNewChunks = true
        activeModelPath = null
        routingSummary.value = "Registering..."
        updateModelCacheSummary(
            message = "Waiting for coordinator shard assignment...",
            source = "pending",
            file = modelFilePath.value?.let(::File)
        )
        appendLog(
            "Starting worker mainline with coordinator=${coordinatorHost.value}:$parsedCoordinatorPort, requestedLocalDataPort=$parsedLocalPort"
        )
        appendLog("Starting worker mainline...")

        workerJob = CoroutineScope(Dispatchers.IO).launch {
            var grpcManager: GrpcManager? = null
            try {
                grpcManager = GrpcManager(
                    appContext = applicationContext,
                    coordinatorHost = coordinatorHost.value,
                    coordinatorPort = parsedCoordinatorPort,
                    requestedLocalServerPort = parsedLocalPort,
                    onChunkReceived = { registration, request ->
                        handleForwardChunk(
                            grpcManager = requireNotNull(activeGrpcManager),
                            registration = registration,
                            modelPath = requireNotNull(activeModelPath) {
                                "No active model path available for stage ${registration.stageId}."
                            },
                            request = request
                        )
                    },
                    onRoutingUpdated = { registration ->
                        runOnUiThread {
                            routingSummary.value = registration.describe()
                        }
                        appendLog(
                            "Route update: stage=${registration.stageId}, ready=${registration.routeReady}, terminal=${registration.isTerminal}, epoch=${registration.routingEpoch}"
                        )
                        registration.nextHop?.let { nextHop ->
                            appendLog(
                                "Route next hop ${nextHop.host}:${nextHop.port} (node ${nextHop.nodeId})"
                            )
                        }
                    },
                    onCoordinatorCommand = { command ->
                        if (command.isNotBlank()) {
                            appendLog("Coordinator command: $command")
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

                            command.equals("SHUTDOWN", ignoreCase = true) -> {
                                stopWorker()
                            }
                        }
                    }
                )

                activeGrpcManager = grpcManager
                val actualLocalPort = grpcManager.startServing()
                appendLog("Local data server is listening on port $actualLocalPort")
                val registration = grpcManager.registerNode(deviceId.value)
                if (registration == null) {
                    appendLog("Registration failed.")
                    routingSummary.value = "Registration failed"
                    return@launch
                }
                grpcManager.setWorkerActive(false)

                appendLog(
                    "Registered node=${registration.nodeId}, stage=${registration.stageId}, terminal=${registration.isTerminal}, routeReady=${registration.routeReady}"
                )
                updateModelCacheSummary(
                    message = "Coordinator assigned shard ${registration.modelShardId}",
                    source = if (registration.modelDownloadUrl.isNotBlank()) "auto-download" else "manual",
                    shardId = registration.modelShardId,
                    expectedBytes = registration.modelBytes,
                    file = modelFilePath.value?.let(::File)?.takeIf { it.exists() }
                )
                registration.nextHop?.let { nextHop ->
                    appendLog("Connected next hop ${nextHop.host}:${nextHop.port} (node ${nextHop.nodeId})")
                }
                val preparedModelPath = prepareModelPath(registration)
                if (preparedModelPath == null) {
                    appendLog("Worker could not prepare a shard file.")
                    routingSummary.value = "Model preparation failed"
                    return@launch
                }
                activeModelPath = preparedModelPath
                grpcManager.setWorkerActive(true)
                appendLog("Worker is active with shard ${registration.modelShardId}.")

                while (isActive) {
                    delay(1_000)
                }
            } catch (cancelled: CancellationException) {
                appendLog("Worker cancelled.")
            } catch (t: Throwable) {
                Log.e(UI_LOG_TAG, "Worker crashed", t)
                appendLog("Worker crashed: ${t.message}")
            } finally {
                activeModelPath = null
                grpcManager?.shutdown()
                activeGrpcManager = null
                isWorkerRunning.value = false
                routingSummary.value = "Stopped"
                appendLog("Worker stopped.")
            }
        }
    }

    private suspend fun prepareModelPath(registration: WorkerRegistration): String? {
        return if (registration.modelDownloadUrl.isNotBlank()) {
            appendLog(
                "Downloading shard ${registration.modelShardId} from ${registration.modelDownloadUrl}"
            )
            updateModelCacheSummary(
                message = "Downloading shard from coordinator",
                source = "auto-download",
                shardId = registration.modelShardId,
                expectedBytes = registration.modelBytes
            )
            try {
                val preparedArtifact = ModelArtifactManager.ensureModel(
                    filesDir = filesDir,
                    registration = registration,
                    onLog = ::appendLog
                )
                val downloadedPath = preparedArtifact.absolutePath
                modelFilePath.value = downloadedPath
                appendLog("Shard ready at $downloadedPath")
                updateModelCacheSummary(
                    message = if (preparedArtifact.cacheHit) {
                        "Using cached shard"
                    } else {
                        "Shard downloaded and verified"
                    },
                    source = "auto-download",
                    shardId = registration.modelShardId,
                    file = File(downloadedPath),
                    expectedBytes = registration.modelBytes,
                    cacheHit = preparedArtifact.cacheHit
                )
                downloadedPath
            } catch (t: Throwable) {
                appendLog("Shard download failed: ${t.message}")
                updateModelCacheSummary(
                    message = "Shard download failed: ${t.message}",
                    source = "auto-download",
                    shardId = registration.modelShardId,
                    expectedBytes = registration.modelBytes
                )
                null
            }
        } else {
            val manualPath = modelFilePath.value
            if (manualPath.isNullOrBlank()) {
                appendLog("No auto-download URL provided and no local shard imported.")
                updateModelCacheSummary(
                    message = "No local shard available",
                    source = "manual",
                    shardId = registration.modelShardId,
                    expectedBytes = registration.modelBytes
                )
                null
            } else {
                appendLog("Using manually imported shard $manualPath")
                updateModelCacheSummary(
                    message = "Using manually imported shard",
                    source = "manual",
                    shardId = registration.modelShardId,
                    file = File(manualPath).takeIf { it.exists() },
                    expectedBytes = registration.modelBytes
                )
                manualPath
            }
        }
    }

    private suspend fun handleForwardChunk(
        grpcManager: GrpcManager,
        registration: WorkerRegistration,
        modelPath: String,
        request: Sid.ForwardChunkRequest
    ): Sid.ForwardChunkResponse {
        val requestId = request.requestId.ifBlank { "batch-${request.batchId}" }
        val requestReceivedEpochMs = System.currentTimeMillis()
        val chunkStartedAtNs = System.nanoTime()
        val localOnly = request.stopAfterLocalStage
        val beliefTransportMode = normalizeBeliefTransportMode(request.beliefTransportMode)
        appendLog(
            "Chunk received request=$requestId chunk=${request.chunkIdx} " +
                "evalOnly=${request.evalOnly} localOnly=$localOnly beliefMode=$beliefTransportMode"
        )
        grpcManager.reportRequestEvent(
            registration = registration,
            requestId = requestId,
            batchId = request.batchId,
            chunkIdx = request.chunkIdx,
            eventType = Sid.RequestEventType.REQUEST_RECEIVED,
            success = true,
            message = "Chunk received on stage ${registration.stageId}; evalOnly=${request.evalOnly}; localOnly=$localOnly beliefMode=$beliefTransportMode",
            terminal = registration.isTerminal
        )

        if (!acceptsNewChunks) {
            appendLog("Rejecting request=$requestId because worker is not accepting new chunks")
            grpcManager.reportRequestEvent(
                registration = registration,
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = "Worker is drained or paused.",
                terminal = registration.isTerminal
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Worker is drained or paused."
            )
        }

        if (!localOnly && !registration.isTerminal && (!registration.routeReady || registration.nextHop == null)) {
            appendLog(
                "No downstream route available for request=$requestId at epoch=${registration.routingEpoch}"
            )
            grpcManager.reportRequestEvent(
                registration = registration,
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = "Downstream route is not ready.",
                terminal = registration.isTerminal
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Downstream route is not ready."
            )
        }

        val localQueueStartedAtNs = System.nanoTime()
        var localQueueWaitMs = 0L
        lateinit var memoryStats: MemoryPeakStats
        val execution = try {
            localExecutionMutex.withLock {
                localQueueWaitMs = elapsedMsSince(localQueueStartedAtNs)
                val memorySampler = MemoryPeakSampler.create(applicationContext).start()
                Trace.beginSection("sid_worker_local_execute")
                try {
                    NativeShardRunner.execute(modelPath, request)
                } finally {
                    Trace.endSection()
                    memoryStats = memorySampler.stop()
                }
            }
        } catch (t: Throwable) {
            appendLog("Native shard execution failed: ${t.message}")
            grpcManager.reportRequestEvent(
                registration = registration,
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = "Native shard execution failed: ${t.message}",
                terminal = registration.isTerminal
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Native shard execution failed: ${t.message}"
            )
        }
        val localElapsedMs = elapsedMsSince(localQueueStartedAtNs)
        val memoryTimingMessage = memoryDeltaMessage(memoryStats)
        val localTimingMessage = "runtime=${execution.runtimeName} method=${execution.methodName} " +
            "inputs=${execution.inputCount} lr=${execution.learningRate} " +
            "localQueueWaitMs=$localQueueWaitMs localMs=$localElapsedMs " +
            "loss=${execution.localLoss} evalOnly=${execution.evalOnly} " +
            "optimizerStepApplied=${execution.optimizerStepApplied} " +
            "checkpointSaved=${execution.checkpointSaved} checkpointIntervalSteps=${execution.checkpointIntervalSteps} " +
            "${execution.timing.describeForLog()} $memoryTimingMessage"

        appendLog(
            "Local shard finished stage=${registration.stageId}, bytes=${execution.outputHiddenStates.data.size()} " +
                localTimingMessage
        )
        grpcManager.reportRequestEvent(
            registration = registration,
            requestId = requestId,
            batchId = request.batchId,
            chunkIdx = request.chunkIdx,
            eventType = Sid.RequestEventType.LOCAL_COMPLETED,
            success = true,
            message = "Local shard finished on stage ${registration.stageId}; $localTimingMessage",
            terminal = registration.isTerminal
        )

        val stageMetricBuilder = Sid.StageExecutionMetrics.newBuilder()
            .setStageId(registration.stageId)
            .setChunkIdx(request.chunkIdx)
            .setNodeId(registration.nodeId)
            .setDeviceId(registration.deviceId)
            .setTerminal(registration.isTerminal)
            .setEvalOnly(execution.evalOnly)
            .setOptimizerStepApplied(execution.optimizerStepApplied)
            .setCheckpointSaved(execution.checkpointSaved)
            .setCheckpointIntervalSteps(execution.checkpointIntervalSteps)
            .setRuntimeName(execution.runtimeName)
            .setMethodName(execution.methodName)
            .setInputCount(execution.inputCount)
            .setLearningRate(execution.learningRate)
            .setLocalLoss(execution.localLoss)
            .setLocalQueueWaitMs(localQueueWaitMs)
            .setLocalElapsedMs(localElapsedMs)
            .setRuntimeAcquireMs(execution.timing.runtimeAcquireMs)
            .setCheckpointRestoreMs(execution.timing.checkpointRestoreMs)
            .setInputBuildMs(execution.timing.inputBuildMs)
            .setExecuteMs(execution.timing.executeMs)
            .setGradientsMs(execution.timing.gradientsMs)
            .setOptimizerCreateMs(execution.timing.optimizerCreateMs)
            .setOptimizerStepMs(execution.timing.optimizerStepMs)
            .setOutputConvertMs(execution.timing.outputConvertMs)
            .setOutputHiddenBytes(execution.outputHiddenStates.data.size().toLong())
            .setOutputShiftLogPBytes(execution.outputShiftLogP.data.size().toLong())
            .setMemorySampleIntervalMs(memoryStats.intervalMs)
            .setMemorySampleCount(memoryStats.sampleCount)
            .setPssBeforeKb(memoryStats.before.appPssKb)
            .setPssAfterKb(memoryStats.after.appPssKb)
            .setPssPeakKb(memoryStats.pssPeakKb)
            .setPrivateDirtyBeforeKb(memoryStats.before.appPrivateDirtyKb)
            .setPrivateDirtyAfterKb(memoryStats.after.appPrivateDirtyKb)
            .setPrivateDirtyPeakKb(memoryStats.privateDirtyPeakKb)
            .setJavaHeapBeforeKb(memoryStats.before.runtimeUsedMemoryKb)
            .setJavaHeapAfterKb(memoryStats.after.runtimeUsedMemoryKb)
            .setJavaHeapPeakKb(memoryStats.javaHeapPeakKb)
            .setRequestReceivedEpochMs(requestReceivedEpochMs)

        val executionShiftLogP = normalizeExecutionBeliefOutput(execution.outputShiftLogP, beliefTransportMode)
        val responseShiftLogP = if (shouldReturnBeliefForStage(beliefTransportMode, registration.isTerminal)) {
            executionShiftLogP
        } else {
            emptyTensorLike(executionShiftLogP)
        }

        val localResponseBuilder = Sid.ForwardChunkResponse.newBuilder()
            .setSuccess(true)
            .setMessage("Stage ${registration.stageId} finished request $requestId")
            .setLocalLoss(execution.localLoss)
            .setOutputHiddenStates(execution.outputHiddenStates)
            .setOutputShiftLogP(responseShiftLogP)
            .setProcessedChunkIdx(request.chunkIdx)
            .setProcessedStageId(registration.stageId)
            .setTerminal(registration.isTerminal)

        if (registration.isTerminal || localOnly) {
            val totalStageMs = elapsedMsSince(chunkStartedAtNs)
            val localResponse = localResponseBuilder
                .setTerminal(registration.isTerminal)
                .addStageMetrics(
                    stageMetricBuilder
                        .setStageTotalMs(totalStageMs)
                        .build()
                )
                .build()
            appendLog(
                "Local response returned for request=$requestId terminal=${registration.isTerminal} " +
                    "localOnly=$localOnly totalStageMs=$totalStageMs"
            )
            grpcManager.reportRequestEvent(
                registration = registration,
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                eventType = if (registration.isTerminal) Sid.RequestEventType.COMPLETED else Sid.RequestEventType.LOCAL_COMPLETED,
                success = true,
                message = "Stage ${registration.stageId} returned locally for request $requestId; localOnly=$localOnly totalStageMs=$totalStageMs",
                terminal = registration.isTerminal
            )
            return localResponse
        }

        val nextHop = requireNotNull(registration.nextHop) {
            "Non-terminal stage ${registration.stageId} lost its downstream route after readiness check."
        }

        val beliefEncodeStartedAtNs = System.nanoTime()
        val rawBeliefForDownstream = if (shouldForwardBeliefToNextStage(beliefTransportMode)) {
            executionShiftLogP
        } else {
            emptyTensorLike(executionShiftLogP)
        }
        val transportedBelief = if (ENABLE_TOPK_BELIEF_TRANSPORT && !rawBeliefForDownstream.data.isEmpty) {
            BeliefTopKCodec.encodeForTransport(rawBeliefForDownstream)
        } else {
            rawBeliefForDownstream
        }
        val beliefEncodeMs = elapsedMsSince(beliefEncodeStartedAtNs)
        val beliefTransportMessage = "beliefTopKEnabled=$ENABLE_TOPK_BELIEF_TRANSPORT " +
            "beliefTopK=${BeliefTopKCodec.DEFAULT_TOP_K} " +
            "beliefDenseBytes=${executionShiftLogP.data.size()} " +
            "beliefTransportBytes=${transportedBelief.data.size()} " +
            "beliefTransportDtype=${transportedBelief.dataType} " +
            "beliefTransportMode=$beliefTransportMode"

        val nextRequest = Sid.ForwardChunkRequest.newBuilder()
            .setBatchId(request.batchId)
            .setChunkIdx(request.chunkIdx + 1)
            .setHiddenStates(execution.outputHiddenStates)
            .setAttentionMask(request.attentionMask)
            .setPositionIds(request.positionIds)
            .setLabels(request.labels)
            .setShiftLogPPrev(transportedBelief)
            .setRequestId(requestId)
            .setEvalOnly(request.evalOnly)
            .setLearningRate(request.learningRate)
            .setBeliefTransportMode(beliefTransportMode)
            .build()

        appendLog(
            "Forwarding request=$requestId to ${nextHop.host}:${nextHop.port}"
        )
        grpcManager.reportRequestEvent(
            registration = registration,
            requestId = requestId,
            batchId = request.batchId,
            chunkIdx = request.chunkIdx,
            eventType = Sid.RequestEventType.FORWARDING,
            success = true,
            message = "Forwarding to ${nextHop.host}:${nextHop.port}; $beliefTransportMessage",
            terminal = false
        )
        val forwardStartedAtNs = System.nanoTime()
        Trace.beginSection("sid_worker_forward_next")
        val downstreamResponse = try {
            grpcManager.sendDataToNextNode(nextRequest)
        } finally {
            Trace.endSection()
        }
        val forwardMs = elapsedMsSince(forwardStartedAtNs)
        val totalStageMs = elapsedMsSince(chunkStartedAtNs)
        val stageMetric = stageMetricBuilder
            .setBeliefEncodeMs(beliefEncodeMs)
            .setBeliefDenseBytes(executionShiftLogP.data.size().toLong())
            .setBeliefTransportBytes(transportedBelief.data.size().toLong())
            .setBeliefTransportDtype(transportedBelief.dataType)
            .setForwardMs(forwardMs)
            .setStageTotalMs(totalStageMs)
            .build()
        val responseWithMetrics = downstreamResponse.toBuilder()
            .clearStageMetrics()
            .addStageMetrics(stageMetric)
            .addAllStageMetrics(downstreamResponse.stageMetricsList)
            .setMessage("${downstreamResponse.message}; forwardMs=$forwardMs totalStageMs=$totalStageMs")
            .build()
        appendLog(
            "Forward completed request=$requestId success=${downstreamResponse.success} " +
                "forwardMs=$forwardMs totalStageMs=$totalStageMs"
        )
        if (!downstreamResponse.success) {
            appendLog("Downstream failed for request=$requestId: ${downstreamResponse.message}")
            grpcManager.reportRequestEvent(
                registration = registration,
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = "${downstreamResponse.message}; forwardMs=$forwardMs totalStageMs=$totalStageMs",
                terminal = downstreamResponse.terminal
            )
            return responseWithMetrics
        } else {
            grpcManager.reportRequestEvent(
                registration = registration,
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                eventType = Sid.RequestEventType.COMPLETED,
                success = true,
                message = "${downstreamResponse.message}; forwardMs=$forwardMs totalStageMs=$totalStageMs",
                terminal = downstreamResponse.terminal
            )
        }
        return responseWithMetrics
    }

    private fun buildFailureResponse(
        request: Sid.ForwardChunkRequest,
        registration: WorkerRegistration,
        message: String
    ): Sid.ForwardChunkResponse {
        return Sid.ForwardChunkResponse.newBuilder()
            .setSuccess(false)
            .setMessage(message)
            .setLocalLoss(0f)
            .setOutputHiddenStates(emptyTensorLike(request.hiddenStates))
            .setOutputShiftLogP(emptyTensorLike(request.shiftLogPPrev))
            .setProcessedChunkIdx(request.chunkIdx)
            .setProcessedStageId(registration.stageId)
            .setTerminal(registration.isTerminal)
            .build()
    }

    private fun emptyTensorLike(reference: Sid.TensorData): Sid.TensorData {
        return Sid.TensorData.newBuilder()
            .setData(ByteString.EMPTY)
            .addAllShape(reference.shapeList)
            .setDataType(reference.dataType)
            .build()
    }

    private fun normalizeBeliefTransportMode(rawMode: String): String {
        return when (rawMode.trim().lowercase()) {
            "", "full", "dense" -> "full"
            "terminal", "terminal_only", "final", "final_only" -> "terminal"
            "none", "off", "disabled", "false" -> "none"
            else -> "full"
        }
    }

    private fun shouldForwardBeliefToNextStage(mode: String): Boolean {
        return mode == "full"
    }

    private fun shouldReturnBeliefForStage(mode: String, terminal: Boolean): Boolean {
        return mode == "full" || (mode == "terminal" && terminal)
    }

    private fun normalizeExecutionBeliefOutput(output: Sid.TensorData, mode: String): Sid.TensorData {
        return if (mode == "full" || output.shapeCount == 3) {
            output
        } else {
            emptyTensorLike(output)
        }
    }

    private fun memoryDeltaMessage(stats: MemoryPeakStats): String {
        return "memorySampleIntervalMs=${stats.intervalMs} memorySampleCount=${stats.sampleCount} " +
            "pssBeforeKb=${stats.before.appPssKb} pssAfterKb=${stats.after.appPssKb} " +
            "pssPeakKb=${stats.pssPeakKb} pssDeltaKb=${stats.after.appPssKb - stats.before.appPssKb} " +
            "privateDirtyBeforeKb=${stats.before.appPrivateDirtyKb} " +
            "privateDirtyAfterKb=${stats.after.appPrivateDirtyKb} " +
            "privateDirtyPeakKb=${stats.privateDirtyPeakKb} " +
            "javaHeapBeforeKb=${stats.before.runtimeUsedMemoryKb} " +
            "javaHeapAfterKb=${stats.after.runtimeUsedMemoryKb} " +
            "javaHeapPeakKb=${stats.javaHeapPeakKb} " +
            "javaHeapDeltaKb=${stats.after.runtimeUsedMemoryKb - stats.before.runtimeUsedMemoryKb}"
    }

    private fun stopWorker() {
        appendLog("Stopping worker...")
        acceptsNewChunks = false
        activeGrpcManager?.setWorkerActive(false)
        workerJob?.cancel()
        workerJob = null
    }

    private fun updateModelCacheSummary(
        message: String,
        source: String,
        shardId: String? = null,
        file: File? = null,
        expectedBytes: Long = 0L,
        cacheHit: Boolean? = null
    ) {
        val lines = mutableListOf<String>()
        lines += message
        lines += "source=$source"
        shardId?.takeIf { it.isNotBlank() }?.let { lines += "shard=$it" }
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
        val summary = lines.joinToString("\n")
        runOnUiThread {
            modelCacheSummary.value = summary
        }
    }

    private fun applyLaunchOverrides(intent: Intent?) {
        val extras = intent?.extras ?: return
        val applied = mutableListOf<String>()

        extras.readStringExtra(EXTRA_COORDINATOR_HOST)?.let {
            coordinatorHost.value = it
            applied += "coordinatorHost=$it"
        }
        extras.readPortExtra(EXTRA_COORDINATOR_PORT)?.let {
            coordinatorPort.value = it
            applied += "coordinatorPort=$it"
        }
        extras.readStringExtra(EXTRA_DEVICE_ID)?.let {
            deviceId.value = it
            applied += "deviceId=$it"
        }
        extras.readPortExtra(EXTRA_LOCAL_PORT)?.let {
            localServerPort.value = it
            applied += "localPort=$it"
        }
        extras.readBooleanExtra(EXTRA_AUTO_START)?.let {
            autoStartOnLaunch = it
            applied += "autoStart=$it"
        }

        if (applied.isNotEmpty()) {
            appendLog("Launch options applied: ${applied.joinToString(", ")}")
        }
    }

    private fun Bundle.readStringExtra(key: String): String? {
        return get(key)?.toString()?.trim()?.takeIf { it.isNotEmpty() }
    }

    private fun Bundle.readPortExtra(key: String): String? {
        return readStringExtra(key)?.filter { it.isDigit() }?.takeIf { it.isNotEmpty() }
    }

    private fun Bundle.readBooleanExtra(key: String): Boolean? {
        return when (val value = get(key)) {
            is Boolean -> value
            is String -> value.equals("true", ignoreCase = true) || value == "1" || value.equals("yes", ignoreCase = true)
            is Number -> value.toInt() != 0
            else -> null
        }
    }

    private fun appendLog(message: String) {
        when {
            message.contains("failed", ignoreCase = true) ||
                message.contains("crashed", ignoreCase = true) ||
                message.contains("rejecting", ignoreCase = true) -> {
                Log.e(UI_LOG_TAG, message)
            }

            message.contains("stopping", ignoreCase = true) ||
                message.contains("stopped", ignoreCase = true) ||
                message.contains("cancelled", ignoreCase = true) -> {
                Log.w(UI_LOG_TAG, message)
            }

            else -> {
                Log.i(UI_LOG_TAG, message)
            }
        }
        runOnUiThread {
            logMessages.add(message)
            if (logMessages.size > 200) {
                logMessages.removeAt(0)
            }
        }
    }

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

    private fun elapsedMsSince(startedAtNs: Long): Long {
        return (System.nanoTime() - startedAtNs) / 1_000_000
    }

    private fun defaultDeviceId(): String {
        val model = Build.MODEL.orEmpty().replace(' ', '_')
        return model.ifBlank { "android_worker" }
    }
}
