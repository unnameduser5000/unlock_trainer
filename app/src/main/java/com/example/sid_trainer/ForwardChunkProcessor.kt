package com.example.sid_trainer

import android.content.Context
import android.os.Build
import android.os.Trace
import com.google.protobuf.ByteString
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import sid.Sid

internal class ForwardChunkProcessor(
    context: Context,
    private val acceptsNewChunks: () -> Boolean,
    private val localLog: (String, Throwable?) -> Unit
) {
    private val appContext = context.applicationContext
    private val localExecutionMutex = Mutex()

    suspend fun process(
        grpcManager: GrpcManager,
        registration: WorkerRegistration,
        modelPath: String,
        request: Sid.ForwardChunkRequest
    ): Sid.ForwardChunkResponse {
        val requestId = request.requestId.ifBlank { "batch-${request.batchId}" }
        val events = WorkerRequestEventRecorder(
            grpcManager = grpcManager,
            registration = registration,
            request = request,
            requestId = requestId,
            localLog = localLog
        )
        val requestReceivedEpochMs = System.currentTimeMillis()
        val chunkStartedAtNs = System.nanoTime()
        val localOnly = request.stopAfterLocalStage
        val durablePipeline = request.durablePipeline && !localOnly
        val dropOnForwardFailure = request.dropOnForwardFailure && !localOnly
        val replayBufferCapacity = request.replayBufferCapacity
            .takeIf { it > 0 }
            ?: DEFAULT_REPLAY_BUFFER_CAPACITY
        val beliefTransportMode = normalizeBeliefTransportMode(request.beliefTransportMode)
        events.record(
            Sid.RequestEventType.REQUEST_RECEIVED,
            success = true,
            message = "Chunk received request=$requestId stage=${registration.stageId} chunk=${request.chunkIdx} " +
                "windowSeq=${request.pipelineWindowSeq} evalOnly=${request.evalOnly} localOnly=$localOnly " +
                "durablePipeline=$durablePipeline dropOnForwardFailure=$dropOnForwardFailure " +
                "replayCapacity=$replayBufferCapacity beliefMode=$beliefTransportMode"
        )

        if (durablePipeline && request.runtimeExecutionMode.equals("atomic", ignoreCase = true)) {
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Durable pipeline mode requires split boundary execution.",
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_INVALID_REQUEST
            )
        }

        if (durablePipeline && dropOnForwardFailure) {
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "durable_pipeline and drop_on_forward_failure are mutually exclusive.",
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_INVALID_REQUEST
            )
        }

        if (!acceptsNewChunks()) {
            events.record(
                Sid.RequestEventType.FAILED,
                success = false,
                message = "Rejecting request=$requestId because worker is drained or paused."
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Worker is drained or paused.",
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_WORKER_UNAVAILABLE
            )
        }

        if (!acceptsIncompleteForwardRoute(request) && !localOnly && !registration.isTerminal &&
            (!registration.routeReady || registration.nextHop == null)
        ) {
            events.record(
                Sid.RequestEventType.FAILED,
                success = false,
                message = "No downstream route available for request=$requestId " +
                    "at epoch=${registration.routingEpoch}."
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Downstream route is not ready.",
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DOWNSTREAM_UNAVAILABLE
            )
        }

        val localQueueStartedAtNs = System.nanoTime()
        var localQueueWaitMs = 0L
        lateinit var memoryStats: MemoryPeakStats
        val nextHopForForward = if (!durablePipeline && !localOnly && !registration.isTerminal) {
            registration.nextHop
        } else {
            null
        }
        val requestScope = CoroutineScope(currentCoroutineContext())
        var overlappedForward: Deferred<DownstreamForwardResult>? = null
        var durableBoundaryPrepare: DurableBoundaryPrepareResult? = null
        var publishedDurableBoundary: DurableBoundaryEntry? = null
        var durableBoundaryPublishedEpochMs = 0L
        var durableBoundaryCallbackHandled = false
        var replayBoundaryState = "none"
        val execution = try {
            localExecutionMutex.withLock {
                localQueueWaitMs = elapsedMsSince(localQueueStartedAtNs)
                val memorySampleIntervalMs = when {
                    request.memorySampleIntervalMs < 0L -> 0L
                    request.memorySampleIntervalMs == 0L -> MemoryPeakSampler.DEFAULT_INTERVAL_MS
                    else -> request.memorySampleIntervalMs
                }
                val memorySampler = MemoryPeakSampler.create(
                    appContext,
                    intervalMs = memorySampleIntervalMs
                ).start()
                Trace.beginSection("sid_worker_local_execute")
                try {
                    val boundaryOutputPolicy = ShardBoundaryOutputPolicy(
                        includeHidden = !registration.isTerminal,
                        includeBelief = shouldForwardBeliefToNextStage(beliefTransportMode) ||
                            shouldReturnBeliefForStage(beliefTransportMode, registration.isTerminal)
                    )
                    NativeShardRunner.execute(
                        modelPath = modelPath,
                        request = request,
                        boundaryOutputPolicy = boundaryOutputPolicy
                    ) { boundary ->
                        if (durablePipeline && !registration.isTerminal) {
                            check(!durableBoundaryCallbackHandled) {
                                "BP-free boundary callback fired more than once for durable request=$requestId"
                            }
                            durableBoundaryCallbackHandled = true
                            when (grpcManager.resolveDurableBoundaryReplay(request)) {
                                is DurableBoundaryReplayResolution.Acknowledged -> {
                                    // A downstream Stage may have committed while this Stage was
                                    // interrupted after publishing its forward boundary. Finish the
                                    // local backward/update without sending that boundary twice.
                                    replayBoundaryState = "acknowledged"
                                }

                                else -> {
                                    val payload = buildDownstreamPayload(
                                        sourceRequest = request,
                                        requestId = requestId,
                                        outputHiddenStates = boundary.outputHiddenStates,
                                        outputShiftLogP = boundary.outputShiftLogP,
                                        evalOnly = boundary.evalOnly,
                                        beliefTransportMode = beliefTransportMode
                                    )
                                    durableBoundaryPrepare = grpcManager.prepareDurableBoundary(
                                        request = payload.request,
                                        capacity = replayBufferCapacity
                                    )
                                    publishedDurableBoundary = grpcManager.publishDurableBoundary(requestId)
                                    durableBoundaryPublishedEpochMs = System.currentTimeMillis()
                                }
                            }
                        } else if (nextHopForForward != null) {
                            check(overlappedForward == null) {
                                "BP-free boundary callback fired more than once for request=$requestId"
                            }
                            // Boundary tensors are copied before this callback, so native backward
                            // can safely resume while the downstream request is encoded and sent.
                            overlappedForward = requestScope.async(Dispatchers.IO) {
                                forwardToNextStage(
                                    grpcManager = grpcManager,
                                    registration = registration,
                                    nextHop = nextHopForForward,
                                    sourceRequest = request,
                                    events = events,
                                    outputHiddenStates = boundary.outputHiddenStates,
                                    outputShiftLogP = boundary.outputShiftLogP,
                                    evalOnly = boundary.evalOnly,
                                    beliefTransportMode = beliefTransportMode
                                )
                            }
                        }
                    }
                } finally {
                    Trace.endSection()
                    memoryStats = memorySampler.stop()
                }
            }
        } catch (t: Throwable) {
            overlappedForward?.cancel(
                CancellationException("Local BP-free execution failed before commit", t)
            )
            if (durablePipeline) {
                grpcManager.abortPreparedDurableBoundary(requestId)
            }
            events.record(
                Sid.RequestEventType.FAILED,
                success = false,
                message = "Native shard execution failed request=$requestId: ${t.message}",
                throwable = t
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Native shard execution failed: ${t.message}",
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_LOCAL_EXECUTION
            )
        }
        if (execution.committedReplayShortCircuited &&
            !durablePipeline && !localOnly && !registration.isTerminal
        ) {
            events.record(
                Sid.RequestEventType.FAILED,
                success = false,
                message = "Committed replay cannot be forwarded without a durable boundary receipt " +
                    "request=$requestId stage=${registration.stageId}.",
                terminal = false
            )
            return buildFailureResponse(
                request = request,
                registration = registration,
                message = "Committed replay requires durable_pipeline at a non-terminal stage.",
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DURABLE_STATE
            )
        }
        val committedBoundary = if (durablePipeline && !registration.isTerminal) {
            try {
                if (execution.committedReplayShortCircuited) {
                    when (val replay = grpcManager.resolveDurableBoundaryReplay(request)) {
                        is DurableBoundaryReplayResolution.Pending -> {
                            replayBoundaryState = replay.entry.state.name.lowercase()
                            if (replay.entry.state == DurableBoundaryState.PREPARED) {
                                grpcManager.publishDurableBoundary(requestId)
                            } else {
                                replay.entry
                            }
                        }

                        is DurableBoundaryReplayResolution.Acknowledged -> {
                            replayBoundaryState = "acknowledged"
                            null
                        }

                        DurableBoundaryReplayResolution.Missing -> {
                            replayBoundaryState = "missing"
                            events.record(
                                Sid.RequestEventType.FAILED,
                                success = false,
                                message = "Committed replay has no pending boundary or ACK receipt " +
                                    "request=$requestId.",
                                terminal = false
                            )
                            return buildFailureResponse(
                                request = request,
                                registration = registration,
                                message = "Committed replay has no durable boundary receipt.",
                                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DURABLE_STATE
                            )
                        }
                    }
                } else {
                    publishedDurableBoundary
                }
            } catch (t: Throwable) {
                events.record(
                    Sid.RequestEventType.FAILED,
                    success = false,
                    message = "Durable boundary commit failed after local commit request=$requestId: ${t.message}",
                    terminal = false,
                    throwable = t
                )
                return buildFailureResponse(
                    request = request,
                    registration = registration,
                    message = "Durable boundary commit failed after local commit: ${t.message}",
                    failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DURABLE_STATE
                )
            }
        } else {
            null
        }
        val localElapsedMs = elapsedMsSince(localQueueStartedAtNs)
        val memoryTimingMessage = memoryDeltaMessage(memoryStats)
        val localTimingMessage = "runtime=${execution.runtimeName} method=${execution.methodName} " +
            "executionMode=${execution.runtimeExecutionMode} " +
            "inputs=${execution.inputCount} lr=${execution.learningRate} " +
            "localQueueWaitMs=$localQueueWaitMs localMs=$localElapsedMs " +
            "loss=${execution.localLoss} evalOnly=${execution.evalOnly} " +
            "optimizerStepApplied=${execution.optimizerStepApplied} " +
            "optimizerStepCommitted=${execution.optimizerStepCommitted} " +
            "optimizerStepCount=${execution.optimizerStepCount} " +
            "checkpointSaved=${execution.checkpointSaved} checkpointStep=${execution.checkpointStep} " +
            "checkpointBytes=${execution.checkpointBytes} checkpointIntervalSteps=${execution.checkpointIntervalSteps} " +
            "replayShortCircuited=${execution.committedReplayShortCircuited} " +
            "${execution.timing.describeForLog()} $memoryTimingMessage"

        if (execution.committedReplayShortCircuited) {
            events.record(
                Sid.RequestEventType.REPLAY_ACKNOWLEDGED,
                success = true,
                message = "Committed request replay acknowledged without forward/backward " +
                    "request=$requestId stage=${registration.stageId} " +
                    "boundaryState=$replayBoundaryState $localTimingMessage"
            )
        } else {
            events.record(
                Sid.RequestEventType.LOCAL_COMPLETED,
                success = true,
                message = "Local shard finished request=$requestId stage=${registration.stageId} " +
                    "bytes=${execution.outputHiddenStates.data.size()} $localTimingMessage"
            )
        }
        if (committedBoundary != null && !execution.committedReplayShortCircuited) {
            val pending = grpcManager.durableBoundaryPendingCount()
            val duplicate = durableBoundaryPrepare?.duplicate == true
            events.record(
                Sid.RequestEventType.BOUNDARY_BUFFERED,
                success = true,
                message = "Durable boundary ready request=$requestId outboxSeq=${committedBoundary.outboxSequence} " +
                    "windowSeq=${request.pipelineWindowSeq} pending=$pending/$replayBufferCapacity " +
                    "duplicate=$duplicate boundaryPublishedEpochMs=$durableBoundaryPublishedEpochMs",
                terminal = false
            )
        }

        val stageMetricBuilder = Sid.StageExecutionMetrics.newBuilder()
            .setStageId(registration.stageId)
            .setChunkIdx(request.chunkIdx)
            .setNodeId(registration.nodeId)
            .setDeviceId(registration.deviceId)
            .setTerminal(registration.isTerminal)
            .setEvalOnly(execution.evalOnly)
            .setOptimizerStepApplied(execution.optimizerStepApplied)
            .setOptimizerStepCommitted(execution.optimizerStepCommitted)
            .setCheckpointSaved(execution.checkpointSaved)
            .setCheckpointIntervalSteps(execution.checkpointIntervalSteps)
            .setRuntimeExecutionMode(execution.runtimeExecutionMode)
            .setOptimizerStepCount(execution.optimizerStepCount)
            .setCheckpointSaveMs(execution.timing.checkpointSaveMs)
            .setCheckpointBytes(execution.checkpointBytes)
            .setCheckpointStep(execution.checkpointStep)
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
            .setForwardBoundaryMs(execution.timing.forwardBoundaryMs)
            .setBackwardMs(execution.timing.backwardMs)
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

        val responseHiddenStates = if (durablePipeline && !registration.isTerminal) {
            emptyTensorLike(execution.outputHiddenStates)
        } else {
            execution.outputHiddenStates
        }
        val localResponseBuilder = Sid.ForwardChunkResponse.newBuilder()
            .setSuccess(true)
            .setMessage(
                if (execution.committedReplayShortCircuited) {
                    "Stage ${registration.stageId} acknowledged committed replay without local execution " +
                        "(step=${execution.optimizerStepCount}, boundaryState=$replayBoundaryState)"
                } else if (committedBoundary != null) {
                    "Stage ${registration.stageId} committed and buffered request $requestId " +
                        "(outboxSeq=${committedBoundary.outboxSequence}, windowSeq=${request.pipelineWindowSeq})"
                } else {
                    "Stage ${registration.stageId} finished request $requestId"
                }
            )
            .setLocalLoss(execution.localLoss)
            .setOutputHiddenStates(responseHiddenStates)
            .setOutputShiftLogP(responseShiftLogP)
            .setProcessedChunkIdx(request.chunkIdx)
            .setProcessedStageId(registration.stageId)
            .setTerminal(registration.isTerminal)

        if (dropOnForwardFailure && !registration.isTerminal && nextHopForForward == null) {
            val totalStageMs = elapsedMsSince(chunkStartedAtNs)
            val localResponse = localResponseBuilder
                .addStageMetrics(
                    stageMetricBuilder
                        .setStageTotalMs(totalStageMs)
                        .build()
                )
                .build()
            return completeForwardDrop(
                events = events,
                registration = registration,
                response = localResponse,
                firstUnprocessedStageId = registration.stageId + 1,
                failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_DOWNSTREAM_UNAVAILABLE,
                cause = "Downstream route is not ready.",
                totalStageMs = totalStageMs
            )
        }

        if (registration.isTerminal || localOnly || durablePipeline) {
            val totalStageMs = elapsedMsSince(chunkStartedAtNs)
            val localResponse = localResponseBuilder
                .setTerminal(registration.isTerminal)
                .addStageMetrics(
                    stageMetricBuilder
                        .setStageTotalMs(totalStageMs)
                        .build()
                )
                .build()
            if (registration.isTerminal) {
                events.record(
                    Sid.RequestEventType.COMPLETED,
                    success = true,
                    message = if (execution.committedReplayShortCircuited) {
                        "Terminal stage ${registration.stageId} returned committed receipt for request $requestId " +
                            "without forward/backward; totalStageMs=$totalStageMs"
                    } else {
                        "Terminal stage ${registration.stageId} completed request $requestId; " +
                            "totalStageMs=$totalStageMs"
                    },
                    terminal = true
                )
            }
            return localResponse
        }

        val nextHop = requireNotNull(nextHopForForward)
        val downstream = overlappedForward?.await() ?: forwardToNextStage(
            grpcManager = grpcManager,
            registration = registration,
            nextHop = nextHop,
            sourceRequest = request,
            events = events,
            outputHiddenStates = execution.outputHiddenStates,
            outputShiftLogP = executionShiftLogP,
            evalOnly = execution.evalOnly,
            beliefTransportMode = beliefTransportMode
        )
        val downstreamResponse = downstream.response
        val payload = downstream.payload
        val forwardMs = downstream.forwardMs
        val transportMessage = downstream.transport.describeForLog()
        val totalStageMs = elapsedMsSince(chunkStartedAtNs)
        val stageMetric = stageMetricBuilder
            .setBeliefEncodeMs(payload.beliefEncodeMs)
            .setBeliefDenseBytes(payload.beliefDenseBytes.toLong())
            .setBeliefTransportBytes(payload.beliefTransportBytes.toLong())
            .setBeliefTransportDtype(payload.beliefTransportDtype)
            .setRpcRequestSerializeMs(downstream.transport.requestSerializeMs)
            .setRpcRequestWriteMs(downstream.transport.requestWriteMs)
            .setRpcResponseWaitMs(downstream.transport.responseWaitMs)
            .setRpcResponseReadMs(downstream.transport.responseReadMs)
            .setRpcResponseParseMs(downstream.transport.responseParseMs)
            .setRpcRequestBytes(downstream.transport.requestBytes.toLong())
            .setRpcResponseBytes(downstream.transport.responseBytes.toLong())
            .setRpcClientSendEpochMs(downstream.transport.clientSendEpochMs)
            .setRpcServerRequestReceivedEpochMs(downstream.transport.serverRequestReceivedEpochMs)
            .setRpcServerRequestReadMs(downstream.transport.serverRequestReadMs)
            .setRpcServerRequestParseMs(downstream.transport.serverRequestParseMs)
            .setRpcServerResponseSerializeMs(downstream.transport.serverResponseSerializeMs)
            .setRpcServerHandlerMs(downstream.transport.serverHandlerMs)
            .setRpcServerResponseReadyEpochMs(downstream.transport.serverResponseReadyEpochMs)
            .setRpcClientResponseReceivedEpochMs(downstream.transport.clientResponseReceivedEpochMs)
            .setForwardMs(forwardMs)
            .setStageTotalMs(totalStageMs)
            .build()
        val responseWithMetrics = downstreamResponse.toBuilder()
            .clearStageMetrics()
            .addStageMetrics(stageMetric)
            .addAllStageMetrics(downstreamResponse.stageMetricsList)
            .setMessage(
                "${downstreamResponse.message}; $transportMessage " +
                    "forwardMs=$forwardMs totalStageMs=$totalStageMs"
            )
            .build()
        if (!downstreamResponse.success) {
            if (shouldDropForwardBoundary(request, downstream.deliveryStatus)) {
                return completeForwardDrop(
                    events = events,
                    registration = registration,
                    response = responseWithMetrics,
                    firstUnprocessedStageId = downstreamResponse.firstUnprocessedStageId,
                    failureKind = downstreamResponse.failureKind,
                    cause = downstreamResponse.message,
                    totalStageMs = totalStageMs
                )
            }
            events.record(
                Sid.RequestEventType.FAILED,
                success = false,
                message = "Downstream failed request=$requestId: ${downstreamResponse.message}; " +
                    "$transportMessage forwardMs=$forwardMs totalStageMs=$totalStageMs",
                terminal = downstreamResponse.terminal
            )
            return responseWithMetrics
        } else if (downstreamResponse.forwardDropped) {
            events.record(
                Sid.RequestEventType.BOUNDARY_DROPPED,
                success = true,
                message = "Forward drop propagated request=$requestId; " +
                    "firstUnprocessedStage=${downstreamResponse.firstUnprocessedStageId} " +
                    "committedPrefixThrough=${responseWithMetrics.processedStageId} totalStageMs=$totalStageMs",
                terminal = false
            )
        } else {
            events.record(
                Sid.RequestEventType.COMPLETED,
                success = true,
                message = "Forward completed request=$requestId: ${downstreamResponse.message}; " +
                    "$transportMessage forwardMs=$forwardMs totalStageMs=$totalStageMs",
                terminal = downstreamResponse.terminal
            )
        }
        return responseWithMetrics
    }

    private suspend fun completeForwardDrop(
        events: WorkerRequestEventRecorder,
        registration: WorkerRegistration,
        response: Sid.ForwardChunkResponse,
        firstUnprocessedStageId: Int,
        failureKind: Sid.RequestFailureKind,
        cause: String,
        totalStageMs: Long
    ): Sid.ForwardChunkResponse {
        val requestId = events.requestId
        val committedStageId = response.stageMetricsList
            .asSequence()
            .filter { metric -> metric.evalOnly || metric.optimizerStepCommitted }
            .maxOfOrNull { metric -> metric.stageId }
            ?: registration.stageId
        val droppedResponse = response.toBuilder()
            .setSuccess(true)
            .setTerminal(false)
            .setForwardDropped(true)
            .setFirstUnprocessedStageId(firstUnprocessedStageId)
            .setFailureKind(failureKind)
            .setProcessedStageId(committedStageId)
            .setOutputHiddenStates(emptyTensorLike(response.outputHiddenStates))
            .setOutputShiftLogP(emptyTensorLike(response.outputShiftLogP))
            .setMessage(
                "Forward boundary dropped without replay before stage $firstUnprocessedStageId; " +
                    "committedPrefixThrough=$committedStageId cause=$cause"
            )
            .build()
        events.record(
            Sid.RequestEventType.BOUNDARY_DROPPED,
            success = true,
            message = "Forward boundary dropped request=$requestId without replay; " +
                "firstUnprocessedStage=$firstUnprocessedStageId " +
                "committedPrefixThrough=$committedStageId totalStageMs=$totalStageMs cause=$cause",
            terminal = false
        )
        return droppedResponse
    }

    private suspend fun forwardToNextStage(
        grpcManager: GrpcManager,
        registration: WorkerRegistration,
        nextHop: NextHopInfo,
        sourceRequest: Sid.ForwardChunkRequest,
        events: WorkerRequestEventRecorder,
        outputHiddenStates: Sid.TensorData,
        outputShiftLogP: Sid.TensorData,
        evalOnly: Boolean,
        beliefTransportMode: String
    ): DownstreamForwardResult {
        val requestId = events.requestId
        val payload = buildDownstreamPayload(
            sourceRequest = sourceRequest,
            requestId = requestId,
            outputHiddenStates = outputHiddenStates,
            outputShiftLogP = outputShiftLogP,
            evalOnly = evalOnly,
            beliefTransportMode = beliefTransportMode
        )
        events.record(
            Sid.RequestEventType.FORWARDING,
            success = true,
            message = "Forwarding request=$requestId to ${nextHop.host}:${nextHop.port}; ${payload.description}",
            terminal = false
        )

        val forwardStartedAtNs = System.nanoTime()
        val traceCookie = requestId.hashCode() xor registration.stageId
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            Trace.beginAsyncSection("sid_worker_forward_next", traceCookie)
        }
        val call = try {
            grpcManager.sendDataToNextNode(payload.request)
        } finally {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                Trace.endAsyncSection("sid_worker_forward_next", traceCookie)
            }
        }
        return DownstreamForwardResult(
            response = call.response,
            payload = payload,
            transport = call.transport,
            deliveryStatus = call.deliveryStatus,
            forwardMs = elapsedMsSince(forwardStartedAtNs)
        )
    }

    private fun buildDownstreamPayload(
        sourceRequest: Sid.ForwardChunkRequest,
        requestId: String,
        outputHiddenStates: Sid.TensorData,
        outputShiftLogP: Sid.TensorData,
        evalOnly: Boolean,
        beliefTransportMode: String
    ): DownstreamPayload {
        val beliefEncodeStartedAtNs = System.nanoTime()
        val normalizedBelief = normalizeExecutionBeliefOutput(outputShiftLogP, beliefTransportMode)
        val rawBelief = if (shouldForwardBeliefToNextStage(beliefTransportMode)) {
            normalizedBelief
        } else {
            emptyTensorLike(normalizedBelief)
        }
        val transportedBelief = if (ENABLE_TOPK_BELIEF_TRANSPORT && !rawBelief.data.isEmpty) {
            BeliefTopKCodec.encodeForTransport(rawBelief)
        } else {
            rawBelief
        }
        val beliefEncodeMs = elapsedMsSince(beliefEncodeStartedAtNs)
        val description = "beliefTopKEnabled=$ENABLE_TOPK_BELIEF_TRANSPORT " +
            "beliefTopK=${BeliefTopKCodec.DEFAULT_TOP_K} " +
            "beliefDenseBytes=${normalizedBelief.data.size()} " +
            "beliefTransportBytes=${transportedBelief.data.size()} " +
            "beliefTransportDtype=${transportedBelief.dataType} " +
            "beliefTransportMode=$beliefTransportMode"
        val nextRequest = Sid.ForwardChunkRequest.newBuilder()
            .setBatchId(sourceRequest.batchId)
            .setChunkIdx(sourceRequest.chunkIdx + 1)
            .setHiddenStates(outputHiddenStates)
            .setAttentionMask(sourceRequest.attentionMask)
            .setPositionIds(sourceRequest.positionIds)
            .setLabels(sourceRequest.labels)
            .setShiftLogPPrev(transportedBelief)
            .setRequestId(requestId)
            .setEvalOnly(evalOnly)
            .setLearningRate(sourceRequest.learningRate)
            .setBeliefTransportMode(beliefTransportMode)
            .setRuntimeExecutionMode(sourceRequest.runtimeExecutionMode)
            .setMemorySampleIntervalMs(sourceRequest.memorySampleIntervalMs)
            .setDurablePipeline(sourceRequest.durablePipeline)
            .setReplayBufferCapacity(sourceRequest.replayBufferCapacity)
            .setPipelineWindowSeq(sourceRequest.pipelineWindowSeq)
            .setDropOnForwardFailure(sourceRequest.dropOnForwardFailure)
            .build()
        return DownstreamPayload(
            request = nextRequest,
            beliefEncodeMs = beliefEncodeMs,
            beliefDenseBytes = normalizedBelief.data.size(),
            beliefTransportBytes = transportedBelief.data.size(),
            beliefTransportDtype = transportedBelief.dataType,
            description = description
        )
    }

    private fun buildFailureResponse(
        request: Sid.ForwardChunkRequest,
        registration: WorkerRegistration,
        message: String,
        failureKind: Sid.RequestFailureKind
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
            .setFirstUnprocessedStageId(registration.stageId)
            .setFailureKind(failureKind)
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
            "", "terminal", "terminal_only", "final", "final_only" -> "terminal"
            "full", "dense" -> "full"
            "none", "off", "disabled", "false" -> "none"
            else -> "terminal"
        }
    }

    private fun shouldForwardBeliefToNextStage(mode: String): Boolean {
        return mode == "full"
    }

    private fun shouldReturnBeliefForStage(mode: String, terminal: Boolean): Boolean {
        return mode == "full" || (mode == "terminal" && terminal)
    }

    private fun normalizeExecutionBeliefOutput(output: Sid.TensorData, mode: String): Sid.TensorData {
        return if (mode != "none" && output.shapeCount == 3) {
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

    private fun elapsedMsSince(startedAtNs: Long): Long {
        return (System.nanoTime() - startedAtNs) / 1_000_000
    }

    private data class DownstreamPayload(
        val request: Sid.ForwardChunkRequest,
        val beliefEncodeMs: Long,
        val beliefDenseBytes: Int,
        val beliefTransportBytes: Int,
        val beliefTransportDtype: String,
        val description: String
    )

    private data class DownstreamForwardResult(
        val response: Sid.ForwardChunkResponse,
        val payload: DownstreamPayload,
        val transport: ForwardChunkTransportMetrics,
        val deliveryStatus: ForwardDeliveryStatus,
        val forwardMs: Long
    )

    private companion object {
        const val ENABLE_TOPK_BELIEF_TRANSPORT = false
        const val DEFAULT_REPLAY_BUFFER_CAPACITY = 4
    }
}
