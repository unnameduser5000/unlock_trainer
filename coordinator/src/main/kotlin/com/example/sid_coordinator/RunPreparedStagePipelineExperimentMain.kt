package com.example.sid_coordinator

import com.google.gson.Gson
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

private data class StagePipelineExperimentArgs(
    val host: String,
    val port: Int,
    val manifestPath: Path,
    val startIndex: Int,
    val maxSubmitted: Int,
    val outputCsvPath: Path,
    val requestPrefix: String,
    val minValidLabels: Int,
    val delayMs: Long,
    val evalOnly: Boolean,
    val stopOnFailure: Boolean,
    val transientRetryCount: Int,
    val transientRetryDelayMs: Long,
    val submitRpcDeadlineMs: Long,
    val maxBufferedPerStage: Int,
    val stageCount: Int,
    val beliefTransportMode: String
)

private data class StagePipelineSubmission(
    val submissionSeq: Int,
    val indexed: IndexedPreparedRequestRecord,
    val validLabels: Int
)

private data class StagePipelineWork(
    val submission: StagePipelineSubmission,
    val originalRequest: Sid.ForwardChunkRequest,
    val currentRequest: Sid.ForwardChunkRequest,
    val submittedEpochMs: Long,
    val startedAtNs: Long,
    val stageMetrics: List<Sid.StageExecutionMetrics>
)

private data class StagePipelineResult(
    val submissionSeq: Int,
    val requestId: String,
    val recordIndex: Int,
    val datasetIndex: Int?,
    val validLabels: Int,
    val evalOnly: Boolean,
    val beliefTransportMode: String,
    val success: Boolean,
    val terminal: Boolean,
    val processedStageId: Int,
    val processedChunkIdx: Int,
    val submittedEpochMs: Long,
    val completedEpochMs: Long,
    val elapsedMs: Long,
    val outputHiddenBytes: Int,
    val outputShiftLogPBytes: Int,
    val localLoss: Float,
    val tokenCorrect: Int,
    val tokenCount: Int,
    val labelChoiceCorrect: Int,
    val labelChoiceCount: Int,
    val stageMetrics: List<Sid.StageExecutionMetrics>,
    val message: String
) {
    val tokenAccuracy: Double
        get() = if (tokenCount == 0) 0.0 else tokenCorrect.toDouble() / tokenCount.toDouble()

    val labelChoiceAccuracy: Double
        get() = if (labelChoiceCount == 0) 0.0 else labelChoiceCorrect.toDouble() / labelChoiceCount.toDouble()

    val countedSuccess: Boolean
        get() = success && terminal

    val firstStageTotalMs: Long
        get() = stageMetrics.firstOrNull()?.stageTotalMs ?: 0L

    val coordinatorOverheadMs: Long
        get() = (elapsedMs - firstStageTotalMs).coerceAtLeast(0L)

    val sumRuntimeAcquireMs: Long
        get() = stageMetrics.sumOf { it.runtimeAcquireMs }

    val sumCheckpointRestoreMs: Long
        get() = stageMetrics.sumOf { it.checkpointRestoreMs }

    val sumLocalQueueWaitMs: Long
        get() = stageMetrics.sumOf { it.localQueueWaitMs }

    val sumLocalElapsedMs: Long
        get() = stageMetrics.sumOf { it.localElapsedMs }

    val sumInputBuildMs: Long
        get() = stageMetrics.sumOf { it.inputBuildMs }

    val sumModelExecuteMs: Long
        get() = stageMetrics.sumOf { it.executeMs }

    val sumGradientsMs: Long
        get() = stageMetrics.sumOf { it.gradientsMs }

    val sumOptimizerCreateMs: Long
        get() = stageMetrics.sumOf { it.optimizerCreateMs }

    val sumOptimizerStepMs: Long
        get() = stageMetrics.sumOf { it.optimizerStepMs }

    val sumOptimizerMs: Long
        get() = sumOptimizerCreateMs + sumOptimizerStepMs

    val sumLocalTrainMs: Long
        get() = sumModelExecuteMs + sumGradientsMs + sumOptimizerMs

    val sumOutputConvertMs: Long
        get() = stageMetrics.sumOf { it.outputConvertMs }

    val sumBeliefEncodeMs: Long
        get() = stageMetrics.sumOf { it.beliefEncodeMs }

    val sumForwardWaitMs: Long
        get() = stageMetrics.sumOf { it.forwardMs }

    val maxPssPeakKb: Long
        get() = stageMetrics.maxOfOrNull { it.pssPeakKb } ?: 0L

    val maxPrivateDirtyPeakKb: Long
        get() = stageMetrics.maxOfOrNull { it.privateDirtyPeakKb } ?: 0L

    val maxJavaHeapPeakKb: Long
        get() = stageMetrics.maxOfOrNull { it.javaHeapPeakKb } ?: 0L

    val endToEndOtherMs: Long
        get() = (elapsedMs - sumLocalTrainMs).coerceAtLeast(0L)

    val stageLearningRates: String
        get() = stageMetrics.joinToString(";") { "${it.stageId}:${it.learningRate}" }
}

fun main(args: Array<String>) = runBlocking {
    val parsed = parseStagePipelineArgs(args)
    val manifestPath = parsed.manifestPath.toAbsolutePath().normalize()
    val manifestDir = manifestPath.parent
    val outputPath = parsed.outputCsvPath.toAbsolutePath().normalize()
    outputPath.parent?.let(Files::createDirectories)
    val maxMessageSizeBytes = 50 * 1024 * 1024

    val records = readManifestRecords(manifestPath)
    val submissions = mutableListOf<StagePipelineSubmission>()
    var skipped = 0
    for (indexed in records) {
        if (indexed.index < parsed.startIndex) {
            continue
        }
        if (parsed.maxSubmitted > 0 && submissions.size >= parsed.maxSubmitted) {
            break
        }
        val validLabels = indexed.record.countValidLabels(manifestDir)
        if (validLabels < parsed.minValidLabels) {
            skipped++
            println(
                "skip index=${indexed.index} requestId=${indexed.record.request_id} validLabels=$validLabels " +
                    "minValidLabels=${parsed.minValidLabels}"
            )
            continue
        }
        submissions += StagePipelineSubmission(
            submissionSeq = submissions.size,
            indexed = indexed,
            validLabels = validLabels
        )
    }

    val channel = ManagedChannelBuilder.forAddress(parsed.host, parsed.port)
        .usePlaintext()
        .maxInboundMessageSize(maxMessageSizeBytes)
        .build()
    val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
    val stageQueues = List(parsed.stageCount) { Channel<StagePipelineWork>(parsed.maxBufferedPerStage) }
    val resultChannel = Channel<StagePipelineResult>(parsed.maxBufferedPerStage)
    val stopLaunching = java.util.concurrent.atomic.AtomicBoolean(false)
    val results = mutableListOf<StagePipelineResult>()

    writeStagePipelineCsv(outputPath, results)
    try {
        val workers = (0 until parsed.stageCount).map { stageId ->
            launch(Dispatchers.IO) {
                for (work in stageQueues[stageId]) {
                    val resultOrNext = runStage(
                        stub = stub,
                        stageId = stageId,
                        lastStageId = parsed.stageCount - 1,
                        work = work,
                        parsed = parsed
                    )
                    val next = resultOrNext.nextWork
                    if (next != null) {
                        stageQueues[stageId + 1].send(next)
                    } else {
                        resultChannel.send(requireNotNull(resultOrNext.result))
                    }
                }
                if (stageId + 1 < parsed.stageCount) {
                    stageQueues[stageId + 1].close()
                }
            }
        }

        val collector = launch {
            for (result in resultChannel) {
                results += result
                writeStagePipelineCsv(outputPath, results)
                println(
                    "requestId=${result.requestId} seq=${result.submissionSeq} index=${result.recordIndex} " +
                        "validLabels=${result.validLabels} evalOnly=${result.evalOnly} success=${result.success} " +
                        "terminal=${result.terminal} elapsedMs=${result.elapsedMs} loss=${result.localLoss} " +
                        "tokenAccuracy=${result.tokenAccuracy} tokens=${result.tokenCount} " +
                        "labelChoiceAccuracy=${result.labelChoiceAccuracy} labelChoiceTokens=${result.labelChoiceCount} " +
                        "stages=${result.stageMetrics.size} localTrainMs=${result.sumLocalTrainMs} " +
                        "otherMs=${result.endToEndOtherMs} lrs=${result.stageLearningRates} " +
                        "message=${result.message}"
                )
                if ((!result.countedSuccess) && parsed.stopOnFailure) {
                    stopLaunching.set(true)
                    println(
                        "stopOnFailure=true; no new stage-pipeline submissions after failed requestId=${result.requestId}; " +
                            "draining queued/in-flight work."
                    )
                }
            }
        }

        launch {
            for (submission in submissions) {
                if (stopLaunching.get()) {
                    break
                }
                val indexed = submission.indexed
                val requestId = "${parsed.requestPrefix}-${indexed.index.toString().padStart(6, '0')}"
                val request = indexed.record.toForwardChunkRequest(
                    manifestDir = manifestDir,
                    requestIdOverride = requestId,
                    evalOnly = parsed.evalOnly,
                    beliefTransportMode = parsed.beliefTransportMode
                ).toBuilder()
                    .setChunkIdx(0)
                    .setStopAfterLocalStage(true)
                    .setBeliefTransportMode(parsed.beliefTransportMode)
                    .build()
                stageQueues[0].send(
                    StagePipelineWork(
                        submission = submission,
                        originalRequest = request,
                        currentRequest = request,
                        submittedEpochMs = System.currentTimeMillis(),
                        startedAtNs = System.nanoTime(),
                        stageMetrics = emptyList()
                    )
                )
                if (parsed.delayMs > 0) {
                    delay(parsed.delayMs)
                }
            }
            stageQueues[0].close()
        }.join()

        workers.joinAll()
        resultChannel.close()
        collector.join()
    } finally {
        channel.shutdownNow()
    }

    val submitted = results.size
    val succeeded = results.count { it.countedSuccess }
    val failed = results.count { !it.countedSuccess }
    val totalLoss = results.filter { it.countedSuccess }.sumOf { it.localLoss.toDouble() }
    val lossRows = results.count { it.countedSuccess }
    val totalCorrect = results.sumOf { it.tokenCorrect }
    val totalTokens = results.sumOf { it.tokenCount }
    val totalLabelChoiceCorrect = results.sumOf { it.labelChoiceCorrect }
    val totalLabelChoiceTokens = results.sumOf { it.labelChoiceCount }

    println("manifest=$manifestPath")
    println("outputCsv=$outputPath")
    println("stageCount=${parsed.stageCount}")
    println("maxBufferedPerStage=${parsed.maxBufferedPerStage}")
    println("beliefTransportMode=${parsed.beliefTransportMode}")
    println("selected=${submissions.size} submitted=$submitted skipped=$skipped succeeded=$succeeded failed=$failed")
    println("avgLocalLoss=${if (lossRows == 0) 0.0 else totalLoss / lossRows.toDouble()}")
    println("tokenAccuracy=${if (totalTokens == 0) 0.0 else totalCorrect.toDouble() / totalTokens.toDouble()} tokens=$totalTokens")
    println(
        "labelChoiceAccuracy=${
            if (totalLabelChoiceTokens == 0) 0.0 else totalLabelChoiceCorrect.toDouble() / totalLabelChoiceTokens.toDouble()
        } labelChoiceTokens=$totalLabelChoiceTokens"
    )
    println("stopOnFailure=${parsed.stopOnFailure}")
    println("transientRetryCount=${parsed.transientRetryCount} transientRetryDelayMs=${parsed.transientRetryDelayMs}")
    println("submitRpcDeadlineMs=${parsed.submitRpcDeadlineMs}")
}

private data class StageRunOutcome(
    val nextWork: StagePipelineWork?,
    val result: StagePipelineResult?
)

private suspend fun runStage(
    stub: CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub,
    stageId: Int,
    lastStageId: Int,
    work: StagePipelineWork,
    parsed: StagePipelineExperimentArgs
): StageRunOutcome {
    val request = work.currentRequest.toBuilder()
        .setChunkIdx(stageId)
        .setStopAfterLocalStage(true)
        .setBeliefTransportMode(parsed.beliefTransportMode)
        .build()
    val response = try {
        submitStageWithTransientRetries(
            stub = stub,
            stageId = stageId,
            request = request,
            requestId = request.requestId,
            recordIndex = work.submission.indexed.index,
            retryCount = parsed.transientRetryCount,
            retryDelayMs = parsed.transientRetryDelayMs,
            submitRpcDeadlineMs = parsed.submitRpcDeadlineMs
        )
    } catch (t: Throwable) {
        return StageRunOutcome(
            nextWork = null,
            result = failedStagePipelineResult(
                work = work,
                stageId = stageId,
                request = request,
                message = "stage $stageId submit failed: ${t.message}"
            )
        )
    }

    val combinedMetrics = work.stageMetrics + response.stageMetricsList
    if (response.success && stageId < lastStageId) {
        val nextRequest = request.toBuilder()
            .setChunkIdx(stageId + 1)
            .setHiddenStates(response.outputHiddenStates)
            .setShiftLogPPrev(stagePipelineBeliefForNextStage(response.outputShiftLogP, parsed.beliefTransportMode))
            .setStopAfterLocalStage(true)
            .setBeliefTransportMode(parsed.beliefTransportMode)
            .build()
        return StageRunOutcome(
            nextWork = work.copy(
                currentRequest = nextRequest,
                stageMetrics = combinedMetrics
            ),
            result = null
        )
    }

    val metrics = if (response.success && stageId == lastStageId) {
        runCatching {
            computeShiftedTokenPredictionMetrics(
                response.outputShiftLogP,
                work.originalRequest.labels,
                work.submission.indexed.record.singleTokenLabelChoices()
            )
        }.getOrElse {
            TokenPredictionMetrics(correct = 0, count = 0)
        }
    } else {
        TokenPredictionMetrics(correct = 0, count = 0)
    }
    val completedEpochMs = System.currentTimeMillis()
    return StageRunOutcome(
        nextWork = null,
        result = StagePipelineResult(
            submissionSeq = work.submission.submissionSeq,
            requestId = request.requestId,
            recordIndex = work.submission.indexed.index,
            datasetIndex = work.submission.indexed.record.dataset_index,
            validLabels = work.submission.validLabels,
            evalOnly = parsed.evalOnly,
            beliefTransportMode = parsed.beliefTransportMode,
            success = response.success,
            terminal = response.terminal,
            processedStageId = response.processedStageId,
            processedChunkIdx = response.processedChunkIdx,
            submittedEpochMs = work.submittedEpochMs,
            completedEpochMs = completedEpochMs,
            elapsedMs = elapsedMsSince(work.startedAtNs),
            outputHiddenBytes = response.outputHiddenStates.data.size(),
            outputShiftLogPBytes = response.outputShiftLogP.data.size(),
            localLoss = response.localLoss,
            tokenCorrect = metrics.correct,
            tokenCount = metrics.count,
            labelChoiceCorrect = metrics.labelChoiceCorrect,
            labelChoiceCount = metrics.labelChoiceCount,
            stageMetrics = combinedMetrics,
            message = response.message
        )
    )
}

private fun failedStagePipelineResult(
    work: StagePipelineWork,
    stageId: Int,
    request: Sid.ForwardChunkRequest,
    message: String
): StagePipelineResult {
    return StagePipelineResult(
        submissionSeq = work.submission.submissionSeq,
        requestId = request.requestId,
        recordIndex = work.submission.indexed.index,
        datasetIndex = work.submission.indexed.record.dataset_index,
        validLabels = work.submission.validLabels,
        evalOnly = request.evalOnly,
        beliefTransportMode = normalizeBeliefTransportMode(request.beliefTransportMode),
        success = false,
        terminal = false,
        processedStageId = stageId,
        processedChunkIdx = request.chunkIdx,
        submittedEpochMs = work.submittedEpochMs,
        completedEpochMs = System.currentTimeMillis(),
        elapsedMs = elapsedMsSince(work.startedAtNs),
        outputHiddenBytes = 0,
        outputShiftLogPBytes = 0,
        localLoss = 0f,
        tokenCorrect = 0,
        tokenCount = 0,
        labelChoiceCorrect = 0,
        labelChoiceCount = 0,
        stageMetrics = work.stageMetrics,
        message = message
    )
}

private fun stagePipelineBeliefForNextStage(
    outputShiftLogP: Sid.TensorData,
    beliefTransportMode: String
): Sid.TensorData {
    return if (normalizeBeliefTransportMode(beliefTransportMode) == "full") {
        outputShiftLogP
    } else {
        emptyTensorLike(outputShiftLogP)
    }
}

private fun emptyTensorLike(reference: Sid.TensorData): Sid.TensorData {
    return Sid.TensorData.newBuilder()
        .addAllShape(reference.shapeList)
        .setDataType(reference.dataType)
        .build()
}

private suspend fun submitStageWithTransientRetries(
    stub: CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub,
    stageId: Int,
    request: Sid.ForwardChunkRequest,
    requestId: String,
    recordIndex: Int,
    retryCount: Int,
    retryDelayMs: Long,
    submitRpcDeadlineMs: Long
): Sid.ForwardChunkResponse {
    var attempt = 0
    var lastFailure: Throwable? = null
    while (attempt <= retryCount) {
        val attemptNumber = attempt + 1
        val startedAtNs = System.nanoTime()
        try {
            val response = stub.withDeadlineAfter(submitRpcDeadlineMs, TimeUnit.MILLISECONDS)
                .submitStageRequest(
                    Sid.StageForwardChunkRequest.newBuilder()
                        .setStageId(stageId)
                        .setRequest(request)
                        .build()
                )
            val attemptElapsedMs = elapsedMsSince(startedAtNs)
            if (response.success || !isStagePipelineTransientFailure(response.message) || attempt >= retryCount) {
                if (!response.success && isStagePipelineTransientFailure(response.message) && attempt >= retryCount) {
                    println(
                        "TRANSIENT exhausted requestId=$requestId index=$recordIndex stage=$stageId " +
                            "attempt=$attemptNumber/${retryCount + 1} elapsedMs=$attemptElapsedMs " +
                            "message=${response.message}"
                    )
                }
                return response
            }
            println(
                "TRANSIENT retry requestId=$requestId index=$recordIndex stage=$stageId " +
                    "attempt=$attemptNumber/${retryCount + 1} elapsedMs=$attemptElapsedMs " +
                    "delayMs=$retryDelayMs message=${response.message}"
            )
        } catch (t: Throwable) {
            lastFailure = t
            val attemptElapsedMs = elapsedMsSince(startedAtNs)
            if (!isStagePipelineTransientFailure(t.message.orEmpty()) || attempt >= retryCount) {
                throw t
            }
            println(
                "TRANSIENT retry requestId=$requestId index=$recordIndex stage=$stageId " +
                    "attempt=$attemptNumber/${retryCount + 1} elapsedMs=$attemptElapsedMs " +
                    "delayMs=$retryDelayMs exception=${t.message}"
            )
        }
        attempt++
        if (retryDelayMs > 0) {
            delay(retryDelayMs)
        }
    }
    throw lastFailure ?: IllegalStateException("stage submit failed after transient retries")
}

private fun parseStagePipelineArgs(args: Array<String>): StagePipelineExperimentArgs {
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
    val stopOnFailure = args.getOrNull(10)?.toBooleanLenient() ?: true
    return StagePipelineExperimentArgs(
        host = args.getOrNull(0) ?: "127.0.0.1",
        port = args.getOrNull(1)?.toIntOrNull() ?: 50051,
        manifestPath = Paths.get(args.getOrNull(2) ?: "data/sft_requests/requests.jsonl"),
        startIndex = args.getOrNull(3)?.toIntOrNull() ?: 0,
        maxSubmitted = args.getOrNull(4)?.toIntOrNull() ?: 0,
        outputCsvPath = Paths.get(
            args.getOrNull(5) ?: "debug_runs/prepared-stage-pipeline-experiment-$timestamp/results.csv"
        ),
        requestPrefix = args.getOrNull(6) ?: "prepared-stage-pipeline-experiment-$timestamp",
        minValidLabels = args.getOrNull(7)?.toIntOrNull() ?: 1,
        delayMs = args.getOrNull(8)?.toLongOrNull() ?: 0L,
        evalOnly = args.getOrNull(9)?.toBooleanLenient() ?: false,
        stopOnFailure = stopOnFailure,
        transientRetryCount = (
            args.getOrNull(11)?.toIntOrNull()
                ?: if (stopOnFailure) 0 else DEFAULT_STAGE_PIPELINE_TRANSIENT_RETRY_COUNT
        ).coerceAtLeast(0),
        transientRetryDelayMs = (
            args.getOrNull(12)?.toLongOrNull()
                ?: DEFAULT_STAGE_PIPELINE_TRANSIENT_RETRY_DELAY_MS
        ).coerceAtLeast(0L),
        submitRpcDeadlineMs = (
            args.getOrNull(13)?.toLongOrNull()
                ?: DEFAULT_STAGE_PIPELINE_SUBMIT_RPC_DEADLINE_MS
        ).coerceAtLeast(1L),
        maxBufferedPerStage = (
            args.getOrNull(14)?.toIntOrNull()
                ?: DEFAULT_STAGE_PIPELINE_BUFFERED_PER_STAGE
        ).coerceAtLeast(1),
        stageCount = (
            args.getOrNull(15)?.toIntOrNull()
                ?: DEFAULT_STAGE_PIPELINE_STAGE_COUNT
        ).coerceAtLeast(1),
        beliefTransportMode = normalizeBeliefTransportMode(args.getOrNull(16))
    )
}

private fun writeStagePipelineCsv(path: Path, results: List<StagePipelineResult>) {
    val rows = mutableListOf(STAGE_PIPELINE_CSV_HEADER)
    rows += results.sortedBy { it.submissionSeq }.map { it.toCsvRow() }
    Files.write(path, rows)
    writeStagePipelineMemoryCsv(stageMemoryCsvPath(path), results)
}

private fun StagePipelineResult.toCsvRow(): String {
    return listOf(
        submissionSeq.toString(),
        requestId,
        recordIndex.toString(),
        datasetIndex?.toString().orEmpty(),
        validLabels.toString(),
        evalOnly.toString(),
        beliefTransportMode,
        success.toString(),
        terminal.toString(),
        processedStageId.toString(),
        processedChunkIdx.toString(),
        submittedEpochMs.toString(),
        completedEpochMs.toString(),
        elapsedMs.toString(),
        outputHiddenBytes.toString(),
        outputShiftLogPBytes.toString(),
        localLoss.toString(),
        tokenCorrect.toString(),
        tokenCount.toString(),
        tokenAccuracy.toString(),
        labelChoiceCorrect.toString(),
        labelChoiceCount.toString(),
        labelChoiceAccuracy.toString(),
        stageMetrics.size.toString(),
        firstStageTotalMs.toString(),
        coordinatorOverheadMs.toString(),
        sumRuntimeAcquireMs.toString(),
        sumCheckpointRestoreMs.toString(),
        sumLocalQueueWaitMs.toString(),
        sumLocalElapsedMs.toString(),
        sumLocalTrainMs.toString(),
        sumModelExecuteMs.toString(),
        sumGradientsMs.toString(),
        sumOptimizerCreateMs.toString(),
        sumOptimizerStepMs.toString(),
        sumOptimizerMs.toString(),
        sumInputBuildMs.toString(),
        sumOutputConvertMs.toString(),
        sumBeliefEncodeMs.toString(),
        sumForwardWaitMs.toString(),
        maxPssPeakKb.toString(),
        maxPrivateDirtyPeakKb.toString(),
        maxJavaHeapPeakKb.toString(),
        endToEndOtherMs.toString(),
        stageLearningRates,
        stageTimingsJson(),
        message
    ).joinToString(",") { it.csvEscape() }
}

private fun StagePipelineResult.stageTimingsJson(): String {
    if (stageMetrics.isEmpty()) {
        return "[]"
    }
    return STAGE_PIPELINE_TIMING_GSON.toJson(
        stageMetrics.map { metric ->
            linkedMapOf<String, Any>(
                "stage_id" to metric.stageId,
                "chunk_idx" to metric.chunkIdx,
                "node_id" to metric.nodeId,
                "device_id" to metric.deviceId,
                "terminal" to metric.terminal,
                "eval_only" to metric.evalOnly,
                "optimizer_step_applied" to metric.optimizerStepApplied,
                "checkpoint_saved" to metric.checkpointSaved,
                "checkpoint_interval_steps" to metric.checkpointIntervalSteps,
                "runtime_name" to metric.runtimeName,
                "method_name" to metric.methodName,
                "input_count" to metric.inputCount,
                "learning_rate" to metric.learningRate,
                "local_loss" to metric.localLoss,
                "local_queue_wait_ms" to metric.localQueueWaitMs,
                "local_elapsed_ms" to metric.localElapsedMs,
                "runtime_acquire_ms" to metric.runtimeAcquireMs,
                "checkpoint_restore_ms" to metric.checkpointRestoreMs,
                "input_build_ms" to metric.inputBuildMs,
                "execute_ms" to metric.executeMs,
                "gradients_ms" to metric.gradientsMs,
                "optimizer_create_ms" to metric.optimizerCreateMs,
                "optimizer_step_ms" to metric.optimizerStepMs,
                "output_convert_ms" to metric.outputConvertMs,
                "output_hidden_bytes" to metric.outputHiddenBytes,
                "output_shift_log_p_bytes" to metric.outputShiftLogPBytes,
                "belief_encode_ms" to metric.beliefEncodeMs,
                "belief_dense_bytes" to metric.beliefDenseBytes,
                "belief_transport_bytes" to metric.beliefTransportBytes,
                "belief_transport_dtype" to metric.beliefTransportDtype,
                "forward_ms" to metric.forwardMs,
                "stage_total_ms" to metric.stageTotalMs,
                "memory_sample_interval_ms" to metric.memorySampleIntervalMs,
                "memory_sample_count" to metric.memorySampleCount,
                "pss_before_kb" to metric.pssBeforeKb,
                "pss_after_kb" to metric.pssAfterKb,
                "pss_peak_kb" to metric.pssPeakKb,
                "private_dirty_before_kb" to metric.privateDirtyBeforeKb,
                "private_dirty_after_kb" to metric.privateDirtyAfterKb,
                "private_dirty_peak_kb" to metric.privateDirtyPeakKb,
                "java_heap_before_kb" to metric.javaHeapBeforeKb,
                "java_heap_after_kb" to metric.javaHeapAfterKb,
                "java_heap_peak_kb" to metric.javaHeapPeakKb,
                "request_received_epoch_ms" to metric.requestReceivedEpochMs
            )
        }
    )
}

private fun writeStagePipelineMemoryCsv(path: Path, results: List<StagePipelineResult>) {
    val rows = mutableListOf(STAGE_PIPELINE_MEMORY_CSV_HEADER)
    rows += results
        .sortedBy { it.submissionSeq }
        .flatMap { it.toStageMemoryCsvRows() }
    Files.write(path, rows)
}

private fun StagePipelineResult.toStageMemoryCsvRows(): List<String> {
    return stageMetrics.map { metric ->
        listOf(
            submissionSeq.toString(),
            requestId,
            recordIndex.toString(),
            datasetIndex?.toString().orEmpty(),
            validLabels.toString(),
            evalOnly.toString(),
            beliefTransportMode,
            success.toString(),
            terminal.toString(),
            elapsedMs.toString(),
            metric.stageId.toString(),
            metric.chunkIdx.toString(),
            metric.nodeId.toString(),
            metric.deviceId,
            metric.runtimeName,
            metric.methodName,
            metric.inputCount.toString(),
            metric.learningRate.toString(),
            metric.localLoss.toString(),
            metric.localQueueWaitMs.toString(),
            metric.localElapsedMs.toString(),
            metric.runtimeAcquireMs.toString(),
            metric.checkpointRestoreMs.toString(),
            metric.inputBuildMs.toString(),
            metric.executeMs.toString(),
            metric.gradientsMs.toString(),
            metric.optimizerCreateMs.toString(),
            metric.optimizerStepMs.toString(),
            metric.outputConvertMs.toString(),
            metric.outputHiddenBytes.toString(),
            metric.outputShiftLogPBytes.toString(),
            metric.beliefEncodeMs.toString(),
            metric.beliefDenseBytes.toString(),
            metric.beliefTransportBytes.toString(),
            metric.beliefTransportDtype,
            metric.forwardMs.toString(),
            metric.stageTotalMs.toString(),
            metric.memorySampleIntervalMs.toString(),
            metric.memorySampleCount.toString(),
            metric.pssBeforeKb.toString(),
            metric.pssAfterKb.toString(),
            metric.pssPeakKb.toString(),
            (metric.pssAfterKb - metric.pssBeforeKb).toString(),
            (metric.pssPeakKb - metric.pssBeforeKb).toString(),
            metric.privateDirtyBeforeKb.toString(),
            metric.privateDirtyAfterKb.toString(),
            metric.privateDirtyPeakKb.toString(),
            (metric.privateDirtyAfterKb - metric.privateDirtyBeforeKb).toString(),
            (metric.privateDirtyPeakKb - metric.privateDirtyBeforeKb).toString(),
            metric.javaHeapBeforeKb.toString(),
            metric.javaHeapAfterKb.toString(),
            metric.javaHeapPeakKb.toString(),
            (metric.javaHeapAfterKb - metric.javaHeapBeforeKb).toString(),
            (metric.javaHeapPeakKb - metric.javaHeapBeforeKb).toString(),
            metric.requestReceivedEpochMs.toString()
        ).joinToString(",") { it.csvEscape() }
    }
}

private fun stageMemoryCsvPath(path: Path): Path {
    val fileName = path.fileName.toString()
    val baseName = if (fileName.lowercase().endsWith(".csv")) {
        fileName.substring(0, fileName.length - 4)
    } else {
        fileName
    }
    return path.resolveSibling("$baseName.stage_memory.csv")
}

private fun isStagePipelineTransientFailure(message: String): Boolean {
    val normalized = message.lowercase()
    return STAGE_PIPELINE_TRANSIENT_FAILURE_MARKERS.any { normalized.contains(it) }
}

private fun String.csvEscape(): String {
    val escaped = replace("\"", "\"\"")
    return if (escaped.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
        "\"$escaped\""
    } else {
        escaped
    }
}

private fun String.toBooleanLenient(): Boolean {
    return equals("true", ignoreCase = true) ||
        equals("1") ||
        equals("yes", ignoreCase = true) ||
        equals("eval", ignoreCase = true) ||
        equals("eval_only", ignoreCase = true)
}

private fun elapsedMsSince(startedAtNs: Long): Long {
    return ((System.nanoTime() - startedAtNs) / 1_000_000L).coerceAtLeast(0L)
}

private const val DEFAULT_STAGE_PIPELINE_STAGE_COUNT = 3
private const val DEFAULT_STAGE_PIPELINE_BUFFERED_PER_STAGE = 3
private const val DEFAULT_STAGE_PIPELINE_TRANSIENT_RETRY_COUNT = 18
private const val DEFAULT_STAGE_PIPELINE_TRANSIENT_RETRY_DELAY_MS = 10_000L
private const val DEFAULT_STAGE_PIPELINE_SUBMIT_RPC_DEADLINE_MS = 420_000L

private val STAGE_PIPELINE_CSV_HEADER = listOf(
    "submission_seq",
    "request_id",
    "record_index",
    "dataset_index",
    "valid_labels",
    "eval_only",
    "belief_transport_mode",
    "success",
    "terminal",
    "processed_stage_id",
    "processed_chunk_idx",
    "submitted_epoch_ms",
    "completed_epoch_ms",
    "elapsed_ms",
    "output_hidden_bytes",
    "output_shift_log_p_bytes",
    "local_loss",
    "token_correct",
    "token_count",
    "token_accuracy",
    "label_choice_correct",
    "label_choice_count",
    "label_choice_accuracy",
    "stage_metrics_count",
    "first_stage_total_ms",
    "coordinator_overhead_ms",
    "sum_runtime_acquire_ms",
    "sum_checkpoint_restore_ms",
    "sum_local_queue_wait_ms",
    "sum_local_elapsed_ms",
    "sum_local_train_ms",
    "sum_model_execute_ms",
    "sum_gradients_ms",
    "sum_optimizer_create_ms",
    "sum_optimizer_step_ms",
    "sum_optimizer_ms",
    "sum_input_build_ms",
    "sum_output_convert_ms",
    "sum_belief_encode_ms",
    "sum_forward_wait_ms",
    "max_pss_peak_kb",
    "max_private_dirty_peak_kb",
    "max_java_heap_peak_kb",
    "end_to_end_other_ms",
    "stage_learning_rates",
    "stage_timings_json",
    "message"
).joinToString(",")

private val STAGE_PIPELINE_MEMORY_CSV_HEADER = listOf(
    "submission_seq",
    "request_id",
    "record_index",
    "dataset_index",
    "valid_labels",
    "eval_only",
    "belief_transport_mode",
    "success",
    "terminal",
    "elapsed_ms",
    "stage_id",
    "chunk_idx",
    "node_id",
    "device_id",
    "runtime_name",
    "method_name",
    "input_count",
    "learning_rate",
    "local_loss",
    "local_queue_wait_ms",
    "local_elapsed_ms",
    "runtime_acquire_ms",
    "checkpoint_restore_ms",
    "input_build_ms",
    "execute_ms",
    "gradients_ms",
    "optimizer_create_ms",
    "optimizer_step_ms",
    "output_convert_ms",
    "output_hidden_bytes",
    "output_shift_log_p_bytes",
    "belief_encode_ms",
    "belief_dense_bytes",
    "belief_transport_bytes",
    "belief_transport_dtype",
    "forward_ms",
    "stage_total_ms",
    "memory_sample_interval_ms",
    "memory_sample_count",
    "pss_before_kb",
    "pss_after_kb",
    "pss_peak_kb",
    "pss_delta_after_kb",
    "pss_delta_peak_kb",
    "private_dirty_before_kb",
    "private_dirty_after_kb",
    "private_dirty_peak_kb",
    "private_dirty_delta_after_kb",
    "private_dirty_delta_peak_kb",
    "java_heap_before_kb",
    "java_heap_after_kb",
    "java_heap_peak_kb",
    "java_heap_delta_after_kb",
    "java_heap_delta_peak_kb",
    "request_received_epoch_ms"
).joinToString(",")

private val STAGE_PIPELINE_TIMING_GSON = Gson()

private val STAGE_PIPELINE_TRANSIENT_FAILURE_MARKERS = listOf(
    "has no live worker",
    "downstream route is not ready",
    "no downstream node connected",
    "coordinator dispatch to stage",
    "downstream forwarding failed",
    "connection reset",
    "connection refused",
    "broken pipe",
    "unavailable",
    "deadline exceeded",
    "timeout",
    "timed out"
)
