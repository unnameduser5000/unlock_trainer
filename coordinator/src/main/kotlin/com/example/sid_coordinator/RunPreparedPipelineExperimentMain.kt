package com.example.sid_coordinator

import com.google.gson.Gson
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.selects.select
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

private data class PreparedPipelineExperimentArgs(
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
    val maxInFlight: Int,
    val beliefTransportMode: String,
    val runtimeExecutionMode: String,
    val memorySampleIntervalMs: Long,
    val durablePipeline: Boolean,
    val replayBufferCapacity: Int,
    val dropOnForwardFailure: Boolean,
    val submitRpcDeadlineOverridesMs: Map<Int, Long>
)

private data class PipelineSubmission(
    val submissionSeq: Int,
    val indexed: IndexedPreparedRequestRecord,
    val validLabels: Int
)

private data class PipelineResult(
    val submissionSeq: Int,
    val requestId: String,
    val recordIndex: Int,
    val datasetIndex: Int?,
    val validLabels: Int,
    val evalOnly: Boolean,
    val beliefTransportMode: String,
    val runtimeExecutionMode: String,
    val memorySampleIntervalMs: Long,
    val dropOnForwardFailure: Boolean,
    val success: Boolean,
    val terminal: Boolean,
    val forwardDropped: Boolean,
    val firstUnprocessedStageId: Int,
    val failureKind: String,
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
        get() = success && terminal && trainCommitValid

    val trainCommitValid: Boolean
        get() = isTrainCommitValid(evalOnly, success, terminal, processedStageId, stageMetrics)

    val droppedPrefixCommitValid: Boolean
        get() = isDroppedPrefixCommitValid(
            evalOnly = evalOnly,
            success = success,
            terminal = terminal,
            forwardDropped = forwardDropped,
            processedStageId = processedStageId,
            firstUnprocessedStageId = firstUnprocessedStageId,
            stageMetrics = stageMetrics
        )

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

    val sumCheckpointSaveMs: Long
        get() = stageMetrics.sumOf { it.checkpointSaveMs }

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

    val stageOptimizerSteps: String
        get() = stageMetrics.joinToString(";") { "${it.stageId}:${it.optimizerStepCount}" }
}

fun main(args: Array<String>) = runBlocking {
    val parsed = parsePipelineArgs(args)
    require(!(parsed.durablePipeline && parsed.dropOnForwardFailure)) {
        "durablePipeline and dropOnForwardFailure are mutually exclusive."
    }
    val manifestPath = parsed.manifestPath.toAbsolutePath().normalize()
    val manifestDir = manifestPath.parent
    val outputPath = parsed.outputCsvPath.toAbsolutePath().normalize()
    outputPath.parent?.let(Files::createDirectories)
    val maxMessageSizeBytes = 50 * 1024 * 1024

    val records = readManifestRecords(manifestPath)
    val submissions = mutableListOf<PipelineSubmission>()
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
        submissions += PipelineSubmission(
            submissionSeq = submissions.size,
            indexed = indexed,
            validLabels = validLabels
        )
    }

    val channel = ManagedChannelBuilder.forAddress(parsed.host, parsed.port)
        .usePlaintext()
        .maxInboundMessageSize(maxMessageSizeBytes)
        .build()
    val results = mutableListOf<PipelineResult>()

    try {
        val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
        var nextSubmission = 0
        var stopLaunching = false
        val active = mutableListOf<Deferred<PipelineResult>>()

        writePipelineCsv(outputPath, results)
        while ((!stopLaunching && nextSubmission < submissions.size) || active.isNotEmpty()) {
            while (!stopLaunching && active.size < parsed.maxInFlight && nextSubmission < submissions.size) {
                val submission = submissions[nextSubmission++]
                active += async(Dispatchers.IO) {
                    runPipelineSubmission(
                        stub = stub,
                        manifestDir = manifestDir,
                        submission = submission,
                        parsed = parsed
                    )
                }
                if (parsed.delayMs > 0 && active.size < parsed.maxInFlight) {
                    delay(parsed.delayMs)
                }
            }

            if (active.isEmpty()) {
                break
            }

            val completed = select<Pair<Deferred<PipelineResult>, PipelineResult>> {
                active.forEach { deferred ->
                    deferred.onAwait { result -> deferred to result }
                }
            }
            active.remove(completed.first)
            val result = completed.second
            results += result
            writePipelineCsv(outputPath, results)

            println(
                "requestId=${result.requestId} seq=${result.submissionSeq} index=${result.recordIndex} " +
                    "validLabels=${result.validLabels} evalOnly=${result.evalOnly} success=${result.success} " +
                    "terminal=${result.terminal} forwardDropped=${result.forwardDropped} " +
                    "firstUnprocessedStage=${result.firstUnprocessedStageId} " +
                    "trainCommitValid=${result.trainCommitValid} " +
                    "droppedPrefixCommitValid=${result.droppedPrefixCommitValid} " +
                    "mode=${result.runtimeExecutionMode} memorySampleMs=${result.memorySampleIntervalMs} " +
                    "elapsedMs=${result.elapsedMs} loss=${result.localLoss} " +
                    "tokenAccuracy=${result.tokenAccuracy} tokens=${result.tokenCount} " +
                    "labelChoiceAccuracy=${result.labelChoiceAccuracy} labelChoiceTokens=${result.labelChoiceCount} " +
                    "stages=${result.stageMetrics.size} localTrainMs=${result.sumLocalTrainMs} " +
                    "otherMs=${result.endToEndOtherMs} lrs=${result.stageLearningRates} " +
                    "message=${result.message}"
            )

            if ((!isSubmissionAccepted(result, parsed.durablePipeline, parsed.dropOnForwardFailure)) && parsed.stopOnFailure) {
                stopLaunching = true
                println(
                    "stopOnFailure=true; no new submissions after failed requestId=${result.requestId}; " +
                        "waiting for ${active.size} in-flight request(s) to drain."
                )
            }
        }
    } finally {
        channel.shutdownNow()
    }

    val submitted = results.size
    val succeeded = results.count {
        isSubmissionAccepted(it, parsed.durablePipeline, parsed.dropOnForwardFailure)
    }
    val failed = results.count {
        !isSubmissionAccepted(it, parsed.durablePipeline, parsed.dropOnForwardFailure)
    }
    val forwardDropped = results.count { it.droppedPrefixCommitValid }
    val totalLoss = results.filter { it.countedSuccess }.sumOf { it.localLoss.toDouble() }
    val lossRows = results.count { it.countedSuccess }
    val totalCorrect = results.sumOf { it.tokenCorrect }
    val totalTokens = results.sumOf { it.tokenCount }
    val totalLabelChoiceCorrect = results.sumOf { it.labelChoiceCorrect }
    val totalLabelChoiceTokens = results.sumOf { it.labelChoiceCount }

    println("manifest=$manifestPath")
    println("outputCsv=$outputPath")
    println("maxInFlight=${parsed.maxInFlight}")
    println("beliefTransportMode=${parsed.beliefTransportMode}")
    println("runtimeExecutionMode=${parsed.runtimeExecutionMode}")
    println("memorySampleIntervalMs=${parsed.memorySampleIntervalMs}")
    println("durablePipeline=${parsed.durablePipeline} replayBufferCapacity=${parsed.replayBufferCapacity}")
    println("dropOnForwardFailure=${parsed.dropOnForwardFailure}")
    println(
        "selected=${submissions.size} submitted=$submitted skipped=$skipped " +
            "succeeded=$succeeded forwardDropped=$forwardDropped failed=$failed"
    )
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
    println("submitRpcDeadlineOverridesMs=${parsed.submitRpcDeadlineOverridesMs.toSortedMap()}")
}

private suspend fun runPipelineSubmission(
    stub: CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub,
    manifestDir: Path,
    submission: PipelineSubmission,
    parsed: PreparedPipelineExperimentArgs
): PipelineResult {
    val indexed = submission.indexed
    val record = indexed.record
    val requestId = "${parsed.requestPrefix}-${indexed.index.toString().padStart(6, '0')}"
    val request = record.toForwardChunkRequest(
        manifestDir = manifestDir,
        requestIdOverride = requestId,
        evalOnly = parsed.evalOnly,
        beliefTransportMode = parsed.beliefTransportMode
    ).toBuilder()
        .setRuntimeExecutionMode(parsed.runtimeExecutionMode)
        .setMemorySampleIntervalMs(parsed.memorySampleIntervalMs)
        .setDurablePipeline(parsed.durablePipeline)
        .setReplayBufferCapacity(parsed.replayBufferCapacity)
        .setPipelineWindowSeq(indexed.index.toLong())
        .setDropOnForwardFailure(parsed.dropOnForwardFailure)
        .build()
    val submittedEpochMs = System.currentTimeMillis()
    val startedAtNs = System.nanoTime()
    val submitRpcDeadlineMs = parsed.submitRpcDeadlineOverridesMs[indexed.index]
        ?: parsed.submitRpcDeadlineMs
    return try {
        val response = submitWithTransientRetries(
            stub = stub,
            request = request,
            requestId = requestId,
            recordIndex = indexed.index,
            retryCount = parsed.transientRetryCount,
            retryDelayMs = parsed.transientRetryDelayMs,
            submitRpcDeadlineMs = submitRpcDeadlineMs
        )
        val completedEpochMs = System.currentTimeMillis()
        val metrics = if (response.success && response.terminal) {
            computeShiftedTokenPredictionMetrics(
                response.outputShiftLogP,
                request.labels,
                record.singleTokenLabelChoices()
            )
        } else {
            TokenPredictionMetrics(correct = 0, count = 0)
        }
        PipelineResult(
            submissionSeq = submission.submissionSeq,
            requestId = requestId,
            recordIndex = indexed.index,
            datasetIndex = record.dataset_index,
            validLabels = submission.validLabels,
            evalOnly = parsed.evalOnly,
            beliefTransportMode = parsed.beliefTransportMode,
            runtimeExecutionMode = parsed.runtimeExecutionMode,
            memorySampleIntervalMs = parsed.memorySampleIntervalMs,
            dropOnForwardFailure = parsed.dropOnForwardFailure,
            success = response.success,
            terminal = response.terminal,
            forwardDropped = response.forwardDropped,
            firstUnprocessedStageId = response.firstUnprocessedStageId,
            failureKind = response.failureKind.name,
            processedStageId = response.processedStageId,
            processedChunkIdx = response.processedChunkIdx,
            submittedEpochMs = submittedEpochMs,
            completedEpochMs = completedEpochMs,
            elapsedMs = elapsedMsSince(startedAtNs),
            outputHiddenBytes = response.outputHiddenStates.data.size(),
            outputShiftLogPBytes = response.outputShiftLogP.data.size(),
            localLoss = response.localLoss,
            tokenCorrect = metrics.correct,
            tokenCount = metrics.count,
            labelChoiceCorrect = metrics.labelChoiceCorrect,
            labelChoiceCount = metrics.labelChoiceCount,
            stageMetrics = response.stageMetricsList,
            message = response.message
        )
    } catch (t: Throwable) {
        PipelineResult(
            submissionSeq = submission.submissionSeq,
            requestId = requestId,
            recordIndex = indexed.index,
            datasetIndex = record.dataset_index,
            validLabels = submission.validLabels,
            evalOnly = parsed.evalOnly,
            beliefTransportMode = parsed.beliefTransportMode,
            runtimeExecutionMode = parsed.runtimeExecutionMode,
            memorySampleIntervalMs = parsed.memorySampleIntervalMs,
            dropOnForwardFailure = parsed.dropOnForwardFailure,
            success = false,
            terminal = false,
            forwardDropped = false,
            firstUnprocessedStageId = 0,
            failureKind = Sid.RequestFailureKind.REQUEST_FAILURE_KIND_UNSPECIFIED.name,
            processedStageId = -1,
            processedChunkIdx = record.chunk_idx,
            submittedEpochMs = submittedEpochMs,
            completedEpochMs = System.currentTimeMillis(),
            elapsedMs = elapsedMsSince(startedAtNs),
            outputHiddenBytes = 0,
            outputShiftLogPBytes = 0,
            localLoss = 0f,
            tokenCorrect = 0,
            tokenCount = 0,
            labelChoiceCorrect = 0,
            labelChoiceCount = 0,
            stageMetrics = emptyList(),
            message = "submit failed: ${t.message}"
        )
    }
}

private suspend fun submitWithTransientRetries(
    stub: CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub,
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
                .submitRequest(request)
            val attemptElapsedMs = elapsedMsSince(startedAtNs)
            if (response.success || !isTransientFailure(response.message) || attempt >= retryCount) {
                if (!response.success && isTransientFailure(response.message) && attempt >= retryCount) {
                    println(
                        "TRANSIENT exhausted requestId=$requestId index=$recordIndex " +
                            "attempt=$attemptNumber/${retryCount + 1} elapsedMs=$attemptElapsedMs " +
                            "message=${response.message}"
                    )
                }
                return response
            }
            println(
                "TRANSIENT retry requestId=$requestId index=$recordIndex " +
                    "attempt=$attemptNumber/${retryCount + 1} elapsedMs=$attemptElapsedMs " +
                    "delayMs=$retryDelayMs message=${response.message}"
            )
        } catch (t: Throwable) {
            lastFailure = t
            val attemptElapsedMs = elapsedMsSince(startedAtNs)
            if (!isTransientFailure(t.message.orEmpty()) || attempt >= retryCount) {
                throw t
            }
            println(
                "TRANSIENT retry requestId=$requestId index=$recordIndex " +
                    "attempt=$attemptNumber/${retryCount + 1} elapsedMs=$attemptElapsedMs " +
                    "delayMs=$retryDelayMs exception=${t.message}"
            )
        }
        attempt++
        if (retryDelayMs > 0) {
            delay(retryDelayMs)
        }
    }
    throw lastFailure ?: IllegalStateException("submit failed after transient retries")
}

private fun parsePipelineArgs(args: Array<String>): PreparedPipelineExperimentArgs {
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
    val stopOnFailure = args.getOrNull(10)?.toBooleanLenient() ?: true
    return PreparedPipelineExperimentArgs(
        host = args.getOrNull(0) ?: "127.0.0.1",
        port = args.getOrNull(1)?.toIntOrNull() ?: 50051,
        manifestPath = Paths.get(args.getOrNull(2) ?: "data/sft_requests/requests.jsonl"),
        startIndex = args.getOrNull(3)?.toIntOrNull() ?: 0,
        maxSubmitted = args.getOrNull(4)?.toIntOrNull() ?: 0,
        outputCsvPath = Paths.get(
            args.getOrNull(5) ?: "debug_runs/prepared-pipeline-experiment-$timestamp/results.csv"
        ),
        requestPrefix = args.getOrNull(6) ?: "prepared-pipeline-experiment-$timestamp",
        minValidLabels = args.getOrNull(7)?.toIntOrNull() ?: 1,
        delayMs = args.getOrNull(8)?.toLongOrNull() ?: 0L,
        evalOnly = args.getOrNull(9)?.toBooleanLenient() ?: false,
        stopOnFailure = stopOnFailure,
        transientRetryCount = (
            args.getOrNull(11)?.toIntOrNull()
                ?: if (stopOnFailure) 0 else DEFAULT_PIPELINE_TRANSIENT_RETRY_COUNT
        ).coerceAtLeast(0),
        transientRetryDelayMs = (
            args.getOrNull(12)?.toLongOrNull()
                ?: DEFAULT_PIPELINE_TRANSIENT_RETRY_DELAY_MS
        ).coerceAtLeast(0L),
        submitRpcDeadlineMs = (
            args.getOrNull(13)?.toLongOrNull()
                ?: DEFAULT_PIPELINE_SUBMIT_RPC_DEADLINE_MS
        ).coerceAtLeast(1L),
        maxInFlight = (args.getOrNull(14)?.toIntOrNull() ?: DEFAULT_MAX_IN_FLIGHT).coerceAtLeast(1),
        beliefTransportMode = normalizeBeliefTransportMode(args.getOrNull(15)),
        runtimeExecutionMode = normalizeRuntimeExecutionMode(args.getOrNull(16)),
        memorySampleIntervalMs = normalizeMemorySampleIntervalMs(args.getOrNull(17)),
        durablePipeline = args.getOrNull(18)?.toBooleanLenient() ?: false,
        replayBufferCapacity = (args.getOrNull(19)?.toIntOrNull() ?: DEFAULT_REPLAY_BUFFER_CAPACITY)
            .coerceAtLeast(1),
        dropOnForwardFailure = args.getOrNull(20)?.toBooleanLenient() ?: false,
        submitRpcDeadlineOverridesMs = parseSubmitRpcDeadlineOverrides(args.getOrNull(21))
    )
}

internal fun parseSubmitRpcDeadlineOverrides(raw: String?): Map<Int, Long> {
    if (raw.isNullOrBlank() || raw.equals("none", ignoreCase = true)) {
        return emptyMap()
    }
    val overrides = linkedMapOf<Int, Long>()
    raw.split(',').forEach { entry ->
        val parts = entry.trim().split(':', limit = 2)
        require(parts.size == 2) {
            "Invalid submit deadline override '$entry'; expected recordIndex:deadlineMs."
        }
        val recordIndex = parts[0].toIntOrNull()
        val deadlineMs = parts[1].toLongOrNull()
        require(recordIndex != null && recordIndex >= 0) {
            "Invalid record index '${parts[0]}' in submit deadline override '$entry'."
        }
        require(deadlineMs != null && deadlineMs >= 1L) {
            "Invalid deadline '${parts[1]}' in submit deadline override '$entry'."
        }
        require(overrides.put(recordIndex, deadlineMs) == null) {
            "Duplicate submit deadline override for record index $recordIndex."
        }
    }
    return overrides
}

private fun isSubmissionAccepted(
    result: PipelineResult,
    durablePipeline: Boolean,
    dropOnForwardFailure: Boolean
): Boolean {
    return if (durablePipeline) {
        result.success && result.processedStageId == 0 && result.stageMetrics.any { metric ->
            metric.stageId == 0 && (result.evalOnly || metric.optimizerStepCommitted)
        }
    } else {
        result.countedSuccess || (dropOnForwardFailure && result.droppedPrefixCommitValid)
    }
}

private fun writePipelineCsv(path: Path, results: List<PipelineResult>) {
    val rows = mutableListOf(PIPELINE_CSV_HEADER)
    rows += results.sortedBy { it.submissionSeq }.map { it.toCsvRow() }
    Files.write(path, rows)
    writePipelineMemoryCsv(stageMemoryCsvPath(path), results)
}

private fun PipelineResult.toCsvRow(): String {
    return listOf(
        submissionSeq.toString(),
        requestId,
        recordIndex.toString(),
        datasetIndex?.toString().orEmpty(),
        validLabels.toString(),
        evalOnly.toString(),
        beliefTransportMode,
        runtimeExecutionMode,
        memorySampleIntervalMs.toString(),
        dropOnForwardFailure.toString(),
        trainCommitValid.toString(),
        droppedPrefixCommitValid.toString(),
        success.toString(),
        terminal.toString(),
        forwardDropped.toString(),
        firstUnprocessedStageId.toString(),
        failureKind,
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
        sumCheckpointSaveMs.toString(),
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
        stageOptimizerSteps,
        stageTimingsJson(),
        message
    ).joinToString(",") { it.csvEscape() }
}

private fun PipelineResult.stageTimingsJson(): String {
    if (stageMetrics.isEmpty()) {
        return "[]"
    }
    return TIMING_GSON.toJson(
        stageMetrics.map { metric ->
            linkedMapOf<String, Any>(
                "stage_id" to metric.stageId,
                "chunk_idx" to metric.chunkIdx,
                "node_id" to metric.nodeId,
                "device_id" to metric.deviceId,
                "terminal" to metric.terminal,
                "eval_only" to metric.evalOnly,
                "optimizer_step_applied" to metric.optimizerStepApplied,
                "optimizer_step_committed" to metric.optimizerStepCommitted,
                "optimizer_step_count" to metric.optimizerStepCount,
                "checkpoint_saved" to metric.checkpointSaved,
                "checkpoint_interval_steps" to metric.checkpointIntervalSteps,
                "checkpoint_save_ms" to metric.checkpointSaveMs,
                "checkpoint_bytes" to metric.checkpointBytes,
                "checkpoint_step" to metric.checkpointStep,
                "runtime_name" to metric.runtimeName,
                "runtime_execution_mode" to metric.runtimeExecutionMode,
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
                "forward_boundary_ms" to metric.forwardBoundaryMs,
                "backward_ms" to metric.backwardMs,
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
                "rpc_request_serialize_ms" to metric.rpcRequestSerializeMs,
                "rpc_request_write_ms" to metric.rpcRequestWriteMs,
                "rpc_response_wait_ms" to metric.rpcResponseWaitMs,
                "rpc_response_read_ms" to metric.rpcResponseReadMs,
                "rpc_response_parse_ms" to metric.rpcResponseParseMs,
                "rpc_request_bytes" to metric.rpcRequestBytes,
                "rpc_response_bytes" to metric.rpcResponseBytes,
                "rpc_client_send_epoch_ms" to metric.rpcClientSendEpochMs,
                "rpc_server_request_received_epoch_ms" to metric.rpcServerRequestReceivedEpochMs,
                "rpc_server_request_read_ms" to metric.rpcServerRequestReadMs,
                "rpc_server_request_parse_ms" to metric.rpcServerRequestParseMs,
                "rpc_server_response_serialize_ms" to metric.rpcServerResponseSerializeMs,
                "rpc_server_handler_ms" to metric.rpcServerHandlerMs,
                "rpc_server_response_ready_epoch_ms" to metric.rpcServerResponseReadyEpochMs,
                "rpc_client_response_received_epoch_ms" to metric.rpcClientResponseReceivedEpochMs,
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

private fun writePipelineMemoryCsv(path: Path, results: List<PipelineResult>) {
    val rows = mutableListOf(PIPELINE_MEMORY_CSV_HEADER)
    rows += results
        .sortedBy { it.submissionSeq }
        .flatMap { it.toStageMemoryCsvRows() }
    Files.write(path, rows)
}

private fun PipelineResult.toStageMemoryCsvRows(): List<String> {
    return stageMetrics.map { metric ->
        listOf(
            submissionSeq.toString(),
            requestId,
            recordIndex.toString(),
            datasetIndex?.toString().orEmpty(),
            validLabels.toString(),
            evalOnly.toString(),
            beliefTransportMode,
            runtimeExecutionMode,
            memorySampleIntervalMs.toString(),
            dropOnForwardFailure.toString(),
            trainCommitValid.toString(),
            droppedPrefixCommitValid.toString(),
            success.toString(),
            terminal.toString(),
            forwardDropped.toString(),
            firstUnprocessedStageId.toString(),
            failureKind,
            elapsedMs.toString(),
            metric.stageId.toString(),
            metric.chunkIdx.toString(),
            metric.nodeId.toString(),
            metric.deviceId,
            metric.runtimeName,
            metric.runtimeExecutionMode,
            metric.methodName,
            metric.inputCount.toString(),
            metric.learningRate.toString(),
            metric.localLoss.toString(),
            metric.evalOnly.toString(),
            metric.optimizerStepApplied.toString(),
            metric.optimizerStepCommitted.toString(),
            metric.checkpointIntervalSteps.toString(),
            metric.localQueueWaitMs.toString(),
            metric.localElapsedMs.toString(),
            metric.runtimeAcquireMs.toString(),
            metric.checkpointRestoreMs.toString(),
            metric.inputBuildMs.toString(),
            metric.executeMs.toString(),
            metric.forwardBoundaryMs.toString(),
            metric.backwardMs.toString(),
            metric.gradientsMs.toString(),
            metric.optimizerCreateMs.toString(),
            metric.optimizerStepMs.toString(),
            metric.optimizerStepCount.toString(),
            metric.checkpointSaved.toString(),
            metric.checkpointSaveMs.toString(),
            metric.checkpointBytes.toString(),
            metric.checkpointStep.toString(),
            metric.outputConvertMs.toString(),
            metric.outputHiddenBytes.toString(),
            metric.outputShiftLogPBytes.toString(),
            metric.beliefEncodeMs.toString(),
            metric.beliefDenseBytes.toString(),
            metric.beliefTransportBytes.toString(),
            metric.beliefTransportDtype,
            metric.rpcRequestSerializeMs.toString(),
            metric.rpcRequestWriteMs.toString(),
            metric.rpcResponseWaitMs.toString(),
            metric.rpcResponseReadMs.toString(),
            metric.rpcResponseParseMs.toString(),
            metric.rpcRequestBytes.toString(),
            metric.rpcResponseBytes.toString(),
            metric.rpcClientSendEpochMs.toString(),
            metric.rpcServerRequestReceivedEpochMs.toString(),
            metric.rpcServerRequestReadMs.toString(),
            metric.rpcServerRequestParseMs.toString(),
            metric.rpcServerResponseSerializeMs.toString(),
            metric.rpcServerHandlerMs.toString(),
            metric.rpcServerResponseReadyEpochMs.toString(),
            metric.rpcClientResponseReceivedEpochMs.toString(),
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

private fun isTransientFailure(message: String): Boolean {
    val normalized = message.lowercase()
    return PIPELINE_TRANSIENT_FAILURE_MARKERS.any { normalized.contains(it) }
}

private fun String.csvEscape(): String {
    val escaped = replace("\"", "\"\"")
    return if (escaped.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
        "\"$escaped\""
    } else {
        escaped
    }
}

private fun normalizeRuntimeExecutionMode(rawMode: String?): String {
    return when (rawMode?.trim()?.lowercase()) {
        null, "", "auto" -> "auto"
        "atomic", "joint", "forward_backward" -> "atomic"
        "split", "boundary", "forward_to_boundary" -> "split"
        else -> error("Unsupported runtime execution mode '$rawMode'; expected auto, atomic, or split.")
    }
}

private fun normalizeMemorySampleIntervalMs(rawValue: String?): Long {
    val normalized = rawValue?.trim()?.lowercase()
    if (normalized.isNullOrEmpty()) return DEFAULT_MEMORY_SAMPLE_INTERVAL_MS
    if (normalized in setOf("off", "none", "disabled")) return -1L
    val intervalMs = normalized.toLongOrNull()
        ?: error("Invalid memory sample interval '$rawValue'; expected milliseconds or off.")
    require(intervalMs == -1L || intervalMs >= 0L) {
        "Memory sample interval must be -1, 0, or a positive millisecond value."
    }
    return intervalMs
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

private const val DEFAULT_MAX_IN_FLIGHT = 3
private const val DEFAULT_PIPELINE_TRANSIENT_RETRY_COUNT = 18
private const val DEFAULT_PIPELINE_TRANSIENT_RETRY_DELAY_MS = 10_000L
private const val DEFAULT_PIPELINE_SUBMIT_RPC_DEADLINE_MS = 420_000L
private const val DEFAULT_MEMORY_SAMPLE_INTERVAL_MS = 100L
private const val DEFAULT_REPLAY_BUFFER_CAPACITY = 4

private val PIPELINE_CSV_HEADER = listOf(
    "submission_seq",
    "request_id",
    "record_index",
    "dataset_index",
    "valid_labels",
    "eval_only",
    "belief_transport_mode",
    "runtime_execution_mode",
    "memory_sample_interval_ms",
    "drop_on_forward_failure",
    "train_commit_valid",
    "dropped_prefix_commit_valid",
    "success",
    "terminal",
    "forward_dropped",
    "first_unprocessed_stage_id",
    "failure_kind",
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
    "sum_checkpoint_save_ms",
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
    "stage_optimizer_steps",
    "stage_timings_json",
    "message"
).joinToString(",")

private val PIPELINE_MEMORY_CSV_HEADER = listOf(
    "submission_seq",
    "request_id",
    "record_index",
    "dataset_index",
    "valid_labels",
    "eval_only",
    "belief_transport_mode",
    "runtime_execution_mode",
    "request_memory_sample_interval_ms",
    "drop_on_forward_failure",
    "train_commit_valid",
    "dropped_prefix_commit_valid",
    "success",
    "terminal",
    "forward_dropped",
    "first_unprocessed_stage_id",
    "failure_kind",
    "elapsed_ms",
    "stage_id",
    "chunk_idx",
    "node_id",
    "device_id",
    "runtime_name",
    "stage_runtime_execution_mode",
    "method_name",
    "input_count",
    "learning_rate",
    "local_loss",
    "stage_eval_only",
    "optimizer_step_applied",
    "optimizer_step_committed",
    "checkpoint_interval_steps",
    "local_queue_wait_ms",
    "local_elapsed_ms",
    "runtime_acquire_ms",
    "checkpoint_restore_ms",
    "input_build_ms",
    "execute_ms",
    "forward_boundary_ms",
    "backward_ms",
    "gradients_ms",
    "optimizer_create_ms",
    "optimizer_step_ms",
    "optimizer_step_count",
    "checkpoint_saved",
    "checkpoint_save_ms",
    "checkpoint_bytes",
    "checkpoint_step",
    "output_convert_ms",
    "output_hidden_bytes",
    "output_shift_log_p_bytes",
    "belief_encode_ms",
    "belief_dense_bytes",
    "belief_transport_bytes",
    "belief_transport_dtype",
    "rpc_request_serialize_ms",
    "rpc_request_write_ms",
    "rpc_response_wait_ms",
    "rpc_response_read_ms",
    "rpc_response_parse_ms",
    "rpc_request_bytes",
    "rpc_response_bytes",
    "rpc_client_send_epoch_ms",
    "rpc_server_request_received_epoch_ms",
    "rpc_server_request_read_ms",
    "rpc_server_request_parse_ms",
    "rpc_server_response_serialize_ms",
    "rpc_server_handler_ms",
    "rpc_server_response_ready_epoch_ms",
    "rpc_client_response_received_epoch_ms",
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

private val TIMING_GSON = Gson()

private val PIPELINE_TRANSIENT_FAILURE_MARKERS = listOf(
    "has no live worker",
    "durable replay buffer is full",
    "downstream route is not ready",
    "no downstream node connected",
    "coordinator dispatch to stage 0 failed",
    "downstream forwarding failed",
    "connection reset",
    "connection refused",
    "broken pipe",
    "unavailable",
    "deadline exceeded",
    "timeout",
    "timed out"
)
