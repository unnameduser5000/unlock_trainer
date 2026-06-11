package com.example.sid_coordinator

import io.grpc.ManagedChannelBuilder
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

private data class PreparedExperimentArgs(
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
    val beliefTransportMode: String
)

fun main(args: Array<String>) {
    val parsed = parseArgs(args)
    val manifestPath = parsed.manifestPath.toAbsolutePath().normalize()
    val manifestDir = manifestPath.parent
    val maxMessageSizeBytes = 50 * 1024 * 1024
    Files.createDirectories(parsed.outputCsvPath.toAbsolutePath().normalize().parent)

    val records = readManifestRecords(manifestPath)
    val channel = ManagedChannelBuilder.forAddress(parsed.host, parsed.port)
        .usePlaintext()
        .maxInboundMessageSize(maxMessageSizeBytes)
        .build()

    val rows = mutableListOf<String>()
    rows += listOf(
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
        "elapsed_ms",
        "output_hidden_bytes",
        "output_shift_log_p_bytes",
        "max_pss_peak_kb",
        "max_private_dirty_peak_kb",
        "max_java_heap_peak_kb",
        "local_loss",
        "token_correct",
        "token_count",
        "token_accuracy",
        "label_choice_correct",
        "label_choice_count",
        "label_choice_accuracy",
        "message"
    ).joinToString(",")
    val memoryRows = mutableListOf(PREPARED_MEMORY_CSV_HEADER)

    var submitted = 0
    var skipped = 0
    var succeeded = 0
    var failed = 0
    var totalCorrect = 0
    var totalTokens = 0
    var totalLabelChoiceCorrect = 0
    var totalLabelChoiceTokens = 0
    var totalLoss = 0.0
    var lossRows = 0

    try {
        val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
        for (indexed in records) {
            if (indexed.index < parsed.startIndex) {
                continue
            }
            if (parsed.maxSubmitted > 0 && submitted >= parsed.maxSubmitted) {
                break
            }

            val record = indexed.record
            val validLabels = record.countValidLabels(manifestDir)
            if (validLabels < parsed.minValidLabels) {
                skipped++
                println(
                    "skip index=${indexed.index} requestId=${record.request_id} validLabels=$validLabels " +
                        "minValidLabels=${parsed.minValidLabels}"
                )
                continue
            }

            val requestId = "${parsed.requestPrefix}-${indexed.index.toString().padStart(6, '0')}"
            val request = record.toForwardChunkRequest(
                manifestDir = manifestDir,
                requestIdOverride = requestId,
                evalOnly = parsed.evalOnly,
                beliefTransportMode = parsed.beliefTransportMode
            )
            var elapsedMs = 0L
            val requestStartedAtNs = System.nanoTime()
            val response = try {
                submitWithTransientRetries(
                    stub = stub,
                    request = request,
                    requestId = requestId,
                    recordIndex = indexed.index,
                    retryCount = parsed.transientRetryCount,
                    retryDelayMs = parsed.transientRetryDelayMs,
                    submitRpcDeadlineMs = parsed.submitRpcDeadlineMs
                )
            } catch (t: Throwable) {
                elapsedMs = elapsedMsSince(requestStartedAtNs)
                failed++
                val message = "submit failed: ${t.message}"
                rows += csvRow(
                    requestId,
                    indexed.index,
                    record.dataset_index,
                    validLabels,
                    parsed.evalOnly,
                    parsed.beliefTransportMode,
                    false,
                    false,
                    -1,
                    record.chunk_idx,
                    elapsedMs,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0f,
                    0,
                    0,
                    0.0,
                    0,
                    0,
                    0.0,
                    message
                )
                println("FAIL requestId=$requestId index=${indexed.index} elapsedMs=$elapsedMs message=$message")
                submitted++
                if (parsed.stopOnFailure) {
                    println("stopOnFailure=true; stopping after failed requestId=$requestId index=${indexed.index}")
                    break
                }
                if (parsed.delayMs > 0) {
                    Thread.sleep(parsed.delayMs)
                }
                continue
            }
            elapsedMs = elapsedMsSince(requestStartedAtNs)

            if (response.success && response.terminal) {
                succeeded++
            } else {
                failed++
            }
            val metrics = if (response.success && response.terminal) {
                computeShiftedTokenPredictionMetrics(
                    response.outputShiftLogP,
                    request.labels,
                    record.singleTokenLabelChoices()
                )
            } else {
                TokenPredictionMetrics(correct = 0, count = 0)
            }
            totalCorrect += metrics.correct
            totalTokens += metrics.count
            totalLabelChoiceCorrect += metrics.labelChoiceCorrect
            totalLabelChoiceTokens += metrics.labelChoiceCount
            if (response.success && response.terminal) {
                totalLoss += response.localLoss.toDouble()
                lossRows++
            }
            val maxPssPeakKb = response.stageMetricsList.maxOfOrNull { it.pssPeakKb } ?: 0L
            val maxPrivateDirtyPeakKb = response.stageMetricsList.maxOfOrNull { it.privateDirtyPeakKb } ?: 0L
            val maxJavaHeapPeakKb = response.stageMetricsList.maxOfOrNull { it.javaHeapPeakKb } ?: 0L
            submitted++
            rows += csvRow(
                requestId,
                indexed.index,
                record.dataset_index,
                validLabels,
                parsed.evalOnly,
                parsed.beliefTransportMode,
                response.success,
                response.terminal,
                response.processedStageId,
                response.processedChunkIdx,
                elapsedMs,
                response.outputHiddenStates.data.size(),
                response.outputShiftLogP.data.size(),
                maxPssPeakKb,
                maxPrivateDirtyPeakKb,
                maxJavaHeapPeakKb,
                response.localLoss,
                metrics.correct,
                metrics.count,
                metrics.accuracy,
                metrics.labelChoiceCorrect,
                metrics.labelChoiceCount,
                metrics.labelChoiceAccuracy,
                response.message
            )
            memoryRows += preparedStageMemoryCsvRows(
                requestId = requestId,
                recordIndex = indexed.index,
                datasetIndex = record.dataset_index,
                validLabels = validLabels,
                evalOnly = parsed.evalOnly,
                beliefTransportMode = parsed.beliefTransportMode,
                success = response.success,
                terminal = response.terminal,
                elapsedMs = elapsedMs,
                response = response
            )
            println(
                "requestId=$requestId index=${indexed.index} validLabels=$validLabels " +
                "evalOnly=${parsed.evalOnly} beliefMode=${parsed.beliefTransportMode} " +
                    "success=${response.success} terminal=${response.terminal} elapsedMs=$elapsedMs " +
                    "loss=${response.localLoss} tokenAccuracy=${metrics.accuracy} " +
                    "tokens=${metrics.count} labelChoiceAccuracy=${metrics.labelChoiceAccuracy} " +
                    "labelChoiceTokens=${metrics.labelChoiceCount} maxPssPeakKb=$maxPssPeakKb " +
                    "maxPrivateDirtyPeakKb=$maxPrivateDirtyPeakKb maxJavaHeapPeakKb=$maxJavaHeapPeakKb " +
                    "message=${response.message}"
            )
            if ((!response.success || !response.terminal) && parsed.stopOnFailure) {
                println("stopOnFailure=true; stopping after failed requestId=$requestId index=${indexed.index}")
                break
            }
            if (parsed.delayMs > 0) {
                Thread.sleep(parsed.delayMs)
            }
        }
    } finally {
        channel.shutdownNow()
        Files.write(parsed.outputCsvPath, rows)
        Files.write(stageMemoryCsvPath(parsed.outputCsvPath), memoryRows)
    }

    println("manifest=$manifestPath")
    println("outputCsv=${parsed.outputCsvPath.toAbsolutePath().normalize()}")
    println("beliefTransportMode=${parsed.beliefTransportMode}")
    println("submitted=$submitted skipped=$skipped succeeded=$succeeded failed=$failed")
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

private fun preparedStageMemoryCsvRows(
    requestId: String,
    recordIndex: Int,
    datasetIndex: Int?,
    validLabels: Int,
    evalOnly: Boolean,
    beliefTransportMode: String,
    success: Boolean,
    terminal: Boolean,
    elapsedMs: Long,
    response: Sid.ForwardChunkResponse
): List<String> {
    return response.stageMetricsList.map { metric ->
        listOf(
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

private fun parseArgs(args: Array<String>): PreparedExperimentArgs {
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
    val stopOnFailure = args.getOrNull(10)?.toBooleanLenient() ?: true
    return PreparedExperimentArgs(
        host = args.getOrNull(0) ?: "127.0.0.1",
        port = args.getOrNull(1)?.toIntOrNull() ?: 50051,
        manifestPath = Paths.get(args.getOrNull(2) ?: "data/sft_requests/requests.jsonl"),
        startIndex = args.getOrNull(3)?.toIntOrNull() ?: 0,
        maxSubmitted = args.getOrNull(4)?.toIntOrNull() ?: 0,
        outputCsvPath = Paths.get(
            args.getOrNull(5) ?: "debug_runs/prepared-experiment-$timestamp/results.csv"
        ),
        requestPrefix = args.getOrNull(6) ?: "prepared-experiment-$timestamp",
        minValidLabels = args.getOrNull(7)?.toIntOrNull() ?: 1,
        delayMs = args.getOrNull(8)?.toLongOrNull() ?: 0L,
        evalOnly = args.getOrNull(9)?.toBooleanLenient() ?: false,
        stopOnFailure = stopOnFailure,
        transientRetryCount = (
            args.getOrNull(11)?.toIntOrNull()
                ?: if (stopOnFailure) 0 else DEFAULT_TRANSIENT_RETRY_COUNT
        ).coerceAtLeast(0),
        transientRetryDelayMs = (
            args.getOrNull(12)?.toLongOrNull()
                ?: DEFAULT_TRANSIENT_RETRY_DELAY_MS
        ).coerceAtLeast(0L),
        submitRpcDeadlineMs = (
            args.getOrNull(13)?.toLongOrNull()
                ?: DEFAULT_SUBMIT_RPC_DEADLINE_MS
        ).coerceAtLeast(1L),
        beliefTransportMode = normalizeBeliefTransportMode(args.getOrNull(14))
    )
}

private fun submitWithTransientRetries(
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
            val response = kotlinx.coroutines.runBlocking {
                stub.withDeadlineAfter(submitRpcDeadlineMs, TimeUnit.MILLISECONDS)
                    .submitRequest(request)
            }
            val attemptElapsedMs = elapsedMsSince(startedAtNs)
            if (response.success || !isTransientFailure(response.message) || attempt >= retryCount) {
                if (!response.success && isTransientFailure(response.message) && attempt >= retryCount) {
                    println(
                        "TRANSIENT exhausted requestId=$requestId index=$recordIndex " +
                            "attempt=$attemptNumber/${
                                retryCount + 1
                            } elapsedMs=$attemptElapsedMs message=${response.message}"
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
            Thread.sleep(retryDelayMs)
        }
    }
    throw lastFailure ?: IllegalStateException("submit failed after transient retries")
}

private fun isTransientFailure(message: String): Boolean {
    val normalized = message.lowercase()
    return TRANSIENT_FAILURE_MARKERS.any { normalized.contains(it) }
}

private fun csvRow(
    requestId: String,
    recordIndex: Int,
    datasetIndex: Int?,
    validLabels: Int,
    evalOnly: Boolean,
    beliefTransportMode: String,
    success: Boolean,
    terminal: Boolean,
    processedStageId: Int,
    processedChunkIdx: Int,
    elapsedMs: Long,
    outputHiddenBytes: Int,
    outputShiftLogPBytes: Int,
    maxPssPeakKb: Long,
    maxPrivateDirtyPeakKb: Long,
    maxJavaHeapPeakKb: Long,
    localLoss: Float,
    tokenCorrect: Int,
    tokenCount: Int,
    tokenAccuracy: Double,
    labelChoiceCorrect: Int,
    labelChoiceCount: Int,
    labelChoiceAccuracy: Double,
    message: String
): String {
    return listOf(
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
        elapsedMs.toString(),
        outputHiddenBytes.toString(),
        outputShiftLogPBytes.toString(),
        maxPssPeakKb.toString(),
        maxPrivateDirtyPeakKb.toString(),
        maxJavaHeapPeakKb.toString(),
        localLoss.toString(),
        tokenCorrect.toString(),
        tokenCount.toString(),
        tokenAccuracy.toString(),
        labelChoiceCorrect.toString(),
        labelChoiceCount.toString(),
        labelChoiceAccuracy.toString(),
        message
    ).joinToString(",") { it.csvEscape() }
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

private const val DEFAULT_TRANSIENT_RETRY_COUNT = 18
private const val DEFAULT_TRANSIENT_RETRY_DELAY_MS = 10_000L
private const val DEFAULT_SUBMIT_RPC_DEADLINE_MS = 420_000L

private val TRANSIENT_FAILURE_MARKERS = listOf(
    "has no live worker",
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

private val PREPARED_MEMORY_CSV_HEADER = listOf(
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
