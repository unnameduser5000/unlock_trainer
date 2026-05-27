package com.example.sid_coordinator

import io.grpc.ManagedChannelBuilder
import sid.CoordinatingServiceGrpcKt
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.system.measureTimeMillis

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
    val stopOnFailure: Boolean
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
        "success",
        "terminal",
        "processed_stage_id",
        "processed_chunk_idx",
        "elapsed_ms",
        "output_hidden_bytes",
        "output_shift_log_p_bytes",
        "local_loss",
        "token_correct",
        "token_count",
        "token_accuracy",
        "message"
    ).joinToString(",")

    var submitted = 0
    var skipped = 0
    var succeeded = 0
    var failed = 0
    var totalCorrect = 0
    var totalTokens = 0
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
            val request = record.toForwardChunkRequest(manifestDir, requestId, parsed.evalOnly)
            var elapsedMs = 0L
            val response = try {
                var received: sid.Sid.ForwardChunkResponse? = null
                elapsedMs = measureTimeMillis {
                    received = kotlinx.coroutines.runBlocking {
                        stub.submitRequest(request)
                    }
                }
                requireNotNull(received)
            } catch (t: Throwable) {
                failed++
                val message = "submit failed: ${t.message}"
                rows += csvRow(
                    requestId,
                    indexed.index,
                    record.dataset_index,
                    validLabels,
                    parsed.evalOnly,
                    false,
                    false,
                    -1,
                    record.chunk_idx,
                    elapsedMs,
                    0,
                    0,
                    0f,
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

            if (response.success && response.terminal) {
                succeeded++
            } else {
                failed++
            }
            val metrics = if (response.success && response.terminal) {
                computeShiftedTokenPredictionMetrics(response.outputShiftLogP, request.labels)
            } else {
                TokenPredictionMetrics(correct = 0, count = 0)
            }
            totalCorrect += metrics.correct
            totalTokens += metrics.count
            if (response.success && response.terminal) {
                totalLoss += response.localLoss.toDouble()
                lossRows++
            }
            submitted++
            rows += csvRow(
                requestId,
                indexed.index,
                record.dataset_index,
                validLabels,
                parsed.evalOnly,
                response.success,
                response.terminal,
                response.processedStageId,
                response.processedChunkIdx,
                elapsedMs,
                response.outputHiddenStates.data.size(),
                response.outputShiftLogP.data.size(),
                response.localLoss,
                metrics.correct,
                metrics.count,
                metrics.accuracy,
                response.message
            )
            println(
                "requestId=$requestId index=${indexed.index} validLabels=$validLabels " +
                "evalOnly=${parsed.evalOnly} success=${response.success} terminal=${response.terminal} elapsedMs=$elapsedMs " +
                    "loss=${response.localLoss} tokenAccuracy=${metrics.accuracy} " +
                "tokens=${metrics.count} message=${response.message}"
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
    }

    println("manifest=$manifestPath")
    println("outputCsv=${parsed.outputCsvPath.toAbsolutePath().normalize()}")
    println("submitted=$submitted skipped=$skipped succeeded=$succeeded failed=$failed")
    println("avgLocalLoss=${if (lossRows == 0) 0.0 else totalLoss / lossRows.toDouble()}")
    println("tokenAccuracy=${if (totalTokens == 0) 0.0 else totalCorrect.toDouble() / totalTokens.toDouble()} tokens=$totalTokens")
    println("stopOnFailure=${parsed.stopOnFailure}")
}

private fun parseArgs(args: Array<String>): PreparedExperimentArgs {
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
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
        stopOnFailure = args.getOrNull(10)?.toBooleanLenient() ?: true
    )
}

private fun csvRow(
    requestId: String,
    recordIndex: Int,
    datasetIndex: Int?,
    validLabels: Int,
    evalOnly: Boolean,
    success: Boolean,
    terminal: Boolean,
    processedStageId: Int,
    processedChunkIdx: Int,
    elapsedMs: Long,
    outputHiddenBytes: Int,
    outputShiftLogPBytes: Int,
    localLoss: Float,
    tokenCorrect: Int,
    tokenCount: Int,
    tokenAccuracy: Double,
    message: String
): String {
    return listOf(
        requestId,
        recordIndex.toString(),
        datasetIndex?.toString().orEmpty(),
        validLabels.toString(),
        evalOnly.toString(),
        success.toString(),
        terminal.toString(),
        processedStageId.toString(),
        processedChunkIdx.toString(),
        elapsedMs.toString(),
        outputHiddenBytes.toString(),
        outputShiftLogPBytes.toString(),
        localLoss.toString(),
        tokenCorrect.toString(),
        tokenCount.toString(),
        tokenAccuracy.toString(),
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
