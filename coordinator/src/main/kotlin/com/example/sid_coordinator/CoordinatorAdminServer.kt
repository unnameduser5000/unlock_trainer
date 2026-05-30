package com.example.sid_coordinator

import com.google.gson.GsonBuilder
import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import org.slf4j.LoggerFactory
import java.io.File
import java.io.FileInputStream
import java.net.InetSocketAddress
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors

class CoordinatorAdminServer(
    private val bindPort: Int,
    private val getStatus: () -> AdminStatusSnapshot,
    private val loadStageArtifact: (Int) -> StageArtifactHandle?,
    private val listRecentRequests: (Int, String?) -> List<AdminRequestStateSnapshot>,
    private val loadRequestDetail: (String, Int) -> AdminRequestDetailSnapshot,
    private val listRecentRuns: (Int) -> List<AdminRunSummarySnapshot>,
    private val loadRunDetail: (String, Int) -> AdminRunDetailSnapshot,
    private val exportRunMetricsCsv: (String, Int) -> String,
    private val exportRunStageTimingsCsv: (String, Int) -> String,
    private val listRecentWorkerTelemetry: (Int, String?) -> List<AdminWorkerTelemetrySnapshot>,
    private val exportWorkerTelemetryCsv: (Int, String?) -> String,
    private val retryRequest: (String) -> AdminMutationResult,
    private val purgeRequest: (String) -> AdminMutationResult,
    private val purgeResolvedRequests: (Long?) -> AdminMutationResult,
    private val drainStage: (Int) -> AdminMutationResult,
    private val resumeStage: (Int) -> AdminMutationResult,
    private val evictNode: (Int) -> AdminMutationResult,
    private val reconcileScheduler: () -> AdminMutationResult,
    private val reloadRouting: () -> AdminMutationResult
) {
    private val logger = LoggerFactory.getLogger(CoordinatorAdminServer::class.java)
    private val gson = GsonBuilder().setPrettyPrinting().create()
    private val server: HttpServer = HttpServer.create(InetSocketAddress(bindPort), 0)

    init {
        server.createContext("/healthz") { exchange ->
            handleRequest(exchange, "text/plain; charset=utf-8") {
                "ok"
            }
        }
        server.createContext("/artifacts/stages/") { exchange ->
            handleArtifactDownload(exchange)
        }
        server.createContext("/api/v1/status") { exchange ->
            handleRequest(exchange, "application/json; charset=utf-8") {
                gson.toJson(getStatus())
            }
        }
        server.createContext("/api/v1/requests") { exchange ->
            handleRequest(exchange, "application/json; charset=utf-8") {
                val limit = queryParam(exchange, "limit")?.toIntOrNull() ?: 50
                val lifecycleState = queryParam(exchange, "state")
                gson.toJson(listRecentRequests(limit, lifecycleState))
            }
        }
        server.createContext("/api/v1/requests/purge-resolved") { exchange ->
            handleMutation(exchange) {
                val olderThanSeconds = queryParam(exchange, "older_than_seconds")?.toLongOrNull()
                purgeResolvedRequests(olderThanSeconds)
            }
        }
        server.createContext("/api/v1/requests/") { exchange ->
            val segments = exchange.requestURI.path.trim('/').split('/')
            if (exchange.requestMethod == "POST" && segments.size == 5 && segments[4] == "retry") {
                handleMutation(exchange) {
                    val requestId = URLDecoder.decode(segments[3], StandardCharsets.UTF_8)
                    retryRequest(requestId)
                }
            } else if (exchange.requestMethod == "POST" && segments.size == 5 && segments[4] == "purge") {
                handleMutation(exchange) {
                    val requestId = URLDecoder.decode(segments[3], StandardCharsets.UTF_8)
                    purgeRequest(requestId)
                }
            } else {
                handleRequest(exchange, "application/json; charset=utf-8") {
                    val requestId = exchange.requestURI.path
                        .removePrefix("/api/v1/requests/")
                        .substringBefore('/')
                        .let { URLDecoder.decode(it, StandardCharsets.UTF_8) }
                    val eventLimit = queryParam(exchange, "events")?.toIntOrNull() ?: 100
                    gson.toJson(loadRequestDetail(requestId, eventLimit))
                }
            }
        }
        server.createContext("/api/v1/runs") { exchange ->
            handleRequest(exchange, "application/json; charset=utf-8") {
                val limit = queryParam(exchange, "limit")?.toIntOrNull() ?: 50
                gson.toJson(listRecentRuns(limit))
            }
        }
        server.createContext("/api/v1/runs/") { exchange ->
            val segments = exchange.requestURI.path.trim('/').split('/')
            if (exchange.requestMethod == "GET" && segments.size == 5 && segments[4] == "metrics.csv") {
                handleRequest(exchange, "text/csv; charset=utf-8") {
                    val runId = URLDecoder.decode(segments[3], StandardCharsets.UTF_8)
                    val limit = queryParam(exchange, "limit")?.toIntOrNull() ?: 100_000
                    exportRunMetricsCsv(runId, limit)
                }
            } else if (exchange.requestMethod == "GET" && segments.size == 5 && segments[4] == "stage-timings.csv") {
                handleRequest(exchange, "text/csv; charset=utf-8") {
                    val runId = URLDecoder.decode(segments[3], StandardCharsets.UTF_8)
                    val limit = queryParam(exchange, "limit")?.toIntOrNull() ?: 100_000
                    exportRunStageTimingsCsv(runId, limit)
                }
            } else {
                handleRequest(exchange, "application/json; charset=utf-8") {
                    val runId = exchange.requestURI.path
                        .removePrefix("/api/v1/runs/")
                        .substringBefore('/')
                        .let { URLDecoder.decode(it, StandardCharsets.UTF_8) }
                    val metricLimit = queryParam(exchange, "metrics")?.toIntOrNull() ?: 1000
                    gson.toJson(loadRunDetail(runId, metricLimit))
                }
            }
        }
        server.createContext("/api/v1/worker-telemetry.csv") { exchange ->
            handleRequest(exchange, "text/csv; charset=utf-8") {
                val limit = queryParam(exchange, "limit")?.toIntOrNull() ?: 100_000
                val deviceId = queryParam(exchange, "device_id")
                exportWorkerTelemetryCsv(limit, deviceId)
            }
        }
        server.createContext("/api/v1/worker-telemetry") { exchange ->
            handleRequest(exchange, "application/json; charset=utf-8") {
                val limit = queryParam(exchange, "limit")?.toIntOrNull() ?: 1000
                val deviceId = queryParam(exchange, "device_id")
                gson.toJson(listRecentWorkerTelemetry(limit, deviceId))
            }
        }
        server.createContext("/api/v1/routing/reload") { exchange ->
            handleMutation(exchange) { reloadRouting() }
        }
        server.createContext("/api/v1/scheduler/reconcile") { exchange ->
            handleMutation(exchange) { reconcileScheduler() }
        }
        server.createContext("/api/v1/stages/") { exchange ->
            handleStageMutation(exchange)
        }
        server.createContext("/api/v1/nodes/") { exchange ->
            handleNodeMutation(exchange)
        }
        server.createContext("/") { exchange ->
            handleRequest(exchange, "application/json; charset=utf-8") {
                gson.toJson(
                    mapOf(
                        "service" to "sid-coordinator-admin",
                        "routes" to listOf(
                            "/healthz",
                            "/artifacts/stages/{stageId}/model",
                            "/api/v1/status",
                            "/api/v1/requests",
                            "/api/v1/requests/{requestId}",
                            "/api/v1/runs",
                            "/api/v1/runs/{runId}",
                            "/api/v1/runs/{runId}/metrics.csv",
                            "/api/v1/runs/{runId}/stage-timings.csv",
                            "/api/v1/worker-telemetry",
                            "/api/v1/worker-telemetry.csv",
                            "POST /api/v1/requests/{requestId}/retry",
                            "POST /api/v1/requests/{requestId}/purge",
                            "POST /api/v1/requests/purge-resolved",
                            "POST /api/v1/stages/{stageId}/drain",
                            "POST /api/v1/stages/{stageId}/resume",
                            "POST /api/v1/nodes/{nodeId}/evict",
                            "POST /api/v1/scheduler/reconcile",
                            "POST /api/v1/routing/reload"
                        )
                    )
                )
            }
        }
        server.executor = Executors.newFixedThreadPool(4) { runnable ->
            Thread(runnable, "sid-coordinator-admin").apply { isDaemon = true }
        }
    }

    fun start() {
        server.start()
        logger.info("Coordinator admin server started on 0.0.0.0:{}", bindPort)
    }

    fun stop() {
        logger.info("Stopping coordinator admin server...")
        server.stop(0)
        (server.executor as? java.util.concurrent.ExecutorService)?.shutdown()
    }

    private fun handleRequest(
        exchange: HttpExchange,
        contentType: String,
        bodyProvider: () -> String
    ) {
        try {
            if (exchange.requestMethod != "GET") {
                writeResponse(exchange, 405, "text/plain; charset=utf-8", "method not allowed")
                return
            }
            writeResponse(exchange, 200, contentType, bodyProvider())
        } catch (t: Throwable) {
            logger.error("Admin request failed for {}", exchange.requestURI, t)
            writeResponse(exchange, 500, "text/plain; charset=utf-8", "internal error")
        } finally {
            exchange.close()
        }
    }

    private fun handleMutation(
        exchange: HttpExchange,
        action: () -> AdminMutationResult
    ) {
        try {
            if (exchange.requestMethod != "POST") {
                writeResponse(exchange, 405, "text/plain; charset=utf-8", "method not allowed")
                return
            }
            val result = action()
            writeResponse(
                exchange,
                if (result.success) 200 else 400,
                "application/json; charset=utf-8",
                gson.toJson(result)
            )
        } catch (t: Throwable) {
            logger.error("Admin mutation failed for {}", exchange.requestURI, t)
            writeResponse(exchange, 500, "text/plain; charset=utf-8", "internal error")
        } finally {
            exchange.close()
        }
    }

    private fun handleStageMutation(exchange: HttpExchange) {
        handleMutation(exchange) {
            val segments = exchange.requestURI.path.trim('/').split('/')
            if (segments.size != 5 || segments[0] != "api" || segments[1] != "v1" || segments[2] != "stages") {
                return@handleMutation invalidMutation("invalid_stage_route", "Unknown stage route")
            }
            val stageId = segments[3].toIntOrNull()
                ?: return@handleMutation invalidMutation("invalid_stage_route", "Stage id must be an integer")
            when (segments[4]) {
                "drain" -> drainStage(stageId)
                "resume" -> resumeStage(stageId)
                else -> invalidMutation("invalid_stage_route", "Unknown stage action ${segments[4]}")
            }
        }
    }

    private fun handleNodeMutation(exchange: HttpExchange) {
        handleMutation(exchange) {
            val segments = exchange.requestURI.path.trim('/').split('/')
            if (segments.size != 5 || segments[0] != "api" || segments[1] != "v1" || segments[2] != "nodes") {
                return@handleMutation invalidMutation("invalid_node_route", "Unknown node route")
            }
            val nodeId = segments[3].toIntOrNull()
                ?: return@handleMutation invalidMutation("invalid_node_route", "Node id must be an integer")
            when (segments[4]) {
                "evict" -> evictNode(nodeId)
                else -> invalidMutation("invalid_node_route", "Unknown node action ${segments[4]}")
            }
        }
    }

    private fun handleArtifactDownload(exchange: HttpExchange) {
        try {
            if (exchange.requestMethod != "GET") {
                writeResponse(exchange, 405, "text/plain; charset=utf-8", "method not allowed")
                return
            }
            val segments = exchange.requestURI.path.trim('/').split('/')
            if (segments.size != 4 || segments[0] != "artifacts" || segments[1] != "stages" || segments[3] != "model") {
                writeResponse(exchange, 404, "text/plain; charset=utf-8", "not found")
                return
            }
            val stageId = segments[2].toIntOrNull()
            if (stageId == null) {
                writeResponse(exchange, 400, "text/plain; charset=utf-8", "stage id must be an integer")
                return
            }
            val artifact = loadStageArtifact(stageId)
            if (artifact == null) {
                writeResponse(exchange, 404, "text/plain; charset=utf-8", "artifact not found")
                return
            }
            val file = File(artifact.filePath)
            if (!file.exists() || !file.isFile) {
                writeResponse(exchange, 404, "text/plain; charset=utf-8", "artifact file not found")
                return
            }

            exchange.responseHeaders.add("Content-Type", "application/octet-stream")
            exchange.responseHeaders.add(
                "Content-Disposition",
                "attachment; filename=\"${artifact.modelShardId}.pte\""
            )
            exchange.responseHeaders.add("X-Model-Sha256", artifact.sha256)
            exchange.responseHeaders.add("X-Model-Bytes", artifact.bytes.toString())
            exchange.sendResponseHeaders(200, file.length())
            FileInputStream(file).use { input ->
                exchange.responseBody.use { output ->
                    input.copyTo(output)
                }
            }
        } catch (t: Throwable) {
            logger.error("Artifact download failed for {}", exchange.requestURI, t)
            writeResponse(exchange, 500, "text/plain; charset=utf-8", "internal error")
        } finally {
            exchange.close()
        }
    }

    private fun writeResponse(
        exchange: HttpExchange,
        statusCode: Int,
        contentType: String,
        body: String
    ) {
        val bytes = body.toByteArray(StandardCharsets.UTF_8)
        exchange.responseHeaders.add("Content-Type", contentType)
        exchange.sendResponseHeaders(statusCode, bytes.size.toLong())
        exchange.responseBody.use { output ->
            output.write(bytes)
        }
    }

    private fun invalidMutation(action: String, message: String): AdminMutationResult {
        return AdminMutationResult(
            action = action,
            success = false,
            message = message,
            status = getStatus()
        )
    }

    private fun queryParam(exchange: HttpExchange, key: String): String? {
        val rawQuery = exchange.requestURI.rawQuery ?: return null
        return rawQuery.split('&')
            .mapNotNull { segment ->
                val parts = segment.split('=', limit = 2)
                if (parts.isEmpty()) {
                    return@mapNotNull null
                }
                val name = URLDecoder.decode(parts[0], StandardCharsets.UTF_8)
                if (name != key) {
                    return@mapNotNull null
                }
                val value = parts.getOrNull(1).orEmpty()
                URLDecoder.decode(value, StandardCharsets.UTF_8)
            }
            .firstOrNull()
    }
}
