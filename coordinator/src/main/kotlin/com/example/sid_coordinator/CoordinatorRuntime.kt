package com.example.sid_coordinator

import io.grpc.Server
import io.grpc.ServerBuilder
import io.grpc.protobuf.services.HealthStatusManager
import io.grpc.protobuf.services.ProtoReflectionService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.slf4j.LoggerFactory
import java.nio.file.Path

class CoordinatorRuntime(
    private val configPath: Path,
    initialConfig: CoordinatorConfig
) {
    private val logger = LoggerFactory.getLogger(CoordinatorRuntime::class.java)
    @Volatile
    private var currentConfig = initialConfig
    private val persistence = CoordinatorPersistence(initialConfig.stateDbPath)
    private val state = CoordinatorState(initialConfig, persistence)
    private val requestOrchestrator = CoordinatorRequestOrchestrator(state)
    private val adminServer = CoordinatorAdminServer(
        bindPort = initialConfig.adminBindPort,
        getStatus = state::adminStatus,
        loadStageArtifact = state::loadStageArtifact,
        listRecentRequests = state::listRecentRequests,
        loadRequestDetail = state::loadRequestDetail,
        listRecentRuns = state::listRecentRuns,
        loadRunDetail = state::loadRunDetail,
        exportRunMetricsCsv = state::exportRunMetricsCsv,
        exportRunStageTimingsCsv = state::exportRunStageTimingsCsv,
        listRecentWorkerTelemetry = state::listRecentWorkerTelemetry,
        exportWorkerTelemetryCsv = state::exportWorkerTelemetryCsv,
        retryRequest = ::retryRequest,
        purgeRequest = state::purgeRequest,
        purgeResolvedRequests = state::purgeResolvedRequests,
        drainStage = state::drainStage,
        resumeStage = state::resumeStage,
        evictNode = state::evictNode,
        reconcileScheduler = state::reconcileScheduler,
        reloadRouting = ::reloadConfigFromDisk
    )
    private val healthStatusManager = HealthStatusManager()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val server: Server = ServerBuilder.forPort(initialConfig.bindPort)
        .addService(CoordinatorService(state, requestOrchestrator))
        .addService(healthStatusManager.healthService)
        .addService(ProtoReflectionService.newInstance())
        .build()

    fun start() {
        server.start()
        adminServer.start()
        healthStatusManager.setStatus("", io.grpc.health.v1.HealthCheckResponse.ServingStatus.SERVING)
        logger.info(
            "Coordinator started on grpc=0.0.0.0:{} admin=0.0.0.0:{} using SQLite {} with {}",
            currentConfig.bindPort,
            currentConfig.adminBindPort,
            currentConfig.stateDbPath,
            state.snapshot()
        )

        scope.launch {
            while (isActive) {
                delay(currentConfig.cleanupIntervalSeconds * 1_000)
                val expiredCount = state.evictExpiredNodes()
                if (expiredCount > 0) {
                    logger.info("Evicted {} expired nodes. {}", expiredCount, state.snapshot())
                }
                val purgedRequests = state.purgeExpiredResolvedRequests()
                if (purgedRequests > 0) {
                    logger.info("Purged {} resolved request histories", purgedRequests)
                }
            }
        }
    }

    fun awaitTermination() {
        server.awaitTermination()
    }

    fun stop() {
        logger.info("Stopping coordinator...")
        scope.cancel()
        healthStatusManager.clearStatus("")
        adminServer.stop()
        server.shutdown()
    }

    private fun retryRequest(requestId: String): AdminMutationResult {
        val response = runBlocking {
            requestOrchestrator.retryRequest(requestId)
        }
        return AdminMutationResult(
            action = "retry_request",
            success = response.success,
            message = response.message,
            status = state.adminStatus()
        )
    }

    private fun reloadConfigFromDisk(): AdminMutationResult {
        val loaded = try {
            CoordinatorConfigLoader.load(configPath)
        } catch (t: Throwable) {
            return AdminMutationResult(
                action = "reload_config",
                success = false,
                message = "Failed to load config: ${t.message}",
                status = state.adminStatus()
            )
        }
        if (loaded.bindPort != currentConfig.bindPort) {
            return AdminMutationResult(
                action = "reload_config",
                success = false,
                message = "bindPort cannot change at runtime",
                status = state.adminStatus()
            )
        }
        if (loaded.adminBindPort != currentConfig.adminBindPort) {
            return AdminMutationResult(
                action = "reload_config",
                success = false,
                message = "adminBindPort cannot change at runtime",
                status = state.adminStatus()
            )
        }
        if (loaded.stateDbPath != currentConfig.stateDbPath) {
            return AdminMutationResult(
                action = "reload_config",
                success = false,
                message = "stateDbPath cannot change at runtime",
                status = state.adminStatus()
            )
        }
        currentConfig = loaded
        return state.reloadConfig(loaded)
    }
}
