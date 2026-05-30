package com.example.sid_coordinator

import sid.Sid
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CoordinatorStateTest {
    @Test
    fun adminStatusCountsMissingStageAsOfflineEvenWhenDownstreamRouteIsReady() {
        val state = CoordinatorState(
            initialConfig = testConfig(),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        state.registerNode(
            Sid.NodeInfo.newBuilder()
                .setDeviceId("stage-1")
                .setIpAddress("127.0.0.2")
                .setGrpcPort(26052)
                .setComputeCapacity(100f)
                .setMemoryGb(4f)
                .build()
        )

        val status = state.adminStatus()
        val stage0 = status.stages.single { it.stageId == 0 }

        assertNull(stage0.assignedNode)
        assertTrue(stage0.routeReady, "stage 0 can still have a ready downstream next hop")
        assertEquals(1, status.summary.liveNodeCount)
        assertEquals(1, status.summary.offlineStageCount)
    }

    @Test
    fun schedulerAssignsUnlistedWorkerToFirstOfflineStage() {
        val state = CoordinatorState(
            initialConfig = testConfig(
                scheduler = SchedulerConfig(enabled = true, policy = "fill-first")
            ),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-scheduler-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        val response = state.registerNode(
            Sid.NodeInfo.newBuilder()
                .setDeviceId("third-phone")
                .setIpAddress("127.0.0.3")
                .setGrpcPort(26052)
                .setComputeCapacity(80f)
                .setMemoryGb(6f)
                .build()
        )

        assertTrue(response.success)
        assertEquals(0, response.stageId)
        assertEquals("chunk-0", response.modelShardId)

        val status = state.adminStatus()
        val stage0 = status.stages.single { it.stageId == 0 }
        assertEquals("third-phone", stage0.assignedNode?.deviceId)
        assertEquals("dynamic-fill-offline-stage", stage0.assignedNode?.assignmentReason)
        assertEquals(1, status.summary.offlineStageCount)
        assertTrue(status.summary.schedulerEnabled)
    }

    @Test
    fun schedulerRelocatesDynamicWorkerWhenPreferredDevicesArrive() {
        val state = CoordinatorState(
            initialConfig = testConfig(
                scheduler = SchedulerConfig(enabled = true, policy = "fill-first"),
                stageCount = 3
            ),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-three-phone-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        val dynamicResponse = state.registerNode(nodeInfo("third-phone", "127.0.0.10", memoryGb = 6f))
        assertTrue(dynamicResponse.success)
        assertEquals(0, dynamicResponse.stageId)

        val stage0Response = state.registerNode(nodeInfo("stage-0", "127.0.0.1", memoryGb = 8f))
        assertTrue(stage0Response.success)
        assertEquals(0, stage0Response.stageId)

        val afterStage0 = state.adminStatus()
        assertEquals("stage-0", afterStage0.stages.single { it.stageId == 0 }.assignedNode?.deviceId)
        assertEquals("third-phone", afterStage0.stages.single { it.stageId == 1 }.assignedNode?.deviceId)

        val stage1Response = state.registerNode(nodeInfo("stage-1", "127.0.0.2", memoryGb = 8f))
        assertTrue(stage1Response.success)
        assertEquals(1, stage1Response.stageId)

        val status = state.adminStatus()
        assertEquals("stage-0", status.stages.single { it.stageId == 0 }.assignedNode?.deviceId)
        assertEquals("stage-1", status.stages.single { it.stageId == 1 }.assignedNode?.deviceId)
        assertEquals("third-phone", status.stages.single { it.stageId == 2 }.assignedNode?.deviceId)
        assertEquals("relocated-for-preferred-device", status.stages.single { it.stageId == 2 }.assignedNode?.assignmentReason)
        assertEquals(0, status.summary.offlineStageCount)
        assertTrue(status.schedulerEvents.any { it.action == "relocate_node" })
    }

    @Test
    fun schedulerReconcileMovesPreferredDeviceBackToPreferredStage() {
        val initialConfig = testConfig(
            scheduler = SchedulerConfig(
                enabled = true,
                policy = "fill-first",
                preferConfiguredDevices = false
            ),
            stageCount = 3
        )
        val state = CoordinatorState(
            initialConfig = initialConfig,
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-reconcile-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        state.registerNode(nodeInfo("third-phone", "127.0.0.10"))
        state.registerNode(nodeInfo("stage-0", "127.0.0.1"))
        assertEquals("third-phone", state.adminStatus().stages.single { it.stageId == 0 }.assignedNode?.deviceId)
        assertEquals("stage-0", state.adminStatus().stages.single { it.stageId == 1 }.assignedNode?.deviceId)

        state.reloadConfig(
            testConfig(
                scheduler = SchedulerConfig(enabled = true, policy = "fill-first"),
                stageCount = 3
            )
        )
        val reconcile = state.reconcileScheduler()
        assertTrue(reconcile.success)

        val status = state.adminStatus()
        assertEquals("stage-0", status.stages.single { it.stageId == 0 }.assignedNode?.deviceId)
        assertEquals("third-phone", status.stages.single { it.stageId == 1 }.assignedNode?.deviceId)
        assertTrue(status.schedulerEvents.any { it.action == "scheduler_reconcile" || it.reason == "reconciled-preferred-device" })
    }

    @Test
    fun staticSchedulerRejectsUnlistedWorker() {
        val state = CoordinatorState(
            initialConfig = testConfig(),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-static-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        val response = state.registerNode(
            Sid.NodeInfo.newBuilder()
                .setDeviceId("third-phone")
                .setIpAddress("127.0.0.3")
                .setGrpcPort(26052)
                .setComputeCapacity(80f)
                .setMemoryGb(6f)
                .build()
        )

        assertFalse(response.success)
        assertEquals(-1, response.stageId)
    }

    @Test
    fun schedulerRejectsWorkerBelowStageMinimum() {
        val state = CoordinatorState(
            initialConfig = testConfig(
                scheduler = SchedulerConfig(enabled = true, policy = "fill-first"),
                stageCount = 1,
                stageOverrides = mapOf(0 to StageRequirement(minMemoryGb = 8f))
            ),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-minimum-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        val response = state.registerNode(nodeInfo("small-phone", "127.0.0.9", memoryGb = 4f))

        assertFalse(response.success)
        assertEquals(1, state.adminStatus().summary.offlineStageCount)
        assertTrue(state.adminStatus().schedulerEvents.any { it.action == "register_rejected" })
    }

    @Test
    fun configValidationDefaultsMissingSchedulingWeightToOne() {
        val validated = testConfig(
            scheduler = SchedulerConfig(enabled = true),
            stageCount = 1,
            stageOverrides = mapOf(0 to StageRequirement(schedulingWeight = 0f))
        ).validated()

        assertEquals(1f, validated.stages.single().schedulingWeight)
    }

    @Test
    fun requestMetricsAreGroupedIntoRunSummaries() {
        val state = CoordinatorState(
            initialConfig = testConfig(),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-runs-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        state.recordRequestMetric(
            PersistedRequestMetric(
                runId = inferRunId("train-000123"),
                requestId = "train-000123",
                requestIndex = inferRequestIndex("train-000123"),
                batchId = 1,
                evalOnly = false,
                submittedEpochMs = 10,
                completedEpochMs = 30,
                elapsedMs = 20,
                success = true,
                terminal = true,
                processedStageId = 1,
                processedChunkIdx = 1,
                outputHiddenBytes = 16,
                outputShiftLogPBytes = 32,
                localLoss = 5.5f,
                tokenCorrect = 2,
                tokenCount = 4,
                message = "ok"
            )
        )

        val runs = state.listRecentRuns(10)
        assertEquals(1, runs.size)
        val run = runs.single()
        assertEquals("train", run.runId)
        assertEquals(1, run.requestRows)
        assertEquals(1, run.successRows)
        assertEquals(0, run.failedRows)
        assertEquals(1, run.trainRows)
        assertEquals(20.0, run.avgElapsedMs)
        assertEquals(5.5, run.avgLoss)
        assertEquals(0.5, run.tokenAccuracy)

        val csv = state.exportRunMetricsCsv("train", 10)
        assertTrue(csv.contains("train-000123"))
        assertTrue(csv.contains("0.5"))
    }

    @Test
    fun runSummaryUsesLatestAttemptForEachRequest() {
        val state = CoordinatorState(
            initialConfig = testConfig(),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-run-attempt-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        state.recordRequestMetric(
            PersistedRequestMetric(
                runId = "retry-train",
                requestId = "retry-train-000001",
                requestIndex = 1,
                batchId = 1,
                evalOnly = false,
                submittedEpochMs = 10,
                completedEpochMs = 20,
                elapsedMs = 10,
                success = false,
                terminal = false,
                processedStageId = -1,
                processedChunkIdx = 0,
                outputHiddenBytes = 0,
                outputShiftLogPBytes = 0,
                localLoss = null,
                tokenCorrect = 0,
                tokenCount = 0,
                message = "Stage 2 has no live worker."
            )
        )
        state.recordRequestMetric(
            PersistedRequestMetric(
                runId = "retry-train",
                requestId = "retry-train-000001",
                requestIndex = 1,
                batchId = 1,
                evalOnly = false,
                submittedEpochMs = 30,
                completedEpochMs = 60,
                elapsedMs = 30,
                success = true,
                terminal = true,
                processedStageId = 2,
                processedChunkIdx = 2,
                outputHiddenBytes = 16,
                outputShiftLogPBytes = 32,
                localLoss = 4.0f,
                tokenCorrect = 3,
                tokenCount = 4,
                message = "ok"
            )
        )

        val run = state.listRecentRuns(10).single()
        assertEquals(1, run.requestRows)
        assertEquals(1, run.successRows)
        assertEquals(0, run.failedRows)
        assertEquals(30.0, run.avgElapsedMs)
        assertEquals(0.75, run.tokenAccuracy)

        val detail = state.loadRunDetail("retry-train", 10)
        assertEquals(2, detail.metrics.size)
        assertEquals(listOf(1, 2), detail.metrics.map { it.attempt })
    }

    @Test
    fun schedulerEventsAreRestoredFromPersistence() {
        val dbPath = Files.createTempDirectory("sid-coordinator-scheduler-events-test")
            .resolve("coordinator.db")
            .toString()
        val persistence = CoordinatorPersistence(dbPath)
        val firstState = CoordinatorState(
            initialConfig = testConfig(scheduler = SchedulerConfig(enabled = true)),
            persistence = persistence
        )

        firstState.registerNode(nodeInfo("stage-0", "127.0.0.1"))
        assertTrue(firstState.adminStatus().schedulerEvents.any { it.action == "register_node" })

        val restoredState = CoordinatorState(
            initialConfig = testConfig(scheduler = SchedulerConfig(enabled = true)),
            persistence = CoordinatorPersistence(dbPath)
        )

        assertTrue(restoredState.adminStatus().schedulerEvents.any { it.action == "register_node" })
    }

    @Test
    fun workerTimingEventsAreParsedIntoStageTimingMetrics() {
        val state = CoordinatorState(
            initialConfig = testConfig(),
            persistence = CoordinatorPersistence(
                Files.createTempDirectory("sid-coordinator-stage-timing-test")
                    .resolve("coordinator.db")
                    .toString()
            )
        )

        val ack = state.reportRequestEvent(
            Sid.RequestEvent.newBuilder()
                .setRequestId("train-000124")
                .setBatchId(1)
                .setChunkIdx(1)
                .setStageId(1)
                .setNodeId(7)
                .setEventType(Sid.RequestEventType.LOCAL_COMPLETED)
                .setSuccess(true)
                .setTerminal(false)
                .setEventEpochMs(1234)
                .setMessage(
                    "Local shard finished stage=1, bytes=262144 runtime=TrainingModule method=forward " +
                        "inputs=5 localMs=5121 loss=5.881173 evalOnly=false optimizerStepApplied=true " +
                        "inputBuildMs=160 executeMs=4955 gradientsMs=0 optimizerCreateMs=0 " +
                        "optimizerStepMs=0 outputConvertMs=3 totalMeasuredMs=5118"
                )
                .build()
        )

        assertTrue(ack.ack)
        val detail = state.loadRunDetail("train", 10)
        val timing = detail.stageTimings.single()
        assertEquals("train-000124", timing.requestId)
        assertEquals(1, timing.stageId)
        assertEquals("TrainingModule", timing.runtime)
        assertEquals(false, timing.evalOnly)
        assertEquals(true, timing.optimizerStepApplied)
        assertEquals(5121L, timing.localMs)
        assertEquals(4955L, timing.executeMs)
        assertEquals(262144, timing.outputBytes)

        val csv = state.exportRunStageTimingsCsv("train", 10)
        assertTrue(csv.contains("optimizer_step_applied"))
        assertTrue(csv.contains("TrainingModule"))
        assertTrue(csv.contains("5121"))
    }

    private fun nodeInfo(
        deviceId: String,
        ipAddress: String,
        computeCapacity: Float = 80f,
        memoryGb: Float = 6f
    ): Sid.NodeInfo {
        return Sid.NodeInfo.newBuilder()
            .setDeviceId(deviceId)
            .setIpAddress(ipAddress)
            .setGrpcPort(26052)
            .setComputeCapacity(computeCapacity)
            .setMemoryGb(memoryGb)
            .build()
    }

    private fun testConfig(
        scheduler: SchedulerConfig = SchedulerConfig(),
        stageCount: Int = 2,
        stageOverrides: Map<Int, StageRequirement> = emptyMap()
    ): CoordinatorConfig {
        return CoordinatorConfig(
            pipelineName = "test-pipeline",
            scheduler = scheduler,
            stages = (0 until stageCount).map { stageId ->
                val override = stageOverrides[stageId] ?: StageRequirement()
                StageConfig(
                    stageId = stageId,
                    deviceId = "stage-$stageId",
                    modelShardId = "chunk-$stageId",
                    minMemoryGb = override.minMemoryGb,
                    minComputeCapacity = override.minComputeCapacity,
                    schedulingWeight = override.schedulingWeight,
                    expectedHost = "127.0.0.${stageId + 1}",
                    expectedPort = 26052
                )
            }
        )
    }

    private data class StageRequirement(
        val minMemoryGb: Float = 0f,
        val minComputeCapacity: Float = 0f,
        val schedulingWeight: Float = 1f
    )
}
