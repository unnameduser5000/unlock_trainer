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
