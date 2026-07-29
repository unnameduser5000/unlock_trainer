package com.example.sid_coordinator

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sid.Sid

class TrainCommitValidationTest {
    @Test
    fun replayedStageCountsAsCommittedWithoutAnotherOptimizerStep() {
        val metrics = listOf(
            stageMetric(stageId = 0, applied = false, committed = true),
            stageMetric(stageId = 1, applied = true, committed = true)
        )

        assertTrue(
            isTrainCommitValid(
                evalOnly = false,
                success = true,
                terminal = true,
                processedStageId = 1,
                stageMetrics = metrics
            )
        )
    }

    @Test
    fun uncommittedStageDoesNotCountAsSuccessfulTraining() {
        val metrics = listOf(
            stageMetric(stageId = 0, applied = false, committed = true),
            stageMetric(stageId = 1, applied = false, committed = false)
        )

        assertFalse(
            isTrainCommitValid(
                evalOnly = false,
                success = true,
                terminal = true,
                processedStageId = 1,
                stageMetrics = metrics
            )
        )
    }

    @Test
    fun evaluationDoesNotRequireOptimizerCommits() {
        assertTrue(
            isTrainCommitValid(
                evalOnly = true,
                success = true,
                terminal = true,
                processedStageId = 1,
                stageMetrics = emptyList()
            )
        )
    }

    @Test
    fun droppedForwardCountsOnlyWithAContiguousCommittedPrefix() {
        val metrics = listOf(
            stageMetric(stageId = 0, applied = true, committed = true),
            stageMetric(stageId = 1, applied = true, committed = true)
        )

        assertTrue(
            isDroppedPrefixCommitValid(
                evalOnly = false,
                success = true,
                terminal = false,
                forwardDropped = true,
                processedStageId = 1,
                firstUnprocessedStageId = 2,
                stageMetrics = metrics
            )
        )
    }

    @Test
    fun droppedForwardRejectsAGapOrUncommittedStage() {
        assertFalse(
            isDroppedPrefixCommitValid(
                evalOnly = false,
                success = true,
                terminal = false,
                forwardDropped = true,
                processedStageId = 1,
                firstUnprocessedStageId = 2,
                stageMetrics = listOf(
                    stageMetric(stageId = 0, applied = true, committed = true),
                    stageMetric(stageId = 1, applied = false, committed = false)
                )
            )
        )
        assertFalse(
            isDroppedPrefixCommitValid(
                evalOnly = false,
                success = true,
                terminal = false,
                forwardDropped = true,
                processedStageId = 0,
                firstUnprocessedStageId = 2,
                stageMetrics = listOf(stageMetric(stageId = 0, applied = true, committed = true))
            )
        )
    }

    private fun stageMetric(stageId: Int, applied: Boolean, committed: Boolean): Sid.StageExecutionMetrics {
        return Sid.StageExecutionMetrics.newBuilder()
            .setStageId(stageId)
            .setEvalOnly(false)
            .setOptimizerStepApplied(applied)
            .setOptimizerStepCommitted(committed)
            .build()
    }
}
