package com.example.sid_trainer

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import sid.Sid

class MobileTrainingStatePolicyTest {
    @Test
    fun duplicateLocalCommitDoesNotTurnTrainingReplayIntoEval() {
        val request = trainingRequest("request-45")

        val decision = decideLocalUpdate(request, alreadyCommitted = true)

        assertFalse(decision.applyOptimizerStep)
        assertTrue(decision.replayWithoutUpdate)
        assertTrue(decision.optimizerStepAlreadyCommitted)
        assertTrue(decision.shortCircuitLocalExecution)
        assertFalse(decision.request.evalOnly)
    }

    @Test
    fun uncommittedDownstreamStageStillAppliesUpdateForSameRequest() {
        val request = trainingRequest("request-45")

        val decision = decideLocalUpdate(request, alreadyCommitted = false)

        assertTrue(decision.applyOptimizerStep)
        assertFalse(decision.replayWithoutUpdate)
        assertFalse(decision.optimizerStepAlreadyCommitted)
        assertFalse(decision.shortCircuitLocalExecution)
        assertFalse(decision.request.evalOnly)
    }

    @Test
    fun realEvalRequestNeverAppliesOptimizerStep() {
        val request = trainingRequest("eval-1").toBuilder().setEvalOnly(true).build()

        val decision = decideLocalUpdate(request, alreadyCommitted = false)

        assertFalse(decision.applyOptimizerStep)
        assertFalse(decision.replayWithoutUpdate)
        assertFalse(decision.optimizerStepAlreadyCommitted)
        assertFalse(decision.shortCircuitLocalExecution)
        assertTrue(decision.request.evalOnly)
    }

    private fun trainingRequest(requestId: String): Sid.ForwardChunkRequest {
        return Sid.ForwardChunkRequest.newBuilder()
            .setRequestId(requestId)
            .setBatchId(45)
            .setChunkIdx(0)
            .setEvalOnly(false)
            .build()
    }
}
