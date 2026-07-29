package com.example.sid_coordinator

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sid.Sid

class PreparedRequestSupportTest {
    @Test
    fun beliefTransportDefaultsToTerminalOnly() {
        assertEquals("terminal", DEFAULT_BELIEF_TRANSPORT_MODE)
        assertEquals("terminal", normalizeBeliefTransportMode(null))
        assertEquals("terminal", normalizeBeliefTransportMode(""))
        assertEquals("terminal", normalizeBeliefTransportMode("terminal"))
    }

    @Test
    fun fullBeliefTransportRequiresAnExplicitMode() {
        assertEquals("full", normalizeBeliefTransportMode("full"))
        assertEquals("full", normalizeBeliefTransportMode("dense"))
        assertEquals("none", normalizeBeliefTransportMode("none"))
    }

    @Test
    fun onlyStrictRequestsRequireACompletePipelineAtAdmission() {
        val strict = Sid.ForwardChunkRequest.getDefaultInstance()
        val durable = strict.toBuilder().setDurablePipeline(true).build()
        val bestEffort = strict.toBuilder().setDropOnForwardFailure(true).build()

        assertTrue(requiresCompletePipeline(strict))
        assertFalse(requiresCompletePipeline(durable))
        assertFalse(requiresCompletePipeline(bestEffort))
    }
}
