package com.example.sid_trainer

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import sid.Sid

class ForwardFailurePolicyTest {
    @Test
    fun bestEffortModeAcceptsAnIncompleteForwardRoute() {
        val request = request(drop = true, durable = false)

        assertTrue(acceptsIncompleteForwardRoute(request))
        assertTrue(
            shouldDropForwardBoundary(
                request,
                ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED
            )
        )
    }

    @Test
    fun bestEffortModeDoesNotDropAnAmbiguousDelivery() {
        val request = request(drop = true, durable = false)

        assertFalse(
            shouldDropForwardBoundary(
                request,
                ForwardDeliveryStatus.DELIVERY_UNKNOWN
            )
        )
    }

    @Test
    fun bestEffortModeDoesNotDropAReceivedFailureResponse() {
        val request = request(drop = true, durable = false)

        assertFalse(
            shouldDropForwardBoundary(
                request,
                ForwardDeliveryStatus.RESPONSE_RECEIVED
            )
        )
    }

    @Test
    fun durableReplayNeverDropsItsBoundary() {
        val request = request(drop = false, durable = true)

        assertTrue(acceptsIncompleteForwardRoute(request))
        assertFalse(
            shouldDropForwardBoundary(
                request,
                ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED
            )
        )
    }

    @Test
    fun strictModeNeverDropsItsBoundary() {
        val request = request(drop = false, durable = false)

        assertFalse(acceptsIncompleteForwardRoute(request))
        assertFalse(
            shouldDropForwardBoundary(
                request,
                ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED
            )
        )
    }

    private fun request(drop: Boolean, durable: Boolean): Sid.ForwardChunkRequest =
        Sid.ForwardChunkRequest.newBuilder()
            .setRequestId("forward-policy")
            .setDropOnForwardFailure(drop)
            .setDurablePipeline(durable)
            .build()
}
