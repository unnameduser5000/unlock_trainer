package com.example.sid_trainer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import sid.Sid
import java.nio.file.Files

class DurableBoundaryOutboxTest {
    @Test
    fun preparedBoundaryIsInvisibleUntilLocalCommit() {
        val outbox = newOutbox()
        val prepared = outbox.prepare(request("r1", 1), capacity = 2)

        assertFalse(prepared.duplicate)
        assertNull(outbox.peekReady())

        outbox.markReady("r1")
        assertEquals("r1", outbox.peekReady()?.request?.requestId)
    }

    @Test
    fun readyBoundariesSurviveRestartAndRemainOrdered() {
        val root = Files.createTempDirectory("sid-boundary-outbox-test").toFile()
        DurableBoundaryOutbox(root).also { first ->
            first.activateStage(0)
            first.prepare(request("r2", 2), capacity = 4)
            first.markReady("r2")
            first.prepare(request("r3", 3), capacity = 4)
            first.markReady("r3")
        }

        val restored = DurableBoundaryOutbox(root).also { it.activateStage(0) }
        val first = requireNotNull(restored.peekReady())
        assertEquals("r2", first.request.requestId)
        restored.acknowledge(first)
        assertEquals("r3", restored.peekReady()?.request?.requestId)
    }

    @Test
    fun acknowledgedBoundarySurvivesRestartAsReplayReceipt() {
        val root = Files.createTempDirectory("sid-boundary-ack-test").toFile()
        DurableBoundaryOutbox(root).also { first ->
            first.activateStage(0)
            first.prepare(request("r1", 1), capacity = 1)
            val ready = first.markReady("r1")
            assertEquals(0, first.acknowledge(ready))
            assertEquals(1, first.acknowledgedCount())
            assertTrue(first.resolveCommittedReplay(sourceRequest("r1", 1)) is
                DurableBoundaryReplayResolution.Acknowledged)
        }

        val restored = DurableBoundaryOutbox(root).also { it.activateStage(0) }
        assertEquals(0, restored.pendingCount())
        assertEquals(1, restored.acknowledgedCount())
        assertTrue(restored.resolveCommittedReplay(sourceRequest("r1", 1)) is
            DurableBoundaryReplayResolution.Acknowledged)
        assertNull(restored.peekReady())
    }

    @Test
    fun committedReplayFindsExistingBoundaryWithoutAddingAnotherEntry() {
        val outbox = newOutbox()
        outbox.prepare(request("r1", 1), capacity = 1)
        outbox.markReady("r1")

        val resolution = outbox.resolveCommittedReplay(sourceRequest("r1", 1))

        assertTrue(resolution is DurableBoundaryReplayResolution.Pending)
        assertEquals(1, outbox.pendingCount())
        assertEquals("r1", outbox.peekReady()?.request?.requestId)
    }

    @Test
    fun duplicateDoesNotConsumeCapacityButChangedPayloadIsRejected() {
        val outbox = newOutbox()
        outbox.prepare(request("r1", 1), capacity = 1)

        val duplicate = outbox.prepare(request("r1", 1), capacity = 1)
        assertTrue(duplicate.duplicate)
        assertEquals(1, duplicate.pendingCount)
        assertThrows(IllegalArgumentException::class.java) {
            outbox.prepare(request("r1", 9), capacity = 1)
        }
        assertThrows(DurableBoundaryBufferFullException::class.java) {
            outbox.prepare(request("r2", 2), capacity = 1)
        }
    }

    @Test
    fun abortOnlyRemovesUncommittedBoundary() {
        val outbox = newOutbox()
        outbox.prepare(request("r1", 1), capacity = 2)
        assertTrue(outbox.abortPrepared("r1"))
        assertEquals(0, outbox.pendingCount())

        outbox.prepare(request("r2", 2), capacity = 2)
        outbox.markReady("r2")
        assertFalse(outbox.abortPrepared("r2"))
        assertEquals(1, outbox.pendingCount())
    }

    private fun newOutbox(): DurableBoundaryOutbox = DurableBoundaryOutbox(
        Files.createTempDirectory("sid-boundary-outbox-test").toFile()
    ).also { it.activateStage(0) }

    private fun request(requestId: String, window: Long): Sid.ForwardChunkRequest =
        Sid.ForwardChunkRequest.newBuilder()
            .setRequestId(requestId)
            .setBatchId(window.toInt())
            .setChunkIdx(1)
            .setDurablePipeline(true)
            .setReplayBufferCapacity(4)
            .setPipelineWindowSeq(window)
            .build()

    private fun sourceRequest(requestId: String, window: Long): Sid.ForwardChunkRequest =
        Sid.ForwardChunkRequest.newBuilder()
            .setRequestId(requestId)
            .setBatchId(window.toInt())
            .setChunkIdx(0)
            .setDurablePipeline(true)
            .setReplayBufferCapacity(4)
            .setPipelineWindowSeq(window)
            .build()
}
