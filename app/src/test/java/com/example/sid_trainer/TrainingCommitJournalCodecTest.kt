package com.example.sid_trainer

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class TrainingCommitJournalCodecTest {
    @Test
    fun roundTripPreservesCheckpointStepAndCommitOrder() {
        val bytes = ByteArrayOutputStream().also { buffer ->
            DataOutputStream(buffer).use { output ->
                TrainingCommitJournalCodec.write(
                    output,
                    checkpointStep = 129L,
                    committedRequestKeys = linkedSetOf("request-128|batch=128|chunk=1", "request-129|batch=129|chunk=1")
                )
            }
        }.toByteArray()

        val restored = DataInputStream(ByteArrayInputStream(bytes)).use { input ->
            TrainingCommitJournalCodec.read(input, expectedCheckpointStep = 129L, maxKeys = 100)
        }

        assertEquals(
            listOf("request-128|batch=128|chunk=1", "request-129|batch=129|chunk=1"),
            restored.toList()
        )
    }

    @Test
    fun rejectsJournalFromDifferentCheckpointGeneration() {
        val bytes = ByteArrayOutputStream().also { buffer ->
            DataOutputStream(buffer).use { output ->
                TrainingCommitJournalCodec.write(output, checkpointStep = 130L, committedRequestKeys = listOf("request-130"))
            }
        }.toByteArray()

        assertThrows(IllegalArgumentException::class.java) {
            DataInputStream(ByteArrayInputStream(bytes)).use { input ->
                TrainingCommitJournalCodec.read(input, expectedCheckpointStep = 129L, maxKeys = 100)
            }
        }
    }

    @Test
    fun rejectsDuplicateCommitKeys() {
        val bytes = ByteArrayOutputStream().also { buffer ->
            DataOutputStream(buffer).use { output ->
                TrainingCommitJournalCodec.write(
                    output,
                    checkpointStep = 5L,
                    committedRequestKeys = listOf("request-5", "request-5")
                )
            }
        }.toByteArray()

        assertThrows(IllegalArgumentException::class.java) {
            DataInputStream(ByteArrayInputStream(bytes)).use { input ->
                TrainingCommitJournalCodec.read(input, expectedCheckpointStep = 5L, maxKeys = 100)
            }
        }
    }
}
