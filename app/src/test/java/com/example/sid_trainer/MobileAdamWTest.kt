package com.example.sid_trainer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.pytorch.executorch.Tensor

class MobileAdamWTest {
    @Test
    fun stepUpdatesParameterTensorInPlace() {
        val parameter = Tensor.fromBlob(floatArrayOf(1.0f, -2.0f), longArrayOf(2))
        val gradient = Tensor.fromBlob(floatArrayOf(0.5f, -0.25f), longArrayOf(2))
        val optimizer = MobileAdamW(mapOf("weight" to parameter), learningRate = 1e-4)

        optimizer.step(mapOf("weight" to gradient))

        val updated = parameter.getDataAsFloatArray()
        assertTrue("Expected first parameter to decrease, got ${updated[0]}", updated[0] < 1.0f)
        assertTrue("Expected second parameter to increase, got ${updated[1]}", updated[1] > -2.0f)
    }

    @Test
    fun restoredSnapshotMatchesContinuousSecondStep() {
        val gradient1 = Tensor.fromBlob(floatArrayOf(0.5f, -0.25f), longArrayOf(2))
        val gradient2 = Tensor.fromBlob(floatArrayOf(0.125f, 0.75f), longArrayOf(2))
        val continuousParameter = Tensor.fromBlob(floatArrayOf(1.0f, -2.0f), longArrayOf(2))
        val continuousOptimizer = MobileAdamW(
            mapOf("weight" to continuousParameter),
            learningRate = 1e-4
        )

        continuousOptimizer.step(mapOf("weight" to gradient1))
        val snapshot = continuousOptimizer.snapshot()
        val afterFirstStep = continuousParameter.getDataAsFloatArray()
        continuousOptimizer.step(mapOf("weight" to gradient2))

        val restoredParameter = Tensor.fromBlob(afterFirstStep, longArrayOf(2))
        val restoredOptimizer = MobileAdamW(
            mapOf("weight" to restoredParameter),
            learningRate = 1e-4
        )
        restoredOptimizer.restore(snapshot)
        restoredOptimizer.step(mapOf("weight" to gradient2))

        val expected = continuousParameter.getDataAsFloatArray()
        val actual = restoredParameter.getDataAsFloatArray()
        assertEquals(expected[0].toDouble(), actual[0].toDouble(), 1e-7)
        assertEquals(expected[1].toDouble(), actual[1].toDouble(), 1e-7)
    }
}
