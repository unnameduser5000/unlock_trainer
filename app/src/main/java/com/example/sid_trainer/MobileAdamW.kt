package com.example.sid_trainer

import org.pytorch.executorch.DType
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.TensorBufferAccess
import java.nio.FloatBuffer
import kotlin.math.pow
import kotlin.math.sqrt

class MobileAdamW(
    private val namedParameters: Map<String, Tensor>,
    private val learningRate: Double,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val eps: Double = 1e-8,
    private val weightDecay: Double = 0.01
) {
    private val states = HashMap<String, ParamState>()
    private var stepCount = 0L

    init {
        require(namedParameters.isNotEmpty()) { "AdamW requires at least one parameter." }
        require(learningRate > 0.0) { "AdamW learning rate must be positive, got $learningRate." }
    }

    fun step(namedGradients: Map<String, Tensor>) {
        require(namedGradients.isNotEmpty()) { "AdamW requires at least one gradient." }
        stepCount += 1
        val biasCorrection1 = 1.0 - beta1.pow(stepCount.toDouble())
        val biasCorrection2 = 1.0 - beta2.pow(stepCount.toDouble())

        for ((name, parameter) in namedParameters) {
            val gradient = namedGradients[name] ?: continue
            require(parameter.dtype() == DType.FLOAT) {
                "AdamW only supports float32 parameters; '$name' is ${parameter.dtype()}."
            }
            require(gradient.dtype() == DType.FLOAT) {
                "AdamW only supports float32 gradients; '$name' is ${gradient.dtype()}."
            }
            require(parameter.shape().contentEquals(gradient.shape())) {
                "AdamW shape mismatch for '$name': parameter=${parameter.shape().contentToString()} " +
                    "gradient=${gradient.shape().contentToString()}"
            }

            val paramBuffer = parameter.rawFloatBuffer()
            val gradBuffer = gradient.rawFloatBuffer()
            val numel = parameter.numel().toInt()
            require(gradBuffer.capacity() >= numel) {
                "AdamW gradient buffer for '$name' has capacity ${gradBuffer.capacity()}, expected $numel."
            }
            require(paramBuffer.capacity() >= numel) {
                "AdamW parameter buffer for '$name' has capacity ${paramBuffer.capacity()}, expected $numel."
            }

            val state = states.getOrPut(name) { ParamState(FloatArray(numel), FloatArray(numel)) }
            require(state.expAvg.size == numel && state.expAvgSq.size == numel) {
                "AdamW state for '$name' has stale size ${state.expAvg.size}, expected $numel."
            }

            for (i in 0 until numel) {
                val grad = gradBuffer.get(i).toDouble()
                val expAvg = beta1 * state.expAvg[i] + (1.0 - beta1) * grad
                val expAvgSq = beta2 * state.expAvgSq[i] + (1.0 - beta2) * grad * grad
                state.expAvg[i] = expAvg.toFloat()
                state.expAvgSq[i] = expAvgSq.toFloat()

                var value = paramBuffer.get(i).toDouble()
                if (weightDecay != 0.0) {
                    value *= 1.0 - learningRate * weightDecay
                }
                val denom = sqrt(expAvgSq / biasCorrection2) + eps
                value -= learningRate * (expAvg / biasCorrection1) / denom
                paramBuffer.put(i, value.toFloat())
            }
        }
    }

    fun snapshot(): Snapshot {
        return Snapshot(
            stepCount = stepCount,
            parameterStates = states.mapValues { (_, state) ->
                ParameterSnapshot(
                    expAvg = state.expAvg.copyOf(),
                    expAvgSq = state.expAvgSq.copyOf()
                )
            }
        )
    }

    fun restore(snapshot: Snapshot) {
        stepCount = snapshot.stepCount
        states.clear()
        for ((name, state) in snapshot.parameterStates) {
            val parameter = namedParameters[name] ?: continue
            val numel = parameter.numel().toInt()
            require(state.expAvg.size == numel && state.expAvgSq.size == numel) {
                "AdamW checkpoint state for '$name' has size expAvg=${state.expAvg.size} " +
                    "expAvgSq=${state.expAvgSq.size}, expected $numel."
            }
            states[name] = ParamState(
                expAvg = state.expAvg.copyOf(),
                expAvgSq = state.expAvgSq.copyOf()
            )
        }
    }

    private fun Tensor.rawFloatBuffer(): FloatBuffer {
        val buffer = TensorBufferAccess.rawDataBuffer(this)
        require(buffer is FloatBuffer) {
            "Expected FloatBuffer backing for tensor dtype=${dtype()}, got ${buffer::class.java.name}."
        }
        return buffer.duplicate()
    }

    private data class ParamState(
        val expAvg: FloatArray,
        val expAvgSq: FloatArray
    )

    data class Snapshot(
        val stepCount: Long,
        val parameterStates: Map<String, ParameterSnapshot>
    )

    data class ParameterSnapshot(
        val expAvg: FloatArray,
        val expAvgSq: FloatArray
    )
}
