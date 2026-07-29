package com.example.sid_trainer

import android.os.Trace
import android.util.Log
import org.pytorch.executorch.training.BPFreeTrainingModule
import sid.Sid
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/** Executes one BP-free request as forward-boundary -> local backward/update. */
internal object BPFreeShardRuntime {
    private const val LOG_TAG = "BPFreeShardRuntime"
    private const val METHOD = "forward"
    private const val DEFAULT_LEARNING_RATE = 1e-5
    private const val CHECKPOINT_INTERVAL_STEPS = 1L

    private data class LoadedRuntime(
        val modelPath: String,
        val module: BPFreeTrainingModule,
        val trainingState: MobileTrainingState
    )

    // The native module can hold exactly one paused Method, so its lifecycle is serialized here.
    private val executionLock = ReentrantLock()
    private var loadedRuntime: LoadedRuntime? = null

    fun execute(
        modelPath: String,
        request: Sid.ForwardChunkRequest,
        outputPolicy: ShardBoundaryOutputPolicy,
        onBoundary: ((ShardBoundaryResult) -> Unit)?
    ): ShardExecutionResult = executionLock.withLock {
        val acquireStartedAtNs = System.nanoTime()
        val runtime = acquire(modelPath)
        val runtimeAcquireMs = elapsedMs(acquireStartedAtNs)
        val prepared = runtime.trainingState.prepare(request)

        if (prepared.shortCircuitLocalExecution) {
            Log.i(
                LOG_TAG,
                "Committed request replay short-circuited before forward/backward " +
                    "request=${prepared.requestKey} step=${runtime.trainingState.currentOptimizerStepCount()}"
            )
            return ShardExecutionResult(
                outputHiddenStates = Sid.TensorData.getDefaultInstance(),
                outputShiftLogP = Sid.TensorData.getDefaultInstance(),
                localLoss = 0f,
                runtimeName = "BPFreeTrainingModule",
                methodName = METHOD,
                inputCount = 0,
                evalOnly = false,
                learningRate = prepared.learningRate,
                optimizerStepApplied = false,
                optimizerStepCommitted = true,
                checkpointSaved = false,
                checkpointIntervalSteps = runtime.trainingState.checkpointIntervalSteps,
                runtimeExecutionMode = "split",
                optimizerStepCount = runtime.trainingState.currentOptimizerStepCount(),
                checkpointBytes = 0L,
                checkpointStep = 0L,
                timing = ShardExecutionTiming(
                    runtimeAcquireMs = runtimeAcquireMs,
                    checkpointRestoreMs = prepared.checkpointRestoreMs
                ),
                committedReplayShortCircuited = true
            )
        }

        val inputStartedAtNs = System.nanoTime()
        val inputs = tracedSection("sid_build_training_inputs") {
            NativeShardRunner.buildBoundaryTrainingInputs(prepared.request)
        }
        val inputBuildMs = elapsedMs(inputStartedAtNs)

        val forwardStartedEpochMs = System.currentTimeMillis()
        val forwardStartedAtNs = System.nanoTime()
        val nativeBoundary = tracedSection("sid_bpfree_forward_boundary") {
            runtime.module.forwardToBoundary(METHOD, *inputs)
        }
        val forwardBoundaryMs = elapsedMs(forwardStartedAtNs)
        val forwardBoundaryEpochMs = System.currentTimeMillis()

        val outputStartedAtNs = System.nanoTime()
        val boundary = ShardBoundaryResult(
            outputHiddenStates = if (outputPolicy.includeHidden) {
                NativeShardRunner.copyBoundaryTensor(nativeBoundary.hidden)
            } else {
                NativeShardRunner.boundaryTensorMetadata(nativeBoundary.hidden)
            },
            outputShiftLogP = if (outputPolicy.includeBelief) {
                NativeShardRunner.copyBoundaryTensor(nativeBoundary.belief)
            } else {
                NativeShardRunner.boundaryTensorMetadata(nativeBoundary.belief)
            },
            localLoss = NativeShardRunner.boundaryScalarValue(nativeBoundary.loss),
            evalOnly = prepared.request.evalOnly
        )
        val outputConvertMs = elapsedMs(outputStartedAtNs)

        var boundaryCallbackFailure: Throwable? = null
        try {
            onBoundary?.invoke(boundary)
        } catch (error: Throwable) {
            boundaryCallbackFailure = error
        }

        // Always finish the paused Method, including when dispatch setup fails in the caller.
        val backwardStartedEpochMs = System.currentTimeMillis()
        val backwardStartedAtNs = System.nanoTime()
        val resumedOutputs = tracedSection("sid_bpfree_resume_backward") {
            runtime.module.resumeBackward(METHOD)
        }
        val backwardMs = elapsedMs(backwardStartedAtNs)
        val backwardCompletedEpochMs = System.currentTimeMillis()
        boundaryCallbackFailure?.let { throw it }

        var gradientsMs = 0L
        var gradientCount = 0
        val optimizerOutcome = if (prepared.applyOptimizerStep) {
            val gradientsStartedAtNs = System.nanoTime()
            val gradients = tracedSection("sid_named_gradients") {
                runtime.module.namedGradients(METHOD)
            }
            gradientsMs = elapsedMs(gradientsStartedAtNs)
            gradientCount = gradients.size
            runtime.trainingState.applyGradients(prepared, gradients)
        } else if (!prepared.request.evalOnly) {
            Log.i(
                LOG_TAG,
                "Replayed committed request without local optimizer step request=${prepared.requestKey}"
            )
            runtime.trainingState.outcomeWithoutOptimizerStep(
                optimizerStepCommitted = prepared.optimizerStepAlreadyCommitted
            )
        } else {
            runtime.trainingState.outcomeWithoutOptimizerStep()
        }

        val executeMs = forwardBoundaryMs + backwardMs
        val timing = ShardExecutionTiming(
            runtimeAcquireMs = runtimeAcquireMs,
            checkpointRestoreMs = prepared.checkpointRestoreMs,
            inputBuildMs = inputBuildMs,
            executeMs = executeMs,
            forwardBoundaryMs = forwardBoundaryMs,
            backwardMs = backwardMs,
            gradientsMs = gradientsMs,
            optimizerCreateMs = optimizerOutcome.optimizerCreateMs,
            optimizerStepMs = optimizerOutcome.optimizerStepMs,
            checkpointSaveMs = optimizerOutcome.checkpointSaveMs,
            outputConvertMs = outputConvertMs,
            forwardStartedEpochMs = forwardStartedEpochMs,
            forwardBoundaryEpochMs = forwardBoundaryEpochMs,
            backwardStartedEpochMs = backwardStartedEpochMs,
            backwardCompletedEpochMs = backwardCompletedEpochMs
        )
        Log.i(
            LOG_TAG,
            "Completed model=$modelPath inputs=${inputs.size} resumedOutputs=${resumedOutputs.size} " +
                "evalOnly=${prepared.request.evalOnly} gradients=$gradientCount " +
                "hiddenCopied=${outputPolicy.includeHidden} beliefCopied=${outputPolicy.includeBelief} " +
                "optimizerStep=${optimizerOutcome.optimizerStepCount} " +
                "checkpointStep=${optimizerOutcome.checkpointStep} checkpointBytes=${optimizerOutcome.checkpointBytes} " +
                "${timing.describeForLog()}"
        )

        ShardExecutionResult(
            outputHiddenStates = boundary.outputHiddenStates,
            outputShiftLogP = boundary.outputShiftLogP,
            localLoss = boundary.localLoss,
            runtimeName = "BPFreeTrainingModule",
            methodName = METHOD,
            inputCount = inputs.size,
            evalOnly = prepared.request.evalOnly,
            learningRate = prepared.learningRate,
            optimizerStepApplied = optimizerOutcome.optimizerStepApplied,
            optimizerStepCommitted = optimizerOutcome.optimizerStepCommitted,
            checkpointSaved = optimizerOutcome.checkpointSaved,
            checkpointIntervalSteps = runtime.trainingState.checkpointIntervalSteps,
            runtimeExecutionMode = "split",
            optimizerStepCount = optimizerOutcome.optimizerStepCount,
            checkpointBytes = optimizerOutcome.checkpointBytes,
            checkpointStep = optimizerOutcome.checkpointStep,
            timing = timing,
            committedReplayShortCircuited = false
        )
    }

    fun release() = executionLock.withLock {
        loadedRuntime?.module?.destroy()
        loadedRuntime = null
    }

    private fun acquire(modelPath: String): LoadedRuntime {
        loadedRuntime?.takeIf { it.modelPath == modelPath }?.let { return it }
        loadedRuntime?.module?.destroy()

        val module = BPFreeTrainingModule.load(modelPath)
        val state = MobileTrainingState(
            modelPath = modelPath,
            parametersProvider = { module.namedParameters(METHOD) },
            defaultLearningRate = DEFAULT_LEARNING_RATE,
            checkpointIntervalSteps = CHECKPOINT_INTERVAL_STEPS
        )
        return LoadedRuntime(modelPath, module, state).also { loadedRuntime = it }
    }

    private inline fun <T> tracedSection(name: String, block: () -> T): T {
        Trace.beginSection(name)
        return try {
            block()
        } finally {
            Trace.endSection()
        }
    }

    private fun elapsedMs(startedAtNs: Long): Long {
        return (System.nanoTime() - startedAtNs) / 1_000_000L
    }
}
