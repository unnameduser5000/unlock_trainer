package com.example.sid_trainer

import android.os.Trace
import android.util.Log
import org.pytorch.executorch.Tensor
import sid.Sid

internal data class PreparedTrainingStep(
    val request: Sid.ForwardChunkRequest,
    val learningRate: Double,
    val requestKey: String,
    val applyOptimizerStep: Boolean,
    val optimizerStepAlreadyCommitted: Boolean,
    val shortCircuitLocalExecution: Boolean,
    val checkpointRestoreMs: Long
)

internal data class LocalUpdateDecision(
    val request: Sid.ForwardChunkRequest,
    val applyOptimizerStep: Boolean,
    val replayWithoutUpdate: Boolean,
    val optimizerStepAlreadyCommitted: Boolean,
    val shortCircuitLocalExecution: Boolean
)

internal fun decideLocalUpdate(
    request: Sid.ForwardChunkRequest,
    alreadyCommitted: Boolean
): LocalUpdateDecision {
    val replayWithoutUpdate = !request.evalOnly && alreadyCommitted
    return LocalUpdateDecision(
        request = request,
        applyOptimizerStep = !request.evalOnly && !alreadyCommitted,
        replayWithoutUpdate = replayWithoutUpdate,
        optimizerStepAlreadyCommitted = replayWithoutUpdate,
        shortCircuitLocalExecution = replayWithoutUpdate
    )
}

internal data class MobileOptimizerOutcome(
    val optimizerStepApplied: Boolean,
    val optimizerStepCommitted: Boolean,
    val checkpointSaved: Boolean,
    val optimizerCreateMs: Long,
    val optimizerStepMs: Long,
    val optimizerStepCount: Long,
    val checkpointSaveMs: Long = 0L,
    val checkpointBytes: Long = 0L,
    val checkpointStep: Long = 0L
)

private data class CheckpointSaveOutcome(
    val saved: Boolean = false,
    val elapsedMs: Long = 0L,
    val bytes: Long = 0L,
    val step: Long = 0L
)

/** Shared per-model AdamW, deduplication, and checkpoint state. */
internal class MobileTrainingState(
    private val modelPath: String,
    private val parametersProvider: () -> Map<String, Tensor>,
    private val defaultLearningRate: Double = 1e-5,
    val checkpointIntervalSteps: Long = 1L
) {
    private var optimizer: MobileAdamW? = null
    private var optimizerLearningRate: Double? = null
    private var parameters: Map<String, Tensor>? = null
    private var pendingOptimizerSnapshot: MobileAdamW.Snapshot? = null
    private var checkpointRestoreAttempted = false
    private var optimizerStepCount = 0L
    private val committedRequestKeys = LinkedHashSet<String>()
    private var commitPersistencePending = false

    init {
        require(checkpointIntervalSteps == 1L) {
            "Durable request commits require a checkpoint after every optimizer step."
        }
    }

    fun prepare(request: Sid.ForwardChunkRequest): PreparedTrainingStep {
        val learningRate = request.trainingLearningRate()
        val restoreStartedAtNs = System.nanoTime()
        ensureCheckpointRestored(learningRate)
        persistPendingCommit()
        val checkpointRestoreMs = elapsedMs(restoreStartedAtNs)
        val requestKey = request.commitKey()
        val decision = decideLocalUpdate(
            request = request,
            alreadyCommitted = committedRequestKeys.contains(requestKey)
        )
        if (decision.replayWithoutUpdate) {
            Log.w(
                LOG_TAG,
                "Duplicate local commit; replaying boundary without optimizer step key=$requestKey"
            )
        }
        return PreparedTrainingStep(
            request = decision.request,
            learningRate = learningRate,
            requestKey = requestKey,
            applyOptimizerStep = decision.applyOptimizerStep,
            optimizerStepAlreadyCommitted = decision.optimizerStepAlreadyCommitted,
            shortCircuitLocalExecution = decision.shortCircuitLocalExecution,
            checkpointRestoreMs = checkpointRestoreMs
        )
    }

    fun applyGradients(
        prepared: PreparedTrainingStep,
        gradients: Map<String, Tensor>
    ): MobileOptimizerOutcome {
        if (!prepared.applyOptimizerStep) {
            return outcomeWithoutOptimizerStep()
        }
        require(gradients.isNotEmpty()) { "Training produced no gradients for $modelPath." }

        var optimizerCreateMs = 0L
        val adamW = optimizer.takeIf { optimizerLearningRate == prepared.learningRate } ?: run {
            val startedAtNs = System.nanoTime()
            tracedSection("sid_create_adamw") {
                createOptimizer(prepared.learningRate)
            }.also {
                optimizer = it
                optimizerLearningRate = prepared.learningRate
                optimizerCreateMs = elapsedMs(startedAtNs)
            }
        }
        val stepStartedAtNs = System.nanoTime()
        tracedSection("sid_adamw_step") {
            adamW.step(gradients)
        }
        val optimizerStepMs = elapsedMs(stepStartedAtNs)

        committedRequestKeys += prepared.requestKey
        optimizerStepCount += 1
        commitPersistencePending = true
        val checkpoint = saveCommitCheckpoint()
        return MobileOptimizerOutcome(
            optimizerStepApplied = true,
            optimizerStepCommitted = true,
            checkpointSaved = checkpoint.saved,
            optimizerCreateMs = optimizerCreateMs,
            optimizerStepMs = optimizerStepMs,
            optimizerStepCount = optimizerStepCount,
            checkpointSaveMs = checkpoint.elapsedMs,
            checkpointBytes = checkpoint.bytes,
            checkpointStep = checkpoint.step
        )
    }

    fun outcomeWithoutOptimizerStep(optimizerStepCommitted: Boolean = false): MobileOptimizerOutcome {
        return MobileOptimizerOutcome(false, optimizerStepCommitted, false, 0, 0, optimizerStepCount)
    }

    fun currentOptimizerStepCount(): Long = optimizerStepCount

    private fun ensureCheckpointRestored(learningRate: Double) {
        if (checkpointRestoreAttempted) return
        var checkpointHasOptimizerState = false
        val result = TrainingCheckpointStore.restoreLatest(modelPath, parameters()) { snapshot ->
            checkpointHasOptimizerState = true
            val existingOptimizer = optimizer
            if (existingOptimizer != null && optimizerLearningRate == learningRate) {
                existingOptimizer.restore(snapshot)
            } else {
                pendingOptimizerSnapshot = snapshot
            }
        }
        check(!result.restoreFailed) {
            "Refusing to train after checkpoint restore failure for $modelPath: ${result.message}"
        }
        checkpointRestoreAttempted = true
        if (result.restored) {
            optimizerStepCount = result.step
            committedRequestKeys.clear()
            committedRequestKeys.addAll(result.committedRequestKeys)
        }
        Log.i(
            LOG_TAG,
            "Checkpoint restore model=$modelPath restored=${result.restored} step=${result.step} " +
                "parameters=${result.parameterCount} optimizerState=$checkpointHasOptimizerState " +
                "committedRequests=${result.committedRequestKeys.size} " +
                "path=${result.checkpointPath} message=${result.message}"
        )
    }

    private fun parameters(): Map<String, Tensor> {
        parameters?.let { return it }
        val loaded = parametersProvider()
        require(loaded.isNotEmpty()) { "Training module has no trainable parameters for $modelPath." }
        parameters = loaded
        return loaded
    }

    private fun createOptimizer(learningRate: Double): MobileAdamW {
        Log.i(LOG_TAG, "Creating AdamW model=$modelPath lr=$learningRate")
        val adamW = MobileAdamW(parameters(), learningRate)
        pendingOptimizerSnapshot?.let { snapshot ->
            adamW.restore(snapshot)
            pendingOptimizerSnapshot = null
            Log.i(
                LOG_TAG,
                "Restored pending AdamW state model=$modelPath step=${snapshot.stepCount}"
            )
        }
        return adamW
    }

    private fun saveCommitCheckpoint(): CheckpointSaveOutcome {
        val startedAtNs = System.nanoTime()
        val checkpointFile = try {
            TrainingCheckpointStore.saveLatest(
                modelPath,
                optimizerStepCount,
                parameters(),
                optimizer?.snapshot(),
                committedRequestKeys
            )
        } catch (error: Throwable) {
            Log.e(LOG_TAG, "Checkpoint save failed model=$modelPath step=$optimizerStepCount", error)
            throw error
        }
        commitPersistencePending = false
        val saveMs = elapsedMs(startedAtNs)
        return CheckpointSaveOutcome(
            saved = true,
            elapsedMs = saveMs,
            bytes = checkpointFile.length(),
            step = optimizerStepCount
        )
    }

    private fun persistPendingCommit() {
        if (!commitPersistencePending) return
        Log.w(LOG_TAG, "Retrying pending durable commit model=$modelPath step=$optimizerStepCount")
        saveCommitCheckpoint()
    }

    private fun Sid.ForwardChunkRequest.commitKey(): String {
        val stableRequestId = requestId.ifBlank { "batch-$batchId" }
        return "$stableRequestId|batch=$batchId|chunk=$chunkIdx"
    }

    private fun Sid.ForwardChunkRequest.trainingLearningRate(): Double {
        return if (learningRate > 0f) learningRate.toDouble() else defaultLearningRate
    }

    private fun elapsedMs(startedAtNs: Long): Long {
        return (System.nanoTime() - startedAtNs) / 1_000_000L
    }

    private inline fun <T> tracedSection(name: String, block: () -> T): T {
        Trace.beginSection(name)
        return try {
            block()
        } finally {
            Trace.endSection()
        }
    }

    private companion object {
        const val LOG_TAG = "MobileTrainingState"
    }
}
