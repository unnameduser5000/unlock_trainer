package com.example.sid_trainer

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicReference

data class MemoryPeakStats(
    val intervalMs: Long,
    val sampleCount: Long,
    val before: WorkerTelemetry,
    val after: WorkerTelemetry,
    val pssPeakKb: Long,
    val privateDirtyPeakKb: Long,
    val javaHeapPeakKb: Long
)

class MemoryPeakSampler private constructor(
    private val appContext: Context,
    private val intervalMs: Long,
    private val before: WorkerTelemetry,
    private val scope: CoroutineScope
) {
    private val state = AtomicReference(
        SampleState(
            sampleCount = 1L,
            pssPeakKb = before.appPssKb,
            privateDirtyPeakKb = before.appPrivateDirtyKb,
            javaHeapPeakKb = before.runtimeUsedMemoryKb
        )
    )
    private var job: Job? = null

    fun start(): MemoryPeakSampler {
        job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                delay(intervalMs)
                sample()
            }
        }
        return this
    }

    suspend fun stop(): MemoryPeakStats {
        job?.cancel()
        job = null
        val after = WorkerTelemetryReader.read(appContext)
        record(after)
        val snapshot = state.get()
        return MemoryPeakStats(
            intervalMs = intervalMs,
            sampleCount = snapshot.sampleCount,
            before = before,
            after = after,
            pssPeakKb = snapshot.pssPeakKb,
            privateDirtyPeakKb = snapshot.privateDirtyPeakKb,
            javaHeapPeakKb = snapshot.javaHeapPeakKb
        )
    }

    private fun sample() {
        record(WorkerTelemetryReader.read(appContext))
    }

    private fun record(telemetry: WorkerTelemetry) {
        while (true) {
            val current = state.get()
            val updated = current.copy(
                sampleCount = current.sampleCount + 1L,
                pssPeakKb = maxOf(current.pssPeakKb, telemetry.appPssKb),
                privateDirtyPeakKb = maxOf(current.privateDirtyPeakKb, telemetry.appPrivateDirtyKb),
                javaHeapPeakKb = maxOf(current.javaHeapPeakKb, telemetry.runtimeUsedMemoryKb)
            )
            if (state.compareAndSet(current, updated)) {
                return
            }
        }
    }

    private data class SampleState(
        val sampleCount: Long,
        val pssPeakKb: Long,
        val privateDirtyPeakKb: Long,
        val javaHeapPeakKb: Long
    )

    companion object {
        fun create(
            context: Context,
            intervalMs: Long = DEFAULT_INTERVAL_MS,
            scope: CoroutineScope = CoroutineScope(Dispatchers.Default)
        ): MemoryPeakSampler {
            val appContext = context.applicationContext
            return MemoryPeakSampler(
                appContext = appContext,
                intervalMs = intervalMs.coerceAtLeast(10L),
                before = WorkerTelemetryReader.read(appContext),
                scope = scope
            )
        }

        const val DEFAULT_INTERVAL_MS = 100L
    }
}
