package com.example.sid_trainer

import sid.Sid

internal class WorkerRequestEventRecorder(
    private val grpcManager: GrpcManager,
    private val registration: WorkerRegistration,
    private val request: Sid.ForwardChunkRequest,
    val requestId: String,
    private val localLog: (String, Throwable?) -> Unit
) {
    suspend fun record(
        eventType: Sid.RequestEventType,
        success: Boolean,
        message: String,
        terminal: Boolean = registration.isTerminal,
        throwable: Throwable? = null
    ) {
        localLog(message, throwable)
        grpcManager.reportRequestEvent(
            registration = registration,
            requestId = requestId,
            batchId = request.batchId,
            chunkIdx = request.chunkIdx,
            eventType = eventType,
            success = success,
            message = message,
            terminal = terminal
        )
    }
}
