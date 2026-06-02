package com.example.sid_coordinator

import com.google.protobuf.ByteString
import org.slf4j.LoggerFactory
import sid.Sid

class CoordinatorRequestOrchestrator(
    private val state: CoordinatorState
) {
    private val logger = LoggerFactory.getLogger(CoordinatorRequestOrchestrator::class.java)

    suspend fun submitRequest(
        request: Sid.ForwardChunkRequest,
        source: String
    ): Sid.ForwardChunkResponse {
        if (request.requestId.isBlank()) {
            return failureResponse(request, "request_id must not be blank for coordinator submission.")
        }

        val requestId = request.requestId
        val plan = state.planRequestSubmission()
        val submittedAtEpochMs = System.currentTimeMillis()
        val submittedAtNs = System.nanoTime()
        state.storeRequestPayload(requestId, request.toByteArray(), submittedAtEpochMs)

        if (!plan.accepted || plan.host == null || plan.port == null) {
            val message = "Coordinator rejected request $requestId from $source: ${plan.message}"
            logger.warn(message)
            state.recordCoordinatorRequestEvent(
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                stageId = plan.stageId,
                nodeId = plan.nodeId,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = message,
                terminal = false
            )
            val completedAtMs = System.currentTimeMillis()
            val response = failureResponse(request, plan.message)
            recordMetric(
                request,
                response,
                submittedAtEpochMs,
                completedAtMs,
                elapsedMsSince(submittedAtNs),
                TokenPredictionMetrics(0, 0)
            )
            return response
        }

        state.recordCoordinatorRequestEvent(
            requestId = requestId,
            batchId = request.batchId,
            chunkIdx = request.chunkIdx,
            stageId = plan.stageId,
            nodeId = plan.nodeId,
            eventType = Sid.RequestEventType.REQUEST_RECEIVED,
            success = true,
            message = "Coordinator accepted request from $source and dispatched it to stage 0 node ${plan.nodeId}; evalOnly=${request.evalOnly}",
            terminal = false
        )

        return try {
            logger.info(
                "Submitting requestId={} batchId={} chunkIdx={} evalOnly={} to stage0 node={} at {}:{} from {}",
                requestId,
                request.batchId,
                request.chunkIdx,
                request.evalOnly,
                plan.nodeId,
                plan.host,
                plan.port,
                source
            )
            val response = ProtoHttpForwardClient.forwardChunk(
                host = plan.host,
                port = plan.port,
                request = request
            )
            val completedAtMs = System.currentTimeMillis()
            val metrics = if (response.success && response.terminal) {
                safeTokenMetrics(request, response)
            } else {
                TokenPredictionMetrics(correct = 0, count = 0)
            }
            recordMetric(request, response, submittedAtEpochMs, completedAtMs, elapsedMsSince(submittedAtNs), metrics)
            response
        } catch (t: Throwable) {
            val completedAtMs = System.currentTimeMillis()
            val message = "Coordinator dispatch to stage 0 failed: ${t.message}"
            logger.error(
                "Dispatch failed for requestId={} node={} host={}:{}",
                requestId,
                plan.nodeId,
                plan.host,
                plan.port,
                t
            )
            state.recordCoordinatorRequestEvent(
                requestId = requestId,
                batchId = request.batchId,
                chunkIdx = request.chunkIdx,
                stageId = plan.stageId,
                nodeId = plan.nodeId,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = message,
                terminal = false
            )
            val response = failureResponse(request, message)
            recordMetric(
                request,
                response,
                submittedAtEpochMs,
                completedAtMs,
                elapsedMsSince(submittedAtNs),
                TokenPredictionMetrics(0, 0)
            )
            response
        }
    }

    suspend fun submitStageRequest(
        stageId: Int,
        request: Sid.ForwardChunkRequest,
        source: String
    ): Sid.ForwardChunkResponse {
        if (request.requestId.isBlank()) {
            return failureResponse(request, "request_id must not be blank for coordinator stage submission.")
        }

        val requestId = request.requestId
        val localOnlyRequest = request.toBuilder()
            .setChunkIdx(stageId)
            .setStopAfterLocalStage(true)
            .build()
        val plan = state.planStageRequestSubmission(stageId)
        val submittedAtEpochMs = System.currentTimeMillis()
        val submittedAtNs = System.nanoTime()
        state.storeRequestPayload(requestId, localOnlyRequest.toByteArray(), submittedAtEpochMs)

        if (!plan.accepted || plan.host == null || plan.port == null) {
            val message = "Coordinator rejected stage request $requestId from $source: ${plan.message}"
            logger.warn(message)
            state.recordCoordinatorRequestEvent(
                requestId = requestId,
                batchId = localOnlyRequest.batchId,
                chunkIdx = localOnlyRequest.chunkIdx,
                stageId = plan.stageId,
                nodeId = plan.nodeId,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = message,
                terminal = false
            )
            val completedAtMs = System.currentTimeMillis()
            val response = failureResponse(localOnlyRequest, plan.message)
            recordMetric(
                localOnlyRequest,
                response,
                submittedAtEpochMs,
                completedAtMs,
                elapsedMsSince(submittedAtNs),
                TokenPredictionMetrics(0, 0)
            )
            return response
        }

        state.recordCoordinatorRequestEvent(
            requestId = requestId,
            batchId = localOnlyRequest.batchId,
            chunkIdx = localOnlyRequest.chunkIdx,
            stageId = plan.stageId,
            nodeId = plan.nodeId,
            eventType = Sid.RequestEventType.REQUEST_RECEIVED,
            success = true,
            message = "Coordinator accepted local-only stage request from $source and dispatched it to stage ${plan.stageId} node ${plan.nodeId}; evalOnly=${localOnlyRequest.evalOnly}",
            terminal = false
        )

        return try {
            logger.info(
                "Submitting local-only requestId={} batchId={} stage={} evalOnly={} node={} at {}:{} from {}",
                requestId,
                localOnlyRequest.batchId,
                stageId,
                localOnlyRequest.evalOnly,
                plan.nodeId,
                plan.host,
                plan.port,
                source
            )
            val response = ProtoHttpForwardClient.forwardChunk(
                host = plan.host,
                port = plan.port,
                request = localOnlyRequest
            )
            val completedAtMs = System.currentTimeMillis()
            val metrics = if (response.success && response.terminal) {
                safeTokenMetrics(localOnlyRequest, response)
            } else {
                TokenPredictionMetrics(correct = 0, count = 0)
            }
            recordMetric(localOnlyRequest, response, submittedAtEpochMs, completedAtMs, elapsedMsSince(submittedAtNs), metrics)
            response
        } catch (t: Throwable) {
            val completedAtMs = System.currentTimeMillis()
            val message = "Coordinator dispatch to stage $stageId failed: ${t.message}"
            logger.error(
                "Stage dispatch failed for requestId={} stage={} node={} host={}:{}",
                requestId,
                stageId,
                plan.nodeId,
                plan.host,
                plan.port,
                t
            )
            state.recordCoordinatorRequestEvent(
                requestId = requestId,
                batchId = localOnlyRequest.batchId,
                chunkIdx = localOnlyRequest.chunkIdx,
                stageId = plan.stageId,
                nodeId = plan.nodeId,
                eventType = Sid.RequestEventType.FAILED,
                success = false,
                message = message,
                terminal = false
            )
            val response = failureResponse(localOnlyRequest, message)
            recordMetric(
                localOnlyRequest,
                response,
                submittedAtEpochMs,
                completedAtMs,
                elapsedMsSince(submittedAtNs),
                TokenPredictionMetrics(0, 0)
            )
            response
        }
    }

    suspend fun retryRequest(requestId: String): Sid.ForwardChunkResponse {
        val payload = state.loadRequestPayload(requestId)
            ?: return Sid.ForwardChunkResponse.newBuilder()
                .setSuccess(false)
                .setMessage("No stored payload found for request $requestId")
                .setProcessedChunkIdx(-1)
                .setProcessedStageId(-1)
                .setTerminal(false)
                .build()

        val request = Sid.ForwardChunkRequest.parseFrom(payload.payloadProto)
        return if (request.stopAfterLocalStage) {
            submitStageRequest(
                stageId = request.chunkIdx,
                request = request,
                source = "admin-stage-retry"
            )
        } else {
            submitRequest(
                request = request,
                source = "admin-retry"
            )
        }
    }

    private fun failureResponse(
        request: Sid.ForwardChunkRequest,
        message: String
    ): Sid.ForwardChunkResponse {
        return Sid.ForwardChunkResponse.newBuilder()
            .setSuccess(false)
            .setMessage(message)
            .setLocalLoss(0f)
            .setOutputHiddenStates(emptyTensorLike(request.hiddenStates))
            .setOutputShiftLogP(emptyTensorLike(request.shiftLogPPrev))
            .setProcessedChunkIdx(request.chunkIdx)
            .setProcessedStageId(-1)
            .setTerminal(false)
            .build()
    }

    private fun emptyTensorLike(reference: Sid.TensorData): Sid.TensorData {
        return Sid.TensorData.newBuilder()
            .setData(ByteString.EMPTY)
            .addAllShape(reference.shapeList)
            .setDataType(reference.dataType)
            .build()
    }

    private fun recordMetric(
        request: Sid.ForwardChunkRequest,
        response: Sid.ForwardChunkResponse,
        submittedAtEpochMs: Long,
        completedAtEpochMs: Long,
        elapsedMs: Long,
        tokenMetrics: TokenPredictionMetrics
    ) {
        state.recordRequestMetric(
            PersistedRequestMetric(
                runId = inferRunId(request.requestId),
                requestId = request.requestId,
                requestIndex = inferRequestIndex(request.requestId),
                batchId = request.batchId,
                evalOnly = request.evalOnly,
                submittedEpochMs = submittedAtEpochMs,
                completedEpochMs = completedAtEpochMs,
                elapsedMs = elapsedMs,
                success = response.success,
                terminal = response.terminal,
                processedStageId = response.processedStageId,
                processedChunkIdx = response.processedChunkIdx,
                outputHiddenBytes = response.outputHiddenStates.data.size(),
                outputShiftLogPBytes = response.outputShiftLogP.data.size(),
                localLoss = response.localLoss.takeIf { response.success && response.terminal },
                tokenCorrect = tokenMetrics.correct,
                tokenCount = tokenMetrics.count,
                message = response.message
            )
        )
    }

    private fun elapsedMsSince(startedAtNs: Long): Long {
        return ((System.nanoTime() - startedAtNs) / 1_000_000L).coerceAtLeast(0L)
    }

    private fun safeTokenMetrics(
        request: Sid.ForwardChunkRequest,
        response: Sid.ForwardChunkResponse
    ): TokenPredictionMetrics {
        return try {
            computeShiftedTokenPredictionMetrics(response.outputShiftLogP, request.labels)
        } catch (t: Throwable) {
            logger.warn("Failed to compute token metrics for requestId={}: {}", request.requestId, t.message)
            TokenPredictionMetrics(correct = 0, count = 0)
        }
    }
}
