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
            return failureResponse(request, plan.message)
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
            ProtoHttpForwardClient.forwardChunk(
                host = plan.host,
                port = plan.port,
                request = request
            )
        } catch (t: Throwable) {
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
            failureResponse(request, message)
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

        return submitRequest(
            request = Sid.ForwardChunkRequest.parseFrom(payload.payloadProto),
            source = "admin-retry"
        )
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
}
