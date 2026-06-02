package com.example.sid_coordinator

import org.slf4j.LoggerFactory
import sid.CoordinatingServiceGrpcKt
import sid.Sid

class CoordinatorService(
    private val state: CoordinatorState,
    private val requestOrchestrator: CoordinatorRequestOrchestrator
) : CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineImplBase() {
    private val logger = LoggerFactory.getLogger(CoordinatorService::class.java)

    override suspend fun registerNode(request: Sid.NodeInfo): Sid.RegistrationResponse {
        logger.info(
            "RegisterNode deviceId={} runtime={}:{} type={} compute={} memoryGb={}",
            request.deviceId,
            request.ipAddress,
            request.grpcPort,
            request.nodeType,
            request.computeCapacity,
            request.memoryGb
        )
        return state.registerNode(request)
    }

    override suspend fun heartbeat(request: Sid.HeartbeatRequest): Sid.HeartbeatResponse {
        logger.debug(
            "Heartbeat deviceId={} nodeId={} battery={} active={}",
            request.deviceId,
            request.nodeId,
            request.batteryLevel,
            request.isActive
        )
        return state.heartbeat(request)
    }

    override suspend fun reportRequestEvent(request: Sid.RequestEvent): Sid.RequestEventAck {
        logger.debug(
            "RequestEvent requestId={} batchId={} chunkIdx={} stageId={} nodeId={} eventType={} success={}",
            request.requestId,
            request.batchId,
            request.chunkIdx,
            request.stageId,
            request.nodeId,
            request.eventType,
            request.success
        )
        return state.reportRequestEvent(request)
    }

    override suspend fun submitRequest(request: Sid.ForwardChunkRequest): Sid.ForwardChunkResponse {
        logger.info(
            "SubmitRequest requestId={} batchId={} chunkIdx={}",
            request.requestId,
            request.batchId,
            request.chunkIdx
        )
        return requestOrchestrator.submitRequest(request, source = "grpc-submit")
    }

    override suspend fun submitStageRequest(request: Sid.StageForwardChunkRequest): Sid.ForwardChunkResponse {
        logger.info(
            "SubmitStageRequest requestId={} batchId={} stageId={} chunkIdx={}",
            request.request.requestId,
            request.request.batchId,
            request.stageId,
            request.request.chunkIdx
        )
        return requestOrchestrator.submitStageRequest(
            stageId = request.stageId,
            request = request.request,
            source = "grpc-stage-submit"
        )
    }
}
