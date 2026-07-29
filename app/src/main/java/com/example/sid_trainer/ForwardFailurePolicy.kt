package com.example.sid_trainer

import sid.Sid

internal fun acceptsIncompleteForwardRoute(request: Sid.ForwardChunkRequest): Boolean =
    request.durablePipeline || request.dropOnForwardFailure

enum class ForwardDeliveryStatus {
    RESPONSE_RECEIVED,
    DEFINITELY_NOT_DELIVERED,
    DELIVERY_UNKNOWN
}

internal fun shouldDropForwardBoundary(
    request: Sid.ForwardChunkRequest,
    deliveryStatus: ForwardDeliveryStatus
): Boolean = request.dropOnForwardFailure &&
    !request.durablePipeline &&
    deliveryStatus == ForwardDeliveryStatus.DEFINITELY_NOT_DELIVERED
