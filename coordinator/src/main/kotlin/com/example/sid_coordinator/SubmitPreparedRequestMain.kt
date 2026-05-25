package com.example.sid_coordinator

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.protobuf.ByteString
import io.grpc.ManagedChannelBuilder
import sid.CoordinatingServiceGrpcKt
import sid.Sid
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

private data class PreparedTensorRecord(
    val path: String,
    val dtype: String,
    val shape: List<Int>
)

private data class PreparedRequestRecord(
    val request_id: String,
    val batch_id: Int = 1,
    val chunk_idx: Int = 0,
    val tensors: Map<String, PreparedTensorRecord>
)

fun main(args: Array<String>) {
    val host = args.getOrNull(0) ?: "127.0.0.1"
    val port = args.getOrNull(1)?.toIntOrNull() ?: 50051
    val manifestPath = Paths.get(args.getOrNull(2) ?: "data/sft_requests/requests.jsonl")
    val recordIndex = args.getOrNull(3)?.toIntOrNull() ?: 0
    val requestIdOverride = args.getOrNull(4).orEmpty()
    val maxMessageSizeBytes = 50 * 1024 * 1024

    val record = readManifestRecord(manifestPath, recordIndex)
    val manifestDir = manifestPath.toAbsolutePath().normalize().parent
    val requestId = requestIdOverride.ifBlank { record.request_id }
    val request = Sid.ForwardChunkRequest.newBuilder()
        .setRequestId(requestId)
        .setBatchId(record.batch_id)
        .setChunkIdx(record.chunk_idx)
        .setHiddenStates(record.requiredTensor("hidden_states", manifestDir))
        .setAttentionMask(record.requiredTensor("attention_mask", manifestDir))
        .setPositionIds(record.requiredTensor("position_ids", manifestDir))
        .setLabels(record.requiredTensor("labels", manifestDir))
        .setShiftLogPPrev(emptyTensor("float32"))
        .build()

    val channel = ManagedChannelBuilder.forAddress(host, port)
        .usePlaintext()
        .maxInboundMessageSize(maxMessageSizeBytes)
        .build()

    try {
        val stub = CoordinatingServiceGrpcKt.CoordinatingServiceCoroutineStub(channel)
        val response = kotlinx.coroutines.runBlocking {
            stub.submitRequest(request)
        }
        println("requestId=$requestId")
        println("manifest=$manifestPath")
        println("recordIndex=$recordIndex")
        println("success=${response.success}")
        println("message=${response.message}")
        println("processedStageId=${response.processedStageId}")
        println("processedChunkIdx=${response.processedChunkIdx}")
        println("terminal=${response.terminal}")
        println("outputHiddenBytes=${response.outputHiddenStates.data.size()}")
    } finally {
        channel.shutdownNow()
    }
}

private fun readManifestRecord(path: Path, index: Int): PreparedRequestRecord {
    require(index >= 0) { "record index must be non-negative" }
    val line = Files.newBufferedReader(path).useLines { lines ->
        lines.drop(index).firstOrNull()
    } ?: error("No record at index $index in $path")
    val type = object : TypeToken<PreparedRequestRecord>() {}.type
    return Gson().fromJson(line, type)
}

private fun PreparedRequestRecord.requiredTensor(
    name: String,
    manifestDir: Path
): Sid.TensorData {
    val record = tensors[name] ?: error("Prepared request is missing tensor '$name'")
    val tensorPath = manifestDir.resolve(record.path).normalize()
    require(Files.exists(tensorPath)) { "Tensor file does not exist: $tensorPath" }
    val bytes = Files.readAllBytes(tensorPath)
    return Sid.TensorData.newBuilder()
        .setData(ByteString.copyFrom(bytes))
        .addAllShape(record.shape)
        .setDataType(record.dtype)
        .build()
}

private fun emptyTensor(dataType: String): Sid.TensorData {
    return Sid.TensorData.newBuilder()
        .setData(ByteString.EMPTY)
        .setDataType(dataType)
        .build()
}
