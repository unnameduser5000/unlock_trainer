package com.example.sid_trainer

import sid.Sid
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.UUID

internal enum class DurableBoundaryState(val suffix: String) {
    PREPARED("prepared"),
    READY("ready")
}

internal data class DurableBoundaryEntry(
    val outboxSequence: Long,
    val createdAtEpochMs: Long,
    val request: Sid.ForwardChunkRequest,
    val state: DurableBoundaryState,
    val file: File
)

internal data class DurableBoundaryPrepareResult(
    val entry: DurableBoundaryEntry,
    val duplicate: Boolean,
    val pendingCount: Int
)

internal data class DurableBoundaryAcknowledgementReceipt(
    val outboxSequence: Long,
    val acknowledgedAtEpochMs: Long,
    val requestId: String,
    val batchId: Int,
    val chunkIdx: Int,
    val pipelineWindowSeq: Long,
    val file: File
)

internal sealed interface DurableBoundaryReplayResolution {
    data class Pending(val entry: DurableBoundaryEntry) : DurableBoundaryReplayResolution
    data class Acknowledged(
        val receipt: DurableBoundaryAcknowledgementReceipt
    ) : DurableBoundaryReplayResolution
    data object Missing : DurableBoundaryReplayResolution
}

internal class DurableBoundaryBufferFullException(message: String) : IllegalStateException(message)

/** Phone-local store-and-forward log for detached pipeline boundaries. */
internal class DurableBoundaryOutbox(private val rootDir: File) {
    private var activeStageId: Int? = null
    private var activeDir: File? = null
    private val entriesByRequestId = linkedMapOf<String, DurableBoundaryEntry>()
    private val acknowledgedByRequestId = linkedMapOf<String, DurableBoundaryAcknowledgementReceipt>()
    private var nextSequence = 1L

    @Synchronized
    fun activateStage(stageId: Int) {
        if (activeStageId == stageId && activeDir != null) return
        val stageDir = File(rootDir, "stage-$stageId")
        require(stageDir.exists() || stageDir.mkdirs()) {
            "Could not create durable boundary outbox ${stageDir.absolutePath}"
        }
        require(stageDir.isDirectory) { "Durable boundary outbox is not a directory: ${stageDir.absolutePath}" }

        val files = stageDir.listFiles().orEmpty()
        val loaded = files
            .orEmpty()
            .filter { it.isFile && (it.extension == DurableBoundaryState.PREPARED.suffix || it.extension == DurableBoundaryState.READY.suffix) }
            .map(::readEntry)
            .sortedBy(DurableBoundaryEntry::outboxSequence)
        val acknowledged = files
            .filter { it.isFile && it.extension == ACKNOWLEDGED_SUFFIX }
            .map(::readAcknowledgementReceipt)
            .sortedBy(DurableBoundaryAcknowledgementReceipt::outboxSequence)

        val duplicateRequestIds = loaded.groupBy { it.request.requestId }.filterValues { it.size > 1 }.keys
        require(duplicateRequestIds.isEmpty()) {
            "Durable boundary outbox contains duplicate request IDs: $duplicateRequestIds"
        }
        val duplicateAcknowledgements = acknowledged.groupBy(DurableBoundaryAcknowledgementReceipt::requestId)
            .filterValues { it.size > 1 }
            .keys
        require(duplicateAcknowledgements.isEmpty()) {
            "Durable boundary outbox contains duplicate acknowledgement receipts: $duplicateAcknowledgements"
        }

        entriesByRequestId.clear()
        acknowledgedByRequestId.clear()
        acknowledged.forEach { receipt -> acknowledgedByRequestId[receipt.requestId] = receipt }
        loaded.forEach { entry ->
            if (acknowledgedByRequestId.containsKey(entry.request.requestId)) {
                require(entry.file.delete() || !entry.file.exists()) {
                    "Could not remove boundary superseded by acknowledgement ${entry.file.absolutePath}"
                }
            } else {
                entriesByRequestId[entry.request.requestId] = entry
            }
        }
        nextSequence = maxOf(
            loaded.maxOfOrNull(DurableBoundaryEntry::outboxSequence) ?: 0L,
            acknowledged.maxOfOrNull(DurableBoundaryAcknowledgementReceipt::outboxSequence) ?: 0L
        ) + 1L
        activeStageId = stageId
        activeDir = stageDir
    }

    @Synchronized
    fun prepare(request: Sid.ForwardChunkRequest, capacity: Int): DurableBoundaryPrepareResult {
        require(request.requestId.isNotBlank()) { "Durable boundary request_id must not be blank." }
        require(capacity > 0) { "Replay buffer capacity must be positive." }
        val stageDir = requireNotNull(activeDir) { "Durable boundary outbox is not activated." }

        require(!acknowledgedByRequestId.containsKey(request.requestId)) {
            "Request ${request.requestId} was already acknowledged; committed replay must be short-circuited."
        }

        entriesByRequestId[request.requestId]?.let { existing ->
            require(existing.request.toByteArray().contentEquals(request.toByteArray())) {
                "Request ${request.requestId} was re-used with a different durable boundary payload."
            }
            return DurableBoundaryPrepareResult(existing, duplicate = true, pendingCount = entriesByRequestId.size)
        }

        if (entriesByRequestId.size >= capacity) {
            throw DurableBoundaryBufferFullException(
                "Durable replay buffer is full (${entriesByRequestId.size}/$capacity); " +
                    "backpressure request ${request.requestId}."
            )
        }

        val sequence = nextSequence++
        val createdAtEpochMs = System.currentTimeMillis()
        val target = File(stageDir, fileName(sequence, DurableBoundaryState.PREPARED))
        writeEntryAtomically(target, sequence, createdAtEpochMs, request)
        val entry = DurableBoundaryEntry(
            outboxSequence = sequence,
            createdAtEpochMs = createdAtEpochMs,
            request = request,
            state = DurableBoundaryState.PREPARED,
            file = target
        )
        entriesByRequestId[request.requestId] = entry
        return DurableBoundaryPrepareResult(entry, duplicate = false, pendingCount = entriesByRequestId.size)
    }

    @Synchronized
    fun markReady(requestId: String): DurableBoundaryEntry {
        val existing = requireNotNull(entriesByRequestId[requestId]) {
            "No prepared durable boundary for request $requestId."
        }
        if (existing.state == DurableBoundaryState.READY) return existing

        val target = File(requireNotNull(activeDir), fileName(existing.outboxSequence, DurableBoundaryState.READY))
        require(existing.file.renameTo(target)) {
            "Could not commit durable boundary ${existing.file.absolutePath} to ${target.absolutePath}"
        }
        return existing.copy(state = DurableBoundaryState.READY, file = target).also {
            entriesByRequestId[requestId] = it
        }
    }

    @Synchronized
    fun abortPrepared(requestId: String): Boolean {
        val existing = entriesByRequestId[requestId] ?: return false
        if (existing.state != DurableBoundaryState.PREPARED) return false
        require(existing.file.delete() || !existing.file.exists()) {
            "Could not remove uncommitted durable boundary ${existing.file.absolutePath}"
        }
        entriesByRequestId.remove(requestId)
        return true
    }

    @Synchronized
    fun peekReady(): DurableBoundaryEntry? = entriesByRequestId.values
        .asSequence()
        .filter { it.state == DurableBoundaryState.READY }
        .minByOrNull(DurableBoundaryEntry::outboxSequence)

    @Synchronized
    fun resolveCommittedReplay(sourceRequest: Sid.ForwardChunkRequest): DurableBoundaryReplayResolution {
        val requestId = sourceRequest.requestId
        acknowledgedByRequestId[requestId]?.let { receipt ->
            validateReplayIdentity(
                sourceRequest = sourceRequest,
                batchId = receipt.batchId,
                downstreamChunkIdx = receipt.chunkIdx,
                pipelineWindowSeq = receipt.pipelineWindowSeq
            )
            return DurableBoundaryReplayResolution.Acknowledged(receipt)
        }
        entriesByRequestId[requestId]?.let { entry ->
            validateReplayIdentity(
                sourceRequest = sourceRequest,
                batchId = entry.request.batchId,
                downstreamChunkIdx = entry.request.chunkIdx,
                pipelineWindowSeq = entry.request.pipelineWindowSeq
            )
            return DurableBoundaryReplayResolution.Pending(entry)
        }
        return DurableBoundaryReplayResolution.Missing
    }

    @Synchronized
    fun acknowledge(entry: DurableBoundaryEntry): Int {
        val current = entriesByRequestId[entry.request.requestId]
            ?: return entriesByRequestId.size
        require(current.outboxSequence == entry.outboxSequence) {
            "Outbox sequence changed for request ${entry.request.requestId}."
        }
        require(current.state == DurableBoundaryState.READY) {
            "Cannot acknowledge an uncommitted durable boundary for ${entry.request.requestId}."
        }
        val target = File(requireNotNull(activeDir), acknowledgementFileName(current.outboxSequence))
        val receipt = if (target.exists()) {
            readAcknowledgementReceipt(target).also { existing ->
                require(existing.requestId == current.request.requestId) {
                    "Acknowledgement sequence ${current.outboxSequence} belongs to ${existing.requestId}, " +
                        "not ${current.request.requestId}."
                }
            }
        } else {
            writeAcknowledgementReceiptAtomically(target, current)
        }
        require(current.file.delete() || !current.file.exists()) {
            "Could not delete acknowledged durable boundary ${current.file.absolutePath}"
        }
        entriesByRequestId.remove(entry.request.requestId)
        acknowledgedByRequestId[receipt.requestId] = receipt
        return entriesByRequestId.size
    }

    @Synchronized
    fun pendingCount(): Int = entriesByRequestId.size

    @Synchronized
    fun acknowledgedCount(): Int = acknowledgedByRequestId.size

    private fun writeEntryAtomically(
        target: File,
        sequence: Long,
        createdAtEpochMs: Long,
        request: Sid.ForwardChunkRequest
    ) {
        val tmp = File(requireNotNull(target.parentFile), ".${target.name}.${UUID.randomUUID()}.tmp")
        val payload = request.toByteArray()
        try {
            FileOutputStream(tmp).use { fileOutput ->
                DataOutputStream(BufferedOutputStream(fileOutput)).use { output ->
                    output.writeInt(MAGIC)
                    output.writeInt(VERSION)
                    output.writeLong(sequence)
                    output.writeLong(createdAtEpochMs)
                    output.writeInt(payload.size)
                    output.write(payload)
                    output.flush()
                    fileOutput.fd.sync()
                }
            }
            require(tmp.renameTo(target)) {
                "Could not move durable boundary ${tmp.absolutePath} to ${target.absolutePath}"
            }
        } finally {
            if (tmp.exists()) tmp.delete()
        }
    }

    private fun readEntry(file: File): DurableBoundaryEntry {
        val state = DurableBoundaryState.entries.firstOrNull { it.suffix == file.extension }
            ?: error("Unsupported durable boundary state for ${file.absolutePath}")
        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
            require(input.readInt() == MAGIC) { "Invalid durable boundary magic in ${file.absolutePath}" }
            require(input.readInt() == VERSION) { "Unsupported durable boundary version in ${file.absolutePath}" }
            val sequence = input.readLong()
            val createdAtEpochMs = input.readLong()
            val payloadSize = input.readInt()
            require(payloadSize in 1..MAX_PAYLOAD_BYTES) {
                "Invalid durable boundary payload size $payloadSize in ${file.absolutePath}"
            }
            val payload = ByteArray(payloadSize)
            input.readFully(payload)
            val request = Sid.ForwardChunkRequest.parseFrom(payload)
            require(request.requestId.isNotBlank()) { "Blank request_id in ${file.absolutePath}" }
            return DurableBoundaryEntry(sequence, createdAtEpochMs, request, state, file)
        }
    }

    private fun writeAcknowledgementReceiptAtomically(
        target: File,
        entry: DurableBoundaryEntry
    ): DurableBoundaryAcknowledgementReceipt {
        val tmp = File(requireNotNull(target.parentFile), ".${target.name}.${UUID.randomUUID()}.tmp")
        val acknowledgedAtEpochMs = System.currentTimeMillis()
        try {
            FileOutputStream(tmp).use { fileOutput ->
                DataOutputStream(BufferedOutputStream(fileOutput)).use { output ->
                    output.writeInt(ACKNOWLEDGED_MAGIC)
                    output.writeInt(ACKNOWLEDGED_VERSION)
                    output.writeLong(entry.outboxSequence)
                    output.writeLong(acknowledgedAtEpochMs)
                    output.writeUTF(entry.request.requestId)
                    output.writeInt(entry.request.batchId)
                    output.writeInt(entry.request.chunkIdx)
                    output.writeLong(entry.request.pipelineWindowSeq)
                    output.flush()
                    fileOutput.fd.sync()
                }
            }
            require(tmp.renameTo(target)) {
                "Could not persist boundary acknowledgement ${tmp.absolutePath} to ${target.absolutePath}"
            }
        } finally {
            if (tmp.exists()) tmp.delete()
        }
        return DurableBoundaryAcknowledgementReceipt(
            outboxSequence = entry.outboxSequence,
            acknowledgedAtEpochMs = acknowledgedAtEpochMs,
            requestId = entry.request.requestId,
            batchId = entry.request.batchId,
            chunkIdx = entry.request.chunkIdx,
            pipelineWindowSeq = entry.request.pipelineWindowSeq,
            file = target
        )
    }

    private fun readAcknowledgementReceipt(file: File): DurableBoundaryAcknowledgementReceipt {
        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
            require(input.readInt() == ACKNOWLEDGED_MAGIC) {
                "Invalid boundary acknowledgement magic in ${file.absolutePath}"
            }
            require(input.readInt() == ACKNOWLEDGED_VERSION) {
                "Unsupported boundary acknowledgement version in ${file.absolutePath}"
            }
            val sequence = input.readLong()
            val acknowledgedAtEpochMs = input.readLong()
            val requestId = input.readUTF()
            require(requestId.isNotBlank()) { "Blank request_id in ${file.absolutePath}" }
            return DurableBoundaryAcknowledgementReceipt(
                outboxSequence = sequence,
                acknowledgedAtEpochMs = acknowledgedAtEpochMs,
                requestId = requestId,
                batchId = input.readInt(),
                chunkIdx = input.readInt(),
                pipelineWindowSeq = input.readLong(),
                file = file
            )
        }
    }

    private fun validateReplayIdentity(
        sourceRequest: Sid.ForwardChunkRequest,
        batchId: Int,
        downstreamChunkIdx: Int,
        pipelineWindowSeq: Long
    ) {
        require(batchId == sourceRequest.batchId &&
            downstreamChunkIdx == sourceRequest.chunkIdx + 1 &&
            pipelineWindowSeq == sourceRequest.pipelineWindowSeq
        ) {
            "Request ${sourceRequest.requestId} was re-used with different pipeline identity."
        }
    }

    private fun fileName(sequence: Long, state: DurableBoundaryState): String =
        "${sequence.toString().padStart(20, '0')}.${state.suffix}"

    private fun acknowledgementFileName(sequence: Long): String =
        "${sequence.toString().padStart(20, '0')}.$ACKNOWLEDGED_SUFFIX"

    private companion object {
        const val MAGIC = 0x53494442
        const val VERSION = 1
        const val ACKNOWLEDGED_MAGIC = 0x53494441
        const val ACKNOWLEDGED_VERSION = 1
        const val ACKNOWLEDGED_SUFFIX = "acked"
        const val MAX_PAYLOAD_BYTES = 50 * 1024 * 1024
    }
}
