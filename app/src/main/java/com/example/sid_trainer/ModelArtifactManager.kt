package com.example.sid_trainer

import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.CancellationException
import kotlin.math.max

data class PreparedModelArtifact(
    val absolutePath: String,
    val fileBytes: Long,
    val cacheHit: Boolean
)

object ModelArtifactManager {
    suspend fun ensureModel(
        filesDir: File,
        registration: WorkerRegistration,
        onLog: (String) -> Unit = {}
    ): PreparedModelArtifact = withContext(Dispatchers.IO) {
        require(registration.modelDownloadUrl.isNotBlank()) {
            "modelDownloadUrl must not be blank"
        }

        val shardDir = File(filesDir, "shards").apply { mkdirs() }
        val finalFile = File(shardDir, "${registration.modelShardId}.pte")
        val tempFile = File(shardDir, "${registration.modelShardId}.download")
        val expectedSha = registration.modelSha256.trim().lowercase()

        if (finalFile.exists() && matchesExpected(finalFile, expectedSha, registration.modelBytes)) {
            onLog("Shard already cached: ${finalFile.absolutePath}")
            return@withContext PreparedModelArtifact(
                absolutePath = finalFile.absolutePath,
                fileBytes = finalFile.length(),
                cacheHit = true
            )
        }

        onLog(
            "Downloading ${registration.modelShardId} (${humanReadableBytes(registration.modelBytes)})"
        )
        try {
            downloadToFile(
                url = registration.modelDownloadUrl,
                destination = tempFile,
                onLog = onLog
            )
        } catch (cancelled: CancellationException) {
            tempFile.delete()
            onLog("Shard download cancelled for ${registration.modelShardId}")
            throw cancelled
        }

        if (!matchesExpected(tempFile, expectedSha, registration.modelBytes)) {
            tempFile.delete()
            error("Downloaded shard failed verification for ${registration.modelShardId}")
        }

        if (finalFile.exists() && !finalFile.delete()) {
            error("Could not replace existing shard file ${finalFile.absolutePath}")
        }
        if (!tempFile.renameTo(finalFile)) {
            tempFile.delete()
            error("Could not move downloaded shard into place for ${registration.modelShardId}")
        }

        PreparedModelArtifact(
            absolutePath = finalFile.absolutePath,
            fileBytes = finalFile.length(),
            cacheHit = false
        )
    }

    private suspend fun downloadToFile(
        url: String,
        destination: File,
        onLog: (String) -> Unit
    ) {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 30_000
            readTimeout = 600_000
            doInput = true
        }
        try {
            currentCoroutineContext().ensureActive()
            connection.connect()
            if (connection.responseCode !in 200..299) {
                error("Artifact download returned HTTP ${connection.responseCode}")
            }

            val totalBytes = max(connection.contentLengthLong, 0L)
            destination.parentFile?.mkdirs()
            connection.inputStream.use { input ->
                FileOutputStream(destination).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var downloadedBytes = 0L
                    var nextProgressBytes = 64L * 1024L * 1024L
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) {
                            break
                        }
                        if (read == 0) {
                            continue
                        }
                        output.write(buffer, 0, read)
                        downloadedBytes += read
                        if (downloadedBytes >= nextProgressBytes) {
                            val totalPart = if (totalBytes > 0) {
                                "/${humanReadableBytes(totalBytes)}"
                            } else {
                                ""
                            }
                            onLog("Downloaded ${humanReadableBytes(downloadedBytes)}$totalPart")
                            nextProgressBytes += 64L * 1024L * 1024L
                        }
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun matchesExpected(file: File, expectedSha: String, expectedBytes: Long): Boolean {
        if (!file.exists()) {
            return false
        }
        if (expectedBytes > 0 && file.length() != expectedBytes) {
            return false
        }
        if (expectedSha.isBlank()) {
            return true
        }
        return sha256Hex(file) == expectedSha
    }

    private fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) {
                    break
                }
                if (read > 0) {
                    digest.update(buffer, 0, read)
                }
            }
        }
        return digest.digest().joinToString("") { byte ->
            "%02x".format(byte)
        }
    }

    private fun humanReadableBytes(bytes: Long): String {
        if (bytes <= 0) {
            return "unknown size"
        }
        val kib = 1024L
        val mib = kib * 1024L
        val gib = mib * 1024L
        return when {
            bytes >= gib -> String.format("%.2f GiB", bytes.toDouble() / gib.toDouble())
            bytes >= mib -> String.format("%.2f MiB", bytes.toDouble() / mib.toDouble())
            bytes >= kib -> String.format("%.2f KiB", bytes.toDouble() / kib.toDouble())
            else -> "$bytes B"
        }
    }
}
