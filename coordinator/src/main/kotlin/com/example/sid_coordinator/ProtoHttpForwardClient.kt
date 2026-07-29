package com.example.sid_coordinator

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sid.Sid
import java.io.FileNotFoundException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

private const val FORWARD_CHUNK_PATH = "/forwardChunk"
private const val CONTENT_TYPE_PROTOBUF = "application/x-protobuf"

object ProtoHttpForwardClient {
    suspend fun forwardChunk(
        host: String,
        port: Int,
        request: Sid.ForwardChunkRequest
    ): Sid.ForwardChunkResponse = withContext(Dispatchers.IO) {
        val requestBytes = request.toByteArray()
        val connection = (URL("http://$host:$port$FORWARD_CHUNK_PATH").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15_000
            readTimeout = 300_000
            doInput = true
            doOutput = true
            useCaches = false
            setRequestProperty("Content-Type", CONTENT_TYPE_PROTOBUF)
            setFixedLengthStreamingMode(requestBytes.size)
        }

        try {
            connection.outputStream.use { output ->
                output.write(requestBytes)
            }

            val responseCode = connection.responseCode
            val responseBytes = (if (responseCode in 200..299) {
                connection.inputStream
            } else {
                connection.errorStream ?: throw FileNotFoundException("HTTP $responseCode")
            }).use { stream ->
                stream.readBytes()
            }

            if (responseCode !in 200..299) {
                throw IllegalStateException(
                    "HTTP $responseCode from downstream: ${responseBytes.toString(StandardCharsets.UTF_8)}"
                )
            }

            Sid.ForwardChunkResponse.parseFrom(responseBytes)
        } finally {
            connection.disconnect()
        }
    }
}
