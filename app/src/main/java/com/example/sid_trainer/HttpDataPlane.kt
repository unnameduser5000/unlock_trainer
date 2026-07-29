package com.example.sid_trainer

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import sid.Sid
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.FileNotFoundException
import java.io.InputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.net.URL
import java.nio.charset.StandardCharsets

private const val FORWARD_CHUNK_PATH = "/forwardChunk"
private const val HEADER_CONTENT_LENGTH = "content-length"
private const val HEADER_CONTENT_TYPE = "content-type"
private const val CONTENT_TYPE_PROTOBUF = "application/x-protobuf"
private const val HEADER_CLIENT_SEND_EPOCH_MS = "x-sid-client-send-epoch-ms"
private const val HEADER_SERVER_REQUEST_RECEIVED_EPOCH_MS = "X-Sid-Server-Request-Received-Epoch-Ms"
private const val HEADER_SERVER_REQUEST_READ_MS = "X-Sid-Server-Request-Read-Ms"
private const val HEADER_SERVER_REQUEST_PARSE_MS = "X-Sid-Server-Request-Parse-Ms"
private const val HEADER_SERVER_RESPONSE_SERIALIZE_MS = "X-Sid-Server-Response-Serialize-Ms"
private const val HEADER_SERVER_HANDLER_MS = "X-Sid-Server-Handler-Ms"
private const val HEADER_SERVER_RESPONSE_READY_EPOCH_MS = "X-Sid-Server-Response-Ready-Epoch-Ms"

data class ForwardChunkTransportMetrics(
    val requestSerializeMs: Long = 0,
    val requestWriteMs: Long = 0,
    val responseWaitMs: Long = 0,
    val responseReadMs: Long = 0,
    val responseParseMs: Long = 0,
    val requestBytes: Int = 0,
    val responseBytes: Int = 0,
    val clientSendEpochMs: Long = 0,
    val serverRequestReceivedEpochMs: Long = 0,
    val serverRequestReadMs: Long = 0,
    val serverRequestParseMs: Long = 0,
    val serverResponseSerializeMs: Long = 0,
    val serverHandlerMs: Long = 0,
    val serverResponseReadyEpochMs: Long = 0,
    val clientResponseReceivedEpochMs: Long = 0
) {
    fun describeForLog(): String =
        "rpcRequestSerializeMs=$requestSerializeMs rpcRequestWriteMs=$requestWriteMs " +
            "rpcResponseWaitMs=$responseWaitMs rpcResponseReadMs=$responseReadMs " +
            "rpcResponseParseMs=$responseParseMs rpcRequestBytes=$requestBytes " +
            "rpcResponseBytes=$responseBytes rpcClientSendEpochMs=$clientSendEpochMs " +
            "rpcServerRequestReceivedEpochMs=$serverRequestReceivedEpochMs " +
            "rpcServerRequestReadMs=$serverRequestReadMs rpcServerRequestParseMs=$serverRequestParseMs " +
            "rpcServerResponseSerializeMs=$serverResponseSerializeMs rpcServerHandlerMs=$serverHandlerMs " +
            "rpcServerResponseReadyEpochMs=$serverResponseReadyEpochMs " +
            "rpcClientResponseReceivedEpochMs=$clientResponseReceivedEpochMs"
}

data class HttpForwardChunkResult(
    val response: Sid.ForwardChunkResponse,
    val transport: ForwardChunkTransportMetrics,
    val deliveryStatus: ForwardDeliveryStatus
)

internal class ForwardConnectFailure(
    host: String,
    port: Int,
    cause: IOException
) : IOException("Failed to connect to $host:$port", cause)

class HttpForwardChunkServer(
    private val bindPort: Int,
    private val onChunkReceived: suspend (Sid.ForwardChunkRequest) -> Sid.ForwardChunkResponse
) {
    private val exceptionHandler = CoroutineExceptionHandler { _, t ->
        Log.e("HttpDataPlane", "Unhandled data-plane coroutine failure", t)
    }
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + exceptionHandler)
    @Volatile
    private var serverSocket: ServerSocket? = null
    @Volatile
    private var acceptJob: Job? = null
    @Volatile
    private var activePort: Int = -1

    fun start(): Int {
        if (acceptJob?.isActive == true) {
            return activePort
        }

        val candidatePorts = buildList {
            add(bindPort)
            if (bindPort != 26052) add(26052)
            if (bindPort != 26062) add(26062)
            if (bindPort != 31052) add(31052)
            if (bindPort != 0) add(0)
        }.distinct()

        var socket: ServerSocket? = null
        var lastError: Throwable? = null
        for (candidatePort in candidatePorts) {
            val candidate = ServerSocket()
            candidate.reuseAddress = true
            try {
                candidate.bind(InetSocketAddress(candidatePort))
                socket = candidate
                break
            } catch (t: Throwable) {
                lastError = t
                Log.w("HttpDataPlane", "Failed to bind local data port $candidatePort", t)
                candidate.close()
            }
        }

        val boundSocket = socket ?: throw IllegalStateException(
            "Could not bind any local data port. Last error: ${lastError?.message}",
            lastError
        )
        activePort = boundSocket.localPort
        serverSocket = boundSocket
        Log.i("HttpDataPlane", "Listening for shard traffic on 0.0.0.0:$activePort")

        acceptJob = scope.launch {
            while (true) {
                try {
                    ensureActive()
                    val client = boundSocket.accept()
                    scope.launch {
                        try {
                            handleClient(client)
                        } catch (cancelled: CancellationException) {
                            throw cancelled
                        } catch (t: Throwable) {
                            Log.e("HttpDataPlane", "Unhandled shard client failure", t)
                        }
                    }
                } catch (closed: SocketException) {
                    if (boundSocket.isClosed) {
                        return@launch
                    }
                    Log.w("HttpDataPlane", "Shard accept failed; retrying", closed)
                    delay(1_000)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (t: Throwable) {
                    Log.e("HttpDataPlane", "Shard accept loop failed; retrying", t)
                    delay(1_000)
                }
            }
        }
        return activePort
    }

    fun stop() {
        acceptJob?.cancel()
        acceptJob = null
        serverSocket?.close()
        serverSocket = null
        activePort = -1
        scope.cancel()
    }

    private suspend fun handleClient(socket: Socket) {
        try {
            socket.use { client ->
                client.soTimeout = 120_000
                val input = BufferedInputStream(client.getInputStream())
                val output = BufferedOutputStream(client.getOutputStream())

                try {
                    val requestLine = readAsciiLine(input)
                    if (!requestLine.startsWith("POST $FORWARD_CHUNK_PATH ")) {
                        writeErrorResponse(output, 404, "Unsupported path")
                        return
                    }

                    val headers = readHeaders(input)
                    val contentLength = headers[HEADER_CONTENT_LENGTH]?.toIntOrNull()
                    if (contentLength == null || contentLength < 0) {
                        writeErrorResponse(output, 411, "Missing content length")
                        return
                    }

                    val requestReceivedEpochMs = System.currentTimeMillis()
                    val requestReadStartedAtNs = System.nanoTime()
                    val requestBytes = readFully(input, contentLength)
                    val requestReadMs = elapsedMsSince(requestReadStartedAtNs)
                    val requestParseStartedAtNs = System.nanoTime()
                    val request = Sid.ForwardChunkRequest.parseFrom(requestBytes)
                    val requestParseMs = elapsedMsSince(requestParseStartedAtNs)
                    val callbackStartedAtNs = System.nanoTime()
                    val response = onChunkReceived(request)
                    val callbackMs = elapsedMsSince(callbackStartedAtNs)
                    val serializeStartedAtNs = System.nanoTime()
                    val responseBytes = response.toByteArray()
                    val serializeMs = elapsedMsSince(serializeStartedAtNs)
                    val responseReadyEpochMs = System.currentTimeMillis()
                    Log.i(
                        "HttpDataPlane",
                        "Response ready requestId=${request.requestId} callbackMs=$callbackMs " +
                            "serializeMs=$serializeMs bytes=${responseBytes.size}"
                    )
                    val writeStartedAtNs = System.nanoTime()
                    writeSuccessResponse(
                        output = output,
                        payload = responseBytes,
                        extraHeaders = mapOf(
                            HEADER_SERVER_REQUEST_RECEIVED_EPOCH_MS to requestReceivedEpochMs.toString(),
                            HEADER_SERVER_REQUEST_READ_MS to requestReadMs.toString(),
                            HEADER_SERVER_REQUEST_PARSE_MS to requestParseMs.toString(),
                            HEADER_SERVER_RESPONSE_SERIALIZE_MS to serializeMs.toString(),
                            HEADER_SERVER_HANDLER_MS to callbackMs.toString(),
                            HEADER_SERVER_RESPONSE_READY_EPOCH_MS to responseReadyEpochMs.toString()
                        )
                    )
                    output.flush()
                    Log.i(
                        "HttpDataPlane",
                        "Response flushed requestId=${request.requestId} writeMs=${elapsedMsSince(writeStartedAtNs)} " +
                            "bytes=${responseBytes.size}"
                    )
                } catch (closed: SocketException) {
                    Log.w("HttpDataPlane", "Shard client disconnected: ${closed.message}")
                    return
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (t: Throwable) {
                    Log.e("HttpDataPlane", "Failed to handle incoming shard request", t)
                    writeErrorResponseQuietly(output, 500, t.message ?: "Internal server error")
                } finally {
                    flushQuietly(output)
                }
            }
        } catch (closed: SocketException) {
            Log.w("HttpDataPlane", "Shard socket closed before response completed: ${closed.message}")
        }
    }

    private fun readHeaders(input: InputStream): Map<String, String> {
        val headers = linkedMapOf<String, String>()
        while (true) {
            val line = readAsciiLine(input)
            if (line.isEmpty()) {
                return headers
            }
            val separator = line.indexOf(':')
            if (separator <= 0) {
                continue
            }
            val key = line.substring(0, separator).trim().lowercase()
            val value = line.substring(separator + 1).trim()
            headers[key] = value
        }
    }

    private fun writeSuccessResponse(
        output: BufferedOutputStream,
        payload: ByteArray,
        extraHeaders: Map<String, String> = emptyMap()
    ) {
        val header = buildString {
            append("HTTP/1.1 200 OK\r\n")
            append("Content-Type: $CONTENT_TYPE_PROTOBUF\r\n")
            append("Content-Length: ${payload.size}\r\n")
            extraHeaders.forEach { (name, value) -> append("$name: $value\r\n") }
            append("Connection: close\r\n")
            append("\r\n")
        }.toByteArray(StandardCharsets.US_ASCII)
        output.write(header)
        output.write(payload)
    }

    private fun writeErrorResponse(output: BufferedOutputStream, status: Int, message: String) {
        val body = message.toByteArray(StandardCharsets.UTF_8)
        val header = buildString {
            append("HTTP/1.1 $status Error\r\n")
            append("Content-Type: text/plain; charset=utf-8\r\n")
            append("Content-Length: ${body.size}\r\n")
            append("Connection: close\r\n")
            append("\r\n")
        }.toByteArray(StandardCharsets.US_ASCII)
        output.write(header)
        output.write(body)
    }

    private fun writeErrorResponseQuietly(output: BufferedOutputStream, status: Int, message: String) {
        try {
            writeErrorResponse(output, status, message)
        } catch (closed: SocketException) {
            Log.w("HttpDataPlane", "Could not write error response; shard client disconnected: ${closed.message}")
        } catch (t: Throwable) {
            Log.e("HttpDataPlane", "Could not write error response", t)
        }
    }

    private fun flushQuietly(output: BufferedOutputStream) {
        try {
            output.flush()
        } catch (closed: SocketException) {
            Log.w("HttpDataPlane", "Could not flush shard response; client disconnected: ${closed.message}")
        } catch (t: Throwable) {
            Log.e("HttpDataPlane", "Could not flush shard response", t)
        }
    }

    private fun elapsedMsSince(startedAtNs: Long): Long =
        (System.nanoTime() - startedAtNs) / 1_000_000L

    private fun readAsciiLine(input: InputStream): String {
        val bytes = ByteArrayOutputStream()
        while (true) {
            val next = input.read()
            if (next < 0) {
                throw EOFException("Unexpected EOF while reading HTTP line")
            }
            if (next == '\n'.code) {
                break
            }
            if (next != '\r'.code) {
                bytes.write(next)
            }
        }
        return bytes.toString(StandardCharsets.US_ASCII.name())
    }

    private fun readFully(input: InputStream, contentLength: Int): ByteArray {
        val buffer = ByteArray(contentLength)
        var offset = 0
        while (offset < contentLength) {
            val read = input.read(buffer, offset, contentLength - offset)
            if (read < 0) {
                throw EOFException("Unexpected EOF while reading HTTP body")
            }
            offset += read
        }
        return buffer
    }
}

object HttpForwardChunkClient {
    suspend fun forwardChunk(
        host: String,
        port: Int,
        request: Sid.ForwardChunkRequest
    ): HttpForwardChunkResult = kotlinx.coroutines.withContext(Dispatchers.IO) {
        val serializeStartedAtNs = System.nanoTime()
        val requestBytes = request.toByteArray()
        val requestSerializeMs = (System.nanoTime() - serializeStartedAtNs) / 1_000_000L
        val clientSendEpochMs = System.currentTimeMillis()
        val connection = (URL("http://$host:$port$FORWARD_CHUNK_PATH").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15_000
            readTimeout = 300_000
            doInput = true
            doOutput = true
            useCaches = false
            setRequestProperty(HEADER_CONTENT_TYPE, CONTENT_TYPE_PROTOBUF)
            setRequestProperty(HEADER_CONTENT_LENGTH, requestBytes.size.toString())
            setRequestProperty(HEADER_CLIENT_SEND_EPOCH_MS, clientSendEpochMs.toString())
            setFixedLengthStreamingMode(requestBytes.size)
        }

        try {
            try {
                connection.connect()
            } catch (failure: IOException) {
                throw ForwardConnectFailure(host, port, failure)
            }

            val writeStartedAtNs = System.nanoTime()
            connection.outputStream.use { output ->
                output.write(requestBytes)
                output.flush()
            }
            val requestWriteMs = (System.nanoTime() - writeStartedAtNs) / 1_000_000L

            val responseWaitStartedAtNs = System.nanoTime()
            val responseCode = connection.responseCode
            val responseWaitMs = (System.nanoTime() - responseWaitStartedAtNs) / 1_000_000L
            val clientResponseReceivedEpochMs = System.currentTimeMillis()
            val bodyStartedAtNs = System.nanoTime()
            val responseBytes = (if (responseCode in 200..299) {
                connection.inputStream
            } else {
                connection.errorStream ?: throw FileNotFoundException("HTTP $responseCode")
            }).use { stream ->
                stream.readBytes()
            }
            val responseReadMs = (System.nanoTime() - bodyStartedAtNs) / 1_000_000L

            if (responseCode !in 200..299) {
                throw IllegalStateException(
                    "HTTP $responseCode from downstream: ${responseBytes.toString(StandardCharsets.UTF_8)}"
                )
            }

            val responseParseStartedAtNs = System.nanoTime()
            val response = Sid.ForwardChunkResponse.parseFrom(responseBytes)
            val responseParseMs = (System.nanoTime() - responseParseStartedAtNs) / 1_000_000L
            val transport = ForwardChunkTransportMetrics(
                requestSerializeMs = requestSerializeMs,
                requestWriteMs = requestWriteMs,
                responseWaitMs = responseWaitMs,
                responseReadMs = responseReadMs,
                responseParseMs = responseParseMs,
                requestBytes = requestBytes.size,
                responseBytes = responseBytes.size,
                clientSendEpochMs = clientSendEpochMs,
                serverRequestReceivedEpochMs = connection.getHeaderField(HEADER_SERVER_REQUEST_RECEIVED_EPOCH_MS)?.toLongOrNull() ?: 0,
                serverRequestReadMs = connection.getHeaderField(HEADER_SERVER_REQUEST_READ_MS)?.toLongOrNull() ?: 0,
                serverRequestParseMs = connection.getHeaderField(HEADER_SERVER_REQUEST_PARSE_MS)?.toLongOrNull() ?: 0,
                serverResponseSerializeMs = connection.getHeaderField(HEADER_SERVER_RESPONSE_SERIALIZE_MS)?.toLongOrNull() ?: 0,
                serverHandlerMs = connection.getHeaderField(HEADER_SERVER_HANDLER_MS)?.toLongOrNull() ?: 0,
                serverResponseReadyEpochMs = connection.getHeaderField(HEADER_SERVER_RESPONSE_READY_EPOCH_MS)?.toLongOrNull() ?: 0,
                clientResponseReceivedEpochMs = clientResponseReceivedEpochMs
            )
            Log.i(
                "HttpDataPlane",
                "Downstream response requestId=${request.requestId} status=$responseCode ${transport.describeForLog()}"
            )
            HttpForwardChunkResult(
                response = response,
                transport = transport,
                deliveryStatus = ForwardDeliveryStatus.RESPONSE_RECEIVED
            )
        } finally {
            connection.disconnect()
        }
    }
}
