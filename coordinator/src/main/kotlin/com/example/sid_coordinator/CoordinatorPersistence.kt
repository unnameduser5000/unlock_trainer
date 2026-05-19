package com.example.sid_coordinator

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.sql.Connection
import java.sql.DriverManager
import java.sql.ResultSet
import java.time.Instant

data class PersistedCoordinatorSnapshot(
    val nodes: List<RegisteredNode>,
    val routingEpoch: Long,
    val nextNodeId: Int,
    val drainedStageIds: Set<Int>
)

class CoordinatorPersistence(
    dbPath: String
) {
    private val jdbcUrl: String

    init {
        Class.forName("org.sqlite.JDBC")
        val path = Paths.get(dbPath).toAbsolutePath().normalize()
        path.parent?.let(Files::createDirectories)
        jdbcUrl = "jdbc:sqlite:${path.toString().replace('\\', '/')}"
        connection().use(::initializeSchema)
    }

    fun loadSnapshot(): PersistedCoordinatorSnapshot {
        connection().use { conn ->
            val routingEpoch = readMetaLong(conn, "routing_epoch") ?: 1L
            val nextNodeId = readMetaLong(conn, "next_node_id")?.toInt() ?: 1
            val nodes = conn.prepareStatement(
                """
                SELECT
                  node_id,
                  stage_id,
                  device_id,
                  ip_address,
                  grpc_port,
                  compute_capacity,
                  memory_gb,
                  registered_at_epoch_ms,
                  last_heartbeat_at_epoch_ms,
                  is_active
                FROM nodes
                ORDER BY stage_id
                """.trimIndent()
            ).use { stmt ->
                stmt.executeQuery().use { rs ->
                    buildList {
                        while (rs.next()) {
                            add(rs.toRegisteredNode())
                        }
                    }
                }
            }
            val drainedStageIds = conn.prepareStatement(
                """
                SELECT stage_id
                FROM stage_overrides
                WHERE drained = 1
                ORDER BY stage_id
                """.trimIndent()
            ).use { stmt ->
                stmt.executeQuery().use { rs ->
                    buildSet {
                        while (rs.next()) {
                            add(rs.getInt("stage_id"))
                        }
                    }
                }
            }
            return PersistedCoordinatorSnapshot(
                nodes = nodes,
                routingEpoch = routingEpoch,
                nextNodeId = maxOf(nextNodeId, (nodes.maxOfOrNull { it.nodeId } ?: 0) + 1),
                drainedStageIds = drainedStageIds
            )
        }
    }

    fun markAllNodesInactive() {
        connection().use { conn ->
            conn.prepareStatement("UPDATE nodes SET is_active = 0").use { stmt ->
                stmt.executeUpdate()
            }
        }
    }

    fun upsertNode(node: RegisteredNode, routingEpoch: Long, nextNodeId: Int) {
        connection().use { conn ->
            conn.autoCommit = false
            conn.prepareStatement(
                """
                INSERT INTO nodes (
                  node_id,
                  stage_id,
                  device_id,
                  ip_address,
                  grpc_port,
                  compute_capacity,
                  memory_gb,
                  registered_at_epoch_ms,
                  last_heartbeat_at_epoch_ms,
                  is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                  stage_id = excluded.stage_id,
                  device_id = excluded.device_id,
                  ip_address = excluded.ip_address,
                  grpc_port = excluded.grpc_port,
                  compute_capacity = excluded.compute_capacity,
                  memory_gb = excluded.memory_gb,
                  registered_at_epoch_ms = excluded.registered_at_epoch_ms,
                  last_heartbeat_at_epoch_ms = excluded.last_heartbeat_at_epoch_ms,
                  is_active = excluded.is_active
                """.trimIndent()
            ).use { stmt ->
                stmt.setInt(1, node.nodeId)
                stmt.setInt(2, node.stageId)
                stmt.setString(3, node.deviceId)
                stmt.setString(4, node.ipAddress)
                stmt.setInt(5, node.grpcPort)
                stmt.setFloat(6, node.computeCapacity)
                stmt.setFloat(7, node.memoryGb)
                stmt.setLong(8, node.registeredAt.toEpochMilli())
                stmt.setLong(9, node.lastHeartbeatAt.toEpochMilli())
                stmt.setInt(10, if (node.isActive) 1 else 0)
                stmt.executeUpdate()
            }
            saveMeta(conn, routingEpoch, nextNodeId)
            conn.commit()
        }
    }

    fun deleteNode(nodeId: Int, routingEpoch: Long, nextNodeId: Int) {
        connection().use { conn ->
            conn.autoCommit = false
            conn.prepareStatement("DELETE FROM nodes WHERE node_id = ?").use { stmt ->
                stmt.setInt(1, nodeId)
                stmt.executeUpdate()
            }
            saveMeta(conn, routingEpoch, nextNodeId)
            conn.commit()
        }
    }

    fun saveMetaOnly(routingEpoch: Long, nextNodeId: Int) {
        connection().use { conn ->
            saveMeta(conn, routingEpoch, nextNodeId)
        }
    }

    fun setStageDrained(stageId: Int, drained: Boolean) {
        connection().use { conn ->
            if (drained) {
                conn.prepareStatement(
                    """
                    INSERT INTO stage_overrides(stage_id, drained)
                    VALUES (?, 1)
                    ON CONFLICT(stage_id) DO UPDATE SET drained = 1
                    """.trimIndent()
                ).use { stmt ->
                    stmt.setInt(1, stageId)
                    stmt.executeUpdate()
                }
            } else {
                conn.prepareStatement("DELETE FROM stage_overrides WHERE stage_id = ?").use { stmt ->
                    stmt.setInt(1, stageId)
                    stmt.executeUpdate()
                }
            }
        }
    }

    fun replaceDrainedStages(stageIds: Set<Int>) {
        connection().use { conn ->
            conn.autoCommit = false
            conn.prepareStatement("DELETE FROM stage_overrides").use { stmt ->
                stmt.executeUpdate()
            }
            conn.prepareStatement(
                """
                INSERT INTO stage_overrides(stage_id, drained)
                VALUES (?, 1)
                """.trimIndent()
            ).use { stmt ->
                stageIds.forEach { stageId ->
                    stmt.setInt(1, stageId)
                    stmt.addBatch()
                }
                stmt.executeBatch()
            }
            conn.commit()
        }
    }

    fun appendRequestEvent(event: PersistedRequestEvent) {
        connection().use { conn ->
            conn.autoCommit = false
            conn.prepareStatement(
                """
                INSERT INTO request_events (
                  request_id,
                  batch_id,
                  chunk_idx,
                  stage_id,
                  node_id,
                  event_type,
                  success,
                  message,
                  event_epoch_ms,
                  terminal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, event.requestId)
                stmt.setInt(2, event.batchId)
                stmt.setInt(3, event.chunkIdx)
                stmt.setInt(4, event.stageId)
                stmt.setInt(5, event.nodeId)
                stmt.setString(6, event.eventType)
                stmt.setInt(7, if (event.success) 1 else 0)
                stmt.setString(8, event.message)
                stmt.setLong(9, event.eventEpochMs)
                stmt.setInt(10, if (event.terminal) 1 else 0)
                stmt.executeUpdate()
            }
            conn.prepareStatement(
                """
                INSERT INTO request_states (
                  request_id,
                  batch_id,
                  latest_chunk_idx,
                  latest_stage_id,
                  latest_node_id,
                  latest_event_type,
                  latest_success,
                  latest_message,
                  first_seen_epoch_ms,
                  last_updated_epoch_ms,
                  terminal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                  batch_id = excluded.batch_id,
                  latest_chunk_idx = excluded.latest_chunk_idx,
                  latest_stage_id = excluded.latest_stage_id,
                  latest_node_id = excluded.latest_node_id,
                  latest_event_type = excluded.latest_event_type,
                  latest_success = excluded.latest_success,
                  latest_message = excluded.latest_message,
                  first_seen_epoch_ms = MIN(request_states.first_seen_epoch_ms, excluded.first_seen_epoch_ms),
                  last_updated_epoch_ms = excluded.last_updated_epoch_ms,
                  terminal = excluded.terminal
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, event.requestId)
                stmt.setInt(2, event.batchId)
                stmt.setInt(3, event.chunkIdx)
                stmt.setInt(4, event.stageId)
                stmt.setInt(5, event.nodeId)
                stmt.setString(6, event.eventType)
                stmt.setInt(7, if (event.success) 1 else 0)
                stmt.setString(8, event.message)
                stmt.setLong(9, event.eventEpochMs)
                stmt.setLong(10, event.eventEpochMs)
                stmt.setInt(11, if (event.terminal) 1 else 0)
                stmt.executeUpdate()
            }
            conn.commit()
        }
    }

    fun upsertRequestPayload(requestId: String, payloadProto: ByteArray, submittedAtEpochMs: Long) {
        connection().use { conn ->
            conn.prepareStatement(
                """
                INSERT INTO request_payloads (
                  request_id,
                  payload_proto,
                  submit_attempts,
                  last_submit_epoch_ms,
                  created_epoch_ms
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                  payload_proto = excluded.payload_proto,
                  submit_attempts = request_payloads.submit_attempts + 1,
                  last_submit_epoch_ms = excluded.last_submit_epoch_ms
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.setBytes(2, payloadProto)
                stmt.setLong(3, submittedAtEpochMs)
                stmt.setLong(4, submittedAtEpochMs)
                stmt.executeUpdate()
            }
        }
    }

    fun loadRequestPayload(requestId: String): PersistedRequestPayload? {
        connection().use { conn ->
            conn.prepareStatement(
                """
                SELECT
                  request_id,
                  payload_proto,
                  submit_attempts,
                  last_submit_epoch_ms,
                  created_epoch_ms
                FROM request_payloads
                WHERE request_id = ?
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.executeQuery().use { rs ->
                    return if (rs.next()) rs.toPersistedRequestPayload() else null
                }
            }
        }
    }

    fun listRecentRequestStates(limit: Int): List<PersistedRequestState> {
        connection().use { conn ->
            conn.prepareStatement(
                """
                SELECT
                  s.request_id,
                  s.batch_id,
                  s.latest_chunk_idx,
                  s.latest_stage_id,
                  s.latest_node_id,
                  s.latest_event_type,
                  s.latest_success,
                  s.latest_message,
                  s.first_seen_epoch_ms,
                  s.last_updated_epoch_ms,
                  s.terminal,
                  CASE WHEN p.request_id IS NULL THEN 0 ELSE 1 END AS stored_payload,
                  COALESCE(p.submit_attempts, 0) AS submit_attempts,
                  p.last_submit_epoch_ms
                FROM request_states s
                LEFT JOIN request_payloads p ON p.request_id = s.request_id
                ORDER BY s.last_updated_epoch_ms DESC
                LIMIT ?
                """.trimIndent()
            ).use { stmt ->
                stmt.setInt(1, limit)
                stmt.executeQuery().use { rs ->
                    return buildList {
                        while (rs.next()) {
                            add(rs.toPersistedRequestState())
                        }
                    }
                }
            }
        }
    }

    fun loadRequestDetail(requestId: String, eventLimit: Int): AdminRequestDetailSnapshot {
        connection().use { conn ->
            val state = conn.prepareStatement(
                """
                SELECT
                  s.request_id,
                  s.batch_id,
                  s.latest_chunk_idx,
                  s.latest_stage_id,
                  s.latest_node_id,
                  s.latest_event_type,
                  s.latest_success,
                  s.latest_message,
                  s.first_seen_epoch_ms,
                  s.last_updated_epoch_ms,
                  s.terminal,
                  CASE WHEN p.request_id IS NULL THEN 0 ELSE 1 END AS stored_payload,
                  COALESCE(p.submit_attempts, 0) AS submit_attempts,
                  p.last_submit_epoch_ms
                FROM request_states s
                LEFT JOIN request_payloads p ON p.request_id = s.request_id
                WHERE s.request_id = ?
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.executeQuery().use { rs ->
                    if (rs.next()) rs.toPersistedRequestState().toAdminSnapshot() else null
                }
            }

            val events = conn.prepareStatement(
                """
                SELECT
                  event_id,
                  request_id,
                  batch_id,
                  chunk_idx,
                  stage_id,
                  node_id,
                  event_type,
                  success,
                  message,
                  event_epoch_ms,
                  terminal
                FROM request_events
                WHERE request_id = ?
                ORDER BY event_epoch_ms DESC, event_id DESC
                LIMIT ?
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.setInt(2, eventLimit)
                stmt.executeQuery().use { rs ->
                    buildList {
                        while (rs.next()) {
                            add(rs.toPersistedRequestEvent().toAdminSnapshot())
                        }
                    }
                }
            }

            return AdminRequestDetailSnapshot(state = state, events = events)
        }
    }

    fun deleteRequest(requestId: String): Boolean {
        connection().use { conn ->
            conn.autoCommit = false
            val deletedStateRows = conn.prepareStatement(
                "DELETE FROM request_states WHERE request_id = ?"
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.executeUpdate()
            }
            conn.prepareStatement(
                "DELETE FROM request_events WHERE request_id = ?"
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.executeUpdate()
            }
            conn.prepareStatement(
                "DELETE FROM request_payloads WHERE request_id = ?"
            ).use { stmt ->
                stmt.setString(1, requestId)
                stmt.executeUpdate()
            }
            conn.commit()
            return deletedStateRows > 0
        }
    }

    fun purgeResolvedRequests(olderThanEpochMs: Long): Int {
        connection().use { conn ->
            conn.autoCommit = false
            val requestIds = conn.prepareStatement(
                """
                SELECT request_id
                FROM request_states
                WHERE last_updated_epoch_ms <= ?
                  AND (
                    latest_event_type = 'FAILED'
                    OR (latest_event_type = 'COMPLETED' AND terminal = 1)
                  )
                """.trimIndent()
            ).use { stmt ->
                stmt.setLong(1, olderThanEpochMs)
                stmt.executeQuery().use { rs ->
                    buildList {
                        while (rs.next()) {
                            add(rs.getString("request_id"))
                        }
                    }
                }
            }
            if (requestIds.isEmpty()) {
                conn.rollback()
                return 0
            }
            conn.prepareStatement(
                "DELETE FROM request_events WHERE request_id = ?"
            ).use { stmt ->
                requestIds.forEach { requestId ->
                    stmt.setString(1, requestId)
                    stmt.addBatch()
                }
                stmt.executeBatch()
            }
            conn.prepareStatement(
                "DELETE FROM request_payloads WHERE request_id = ?"
            ).use { stmt ->
                requestIds.forEach { requestId ->
                    stmt.setString(1, requestId)
                    stmt.addBatch()
                }
                stmt.executeBatch()
            }
            conn.prepareStatement(
                "DELETE FROM request_states WHERE request_id = ?"
            ).use { stmt ->
                requestIds.forEach { requestId ->
                    stmt.setString(1, requestId)
                    stmt.addBatch()
                }
                stmt.executeBatch()
            }
            conn.commit()
            return requestIds.size
        }
    }

    private fun connection(): Connection {
        return DriverManager.getConnection(jdbcUrl)
    }

    private fun initializeSchema(conn: Connection) {
        conn.createStatement().use { stmt ->
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS coordinator_meta (
                  meta_key TEXT PRIMARY KEY,
                  meta_value TEXT NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                  node_id INTEGER PRIMARY KEY,
                  stage_id INTEGER NOT NULL UNIQUE,
                  device_id TEXT NOT NULL UNIQUE,
                  ip_address TEXT NOT NULL,
                  grpc_port INTEGER NOT NULL,
                  compute_capacity REAL NOT NULL,
                  memory_gb REAL NOT NULL,
                  registered_at_epoch_ms INTEGER NOT NULL,
                  last_heartbeat_at_epoch_ms INTEGER NOT NULL,
                  is_active INTEGER NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_overrides (
                  stage_id INTEGER PRIMARY KEY,
                  drained INTEGER NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS request_states (
                  request_id TEXT PRIMARY KEY,
                  batch_id INTEGER NOT NULL,
                  latest_chunk_idx INTEGER NOT NULL,
                  latest_stage_id INTEGER NOT NULL,
                  latest_node_id INTEGER NOT NULL,
                  latest_event_type TEXT NOT NULL,
                  latest_success INTEGER NOT NULL,
                  latest_message TEXT NOT NULL,
                  first_seen_epoch_ms INTEGER NOT NULL,
                  last_updated_epoch_ms INTEGER NOT NULL,
                  terminal INTEGER NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS request_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  request_id TEXT NOT NULL,
                  batch_id INTEGER NOT NULL,
                  chunk_idx INTEGER NOT NULL,
                  stage_id INTEGER NOT NULL,
                  node_id INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  success INTEGER NOT NULL,
                  message TEXT NOT NULL,
                  event_epoch_ms INTEGER NOT NULL,
                  terminal INTEGER NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS request_payloads (
                  request_id TEXT PRIMARY KEY,
                  payload_proto BLOB NOT NULL,
                  submit_attempts INTEGER NOT NULL,
                  last_submit_epoch_ms INTEGER NOT NULL,
                  created_epoch_ms INTEGER NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_events_request_id_epoch
                ON request_events(request_id, event_epoch_ms DESC, event_id DESC)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_states_last_updated
                ON request_states(last_updated_epoch_ms DESC)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_payloads_last_submit
                ON request_payloads(last_submit_epoch_ms DESC)
                """.trimIndent()
            )
        }
    }

    private fun saveMeta(conn: Connection, routingEpoch: Long, nextNodeId: Int) {
        saveMetaValue(conn, "routing_epoch", routingEpoch.toString())
        saveMetaValue(conn, "next_node_id", nextNodeId.toString())
    }

    private fun saveMetaValue(conn: Connection, key: String, value: String) {
        conn.prepareStatement(
            """
            INSERT INTO coordinator_meta(meta_key, meta_value)
            VALUES (?, ?)
            ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value
            """.trimIndent()
        ).use { stmt ->
            stmt.setString(1, key)
            stmt.setString(2, value)
            stmt.executeUpdate()
        }
    }

    private fun readMetaLong(conn: Connection, key: String): Long? {
        conn.prepareStatement(
            "SELECT meta_value FROM coordinator_meta WHERE meta_key = ?"
        ).use { stmt ->
            stmt.setString(1, key)
            stmt.executeQuery().use { rs ->
                if (!rs.next()) {
                    return null
                }
                return rs.getString("meta_value").toLongOrNull()
            }
        }
    }

    private fun ResultSet.toRegisteredNode(): RegisteredNode {
        return RegisteredNode(
            nodeId = getInt("node_id"),
            stageId = getInt("stage_id"),
            deviceId = getString("device_id"),
            ipAddress = getString("ip_address"),
            grpcPort = getInt("grpc_port"),
            computeCapacity = getFloat("compute_capacity"),
            memoryGb = getFloat("memory_gb"),
            registeredAt = Instant.ofEpochMilli(getLong("registered_at_epoch_ms")),
            lastHeartbeatAt = Instant.ofEpochMilli(getLong("last_heartbeat_at_epoch_ms")),
            isActive = getInt("is_active") != 0
        )
    }

    private fun ResultSet.toPersistedRequestState(): PersistedRequestState {
        return PersistedRequestState(
            requestId = getString("request_id"),
            batchId = getInt("batch_id"),
            latestChunkIdx = getInt("latest_chunk_idx"),
            latestStageId = getInt("latest_stage_id"),
            latestNodeId = getInt("latest_node_id"),
            latestEventType = getString("latest_event_type"),
            latestSuccess = getInt("latest_success") != 0,
            latestMessage = getString("latest_message"),
            firstSeenEpochMs = getLong("first_seen_epoch_ms"),
            lastUpdatedEpochMs = getLong("last_updated_epoch_ms"),
            terminal = getInt("terminal") != 0,
            storedPayload = getInt("stored_payload") != 0,
            submitAttempts = getInt("submit_attempts"),
            lastSubmitEpochMs = getLong("last_submit_epoch_ms").takeIf { !wasNull() }
        )
    }

    private fun ResultSet.toPersistedRequestEvent(): PersistedRequestEvent {
        return PersistedRequestEvent(
            eventId = getLong("event_id"),
            requestId = getString("request_id"),
            batchId = getInt("batch_id"),
            chunkIdx = getInt("chunk_idx"),
            stageId = getInt("stage_id"),
            nodeId = getInt("node_id"),
            eventType = getString("event_type"),
            success = getInt("success") != 0,
            message = getString("message"),
            eventEpochMs = getLong("event_epoch_ms"),
            terminal = getInt("terminal") != 0
        )
    }

    private fun ResultSet.toPersistedRequestPayload(): PersistedRequestPayload {
        return PersistedRequestPayload(
            requestId = getString("request_id"),
            payloadProto = getBytes("payload_proto"),
            submitAttempts = getInt("submit_attempts"),
            lastSubmitEpochMs = getLong("last_submit_epoch_ms"),
            createdEpochMs = getLong("created_epoch_ms")
        )
    }

    private fun PersistedRequestState.toAdminSnapshot(): AdminRequestStateSnapshot {
        return AdminRequestStateSnapshot(
            requestId = requestId,
            batchId = batchId,
            latestChunkIdx = latestChunkIdx,
            latestStageId = latestStageId,
            latestNodeId = latestNodeId,
            latestEventType = latestEventType,
            latestSuccess = latestSuccess,
            latestMessage = latestMessage,
            firstSeenEpochMs = firstSeenEpochMs,
            lastUpdatedEpochMs = lastUpdatedEpochMs,
            terminal = terminal,
            lifecycleState = "UNKNOWN",
            stalled = false,
            lastUpdatedAgeSeconds = 0,
            storedPayload = storedPayload,
            submitAttempts = submitAttempts,
            lastSubmitEpochMs = lastSubmitEpochMs
        )
    }

    private fun PersistedRequestEvent.toAdminSnapshot(): AdminRequestEventSnapshot {
        return AdminRequestEventSnapshot(
            eventId = eventId,
            requestId = requestId,
            batchId = batchId,
            chunkIdx = chunkIdx,
            stageId = stageId,
            nodeId = nodeId,
            eventType = eventType,
            success = success,
            message = message,
            eventEpochMs = eventEpochMs,
            terminal = terminal
        )
    }
}
