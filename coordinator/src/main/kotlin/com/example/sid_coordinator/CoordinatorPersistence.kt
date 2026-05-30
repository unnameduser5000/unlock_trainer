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
            val insertedEventId = conn.prepareStatement(
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
                """.trimIndent(),
                java.sql.Statement.RETURN_GENERATED_KEYS
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
                stmt.generatedKeys.use { keys ->
                    if (keys.next()) keys.getLong(1) else 0L
                }
            }
            parseStageTimingMetric(event.copy(eventId = insertedEventId))?.let { timing ->
                insertStageTimingMetric(conn, timing)
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

    fun appendSchedulerEvent(event: AdminSchedulerEventSnapshot): AdminSchedulerEventSnapshot {
        connection().use { conn ->
            conn.prepareStatement(
                """
                INSERT INTO scheduler_events (
                  event_epoch_ms,
                  action,
                  stage_id,
                  node_id,
                  device_id,
                  reason,
                  message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """.trimIndent(),
                java.sql.Statement.RETURN_GENERATED_KEYS
            ).use { stmt ->
                stmt.setLong(1, event.eventEpochMs)
                stmt.setString(2, event.action)
                setNullableInt(stmt, 3, event.stageId)
                setNullableInt(stmt, 4, event.nodeId)
                stmt.setString(5, event.deviceId)
                stmt.setString(6, event.reason)
                stmt.setString(7, event.message)
                stmt.executeUpdate()
                stmt.generatedKeys.use { keys ->
                    val eventId = if (keys.next()) keys.getLong(1) else 0L
                    return event.copy(eventId = eventId)
                }
            }
        }
    }

    fun listRecentSchedulerEvents(limit: Int): List<AdminSchedulerEventSnapshot> {
        connection().use { conn ->
            conn.prepareStatement(
                """
                SELECT
                  event_id,
                  event_epoch_ms,
                  action,
                  stage_id,
                  node_id,
                  device_id,
                  reason,
                  message
                FROM scheduler_events
                ORDER BY event_epoch_ms DESC, event_id DESC
                LIMIT ?
                """.trimIndent()
            ).use { stmt ->
                stmt.setInt(1, limit)
                stmt.executeQuery().use { rs ->
                    return buildList {
                        while (rs.next()) {
                            add(rs.toAdminSchedulerEventSnapshot())
                        }
                    }
                }
            }
        }
    }

    fun appendWorkerTelemetry(telemetry: PersistedWorkerTelemetry) {
        connection().use { conn ->
            conn.prepareStatement(
                """
                INSERT INTO worker_telemetry (
                  observed_epoch_ms,
                  device_id,
                  node_id,
                  stage_id,
                  is_active,
                  battery_level,
                  is_charging,
                  power_source,
                  battery_status,
                  battery_temp_c,
                  battery_voltage_mv,
                  battery_current_ua,
                  thermal_status,
                  app_pss_kb,
                  app_private_dirty_kb,
                  runtime_used_memory_kb,
                  worker_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.trimIndent()
            ).use { stmt ->
                stmt.setLong(1, telemetry.observedEpochMs)
                stmt.setString(2, telemetry.deviceId)
                stmt.setInt(3, telemetry.nodeId)
                stmt.setInt(4, telemetry.stageId)
                stmt.setInt(5, if (telemetry.isActive) 1 else 0)
                stmt.setFloat(6, telemetry.batteryLevel)
                stmt.setInt(7, if (telemetry.isCharging) 1 else 0)
                stmt.setString(8, telemetry.powerSource)
                stmt.setInt(9, telemetry.batteryStatus)
                setNullableFloat(stmt, 10, telemetry.batteryTempC)
                setNullableInt(stmt, 11, telemetry.batteryVoltageMv)
                setNullableLong(stmt, 12, telemetry.batteryCurrentUa)
                stmt.setString(13, telemetry.thermalStatus)
                setNullableLong(stmt, 14, telemetry.appPssKb)
                setNullableLong(stmt, 15, telemetry.appPrivateDirtyKb)
                setNullableLong(stmt, 16, telemetry.runtimeUsedMemoryKb)
                stmt.setString(17, telemetry.workerState)
                stmt.executeUpdate()
            }
        }
    }

    fun listRecentWorkerTelemetry(limit: Int, deviceId: String?): List<PersistedWorkerTelemetry> {
        connection().use { conn ->
            val filteredDeviceId = deviceId?.trim()?.takeIf { it.isNotBlank() }
            val whereClause = if (filteredDeviceId == null) "" else "WHERE device_id = ?"
            conn.prepareStatement(
                """
                SELECT
                  telemetry_id,
                  observed_epoch_ms,
                  device_id,
                  node_id,
                  stage_id,
                  is_active,
                  battery_level,
                  is_charging,
                  power_source,
                  battery_status,
                  battery_temp_c,
                  battery_voltage_mv,
                  battery_current_ua,
                  thermal_status,
                  app_pss_kb,
                  app_private_dirty_kb,
                  runtime_used_memory_kb,
                  worker_state
                FROM worker_telemetry
                $whereClause
                ORDER BY observed_epoch_ms DESC, telemetry_id DESC
                LIMIT ?
                """.trimIndent()
            ).use { stmt ->
                var index = 1
                if (filteredDeviceId != null) {
                    stmt.setString(index++, filteredDeviceId)
                }
                stmt.setInt(index, limit)
                stmt.executeQuery().use { rs ->
                    return buildList {
                        while (rs.next()) {
                            add(rs.toPersistedWorkerTelemetry())
                        }
                    }
                }
            }
        }
    }

    fun appendRequestMetric(
        metric: PersistedRequestMetric,
        pipelineName: String,
        modelShards: String,
        configHash: String
    ) {
        connection().use { conn ->
            conn.autoCommit = false
            val attempt = if (metric.attempt > 0) {
                metric.attempt
            } else {
                nextMetricAttempt(conn, metric.requestId)
            }
            conn.prepareStatement(
                """
                INSERT INTO runs (
                  run_id,
                  pipeline_name,
                  model_shards,
                  config_hash,
                  first_seen_epoch_ms,
                  last_updated_epoch_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  pipeline_name = excluded.pipeline_name,
                  model_shards = excluded.model_shards,
                  config_hash = excluded.config_hash,
                  first_seen_epoch_ms = MIN(runs.first_seen_epoch_ms, excluded.first_seen_epoch_ms),
                  last_updated_epoch_ms = MAX(runs.last_updated_epoch_ms, excluded.last_updated_epoch_ms)
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, metric.runId)
                stmt.setString(2, pipelineName)
                stmt.setString(3, modelShards)
                stmt.setString(4, configHash)
                stmt.setLong(5, metric.submittedEpochMs)
                stmt.setLong(6, metric.completedEpochMs)
                stmt.executeUpdate()
            }
            conn.prepareStatement(
                """
                INSERT INTO request_metrics (
                  run_id,
                  request_id,
                  request_index,
                  attempt,
                  batch_id,
                  eval_only,
                  submitted_epoch_ms,
                  completed_epoch_ms,
                  elapsed_ms,
                  success,
                  terminal,
                  processed_stage_id,
                  processed_chunk_idx,
                  output_hidden_bytes,
                  output_shift_log_p_bytes,
                  local_loss,
                  token_correct,
                  token_count,
                  message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.trimIndent()
            ).use { stmt ->
                stmt.setString(1, metric.runId)
                stmt.setString(2, metric.requestId)
                setNullableInt(stmt, 3, metric.requestIndex)
                stmt.setInt(4, attempt)
                stmt.setInt(5, metric.batchId)
                stmt.setInt(6, if (metric.evalOnly) 1 else 0)
                stmt.setLong(7, metric.submittedEpochMs)
                stmt.setLong(8, metric.completedEpochMs)
                stmt.setLong(9, metric.elapsedMs)
                stmt.setInt(10, if (metric.success) 1 else 0)
                stmt.setInt(11, if (metric.terminal) 1 else 0)
                stmt.setInt(12, metric.processedStageId)
                stmt.setInt(13, metric.processedChunkIdx)
                stmt.setInt(14, metric.outputHiddenBytes)
                stmt.setInt(15, metric.outputShiftLogPBytes)
                if (metric.localLoss == null) {
                    stmt.setNull(16, java.sql.Types.REAL)
                } else {
                    stmt.setFloat(16, metric.localLoss)
                }
                stmt.setInt(17, metric.tokenCorrect)
                stmt.setInt(18, metric.tokenCount)
                stmt.setString(19, metric.message)
                stmt.executeUpdate()
            }
            conn.commit()
        }
    }

    fun listRecentRuns(limit: Int): List<PersistedRunSummary> {
        connection().use { conn ->
            conn.prepareStatement(
                runSummarySelect(
                    whereClause = "",
                    orderLimitClause = "ORDER BY r.last_updated_epoch_ms DESC LIMIT ?"
                )
            ).use { stmt ->
                stmt.setInt(1, limit)
                stmt.executeQuery().use { rs ->
                    return buildList {
                        while (rs.next()) {
                            add(rs.toPersistedRunSummary())
                        }
                    }
                }
            }
        }
    }

    fun loadRunDetail(runId: String, metricLimit: Int): AdminRunDetailSnapshot {
        connection().use { conn ->
            val summary = conn.prepareStatement(
                runSummarySelect(
                    whereClause = "WHERE r.run_id = ?",
                    orderLimitClause = ""
                )
            ).use { stmt ->
                stmt.setString(1, runId)
                stmt.executeQuery().use { rs ->
                    if (rs.next()) rs.toPersistedRunSummary().toAdminSnapshot() else null
                }
            }
            val metrics = listRunMetrics(conn, runId, metricLimit)
                .map { it.toAdminSnapshot() }
            val stageTimings = listStageTimingMetrics(conn, runId, metricLimit)
                .map { it.toAdminSnapshot() }
            return AdminRunDetailSnapshot(summary = summary, metrics = metrics, stageTimings = stageTimings)
        }
    }

    fun listRunMetrics(runId: String, metricLimit: Int): List<PersistedRequestMetric> {
        connection().use { conn ->
            return listRunMetrics(conn, runId, metricLimit)
        }
    }

    fun listStageTimingMetrics(runId: String, metricLimit: Int): List<PersistedStageTimingMetric> {
        connection().use { conn ->
            return listStageTimingMetrics(conn, runId, metricLimit)
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
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  pipeline_name TEXT NOT NULL,
                  model_shards TEXT NOT NULL,
                  config_hash TEXT NOT NULL,
                  first_seen_epoch_ms INTEGER NOT NULL,
                  last_updated_epoch_ms INTEGER NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS request_metrics (
                  metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  request_id TEXT NOT NULL,
                  request_index INTEGER,
                  attempt INTEGER NOT NULL,
                  batch_id INTEGER NOT NULL,
                  eval_only INTEGER NOT NULL,
                  submitted_epoch_ms INTEGER NOT NULL,
                  completed_epoch_ms INTEGER NOT NULL,
                  elapsed_ms INTEGER NOT NULL,
                  success INTEGER NOT NULL,
                  terminal INTEGER NOT NULL,
                  processed_stage_id INTEGER NOT NULL,
                  processed_chunk_idx INTEGER NOT NULL,
                  output_hidden_bytes INTEGER NOT NULL,
                  output_shift_log_p_bytes INTEGER NOT NULL,
                  local_loss REAL,
                  token_correct INTEGER NOT NULL,
                  token_count INTEGER NOT NULL,
                  message TEXT NOT NULL,
                  FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_timing_metrics (
                  stage_metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  request_event_id INTEGER NOT NULL,
                  run_id TEXT NOT NULL,
                  request_id TEXT NOT NULL,
                  batch_id INTEGER NOT NULL,
                  chunk_idx INTEGER NOT NULL,
                  stage_id INTEGER NOT NULL,
                  node_id INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  event_epoch_ms INTEGER NOT NULL,
                  runtime TEXT,
                  method TEXT,
                  input_count INTEGER,
                  eval_only INTEGER,
                  optimizer_step_applied INTEGER,
                  local_loss REAL,
                  local_ms INTEGER,
                  input_build_ms INTEGER,
                  execute_ms INTEGER,
                  gradients_ms INTEGER,
                  optimizer_create_ms INTEGER,
                  optimizer_step_ms INTEGER,
                  output_convert_ms INTEGER,
                  total_measured_ms INTEGER,
                  forward_ms INTEGER,
                  total_stage_ms INTEGER,
                  output_bytes INTEGER,
                  message TEXT NOT NULL,
                  FOREIGN KEY(request_event_id) REFERENCES request_events(event_id)
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduler_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_epoch_ms INTEGER NOT NULL,
                  action TEXT NOT NULL,
                  stage_id INTEGER,
                  node_id INTEGER,
                  device_id TEXT,
                  reason TEXT NOT NULL,
                  message TEXT NOT NULL
                )
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_telemetry (
                  telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  observed_epoch_ms INTEGER NOT NULL,
                  device_id TEXT NOT NULL,
                  node_id INTEGER NOT NULL,
                  stage_id INTEGER NOT NULL,
                  is_active INTEGER NOT NULL,
                  battery_level REAL NOT NULL,
                  is_charging INTEGER NOT NULL,
                  power_source TEXT NOT NULL,
                  battery_status INTEGER NOT NULL,
                  battery_temp_c REAL,
                  battery_voltage_mv INTEGER,
                  battery_current_ua INTEGER,
                  thermal_status TEXT NOT NULL,
                  app_pss_kb INTEGER,
                  app_private_dirty_kb INTEGER,
                  runtime_used_memory_kb INTEGER,
                  worker_state TEXT NOT NULL
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
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_metrics_run_id_index
                ON request_metrics(run_id, request_index, metric_id)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_request_metrics_request_id_attempt
                ON request_metrics(request_id, attempt DESC)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stage_timing_metrics_run_stage
                ON stage_timing_metrics(run_id, stage_id, event_epoch_ms)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stage_timing_metrics_request
                ON stage_timing_metrics(request_id, event_epoch_ms, stage_metric_id)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_last_updated
                ON runs(last_updated_epoch_ms DESC)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduler_events_epoch
                ON scheduler_events(event_epoch_ms DESC, event_id DESC)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_worker_telemetry_epoch
                ON worker_telemetry(observed_epoch_ms DESC, telemetry_id DESC)
                """.trimIndent()
            )
            stmt.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_worker_telemetry_device_epoch
                ON worker_telemetry(device_id, observed_epoch_ms DESC, telemetry_id DESC)
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

    private fun nextMetricAttempt(conn: Connection, requestId: String): Int {
        conn.prepareStatement(
            "SELECT COALESCE(MAX(attempt), 0) + 1 AS next_attempt FROM request_metrics WHERE request_id = ?"
        ).use { stmt ->
            stmt.setString(1, requestId)
            stmt.executeQuery().use { rs ->
                return if (rs.next()) rs.getInt("next_attempt") else 1
            }
        }
    }

    private fun listRunMetrics(
        conn: Connection,
        runId: String,
        metricLimit: Int
    ): List<PersistedRequestMetric> {
        conn.prepareStatement(
            """
            SELECT
              metric_id,
              run_id,
              request_id,
              request_index,
              attempt,
              batch_id,
              eval_only,
              submitted_epoch_ms,
              completed_epoch_ms,
              elapsed_ms,
              success,
              terminal,
              processed_stage_id,
              processed_chunk_idx,
              output_hidden_bytes,
              output_shift_log_p_bytes,
              local_loss,
              token_correct,
              token_count,
              message
            FROM request_metrics
            WHERE run_id = ?
            ORDER BY
              CASE WHEN request_index IS NULL THEN 1 ELSE 0 END,
              request_index ASC,
              metric_id ASC
            LIMIT ?
            """.trimIndent()
        ).use { stmt ->
            stmt.setString(1, runId)
            stmt.setInt(2, metricLimit)
            stmt.executeQuery().use { rs ->
                return buildList {
                    while (rs.next()) {
                        add(rs.toPersistedRequestMetric())
                    }
                }
            }
        }
    }

    private fun listStageTimingMetrics(
        conn: Connection,
        runId: String,
        metricLimit: Int
    ): List<PersistedStageTimingMetric> {
        conn.prepareStatement(
            """
            SELECT
              stage_metric_id,
              request_event_id,
              run_id,
              request_id,
              batch_id,
              chunk_idx,
              stage_id,
              node_id,
              event_type,
              event_epoch_ms,
              runtime,
              method,
              input_count,
              eval_only,
              optimizer_step_applied,
              local_loss,
              local_ms,
              input_build_ms,
              execute_ms,
              gradients_ms,
              optimizer_create_ms,
              optimizer_step_ms,
              output_convert_ms,
              total_measured_ms,
              forward_ms,
              total_stage_ms,
              output_bytes,
              message
            FROM stage_timing_metrics
            WHERE run_id = ?
            ORDER BY event_epoch_ms ASC, stage_metric_id ASC
            LIMIT ?
            """.trimIndent()
        ).use { stmt ->
            stmt.setString(1, runId)
            stmt.setInt(2, metricLimit)
            stmt.executeQuery().use { rs ->
                return buildList {
                    while (rs.next()) {
                        add(rs.toPersistedStageTimingMetric())
                    }
                }
            }
        }
    }

    private fun insertStageTimingMetric(conn: Connection, timing: PersistedStageTimingMetric) {
        conn.prepareStatement(
            """
            INSERT INTO stage_timing_metrics (
              request_event_id,
              run_id,
              request_id,
              batch_id,
              chunk_idx,
              stage_id,
              node_id,
              event_type,
              event_epoch_ms,
              runtime,
              method,
              input_count,
              eval_only,
              optimizer_step_applied,
              local_loss,
              local_ms,
              input_build_ms,
              execute_ms,
              gradients_ms,
              optimizer_create_ms,
              optimizer_step_ms,
              output_convert_ms,
              total_measured_ms,
              forward_ms,
              total_stage_ms,
              output_bytes,
              message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.trimIndent()
        ).use { stmt ->
            stmt.setLong(1, timing.requestEventId)
            stmt.setString(2, timing.runId)
            stmt.setString(3, timing.requestId)
            stmt.setInt(4, timing.batchId)
            stmt.setInt(5, timing.chunkIdx)
            stmt.setInt(6, timing.stageId)
            stmt.setInt(7, timing.nodeId)
            stmt.setString(8, timing.eventType)
            stmt.setLong(9, timing.eventEpochMs)
            stmt.setString(10, timing.runtime)
            stmt.setString(11, timing.method)
            setNullableInt(stmt, 12, timing.inputCount)
            setNullableBoolean(stmt, 13, timing.evalOnly)
            setNullableBoolean(stmt, 14, timing.optimizerStepApplied)
            setNullableFloat(stmt, 15, timing.localLoss)
            setNullableLong(stmt, 16, timing.localMs)
            setNullableLong(stmt, 17, timing.inputBuildMs)
            setNullableLong(stmt, 18, timing.executeMs)
            setNullableLong(stmt, 19, timing.gradientsMs)
            setNullableLong(stmt, 20, timing.optimizerCreateMs)
            setNullableLong(stmt, 21, timing.optimizerStepMs)
            setNullableLong(stmt, 22, timing.outputConvertMs)
            setNullableLong(stmt, 23, timing.totalMeasuredMs)
            setNullableLong(stmt, 24, timing.forwardMs)
            setNullableLong(stmt, 25, timing.totalStageMs)
            setNullableInt(stmt, 26, timing.outputBytes)
            stmt.setString(27, timing.message)
            stmt.executeUpdate()
        }
    }

    private fun parseStageTimingMetric(event: PersistedRequestEvent): PersistedStageTimingMetric? {
        if (event.eventType !in setOf("LOCAL_COMPLETED", "FORWARDING", "COMPLETED")) {
            return null
        }
        val pairs = parseKeyValuePairs(event.message)
        val hasTiming =
            pairs.containsKey("localMs") ||
                pairs.containsKey("executeMs") ||
                pairs.containsKey("forwardMs") ||
                pairs.containsKey("totalStageMs")
        if (!hasTiming) {
            return null
        }
        return PersistedStageTimingMetric(
            requestEventId = event.eventId,
            runId = inferRunId(event.requestId),
            requestId = event.requestId,
            batchId = event.batchId,
            chunkIdx = event.chunkIdx,
            stageId = event.stageId,
            nodeId = event.nodeId,
            eventType = event.eventType,
            eventEpochMs = event.eventEpochMs,
            runtime = pairs["runtime"],
            method = pairs["method"],
            inputCount = pairs["inputs"]?.toIntOrNull(),
            evalOnly = pairs["evalOnly"]?.toBooleanStrictOrNull(),
            optimizerStepApplied = pairs["optimizerStepApplied"]?.toBooleanStrictOrNull(),
            localLoss = pairs["loss"]?.toFloatOrNull(),
            localMs = pairs["localMs"]?.toLongOrNull(),
            inputBuildMs = pairs["inputBuildMs"]?.toLongOrNull(),
            executeMs = pairs["executeMs"]?.toLongOrNull(),
            gradientsMs = pairs["gradientsMs"]?.toLongOrNull(),
            optimizerCreateMs = pairs["optimizerCreateMs"]?.toLongOrNull(),
            optimizerStepMs = pairs["optimizerStepMs"]?.toLongOrNull(),
            outputConvertMs = pairs["outputConvertMs"]?.toLongOrNull(),
            totalMeasuredMs = pairs["totalMeasuredMs"]?.toLongOrNull(),
            forwardMs = pairs["forwardMs"]?.toLongOrNull(),
            totalStageMs = pairs["totalStageMs"]?.toLongOrNull(),
            outputBytes = pairs["bytes"]?.toIntOrNull(),
            message = event.message
        )
    }

    private fun parseKeyValuePairs(message: String): Map<String, String> {
        return KEY_VALUE_PATTERN.findAll(message)
            .associate { match -> match.groupValues[1] to match.groupValues[2] }
    }

    private fun runSummarySelect(whereClause: String, orderLimitClause: String): String {
        return """
            SELECT
              r.run_id,
              r.pipeline_name,
              r.model_shards,
              r.config_hash,
              r.first_seen_epoch_ms,
              r.last_updated_epoch_ms,
              COUNT(m.metric_id) AS request_rows,
              COALESCE(SUM(CASE WHEN m.success = 1 AND m.terminal = 1 THEN 1 ELSE 0 END), 0) AS success_rows,
              COALESCE(SUM(CASE WHEN m.success = 0 OR m.terminal = 0 THEN 1 ELSE 0 END), 0) AS failed_rows,
              COALESCE(SUM(CASE WHEN m.eval_only = 1 THEN 1 ELSE 0 END), 0) AS eval_only_rows,
              COALESCE(SUM(CASE WHEN m.eval_only = 0 THEN 1 ELSE 0 END), 0) AS train_rows,
              AVG(CASE WHEN m.success = 1 AND m.terminal = 1 THEN m.elapsed_ms END) AS avg_elapsed_ms,
              AVG(CASE WHEN m.success = 1 AND m.terminal = 1 THEN m.local_loss END) AS avg_loss,
              COALESCE(SUM(CASE WHEN m.success = 1 AND m.terminal = 1 THEN m.token_correct ELSE 0 END), 0) AS token_correct,
              COALESCE(SUM(CASE WHEN m.success = 1 AND m.terminal = 1 THEN m.token_count ELSE 0 END), 0) AS token_count
            FROM runs r
            LEFT JOIN (
              SELECT rm.*
              FROM request_metrics rm
              INNER JOIN (
                SELECT run_id, request_id, MAX(metric_id) AS latest_metric_id
                FROM request_metrics
                GROUP BY run_id, request_id
              ) latest
                ON latest.run_id = rm.run_id
               AND latest.request_id = rm.request_id
               AND latest.latest_metric_id = rm.metric_id
            ) m ON m.run_id = r.run_id
            $whereClause
            GROUP BY r.run_id
            $orderLimitClause
        """.trimIndent()
    }

    private fun setNullableInt(stmt: java.sql.PreparedStatement, index: Int, value: Int?) {
        if (value == null) {
            stmt.setNull(index, java.sql.Types.INTEGER)
        } else {
            stmt.setInt(index, value)
        }
    }

    private fun setNullableLong(stmt: java.sql.PreparedStatement, index: Int, value: Long?) {
        if (value == null) {
            stmt.setNull(index, java.sql.Types.INTEGER)
        } else {
            stmt.setLong(index, value)
        }
    }

    private fun setNullableFloat(stmt: java.sql.PreparedStatement, index: Int, value: Float?) {
        if (value == null) {
            stmt.setNull(index, java.sql.Types.REAL)
        } else {
            stmt.setFloat(index, value)
        }
    }

    private fun setNullableBoolean(stmt: java.sql.PreparedStatement, index: Int, value: Boolean?) {
        if (value == null) {
            stmt.setNull(index, java.sql.Types.INTEGER)
        } else {
            stmt.setInt(index, if (value) 1 else 0)
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

    private fun ResultSet.toAdminSchedulerEventSnapshot(): AdminSchedulerEventSnapshot {
        val stage = getInt("stage_id").takeIf { !wasNull() }
        val node = getInt("node_id").takeIf { !wasNull() }
        val device = getString("device_id").takeIf { !wasNull() }
        return AdminSchedulerEventSnapshot(
            eventId = getLong("event_id"),
            eventEpochMs = getLong("event_epoch_ms"),
            action = getString("action"),
            stageId = stage,
            nodeId = node,
            deviceId = device,
            reason = getString("reason"),
            message = getString("message")
        )
    }

    private fun ResultSet.toPersistedRequestMetric(): PersistedRequestMetric {
        val requestIndex = getInt("request_index").takeIf { !wasNull() }
        val localLoss = getFloat("local_loss").takeIf { !wasNull() }
        return PersistedRequestMetric(
            metricId = getLong("metric_id"),
            runId = getString("run_id"),
            requestId = getString("request_id"),
            requestIndex = requestIndex,
            attempt = getInt("attempt"),
            batchId = getInt("batch_id"),
            evalOnly = getInt("eval_only") != 0,
            submittedEpochMs = getLong("submitted_epoch_ms"),
            completedEpochMs = getLong("completed_epoch_ms"),
            elapsedMs = getLong("elapsed_ms"),
            success = getInt("success") != 0,
            terminal = getInt("terminal") != 0,
            processedStageId = getInt("processed_stage_id"),
            processedChunkIdx = getInt("processed_chunk_idx"),
            outputHiddenBytes = getInt("output_hidden_bytes"),
            outputShiftLogPBytes = getInt("output_shift_log_p_bytes"),
            localLoss = localLoss,
            tokenCorrect = getInt("token_correct"),
            tokenCount = getInt("token_count"),
            message = getString("message")
        )
    }

    private fun ResultSet.toPersistedRunSummary(): PersistedRunSummary {
        val avgElapsed = getDouble("avg_elapsed_ms").takeIf { !wasNull() }
        val avgLoss = getDouble("avg_loss").takeIf { !wasNull() }
        return PersistedRunSummary(
            runId = getString("run_id"),
            pipelineName = getString("pipeline_name"),
            modelShards = getString("model_shards"),
            configHash = getString("config_hash"),
            firstSeenEpochMs = getLong("first_seen_epoch_ms"),
            lastUpdatedEpochMs = getLong("last_updated_epoch_ms"),
            requestRows = getInt("request_rows"),
            successRows = getInt("success_rows"),
            failedRows = getInt("failed_rows"),
            evalOnlyRows = getInt("eval_only_rows"),
            trainRows = getInt("train_rows"),
            avgElapsedMs = avgElapsed,
            avgLoss = avgLoss,
            tokenCorrect = getLong("token_correct"),
            tokenCount = getLong("token_count")
        )
    }

    private fun ResultSet.toPersistedStageTimingMetric(): PersistedStageTimingMetric {
        return PersistedStageTimingMetric(
            stageMetricId = getLong("stage_metric_id"),
            requestEventId = getLong("request_event_id"),
            runId = getString("run_id"),
            requestId = getString("request_id"),
            batchId = getInt("batch_id"),
            chunkIdx = getInt("chunk_idx"),
            stageId = getInt("stage_id"),
            nodeId = getInt("node_id"),
            eventType = getString("event_type"),
            eventEpochMs = getLong("event_epoch_ms"),
            runtime = getString("runtime").takeIf { !wasNull() },
            method = getString("method").takeIf { !wasNull() },
            inputCount = getInt("input_count").takeIf { !wasNull() },
            evalOnly = getInt("eval_only").takeIf { !wasNull() }?.let { it != 0 },
            optimizerStepApplied = getInt("optimizer_step_applied").takeIf { !wasNull() }?.let { it != 0 },
            localLoss = getFloat("local_loss").takeIf { !wasNull() },
            localMs = getLong("local_ms").takeIf { !wasNull() },
            inputBuildMs = getLong("input_build_ms").takeIf { !wasNull() },
            executeMs = getLong("execute_ms").takeIf { !wasNull() },
            gradientsMs = getLong("gradients_ms").takeIf { !wasNull() },
            optimizerCreateMs = getLong("optimizer_create_ms").takeIf { !wasNull() },
            optimizerStepMs = getLong("optimizer_step_ms").takeIf { !wasNull() },
            outputConvertMs = getLong("output_convert_ms").takeIf { !wasNull() },
            totalMeasuredMs = getLong("total_measured_ms").takeIf { !wasNull() },
            forwardMs = getLong("forward_ms").takeIf { !wasNull() },
            totalStageMs = getLong("total_stage_ms").takeIf { !wasNull() },
            outputBytes = getInt("output_bytes").takeIf { !wasNull() },
            message = getString("message")
        )
    }

    private fun ResultSet.toPersistedWorkerTelemetry(): PersistedWorkerTelemetry {
        return PersistedWorkerTelemetry(
            telemetryId = getLong("telemetry_id"),
            observedEpochMs = getLong("observed_epoch_ms"),
            deviceId = getString("device_id"),
            nodeId = getInt("node_id"),
            stageId = getInt("stage_id"),
            isActive = getInt("is_active") != 0,
            batteryLevel = getFloat("battery_level"),
            isCharging = getInt("is_charging") != 0,
            powerSource = getString("power_source"),
            batteryStatus = getInt("battery_status"),
            batteryTempC = getFloat("battery_temp_c").takeIf { !wasNull() },
            batteryVoltageMv = getInt("battery_voltage_mv").takeIf { !wasNull() },
            batteryCurrentUa = getLong("battery_current_ua").takeIf { !wasNull() },
            thermalStatus = getString("thermal_status"),
            appPssKb = getLong("app_pss_kb").takeIf { !wasNull() },
            appPrivateDirtyKb = getLong("app_private_dirty_kb").takeIf { !wasNull() },
            runtimeUsedMemoryKb = getLong("runtime_used_memory_kb").takeIf { !wasNull() },
            workerState = getString("worker_state")
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

    private fun PersistedRequestMetric.toAdminSnapshot(): AdminRequestMetricSnapshot {
        return AdminRequestMetricSnapshot(
            metricId = metricId,
            runId = runId,
            requestId = requestId,
            requestIndex = requestIndex,
            attempt = attempt,
            batchId = batchId,
            evalOnly = evalOnly,
            submittedEpochMs = submittedEpochMs,
            completedEpochMs = completedEpochMs,
            elapsedMs = elapsedMs,
            success = success,
            terminal = terminal,
            processedStageId = processedStageId,
            processedChunkIdx = processedChunkIdx,
            outputHiddenBytes = outputHiddenBytes,
            outputShiftLogPBytes = outputShiftLogPBytes,
            localLoss = localLoss,
            tokenCorrect = tokenCorrect,
            tokenCount = tokenCount,
            tokenAccuracy = if (tokenCount == 0) 0.0 else tokenCorrect.toDouble() / tokenCount.toDouble(),
            message = message
        )
    }

    private fun PersistedRunSummary.toAdminSnapshot(): AdminRunSummarySnapshot {
        return AdminRunSummarySnapshot(
            runId = runId,
            pipelineName = pipelineName,
            modelShards = modelShards,
            configHash = configHash,
            firstSeenEpochMs = firstSeenEpochMs,
            lastUpdatedEpochMs = lastUpdatedEpochMs,
            requestRows = requestRows,
            successRows = successRows,
            failedRows = failedRows,
            evalOnlyRows = evalOnlyRows,
            trainRows = trainRows,
            avgElapsedMs = avgElapsedMs,
            avgLoss = avgLoss,
            tokenCorrect = tokenCorrect,
            tokenCount = tokenCount,
            tokenAccuracy = tokenAccuracy
        )
    }

    private fun PersistedStageTimingMetric.toAdminSnapshot(): AdminStageTimingMetricSnapshot {
        return AdminStageTimingMetricSnapshot(
            stageMetricId = stageMetricId,
            requestEventId = requestEventId,
            runId = runId,
            requestId = requestId,
            batchId = batchId,
            chunkIdx = chunkIdx,
            stageId = stageId,
            nodeId = nodeId,
            eventType = eventType,
            eventEpochMs = eventEpochMs,
            runtime = runtime,
            method = method,
            inputCount = inputCount,
            evalOnly = evalOnly,
            optimizerStepApplied = optimizerStepApplied,
            localLoss = localLoss,
            localMs = localMs,
            inputBuildMs = inputBuildMs,
            executeMs = executeMs,
            gradientsMs = gradientsMs,
            optimizerCreateMs = optimizerCreateMs,
            optimizerStepMs = optimizerStepMs,
            outputConvertMs = outputConvertMs,
            totalMeasuredMs = totalMeasuredMs,
            forwardMs = forwardMs,
            totalStageMs = totalStageMs,
            outputBytes = outputBytes,
            message = message
        )
    }

    private companion object {
        private val KEY_VALUE_PATTERN = Regex("""([A-Za-z][A-Za-z0-9_]*)=([^,\s]+)""")
    }
}
