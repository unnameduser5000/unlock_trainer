package com.example.sid_trainer

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf

class MainActivity : ComponentActivity() {

    companion object {
        private const val UI_LOG_TAG = "SidWorkerUi"
        private const val EXTRA_COORDINATOR_HOST = "sid.coordinator_host"
        private const val EXTRA_COORDINATOR_PORT = "sid.coordinator_port"
        private const val EXTRA_DEVICE_ID = "sid.device_id"
        private const val EXTRA_LOCAL_PORT = "sid.local_port"
        private const val EXTRA_AUTO_START = "sid.auto_start"
    }

    private val logMessages = mutableStateListOf<String>()
    private val isWorkerRunning = mutableStateOf(false)
    private val modelFilePath = mutableStateOf<String?>(null)
    private val modelCacheSummary = mutableStateOf("No shard prepared")
    private val coordinatorHost = mutableStateOf("192.168.1.10")
    private val coordinatorPort = mutableStateOf("50051")
    private val deviceId = mutableStateOf(defaultDeviceId())
    private val localServerPort = mutableStateOf("26052")
    private val routingSummary = mutableStateOf("Not registered")
    private var autoStartRequested = false
    private var autoStartOnLaunch = false
    private val workerController by lazy {
        WorkerController(
            context = applicationContext,
            onRunningChanged = { isWorkerRunning.value = it },
            onModelPathChanged = { modelFilePath.value = it },
            onModelCacheChanged = { modelCacheSummary.value = it },
            onRoutingChanged = { routingSummary.value = it },
            localLog = ::appendLog
        )
    }

    private val pickModelLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { importModel(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyLaunchOverrides(intent)
        setContent {
            WorkerConsoleScreen(
                coordinatorHost = coordinatorHost.value,
                coordinatorPort = coordinatorPort.value,
                deviceId = deviceId.value,
                localServerPort = localServerPort.value,
                modelPath = modelFilePath.value,
                modelCacheSummary = modelCacheSummary.value,
                routingSummary = routingSummary.value,
                isWorkerRunning = isWorkerRunning.value,
                logMessages = logMessages,
                onCoordinatorHostChange = { coordinatorHost.value = it },
                onCoordinatorPortChange = { coordinatorPort.value = it },
                onDeviceIdChange = { deviceId.value = it },
                onLocalServerPortChange = { localServerPort.value = it },
                onImportModel = { pickModelLauncher.launch("*/*") },
                onStartWorker = ::startWorker,
                onStopWorker = ::stopWorker,
                onReady = ::handleConsoleReady
            )
        }
        appendLog("SID worker ready.")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        applyLaunchOverrides(intent)
        appendLog("Launch options updated from adb intent.")
        if (autoStartOnLaunch && !isWorkerRunning.value) {
            startWorker()
        }
    }

    private fun handleConsoleReady() {
        if (autoStartRequested) {
            return
        }
        autoStartRequested = true
        if (autoStartOnLaunch) {
            appendLog(
                "Auto-starting worker with coordinator=${coordinatorHost.value}:${coordinatorPort.value}, " +
                    "deviceId=${deviceId.value}, localPort=${localServerPort.value}"
            )
            startWorker()
        } else {
            appendLog("Auto-start disabled. Configure the fields and press Start Worker, or launch with sid.auto_start=true.")
        }
    }

    private fun importModel(uri: Uri) {
        workerController.importModel(uri)
    }

    private fun startWorker() {
        val parsedCoordinatorPort = coordinatorPort.value.toIntOrNull()
        val parsedLocalPort = localServerPort.value.toIntOrNull()
        if (parsedCoordinatorPort == null || parsedLocalPort == null) {
            appendLog("Coordinator port and local data port must be valid numbers.")
            return
        }

        workerController.start(
            WorkerStartConfig(
                coordinatorHost = coordinatorHost.value,
                coordinatorPort = parsedCoordinatorPort,
                deviceId = deviceId.value,
                localServerPort = parsedLocalPort
            )
        )
    }

    private fun stopWorker() {
        workerController.stop()
    }

    private fun applyLaunchOverrides(intent: Intent?) {
        val extras = intent?.extras ?: return
        val applied = mutableListOf<String>()

        extras.readStringExtra(EXTRA_COORDINATOR_HOST)?.let {
            coordinatorHost.value = it
            applied += "coordinatorHost=$it"
        }
        extras.readPortExtra(EXTRA_COORDINATOR_PORT)?.let {
            coordinatorPort.value = it
            applied += "coordinatorPort=$it"
        }
        extras.readStringExtra(EXTRA_DEVICE_ID)?.let {
            deviceId.value = it
            applied += "deviceId=$it"
        }
        extras.readPortExtra(EXTRA_LOCAL_PORT)?.let {
            localServerPort.value = it
            applied += "localPort=$it"
        }
        extras.readBooleanExtra(EXTRA_AUTO_START)?.let {
            autoStartOnLaunch = it
            applied += "autoStart=$it"
        }

        if (applied.isNotEmpty()) {
            appendLog("Launch options applied: ${applied.joinToString(", ")}")
        }
    }

    private fun Bundle.readStringExtra(key: String): String? {
        return get(key)?.toString()?.trim()?.takeIf { it.isNotEmpty() }
    }

    private fun Bundle.readPortExtra(key: String): String? {
        return readStringExtra(key)?.filter { it.isDigit() }?.takeIf { it.isNotEmpty() }
    }

    private fun Bundle.readBooleanExtra(key: String): Boolean? {
        return when (val value = get(key)) {
            is Boolean -> value
            is String -> value.equals("true", ignoreCase = true) || value == "1" || value.equals("yes", ignoreCase = true)
            is Number -> value.toInt() != 0
            else -> null
        }
    }

    private fun appendLog(message: String, throwable: Throwable? = null) {
        when {
            throwable != null ||
                message.contains("failed", ignoreCase = true) ||
                message.contains("crashed", ignoreCase = true) ||
                message.contains("rejecting", ignoreCase = true) -> {
                if (throwable == null) {
                    Log.e(UI_LOG_TAG, message)
                } else {
                    Log.e(UI_LOG_TAG, message, throwable)
                }
            }

            message.contains("stopping", ignoreCase = true) ||
                message.contains("stopped", ignoreCase = true) ||
                message.contains("cancelled", ignoreCase = true) -> {
                Log.w(UI_LOG_TAG, message)
            }

            else -> {
                Log.i(UI_LOG_TAG, message)
            }
        }
        runOnUiThread {
            logMessages.add(message)
            if (logMessages.size > 200) {
                logMessages.removeAt(0)
            }
        }
    }

    private fun defaultDeviceId(): String {
        val model = Build.MODEL.orEmpty().replace(' ', '_')
        return model.ifBlank { "android_worker" }
    }
}
