package com.example.sid_trainer

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp

@Composable
internal fun WorkerConsoleScreen(
    coordinatorHost: String,
    coordinatorPort: String,
    deviceId: String,
    localServerPort: String,
    modelPath: String?,
    modelCacheSummary: String,
    routingSummary: String,
    isWorkerRunning: Boolean,
    logMessages: List<String>,
    onCoordinatorHostChange: (String) -> Unit,
    onCoordinatorPortChange: (String) -> Unit,
    onDeviceIdChange: (String) -> Unit,
    onLocalServerPortChange: (String) -> Unit,
    onImportModel: () -> Unit,
    onStartWorker: () -> Unit,
    onStopWorker: () -> Unit,
    onReady: () -> Unit
) {
    LaunchedEffect(Unit) {
        onReady()
    }

    MaterialTheme {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text("SID Worker", style = MaterialTheme.typography.headlineSmall)
            Spacer(modifier = Modifier.height(12.dp))

            OutlinedTextField(
                value = coordinatorHost,
                onValueChange = onCoordinatorHostChange,
                label = { Text("Coordinator Host") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = coordinatorPort,
                onValueChange = { onCoordinatorPortChange(it.filter(Char::isDigit)) },
                label = { Text("Coordinator Port") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = deviceId,
                onValueChange = onDeviceIdChange,
                label = { Text("Device ID") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = localServerPort,
                onValueChange = { onLocalServerPortChange(it.filter(Char::isDigit)) },
                label = { Text("Local Data Port") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text("Model: ${modelPath ?: "Auto-download if coordinator provides one"}")
            Spacer(modifier = Modifier.height(4.dp))
            Text("Model Cache:\n$modelCacheSummary", style = MaterialTheme.typography.bodySmall)
            Spacer(modifier = Modifier.height(4.dp))
            Text("Route: $routingSummary")

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Button(onClick = onImportModel) {
                    Text("Import .pte")
                }

                Button(
                    onClick = onStartWorker,
                    enabled = !isWorkerRunning,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
                ) {
                    Text("Start Worker")
                }

                Button(
                    onClick = onStopWorker,
                    enabled = isWorkerRunning,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828))
                ) {
                    Text("Stop Worker")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Text("Logs", style = MaterialTheme.typography.titleMedium)
            Spacer(modifier = Modifier.height(8.dp))

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
                    .padding(8.dp)
            ) {
                LazyColumn {
                    items(logMessages) { message ->
                        Text(
                            text = message,
                            color = Color(0xFF7CFF7C),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
    }
}
