package com.example.sid_trainer

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.Process
import sid.Sid

data class WorkerTelemetry(
    val batteryLevel: Float,
    val isCharging: Boolean,
    val powerSource: String,
    val batteryStatus: Int,
    val batteryTempC: Float,
    val batteryVoltageMv: Int,
    val batteryCurrentUa: Long,
    val thermalStatus: String,
    val appPssKb: Long,
    val appPrivateDirtyKb: Long,
    val runtimeUsedMemoryKb: Long
) {
    fun applyTo(builder: Sid.HeartbeatRequest.Builder, workerState: String): Sid.HeartbeatRequest.Builder {
        return builder
            .setBatteryLevel(batteryLevel)
            .setIsCharging(isCharging)
            .setPowerSource(powerSource)
            .setBatteryStatus(batteryStatus)
            .setBatteryTempC(batteryTempC)
            .setBatteryVoltageMv(batteryVoltageMv)
            .setBatteryCurrentUa(batteryCurrentUa)
            .setThermalStatus(thermalStatus)
            .setAppPssKb(appPssKb)
            .setAppPrivateDirtyKb(appPrivateDirtyKb)
            .setRuntimeUsedMemoryKb(runtimeUsedMemoryKb)
            .setWorkerState(workerState)
    }
}

object WorkerTelemetryReader {
    fun read(context: Context): WorkerTelemetry {
        val appContext = context.applicationContext
        val battery = appContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val status = battery?.getIntExtra(BatteryManager.EXTRA_STATUS, BatteryManager.BATTERY_STATUS_UNKNOWN)
            ?: BatteryManager.BATTERY_STATUS_UNKNOWN
        val plugged = battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val tempTenthsC = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        val voltageMv = battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0

        val batteryManager = appContext.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
        val currentUa = batteryManager
            ?.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            ?.takeIf { it != Long.MIN_VALUE }
            ?: 0L

        val memoryInfo = (appContext.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager)
            ?.getProcessMemoryInfo(intArrayOf(Process.myPid()))
            ?.firstOrNull()
        val runtime = Runtime.getRuntime()
        val runtimeUsedMemoryKb = (runtime.totalMemory() - runtime.freeMemory()) / 1024L

        return WorkerTelemetry(
            batteryLevel = if (level >= 0 && scale > 0) level.toFloat() * 100f / scale.toFloat() else -1f,
            isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL ||
                plugged != 0,
            powerSource = pluggedSource(plugged),
            batteryStatus = status,
            batteryTempC = if (tempTenthsC == 0) 0f else tempTenthsC.toFloat() / 10f,
            batteryVoltageMv = voltageMv,
            batteryCurrentUa = currentUa,
            thermalStatus = thermalStatus(appContext),
            appPssKb = memoryInfo?.totalPss?.toLong() ?: 0L,
            appPrivateDirtyKb = memoryInfo?.totalPrivateDirty?.toLong() ?: 0L,
            runtimeUsedMemoryKb = runtimeUsedMemoryKb
        )
    }

    private fun pluggedSource(plugged: Int): String {
        return when (plugged) {
            BatteryManager.BATTERY_PLUGGED_AC -> "AC"
            BatteryManager.BATTERY_PLUGGED_USB -> "USB"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "WIRELESS"
            else -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && plugged == BatteryManager.BATTERY_PLUGGED_DOCK) {
                "DOCK"
            } else {
                "BATTERY"
            }
        }
    }

    private fun thermalStatus(context: Context): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return ""
        }
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return ""
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
            else -> "UNKNOWN"
        }
    }
}
