param(
    [string[]]$Serials = @("91260221021D", "ZY22G2HC5C"),
    [string]$PackageName = "com.example.sid_trainer",
    [int]$DurationSec = 60,
    [string]$Phase = "run",
    [string]$OutDir = "debug_runs"
)

$ErrorActionPreference = "Continue"

if ($DurationSec -lt 1) {
    throw "DurationSec must be >= 1."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bundle = Join-Path $OutDir "perfetto-$timestamp"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null

adb devices -l | Out-File -Encoding utf8 (Join-Path $bundle "adb_devices.txt")

function Write-TraceConfig([string]$Path, [int]$DurationMs, [string]$PackageName) {
    $config = @"
buffers {
  size_kb: 65536
  fill_policy: RING_BUFFER
}
duration_ms: $DurationMs
data_sources {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_wakeup"
      ftrace_events: "power/cpu_frequency"
      ftrace_events: "power/cpu_idle"
      atrace_categories: "am"
      atrace_categories: "binder_driver"
      atrace_categories: "dalvik"
      atrace_categories: "freq"
      atrace_categories: "sched"
      atrace_categories: "view"
      atrace_categories: "wm"
      atrace_apps: "$PackageName"
    }
  }
}
data_sources {
  config {
    name: "linux.process_stats"
    process_stats_config {
      scan_all_processes_on_start: true
      proc_stats_poll_ms: 1000
    }
  }
}
data_sources {
  config {
    name: "android.power"
    android_power_config {
      battery_poll_ms: 1000
      battery_counters: BATTERY_COUNTER_CHARGE
      battery_counters: BATTERY_COUNTER_CURRENT
      battery_counters: BATTERY_COUNTER_CAPACITY_PERCENT
    }
  }
}
data_sources {
  config {
    name: "android.log"
    android_log_config {
      log_ids: LID_DEFAULT
      filter_tags: "SidWorkerUi"
      filter_tags: "ExecuTorchShardRunner"
      filter_tags: "GrpcManager"
      filter_tags: "AndroidRuntime"
    }
  }
}
"@
    Set-Content -Encoding ascii -Path $Path -Value $config
}

$traceJobs = @()

foreach ($serial in $Serials) {
    $deviceDir = Join-Path $bundle "$Phase-$serial"
    New-Item -ItemType Directory -Force -Path $deviceDir | Out-Null

    $configPath = Join-Path $deviceDir "perfetto_config.pbtxt"
    $deviceConfigPath = "/data/misc/perfetto-configs/sid_perfetto_config.pbtxt"
    $deviceTracePath = "/data/misc/perfetto-traces/sid_${Phase}_${timestamp}.pftrace"
    $localTracePath = Join-Path $deviceDir "trace.pftrace"
    $durationMs = $DurationSec * 1000

    Write-TraceConfig -Path $configPath -DurationMs $durationMs -PackageName $PackageName

    adb -s $serial shell getprop ro.product.model | Out-File -Encoding utf8 (Join-Path $deviceDir "model.txt")
    adb -s $serial shell pidof $PackageName | Out-File -Encoding utf8 (Join-Path $deviceDir "pid.txt")
    adb -s $serial shell mkdir -p /data/misc/perfetto-configs /data/misc/perfetto-traces |
        Out-File -Encoding utf8 (Join-Path $deviceDir "mkdir_perfetto_dirs.txt")
    adb -s $serial push $configPath $deviceConfigPath | Out-File -Encoding utf8 (Join-Path $deviceDir "push_config.txt")

    Write-Host "Recording Perfetto trace on $serial for ${DurationSec}s..."
    $outputPath = Join-Path $deviceDir "perfetto_output.txt"
    $job = Start-Job -ScriptBlock {
        param($JobSerial, $JobDeviceConfigPath, $JobDeviceTracePath)
        & adb -s $JobSerial shell perfetto --txt -c $JobDeviceConfigPath -o $JobDeviceTracePath 2>&1
    } -ArgumentList $serial, $deviceConfigPath, $deviceTracePath

    $traceJobs += [pscustomobject]@{
        Serial = $serial
        Job = $job
        DeviceDir = $deviceDir
        DeviceConfigPath = $deviceConfigPath
        DeviceTracePath = $deviceTracePath
        LocalTracePath = $localTracePath
        OutputPath = $outputPath
    }
}

foreach ($job in $traceJobs) {
    Wait-Job -Job $job.Job | Out-Null
    Receive-Job -Job $job.Job | Out-File -Encoding utf8 $job.OutputPath
    Remove-Job -Job $job.Job
}

foreach ($job in $traceJobs) {
    adb -s $job.Serial pull $job.DeviceTracePath $job.LocalTracePath |
        Out-File -Encoding utf8 (Join-Path $job.DeviceDir "pull_trace.txt")
    adb -s $job.Serial shell rm $job.DeviceConfigPath $job.DeviceTracePath |
        Out-File -Encoding utf8 (Join-Path $job.DeviceDir "cleanup.txt")

    adb -s $job.Serial shell dumpsys battery | Out-File -Encoding utf8 (Join-Path $job.DeviceDir "battery_after.txt")
    adb -s $job.Serial shell dumpsys thermalservice | Out-File -Encoding utf8 (Join-Path $job.DeviceDir "thermal_after.txt")
}

Write-Host "Perfetto traces written to $bundle"
