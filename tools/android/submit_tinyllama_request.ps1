param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 50051,
    [string]$RequestId = "",
    [string]$Model = "tinyllama",
    [int]$SeqLen = 64,
    [int]$BatchSize = 1,
    [int]$ChunkIdx = 0
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RequestId)) {
    $RequestId = "tinyllama-mainline-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

Write-Host "Submitting requestId=$RequestId model=$Model seqLen=$SeqLen batchSize=$BatchSize chunkIdx=$ChunkIdx"
.\gradlew.bat :coordinator:runSubmitDemo --args="$HostName $Port $RequestId $Model $SeqLen $BatchSize $ChunkIdx"

Write-Host ""
Write-Host "Request detail:"
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:18080/api/v1/requests/$RequestId`?eventLimit=50" |
    Select-Object -ExpandProperty Content
