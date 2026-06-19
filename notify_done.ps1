# Wait for epoch 3 to finish
while (-not (Select-String -Path "C:\Users\Admin\kaskor-asr-v0.0\train_log.txt" -Pattern "train_runtime" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 20
}

Start-Sleep -Seconds 3

# Get result line
$result = (Select-String -Path "C:\Users\Admin\kaskor-asr-v0.0\train_log.txt" -Pattern "train_runtime").Line | Select-Object -Last 1

# Send Telegram (reads from .env — see .env.example)
Get-Content "$PSScriptRoot\.env" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
$token = $env:KASEKOR_TG_TOKEN
$chatId = $env:KASEKOR_TG_CHAT
$message = "Epoch 3 DONE!`n`n$result`n`nPC is shutting down. Rest well Hongleng!"

$body = @{ chat_id = $chatId; text = $message } | ConvertTo-Json
Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" -Method Post -Body $body -ContentType "application/json"
