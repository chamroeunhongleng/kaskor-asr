Clear-Host
Write-Host "KASEKOR Live Monitor — watching train_err.txt" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to exit (training continues in background)" -ForegroundColor DarkGray
Write-Host ""

$errFile = "C:\Users\Admin\kaskor-asr-v0.0\train_err.txt"
$lastLine = ""

while ($true) {
    $line = Get-Content $errFile -Tail 1 2>$null
    if ($line -and $line -ne $lastLine) {
        $lastLine = $line
        $ts = Get-Date -Format "HH:mm:ss"

        # Parse tqdm line: step/total [elapsed<remaining, speed]
        if ($line -match "(\d+)/(\d+) \[(\S+)<(\S+),\s*([\d.]+)s/it\]") {
            $step    = [int]$Matches[1]
            $total   = [int]$Matches[2]
            $elapsed = $Matches[3]
            $remain  = $Matches[4]
            $speed   = $Matches[5]
            $pct     = [math]::Round($step / $total * 100, 1)

            $bar = "#" * [math]::Floor($pct / 5)
            $empty = "-" * (20 - [math]::Floor($pct / 5))

            Clear-Host
            Write-Host "=====================================" -ForegroundColor Cyan
            Write-Host "  KASEKOR Epoch 2 — LIVE MONITOR" -ForegroundColor Cyan
            Write-Host "=====================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host ("  Step      : {0} / {1}" -f $step, $total)
            Write-Host ("  Progress  : {0}%  [{1}{2}]" -f $pct, $bar, $empty) -ForegroundColor Green
            Write-Host ("  Elapsed   : {0}" -f $elapsed)
            Write-Host ("  Remaining : {0}" -f $remain) -ForegroundColor Yellow
            Write-Host ("  Speed     : {0} sec/step" -f $speed)
            Write-Host ""
            Write-Host ("  Updated   : {0}" -f $ts) -ForegroundColor DarkGray
            Write-Host ""
            Write-Host "  Auto-shutdown + Telegram notification on finish." -ForegroundColor DarkGray
            Write-Host "=====================================" -ForegroundColor Cyan

            if ($step -ge $total) {
                Write-Host ""
                Write-Host "  TRAINING COMPLETE!" -ForegroundColor Green
                Write-Host "  PC will shutdown in ~60 seconds." -ForegroundColor Yellow
                break
            }
        }
    }
    Start-Sleep -Seconds 3
}
