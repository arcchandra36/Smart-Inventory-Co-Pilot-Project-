# Quick Recovery Script for Inventory App
# Run this if your main app.py gets corrupted

Write-Host "=== INVENTORY APP RECOVERY TOOL ==="
Write-Host ""

# Check if main app exists
if (Test-Path "src\app.py") {
    Write-Host "Current app.py found. Creating emergency backup..."
    Copy-Item "src\app.py" "src\app_emergency_backup.py"
}

# Show available backups
Write-Host "Available backup files:"
Write-Host "1. app_backup_final.py (Latest working version)"
Write-Host "2. app_backup_v2.py (Previous version)"
Write-Host "3. app_backup_v1.py (Original version)"

# Get available timestamped backups
if (Test-Path "backups") {
    $timestampedBackups = Get-ChildItem "backups\app_*.py" | Sort-Object LastWriteTime -Descending
    $count = 4
    foreach ($backup in $timestampedBackups) {
        Write-Host "$count. $($backup.Name) (Created: $($backup.LastWriteTime))"
        $count++
    }
}

Write-Host ""
$choice = Read-Host "Which backup would you like to restore? (1-3, or filename)"

switch ($choice) {
    "1" { 
        Copy-Item "src\app_backup_final.py" "src\app.py"
        Write-Host "✅ Restored from app_backup_final.py"
    }
    "2" { 
        Copy-Item "src\app_backup_v2.py" "src\app.py"
        Write-Host "✅ Restored from app_backup_v2.py"
    }
    "3" { 
        Copy-Item "src\app_backup_v1.py" "src\app.py"
        Write-Host "✅ Restored from app_backup_v1.py"
    }
    default {
        if (Test-Path "backups\$choice") {
            Copy-Item "backups\$choice" "src\app.py"
            Write-Host "✅ Restored from $choice"
        } else {
            Write-Host "❌ File not found. Please check filename and try again."
        }
    }
}

Write-Host ""
Write-Host "Recovery complete! Test your app with:"
Write-Host "python -m streamlit run src/app.py"
