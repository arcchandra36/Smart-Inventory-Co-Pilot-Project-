# Inventory App Backup Script
# Run this script regularly to create timestamped backups

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$backupFolder = "backups"

# Create backup folder if it doesn't exist
if (!(Test-Path $backupFolder)) {
    New-Item -ItemType Directory -Path $backupFolder
    Write-Host "Created backup folder: $backupFolder"
}

# Copy main files with timestamp
Copy-Item "src\app.py" "$backupFolder\app_$timestamp.py"
Copy-Item "requirements.txt" "$backupFolder\requirements_$timestamp.txt"
Copy-Item "data\retail_store_inventory_with_vendors.csv" "$backupFolder\data_$timestamp.csv" -ErrorAction SilentlyContinue

# Create backup info file
$backupInfo = @"
Backup Created: $(Get-Date)
Files Backed Up:
- src/app.py -> app_$timestamp.py
- requirements.txt -> requirements_$timestamp.txt
- data/retail_store_inventory_with_vendors.csv -> data_$timestamp.csv

Status: ✅ Backup Complete
"@

$backupInfo | Out-File "$backupFolder\backup_info_$timestamp.txt"

Write-Host "Backup completed successfully!"
Write-Host "Files saved in: $backupFolder"
Write-Host "Timestamp: $timestamp"
