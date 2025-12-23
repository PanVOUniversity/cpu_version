$container = "detectron2-cpu"

$validationPath = "/home/appuser/detectron2_repo/validation"
$imagesPath = "/home/appuser/detectron2_repo/validation/images"
$outputPath = "/home/appuser/detectron2_repo/validation/output"
$configPath = "/home/appuser/detectron2_repo/validation/config"
$dataPath = "/home/appuser/detectron2_repo/validation/data"
$cpuVersionPath = "/tmp/cpu-version"

# Get Desktop path
$desktopPath = [Environment]::GetFolderPath("Desktop")
$cpuDataPath = Join-Path $desktopPath "cpu\data"

Write-Host "Removing old cpu-version folder in container..." -ForegroundColor Yellow
docker exec $container bash -lc "sudo rm -rf $cpuVersionPath"

Write-Host "Copying only necessary files to container..." -ForegroundColor Yellow

# Check if model_final.pth is a Git LFS pointer and pull if needed
$modelFile = Get-Item "model_final.pth" -ErrorAction SilentlyContinue
if ($modelFile) {
    $firstLine = Get-Content "model_final.pth" -TotalCount 1 -ErrorAction SilentlyContinue
    if ($firstLine -match "version https://git-lfs") {
        Write-Host "model_final.pth is a Git LFS pointer. Pulling actual file..." -ForegroundColor Yellow
        git lfs pull
        $modelFile = Get-Item "model_final.pth"
    }
    $fileSizeMB = [math]::Round($modelFile.Length / 1MB, 2)
    Write-Host "model_final.pth size: $fileSizeMB MB" -ForegroundColor Cyan
    if ($fileSizeMB -lt 1) {
        Write-Host "WARNING: model_final.pth seems too small. Make sure Git LFS file is pulled." -ForegroundColor Red
    }
}

# Create temp directory
docker exec $container bash -lc "sudo mkdir -p $cpuVersionPath && sudo chown -R appuser $cpuVersionPath"

# Copy only needed files (excluding .git)
docker cp inference.py "$($container):$cpuVersionPath/inference.py"
docker cp model_final.pth "$($container):$cpuVersionPath/model_final.pth"
docker cp config "$($container):$cpuVersionPath/config"

# Copy images if exists
if (Test-Path "images") {
    docker cp images "$($container):$cpuVersionPath/images"
}

# Copy cpu/data from Desktop if exists
if (Test-Path $cpuDataPath) {
    Write-Host "Copying cpu/data from Desktop to container..." -ForegroundColor Yellow
    docker cp $cpuDataPath "$($container):$cpuVersionPath/data"
} else {
    Write-Host "Warning: cpu/data folder not found at $cpuDataPath" -ForegroundColor Yellow
}

# Also check for local data folder in current directory
if (Test-Path "data") {
    Write-Host "Copying local data folder to container..." -ForegroundColor Yellow
    docker cp data "$($container):$cpuVersionPath/data"
}

# Fix permissions after copying (docker cp may set root ownership)
docker exec $container bash -lc "sudo chown -R appuser $cpuVersionPath"

Write-Host "Cleaning and creating validation directories..." -ForegroundColor Yellow
# Create directories if they don't exist (don't remove mounted volumes)
docker exec $container bash -lc "sudo mkdir -p $validationPath $imagesPath $outputPath $configPath $dataPath && sudo chown -R appuser $validationPath"

Write-Host "Moving files to validation..." -ForegroundColor Yellow
# Use sudo cp to ensure we can write, then fix ownership
docker exec $container bash -lc "sudo cp $cpuVersionPath/inference.py $validationPath/inference.py"
docker exec $container bash -lc "sudo cp $cpuVersionPath/model_final.pth $validationPath/model_final.pth"
docker exec $container bash -lc "sudo cp -r $cpuVersionPath/config $validationPath/config"

# Copy data folder if it exists
Write-Host "Copying data folder to validation..." -ForegroundColor Yellow
docker exec $container bash -lc "if [ -d `"$cpuVersionPath/data`" ]; then sudo cp -r $cpuVersionPath/data/. $dataPath/ 2>/dev/null; fi"

# Copy images from cpu/data/images to validation/images
Write-Host "Copying images from cpu/data/images to validation/images..." -ForegroundColor Yellow
docker exec $container bash -lc "if [ -d `"$cpuVersionPath/data/images`" ]; then sudo cp -r $cpuVersionPath/data/images/. $imagesPath/ 2>/dev/null; fi"

# Verify model file size in container
Write-Host "Verifying model file in container..." -ForegroundColor Yellow
docker exec $container bash -lc "ls -lh $validationPath/model_final.pth"

Write-Host "Moving images folder (local images if exists)..." -ForegroundColor Yellow
docker exec $container bash -lc "if [ -d `"$cpuVersionPath/images`" ]; then sudo cp -r $cpuVersionPath/images/. $imagesPath/ 2>/dev/null; fi"

# Verify images were copied
Write-Host "Verifying images in container..." -ForegroundColor Yellow
docker exec $container bash -lc "ls -lh $imagesPath/ | head -10"

# Fix ownership of all copied files
docker exec $container bash -lc "sudo chown -R appuser $validationPath"

Write-Host "Cleaning up temporary cpu-version folder..." -ForegroundColor Yellow
docker exec $container bash -lc "sudo rm -rf $cpuVersionPath"

Write-Host "Done! Files synchronized." -ForegroundColor Green
Write-Host "Data folder location in container: $dataPath" -ForegroundColor Cyan
