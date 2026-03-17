# setup_data.ps1
# Prepare dataset folder structure for the Rakuten project

Write-Host ""
Write-Host "Rakuten dataset setup starting..."
Write-Host ""

# Determine project root (one level above scripts)
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Define paths
$DataFolder = Join-Path $ProjectRoot "data"
$RawFolder = Join-Path $DataFolder "raw"
$ImagesFolder = Join-Path $RawFolder "images"
$DownloadFolder = Join-Path $DataFolder "_downloads"

# Create folders if they don't exist
$folders = @(
    $DataFolder,
    $RawFolder,
    $ImagesFolder,
    $DownloadFolder
)

foreach ($folder in $folders) {
    if (-Not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created folder: $folder"
    }
    else {
        Write-Host "Folder already exists: $folder"
    }
}

Write-Host ""
Write-Host "Dataset folder structure is ready."
Write-Host ""

# --------------------------------------------------
# OPTIONAL: Kaggle dataset download
# --------------------------------------------------
<#
This optional section downloads the dataset from Kaggle.

Dataset:
arturillenseer/rakuten-product-images-ml

Standard command (most systems):

kaggle datasets download -d arturillenseer/rakuten-product-images-ml -p data/_downloads

Fallback for restricted Windows systems:

uv run python -c "from kaggle.cli import main; main()" datasets download `
    -d arturillenseer/rakuten-product-images-ml `
    -p data/_downloads

Example:

$DownloadPath = Join-Path $ProjectRoot "data\_downloads"

uv run python -c "from kaggle.cli import main; main()" datasets download `
    -d arturillenseer/rakuten-product-images-ml `
    -p $DownloadPath
#>

# --------------------------------------------------
# Extract zip files from data/_downloads
# --------------------------------------------------

Write-Host "Checking for zip files in download folder..."

$ZipFiles = Get-ChildItem -Path $DownloadFolder -Filter "*.zip" -File -ErrorAction SilentlyContinue

if (-not $ZipFiles) {
    Write-Host "No zip files found in: $DownloadFolder"
}
else {
    foreach ($zip in $ZipFiles) {
        Write-Host "Extracting: $($zip.Name)"
        Expand-Archive -Path $zip.FullName -DestinationPath $DownloadFolder -Force
    }
    Write-Host "Zip extraction finished."
}

# --------------------------------------------------
# Move extracted files into data/raw
# --------------------------------------------------

Write-Host "Organizing extracted dataset files..."

$XTrainTarget = Join-Path $RawFolder "x_train.csv"
$YTrainTarget = Join-Path $RawFolder "y_train.csv"
$XTestTarget  = Join-Path $RawFolder "x_test.csv"

# Find CSV files
$xTrainFile = Get-ChildItem -Path $DownloadFolder -Recurse -File | Where-Object { $_.Name -match "^x_train.*\.csv$" } | Select-Object -First 1
$yTrainFile = Get-ChildItem -Path $DownloadFolder -Recurse -File | Where-Object { $_.Name -match "^y_train.*\.csv$" } | Select-Object -First 1
$xTestFile  = Get-ChildItem -Path $DownloadFolder -Recurse -File | Where-Object { $_.Name -match "^x_test.*\.csv$" }  | Select-Object -First 1

if ($null -ne $xTrainFile) {
    Copy-Item -Path $xTrainFile.FullName -Destination $XTrainTarget -Force
    Write-Host "Copied x_train to: $XTrainTarget"
}
else {
    Write-Host "x_train CSV not found."
}

if ($null -ne $yTrainFile) {
    Copy-Item -Path $yTrainFile.FullName -Destination $YTrainTarget -Force
    Write-Host "Copied y_train to: $YTrainTarget"
}
else {
    Write-Host "y_train CSV not found."
}

if ($null -ne $xTestFile) {
    Copy-Item -Path $xTestFile.FullName -Destination $XTestTarget -Force
    Write-Host "Copied x_test to: $XTestTarget"
}
else {
    Write-Host "x_test CSV not found."
}

# Find image folders
$ImageTrainingSource = Get-ChildItem -Path $DownloadFolder -Recurse -Directory | Where-Object { $_.Name -eq "image_training" } | Select-Object -First 1
$ImageTestSource     = Get-ChildItem -Path $DownloadFolder -Recurse -Directory | Where-Object { $_.Name -eq "image_test" } | Select-Object -First 1

if ($null -ne $ImageTrainingSource) {
    $ImageTrainingTarget = Join-Path $ImagesFolder "image_training"
    if (-not (Test-Path $ImageTrainingTarget)) {
        New-Item -ItemType Directory -Path $ImageTrainingTarget | Out-Null
    }
    Copy-Item -Path (Join-Path $ImageTrainingSource.FullName "*") -Destination $ImageTrainingTarget -Recurse -Force
    Write-Host "Copied image_training to: $ImageTrainingTarget"
}
else {
    Write-Host "image_training folder not found."
}

if ($null -ne $ImageTestSource) {
    $ImageTestTarget = Join-Path $ImagesFolder "image_test"
    if (-not (Test-Path $ImageTestTarget)) {
        New-Item -ItemType Directory -Path $ImageTestTarget | Out-Null
    }
    Copy-Item -Path (Join-Path $ImageTestSource.FullName "*") -Destination $ImageTestTarget -Recurse -Force
    Write-Host "Copied image_test to: $ImageTestTarget"
}
else {
    Write-Host "image_test folder not found."
}

Write-Host "Dataset organization step finished."

Write-Host ""
Write-Host "Current data folder structure:"
tree $DataFolder