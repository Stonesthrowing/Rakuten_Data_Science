# setup_data.ps1
# Prepare dataset folder structure for the Rakuten project

$ErrorActionPreference = "Stop"

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

#This optional section downloads the datasets from Kaggle.

#Images dataset:
#arturillenseer/rakuten-product-images-ml

#CSV dataset:
#arturillenseer/csv-files

#Standard commands (most systems):

#kaggle datasets download -d arturillenseer/rakuten-product-images-ml -p data/_downloads
#kaggle datasets download -d arturillenseer/csv-files -p data/_downloads

#Fallback for restricted Windows systems:

uv run python -c "from kaggle.cli import main; main()" datasets download `
    -d arturillenseer/rakuten-product-images-ml `
    -p data/_downloads

uv run python -c "from kaggle.cli import main; main()" datasets download `
    -d arturillenseer/csv-files `
    -p data/_downloads

#Example:

$DownloadPath = Join-Path $ProjectRoot "data\_downloads"

uv run python -c "from kaggle.cli import main; main()" datasets download `
    -d arturillenseer/rakuten-product-images-ml `
    -p $DownloadPath

uv run python -c "from kaggle.cli import main; main()" datasets download `
    -d arturillenseer/csv-files `
    -p $DownloadPath


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

$XTrainTarget = Join-Path $RawFolder "X_train.csv"
$YTrainTarget = Join-Path $RawFolder "Y_train.csv"
$XTestTarget  = Join-Path $RawFolder "X_test.csv"

# Find CSV files
$xTrainFile = Get-ChildItem -Path $DownloadFolder -Recurse -File | Where-Object { $_.Name -ieq "X_train.csv" } | Select-Object -First 1
$yTrainFile = Get-ChildItem -Path $DownloadFolder -Recurse -File | Where-Object { $_.Name -ieq "Y_train.csv" } | Select-Object -First 1
$xTestFile  = Get-ChildItem -Path $DownloadFolder -Recurse -File | Where-Object { $_.Name -ieq "X_test.csv" }  | Select-Object -First 1

if ($null -ne $xTrainFile) {
    Copy-Item -Path $xTrainFile.FullName -Destination $XTrainTarget -Force
    Write-Host "Copied X_train.csv to: $XTrainTarget"
}
else {
    Write-Host "X_train.csv not found."
}

if ($null -ne $yTrainFile) {
    Copy-Item -Path $yTrainFile.FullName -Destination $YTrainTarget -Force
    Write-Host "Copied Y_train.csv to: $YTrainTarget"
}
else {
    Write-Host "Y_train.csv not found."
}

if ($null -ne $xTestFile) {
    Copy-Item -Path $xTestFile.FullName -Destination $XTestTarget -Force
    Write-Host "Copied X_test.csv to: $XTestTarget"
}
else {
    Write-Host "X_test.csv not found."
}

# Find image folders
$ImageTrainSource = Get-ChildItem -Path $DownloadFolder -Recurse -Directory | Where-Object { $_.Name -eq "image_train" } | Select-Object -First 1
$ImageTestSource  = Get-ChildItem -Path $DownloadFolder -Recurse -Directory | Where-Object { $_.Name -eq "image_test" } | Select-Object -First 1

if ($null -ne $ImageTrainSource) {
    $ImageTrainTarget = Join-Path $ImagesFolder "image_train"
    if (-not (Test-Path $ImageTrainTarget)) {
        New-Item -ItemType Directory -Path $ImageTrainTarget | Out-Null
    }
    Copy-Item -Path (Join-Path $ImageTrainSource.FullName "*") -Destination $ImageTrainTarget -Recurse -Force
    Write-Host "Copied image_train to: $ImageTrainTarget"
}
else {
    Write-Host "image_train folder not found."
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