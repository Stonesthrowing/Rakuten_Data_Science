#!/usr/bin/env bash
set -euo pipefail

echo
echo "Rakuten dataset setup starting..."
echo

# Determine project root (one level above scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Define paths
DATA_FOLDER="${PROJECT_ROOT}/data"
RAW_FOLDER="${DATA_FOLDER}/raw"
IMAGES_FOLDER="${RAW_FOLDER}/images"
DOWNLOAD_FOLDER="${DATA_FOLDER}/_downloads"

# Create folders if they don't exist
mkdir -p "${DATA_FOLDER}"
mkdir -p "${RAW_FOLDER}"
mkdir -p "${IMAGES_FOLDER}"
mkdir -p "${DOWNLOAD_FOLDER}"
mkdir -p "${DATA_FOLDER}/processed"
mkdir -p "${DATA_FOLDER}/splits"

echo "Dataset folder structure is ready."
echo

# --------------------------------------------------
# OPTIONAL: Kaggle dataset download
# --------------------------------------------------
: '
This optional section downloads the datasets from Kaggle.

Images dataset:
arturillenseer/rakuten-product-images-ml

CSV dataset:
arturillenseer/csv-files

Standard commands:

kaggle datasets download -d arturillenseer/rakuten-product-images-ml -p data/_downloads
kaggle datasets download -d arturillenseer/csv-files -p data/_downloads
'

# --------------------------------------------------
# Extract zip files from data/_downloads
# --------------------------------------------------

echo "Checking for zip files in download folder..."

shopt_zip=false
if compgen -G "${DOWNLOAD_FOLDER}"/*.zip > /dev/null; then
  for zipfile in "${DOWNLOAD_FOLDER}"/*.zip; do
    echo "Extracting: $(basename "${zipfile}")"
    unzip -o "${zipfile}" -d "${DOWNLOAD_FOLDER}" > /dev/null
  done
  echo "Zip extraction finished."
else
  echo "No zip files found in: ${DOWNLOAD_FOLDER}"
fi

# --------------------------------------------------
# Move extracted files into data/raw
# --------------------------------------------------

echo "Organizing extracted dataset files..."

XTRAIN_TARGET="${RAW_FOLDER}/X_train.csv"
YTRAIN_TARGET="${RAW_FOLDER}/Y_train.csv"
XTEST_TARGET="${RAW_FOLDER}/X_test.csv"

find_first_file() {
  local filename="$1"
  find "${DOWNLOAD_FOLDER}" -type f \( -iname "${filename}" \) | head -n 1
}

find_first_dir() {
  local dirname="$1"
  find "${DOWNLOAD_FOLDER}" -type d \( -iname "${dirname}" \) | head -n 1
}

XTRAIN_FILE="$(find_first_file "X_train.csv")"
YTRAIN_FILE="$(find_first_file "Y_train.csv")"
XTEST_FILE="$(find_first_file "X_test.csv")"

if [[ -n "${XTRAIN_FILE}" ]]; then
  cp -f "${XTRAIN_FILE}" "${XTRAIN_TARGET}"
  echo "Copied X_train.csv to: ${XTRAIN_TARGET}"
else
  echo "X_train.csv not found."
fi

if [[ -n "${YTRAIN_FILE}" ]]; then
  cp -f "${YTRAIN_FILE}" "${YTRAIN_TARGET}"
  echo "Copied Y_train.csv to: ${YTRAIN_TARGET}"
else
  echo "Y_train.csv not found."
fi

if [[ -n "${XTEST_FILE}" ]]; then
  cp -f "${XTEST_FILE}" "${XTEST_TARGET}"
  echo "Copied X_test.csv to: ${XTEST_TARGET}"
else
  echo "X_test.csv not found."
fi

IMAGE_TRAIN_SOURCE="$(find_first_dir "image_train")"
IMAGE_TEST_SOURCE="$(find_first_dir "image_test")"

if [[ -n "${IMAGE_TRAIN_SOURCE}" ]]; then
  IMAGE_TRAIN_TARGET="${IMAGES_FOLDER}/image_train"
  mkdir -p "${IMAGE_TRAIN_TARGET}"
  cp -R "${IMAGE_TRAIN_SOURCE}"/. "${IMAGE_TRAIN_TARGET}/"
  echo "Copied image_train to: ${IMAGE_TRAIN_TARGET}"
else
  echo "image_train folder not found."
fi

if [[ -n "${IMAGE_TEST_SOURCE}" ]]; then
  IMAGE_TEST_TARGET="${IMAGES_FOLDER}/image_test"
  mkdir -p "${IMAGE_TEST_TARGET}"
  cp -R "${IMAGE_TEST_SOURCE}"/. "${IMAGE_TEST_TARGET}/"
  echo "Copied image_test to: ${IMAGE_TEST_TARGET}"
else
  echo "image_test folder not found."
fi

echo "Dataset organization step finished."
echo
echo "Current data folder structure:"
find "${DATA_FOLDER}" -maxdepth 3 | sort