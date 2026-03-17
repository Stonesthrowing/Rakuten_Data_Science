# Rakuten Data Science

Team repository for the Rakuten multimodal project.

## Project setup

We use:

- Python 3.12
- uv for dependency management
- Kaggle for dataset sharing
- GitHub for code, configuration and documentation

## Repository structure

docs/
data/
notebooks/
scripts/
src/
tests/

## Environment setup

Clone the repository and install the project environment:

git clone <repository-url>
cd Rakuten_Data_Science
uv sync

## Data setup

Raw data is not stored in Git.

The dataset is shared via Kaggle and should be downloaded locally.
Use the setup script:

.\scripts\setup_data.ps1

This script prepares the local data structure under:

data/
└── raw/

## Rules

- Do not commit raw data
- Do not commit large image files
- Do not commit secrets or Kaggle credentials
- Use notebooks for exploration
- Move reusable code into src/

## Workflow

Additional setup and workflow details are documented in:

docs/WORKFLOW.md