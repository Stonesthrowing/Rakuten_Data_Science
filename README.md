# Rakuten Data Science

Multimodal machine learning project for product classification using text and images.

## Overview

This repository contains the codebase for the Rakuten project.

We focus on:

- text-based models
- image-based models
- multimodal approaches combining both

## Tech stack

- Python 3.12
- uv (dependency management)
- scikit-learn / ML libraries
- Kaggle (dataset hosting)
- GitHub (code & collaboration)

## Repository structure

docs/ → project documentation  
data/ → local dataset (not tracked by Git)  
notebooks/ → exploration and experiments  
scripts/ → setup and utility scripts  
src/ → reusable code  
tests/ → validation and tests  

## Quick start

Clone the repository:

git clone <repository-url>  
cd Rakuten_Data_Science

Install dependencies:

uv sync

Prepare data:

.\scripts\setup_data.ps1

## Data handling

- Raw data is **not stored in Git**
- Dataset is shared via Kaggle
- Data is stored locally in:

data/raw/

## Key principles

- Separation of code and data
- Reproducible environment via uv
- Clean and modular project structure

## Workflow

Detailed workflow instructions are available in:

docs/WORKFLOW.md

## Environment Setup and Running the Streamlit App

There should be only one `.venv` folder in the project root.

Make sure `streamlit` is listed in the `pyproject.toml` file under `dependencies`.

Install all project dependencies:

```bash
uv sync

Activate the virtual environment: