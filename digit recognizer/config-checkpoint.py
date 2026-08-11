"""
Central configuration for paths used across the project.

Every other module imports paths from here instead of hard-coding
relative paths like "../models" — that's what broke things originally:
predict scripts only worked if you launched Python from a specific folder.
"""
from pathlib import Path

# Project root = the folder that contains app/, models/, notebooks/, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
SVM_MODEL_PATH = MODELS_DIR / "svm_hog_model.pkl"

DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
