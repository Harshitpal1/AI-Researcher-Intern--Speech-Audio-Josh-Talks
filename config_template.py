"""
Configuration Template for Hindi ASR Fine-tuning Pipeline

This file contains all configurable parameters for the pipeline.
Copy this to config.py and modify as needed.
"""

# ============================================================================
# DATA PREPROCESSING CONFIGURATION
# ============================================================================

# Path to training data CSV file
TRAINING_DATA_PATH = "training_data.csv"

# Cache directory for downloaded files
DATA_CACHE_DIR = "./data_cache"

# Duration filtering (in seconds)
MIN_AUDIO_DURATION = 1.0
MAX_AUDIO_DURATION = 30.0

# Sample limit (set to None to process all data)
PREPROCESS_SAMPLE_LIMIT = None  # or e.g., 1000 for testing

# Output directory for preprocessed data
PREPROCESSED_DATA_DIR = "./preprocessed_data"


# ============================================================================
# MODEL FINE-TUNING CONFIGURATION
# ============================================================================

# Base model
BASE_MODEL_NAME = "openai/whisper-small"

# Target language and task
LANGUAGE = "hi"  # Hindi
TASK = "transcribe"

# Output directory for fine-tuned model
FINETUNED_MODEL_DIR = "./whisper-small-hi-finetuned"

# Training hyperparameters
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
WARMUP_STEPS = 100

# Training/validation split
TRAIN_TEST_SPLIT_RATIO = 0.1  # 10% for validation

# Evaluation during training
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 100
SAVE_STEPS = 100
LOGGING_STEPS = 25

# Mixed precision training (set to False if GPU doesn't support FP16)
USE_FP16 = True

# Maximum generation length for predictions
GENERATION_MAX_LENGTH = 225

# Number of checkpoints to keep
SAVE_TOTAL_LIMIT = 2


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# FLEURS dataset configuration
FLEURS_LANGUAGE = "hi_in"  # Hindi India
FLEURS_SPLIT = "test"

# Evaluation batch size
EVAL_BATCH_SIZE = 8

# Sample limit for evaluation (set to None to evaluate on full test set)
EVAL_SAMPLE_LIMIT = None  # or e.g., 50 for quick testing

# Output file for evaluation results
EVALUATION_RESULTS_FILE = "evaluation_results.csv"


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

# Device selection (None = auto-detect, "cuda", or "cpu")
DEVICE = None

# Number of CPU workers for data loading
NUM_WORKERS = 4

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
LOG_LEVEL = "INFO"


# ============================================================================
# NOTES
# ============================================================================

"""
Usage:
1. Copy this file to config.py
2. Modify parameters as needed
3. Import in your scripts: import config

Example:
    import config
    model = WhisperFineTuner(model_name=config.BASE_MODEL_NAME)

Example modifications for different scenarios:

QUICK TESTING:
- Set PREPROCESS_SAMPLE_LIMIT = 100
- Set NUM_TRAIN_EPOCHS = 1
- Set EVAL_SAMPLE_LIMIT = 50
- Set PER_DEVICE_TRAIN_BATCH_SIZE = 2

LOW MEMORY (4GB GPU):
- Set PER_DEVICE_TRAIN_BATCH_SIZE = 2
- Set GRADIENT_ACCUMULATION_STEPS = 8
- Set USE_FP16 = True

HIGH MEMORY (24GB+ GPU):
- Set PER_DEVICE_TRAIN_BATCH_SIZE = 16
- Set GRADIENT_ACCUMULATION_STEPS = 1
- Set EVAL_BATCH_SIZE = 32

PRODUCTION:
- Set PREPROCESS_SAMPLE_LIMIT = None
- Set NUM_TRAIN_EPOCHS = 5
- Set EVAL_SAMPLE_LIMIT = None
- Adjust batch size based on your GPU
"""
