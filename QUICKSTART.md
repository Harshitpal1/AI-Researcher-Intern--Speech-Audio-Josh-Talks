# Quick Start Guide

This guide will help you get started with the Hindi ASR fine-tuning pipeline.

## Prerequisites

1. Python 3.8+ installed
2. Access to Hindi ASR training data (CSV file with required schema)
3. GPU with 8GB+ VRAM (recommended) or CPU with 16GB+ RAM

## Step-by-Step Instructions

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/Harshitpal1/AI-Researcher-Intern--Speech-Audio-Josh-Talks.git
cd AI-Researcher-Intern--Speech-Audio-Josh-Talks

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your training data should be in CSV format with the following columns:
- `user_id`: Speaker identifier
- `recording_id`: Unique recording ID
- `language`: Language code (e.g., "hi" for Hindi)
- `duration`: Audio duration in seconds
- `rec_url_gcp`: URL to audio file
- `transcription_url`: URL to transcription text file
- `metadata_url`: URL to metadata (optional)

See `training_data_template.csv` for an example format.

Place your training data CSV file in the project directory and name it `training_data.csv` (or update the path in `preprocess_data.py`).

### 3. Run Data Preprocessing

```bash
python preprocess_data.py
```

This will:
- Load your training data
- Download and validate audio/transcriptions
- Clean and filter the data
- Save preprocessed data to `./preprocessed_data/`

**Expected output:**
- Dataset statistics (total samples, duration, etc.)
- Preprocessed dataset saved to disk

### 4. Fine-tune Whisper-small

```bash
python finetune_whisper.py
```

This will:
- Load the preprocessed data
- Fine-tune Whisper-small model on Hindi data
- Save checkpoints during training
- Save final model to `./whisper-small-hi-finetuned/`

**Expected duration:** 2-4 hours (depending on dataset size and hardware)

**Monitor training:**
```bash
# In a separate terminal
tensorboard --logdir ./whisper-small-hi-finetuned
```

### 5. Evaluate Models

```bash
python evaluate_model.py
```

This will:
- Download FLEURS Hindi test dataset
- Evaluate pretrained Whisper-small baseline
- Evaluate your fine-tuned model
- Generate WER comparison table
- Save results to `evaluation_results.csv`

**Expected output:**
```
==============================================================
WER Comparison Table
==============================================================
                        Model  WER (%)  Samples Evaluated  ...
  Whisper-small (Pretrained)    XX.XX                XXX  ...
   Whisper-small (Fine-tuned)    XX.XX                XXX  ...
==============================================================
```

## Troubleshooting

### Issue: Out of memory during training

**Solution:**
Edit `finetune_whisper.py` and reduce batch size:
```python
per_device_train_batch_size=2,  # Reduced from 4
gradient_accumulation_steps=8,  # Increased from 4
```

### Issue: Dataset download fails

**Solution:**
- Check internet connection
- Verify URLs are accessible
- Check firewall settings
- Ensure sufficient disk space

### Issue: CUDA not available

**Solution:**
Training will automatically use CPU if CUDA is not available. To explicitly disable GPU:
```python
# In finetune_whisper.py, modify:
fp16=False,  # Disable mixed precision
```

Or set environment variable:
```bash
CUDA_VISIBLE_DEVICES="" python finetune_whisper.py
```

## Testing with Small Dataset

For quick testing, limit the number of samples:

In `preprocess_data.py`:
```python
dataset = preprocessor.prepare_dataset(df_filtered, sample_limit=100)
```

In `finetune_whisper.py`:
```python
if len(dataset) > 100:
    dataset = dataset.select(range(100))
```

In `evaluate_model.py`:
```python
max_samples = 50  # Already set by default
```

## Next Steps

After successful evaluation:
1. Analyze the WER comparison results
2. Review example predictions for quality assessment
3. Experiment with hyperparameters for better results
4. Scale up to full dataset for production model

## Getting Help

If you encounter issues:
1. Check the detailed logs in console output
2. Review the main README.md for detailed documentation
3. Ensure all prerequisites are met
4. Verify your training data format matches the template

## Tips for Better Results

1. **Data Quality:**
   - Ensure transcriptions are accurate
   - Remove noisy or low-quality audio samples
   - Balance speaker diversity

2. **Training:**
   - Use more epochs for larger datasets
   - Adjust learning rate if training is unstable
   - Monitor validation WER during training

3. **Evaluation:**
   - Test on diverse audio conditions
   - Compare with other benchmarks
   - Analyze error patterns for insights
