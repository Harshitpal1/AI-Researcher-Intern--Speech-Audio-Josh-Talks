# Hindi ASR Fine-tuning with Whisper-small

This repository contains the complete pipeline for fine-tuning OpenAI's Whisper-small model on Hindi Automatic Speech Recognition (ASR) data and evaluating it on the FLEURS Hindi test dataset.

## Project Overview

This project implements:
1. **Data Preprocessing**: Process ~10 hours of Hindi ASR training data
2. **Model Fine-tuning**: Fine-tune Whisper-small on Hindi ASR data
3. **Evaluation**: Evaluate both pretrained and fine-tuned models on FLEURS Hindi test set
4. **WER Reporting**: Generate structured WER (Word Error Rate) comparison results

## Dataset

### Training Dataset
- **Source**: Hindi ASR training data (~10 hours)
- **Format**: Audio files with transcription metadata
- **Schema**:
  - `user_id`: Speaker/user identifier (anonymized)
  - `recording_id`: Unique recording identifier
  - `language`: Language label (e.g., "hi" for Hindi)
  - `duration`: Audio duration in seconds
  - `rec_url_gcp`: URL to audio file on cloud storage
  - `transcription_url`: URL to ground-truth transcription
  - `metadata_url`: URL to additional metadata

### Test Dataset
- **Source**: FLEURS Hindi test dataset (from Google)
- **Purpose**: Evaluation of pretrained and fine-tuned models

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 10GB+ disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Harshitpal1/AI-Researcher-Intern--Speech-Audio-Josh-Talks.git
cd AI-Researcher-Intern--Speech-Audio-Josh-Talks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

The preprocessing script handles:
- Loading audio files and transcriptions from URLs
- Data validation and cleaning
- Duration-based filtering
- Dataset preparation for training

**Run preprocessing:**
```bash
python preprocess_data.py
```

**What the preprocessing does:**
- Loads dataset metadata from CSV file
- Downloads transcriptions from provided URLs
- Cleans and validates transcription text
- Filters samples by duration (1-30 seconds recommended)
- Creates HuggingFace Dataset format
- Saves preprocessed data to disk

**Configuration:**
- Modify `data_path` in the script to point to your dataset CSV
- Adjust `min_duration` and `max_duration` for filtering
- Set `sample_limit` to None to process all data (or a number for testing)

### 2. Model Fine-tuning

The fine-tuning script:
- Loads preprocessed dataset
- Configures Whisper-small for Hindi ASR
- Fine-tunes with appropriate hyperparameters
- Saves the fine-tuned model

**Run fine-tuning:**
```bash
python finetune_whisper.py
```

**Training configuration:**
- Model: `openai/whisper-small`
- Language: Hindi (`hi`)
- Task: Transcribe
- Default hyperparameters:
  - Epochs: 3
  - Batch size: 4 (per device)
  - Gradient accumulation: 4 steps
  - Learning rate: 1e-5
  - Warmup steps: 100
  - FP16 training: Enabled (if GPU available)

**Outputs:**
- Fine-tuned model saved to `./whisper-small-hi-finetuned/`
- Training logs in TensorBoard format
- Checkpoint saves every 100 steps

### 3. Model Evaluation

The evaluation script:
- Loads FLEURS Hindi test dataset
- Evaluates pretrained Whisper-small baseline
- Evaluates fine-tuned model
- Compares WER and generates results table

**Run evaluation:**
```bash
python evaluate_model.py
```

**Evaluation process:**
- Downloads FLEURS Hindi test set automatically
- Runs inference on both models
- Calculates Word Error Rate (WER)
- Generates comparison table
- Saves results to CSV

**Outputs:**
- WER comparison table (printed to console)
- `evaluation_results.csv`: Detailed results
- Example predictions from both models

## Results Format

The evaluation generates a structured table with:

| Model | WER (%) | Samples Evaluated | Absolute Improvement (pp) | Relative Improvement (%) |
|-------|---------|-------------------|---------------------------|-------------------------|
| Whisper-small (Pretrained) | XX.XX | XXX | - | - |
| Whisper-small (Fine-tuned) | XX.XX | XXX | XX.XX | XX.XX |

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── preprocess_data.py                 # Data preprocessing script
├── finetune_whisper.py               # Model fine-tuning script
├── evaluate_model.py                 # Evaluation script
├── FT Data - Google Sheets.pdf       # Training data reference
├── FT Result - Google Sheets.pdf     # Results reference
└── Task Assignment _ AI Researcher Intern- Speech & Audio _ Josh Talks - Google Docs.pdf
```

## Implementation Details

### Data Preprocessing
- **Audio validation**: Checks audio file integrity and sample rates
- **Transcription cleaning**: Removes extra whitespace, normalizes text
- **Duration filtering**: Removes very short (<1s) and very long (>30s) samples
- **Format conversion**: Converts to HuggingFace Dataset format with Audio feature

### Fine-tuning Strategy
- **Base model**: Whisper-small (244M parameters)
- **Approach**: Full model fine-tuning (not LoRA or adapter-based)
- **Language-specific**: Configured for Hindi transcription
- **Optimization**: Mixed precision (FP16) training for efficiency
- **Validation**: 10% of data held out for validation during training

### Evaluation Methodology
- **Dataset**: FLEURS Hindi test set (standardized benchmark)
- **Metric**: Word Error Rate (WER) - industry standard for ASR
- **Comparison**: Direct comparison with pretrained baseline
- **Reproducibility**: Fixed random seeds and deterministic evaluation

## Hardware Requirements

### Minimum (CPU-only):
- CPU: 4+ cores
- RAM: 16GB
- Disk: 10GB

### Recommended (GPU):
- GPU: 8GB+ VRAM (e.g., RTX 3070, V100)
- CPU: 8+ cores
- RAM: 32GB
- Disk: 20GB

## Troubleshooting

### Common Issues

1. **Out of Memory during training:**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Disable FP16 training

2. **Dataset download errors:**
   - Check internet connection
   - Verify URL accessibility
   - Check available disk space

3. **CUDA errors:**
   - Update CUDA drivers
   - Verify PyTorch CUDA compatibility
   - Try CPU-only mode

## Dependencies

Key dependencies:
- `torch>=2.0.0`: PyTorch deep learning framework
- `transformers>=4.35.0`: Hugging Face transformers library
- `datasets>=2.14.0`: Hugging Face datasets library
- `evaluate>=0.4.0`: Evaluation metrics
- `jiwer>=3.0.0`: WER calculation
- `librosa>=0.10.0`: Audio processing

See `requirements.txt` for complete list.

## Notes

- The preprocessing script includes URL-based audio downloading. In production, you may want to cache audio files locally.
- Training time depends on dataset size and hardware. Expect 2-4 hours on a modern GPU for ~10 hours of audio data.
- The evaluation script can be configured to evaluate on a subset of the test set for faster iteration.
- All scripts include detailed logging for debugging and monitoring.

## License

This project is for educational and research purposes.

## Acknowledgments

- OpenAI for the Whisper model
- Google for the FLEURS dataset
- Hugging Face for the transformers library