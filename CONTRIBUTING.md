# Contributing Guide

This guide explains how to customize and extend the Hindi ASR fine-tuning pipeline.

## Customization Options

### 1. Adjusting Hyperparameters

**Location:** `finetune_whisper.py` - `train()` method or `main()` function

**Common adjustments:**

```python
# For better quality (longer training)
num_train_epochs=5  # Instead of 3

# For faster iteration (testing)
num_train_epochs=1
per_device_train_batch_size=2

# For larger GPU memory (e.g., A100, V100)
per_device_train_batch_size=16
gradient_accumulation_steps=1

# For limited GPU memory (e.g., 4GB)
per_device_train_batch_size=2
gradient_accumulation_steps=8
```

### 2. Changing Data Filters

**Location:** `preprocess_data.py` - `filter_by_duration()` method

```python
# To include longer audio samples
df_filtered = preprocessor.filter_by_duration(
    df, 
    min_duration=1.0,    # Minimum 1 second
    max_duration=60.0    # Maximum 60 seconds (instead of 30)
)

# To focus on shorter samples only
df_filtered = preprocessor.filter_by_duration(
    df,
    min_duration=2.0,    # Minimum 2 seconds
    max_duration=10.0    # Maximum 10 seconds
)
```

### 3. Using Different Base Models

**Location:** `finetune_whisper.py` - `main()` function

```python
# Use Whisper-tiny (faster, lower accuracy)
model_name = "openai/whisper-tiny"

# Use Whisper-base (balanced)
model_name = "openai/whisper-base"

# Use Whisper-medium (better accuracy, slower)
model_name = "openai/whisper-medium"

# Use Whisper-large (best accuracy, very slow)
model_name = "openai/whisper-large-v2"
```

### 4. Custom Data Augmentation

Add to `preprocess_data.py` after loading audio:

```python
def augment_audio(self, audio_bytes):
    """Apply data augmentation to audio"""
    import librosa
    
    audio_io = io.BytesIO(audio_bytes)
    data, sr = sf.read(audio_io)
    
    # Time stretching
    data_stretched = librosa.effects.time_stretch(data, rate=1.1)
    
    # Pitch shifting
    data_shifted = librosa.effects.pitch_shift(data, sr=sr, n_steps=2)
    
    return data_augmented
```

### 5. Custom Evaluation Metrics

Add to `evaluate_model.py`:

```python
import evaluate

# Add Character Error Rate (CER)
cer_metric = evaluate.load("cer")

def compute_metrics(predictions, references):
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    return {
        "wer": wer * 100,
        "cer": cer * 100
    }
```

## Extension Ideas

### 1. Add Resume from Checkpoint

In `finetune_whisper.py`:

```python
def train(self, ..., resume_from_checkpoint=None):
    training_args = Seq2SeqTrainingArguments(
        # ... existing args ...
        resume_from_checkpoint=resume_from_checkpoint,
    )
```

### 2. Add Model Quantization for Deployment

Create `quantize_model.py`:

```python
from transformers import WhisperForConditionalGeneration
import torch

def quantize_model(model_path, output_path):
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    quantized_model.save_pretrained(output_path)
```

### 3. Add Multi-GPU Training Support

In `finetune_whisper.py`:

```python
training_args = Seq2SeqTrainingArguments(
    # ... existing args ...
    local_rank=-1,  # For distributed training
    ddp_find_unused_parameters=False,
)

# Run with:
# python -m torch.distributed.launch --nproc_per_node=2 finetune_whisper.py
```

### 4. Add Real-time Inference API

Create `inference_api.py`:

```python
from flask import Flask, request, jsonify
from transformers import pipeline
import soundfile as sf
import io

app = Flask(__name__)

# Load model once at startup
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="./whisper-small-hi-finetuned"
)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    # Process audio
    audio_io = io.BytesIO(audio_bytes)
    audio, sr = sf.read(audio_io)
    
    # Transcribe
    result = asr_pipeline({"array": audio, "sampling_rate": sr})
    
    return jsonify({"transcription": result["text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 5. Add Training Progress Visualization

Create `visualize_training.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def visualize_training_logs(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get training loss
    loss_events = ea.Scalars('train/loss')
    steps = [e.step for e in loss_events]
    losses = [e.value for e in loss_events]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
visualize_training_logs('./whisper-small-hi-finetuned/runs')
```

## Testing Your Changes

### Unit Testing Example

Create `tests/test_preprocessing.py`:

```python
import unittest
from preprocess_data import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor("test_data.csv")
    
    def test_clean_transcription(self):
        # Test with extra whitespace
        text = "  hello   world  "
        cleaned = self.preprocessor.clean_transcription(text)
        self.assertEqual(cleaned, "hello world")
    
    def test_validate_audio(self):
        # Test with valid audio bytes
        # Add your test implementation
        pass

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

Create `tests/test_pipeline.py`:

```python
def test_complete_pipeline():
    # Test preprocessing
    preprocessor = DataPreprocessor("test_data.csv")
    dataset = preprocessor.prepare_dataset(test_df, sample_limit=10)
    assert len(dataset) == 10
    
    # Test fine-tuning (minimal)
    fine_tuner = WhisperFineTuner()
    # Add training test with 1 epoch, small data
    
    # Test evaluation
    evaluator = WhisperEvaluator()
    # Add evaluation test
```

## Performance Optimization Tips

### 1. Faster Data Loading

```python
# In finetune_whisper.py
training_args = Seq2SeqTrainingArguments(
    # ... existing args ...
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster GPU transfer
)
```

### 2. Gradient Checkpointing (Save Memory)

```python
# In finetune_whisper.py
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

### 3. Mixed Precision Training

Already enabled by default with `fp16=True`. For newer GPUs:

```python
training_args = Seq2SeqTrainingArguments(
    # ... existing args ...
    bf16=True,  # Use bfloat16 instead of fp16 (for Ampere+ GPUs)
)
```

## Troubleshooting Common Issues

### Issue: Training is very slow

**Solutions:**
1. Increase batch size and reduce gradient accumulation
2. Use multiple GPUs with distributed training
3. Use a smaller base model (whisper-tiny or whisper-base)
4. Enable gradient checkpointing to allow larger batches

### Issue: High WER after fine-tuning

**Solutions:**
1. Train for more epochs
2. Increase training data size
3. Adjust learning rate (try 5e-6 or 2e-5)
4. Check data quality and transcription accuracy
5. Try different train/val split ratios

### Issue: Model overfitting (low train WER, high val WER)

**Solutions:**
1. Add dropout (modify model config)
2. Use more training data
3. Reduce number of epochs
4. Add data augmentation
5. Increase validation set size

## Best Practices

1. **Always validate your data first**: Run preprocessing with a small sample to check data quality
2. **Monitor training**: Use TensorBoard to track loss and WER during training
3. **Save checkpoints frequently**: Don't lose progress if training crashes
4. **Test on multiple datasets**: Don't rely only on FLEURS for evaluation
5. **Version your models**: Tag each trained model with date and config
6. **Document your changes**: Keep notes on what hyperparameters work best

## Getting Help

If you need help with customization:
1. Check the official Transformers documentation
2. Review the Whisper paper for understanding model architecture
3. Search for similar issues on Hugging Face forums
4. Experiment with small changes first before major modifications

## Contributing Back

If you develop useful extensions:
1. Document your changes clearly
2. Test thoroughly
3. Consider sharing as a pull request
4. Include examples and usage instructions
