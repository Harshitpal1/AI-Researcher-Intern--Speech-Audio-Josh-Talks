"""
Fine-tuning Script for Whisper-small on Hindi ASR Data

This script fine-tunes the Whisper-small model on preprocessed Hindi ASR data.
"""

import os
import torch
from datasets import load_from_disk, load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech-to-text tasks
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class WhisperFineTuner:
    """Fine-tuner for Whisper model on Hindi ASR data"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "hi",
        task: str = "transcribe",
        output_dir: str = "./whisper-small-hi",
    ):
        """
        Initialize the fine-tuner
        
        Args:
            model_name: Pretrained Whisper model name
            language: Target language code
            task: Task type (transcribe or translate)
            output_dir: Directory to save fine-tuned model
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.output_dir = output_dir
        
        logger.info(f"Initializing Whisper fine-tuner with model: {model_name}")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Configure model for target language
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        # Set language and task
        self.model.generation_config.language = language
        self.model.generation_config.task = task
        
        # Load metric
        self.metric = evaluate.load("wer")
        
    def prepare_dataset(self, batch):
        """
        Prepare dataset batch for training
        
        Args:
            batch: Batch of data
            
        Returns:
            Processed batch
        """
        # Load and resample audio
        audio = batch["audio"]
        
        # Compute input features
        batch["input_features"] = self.processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Encode transcription
        batch["labels"] = self.processor.tokenizer(batch["transcription"]).input_ids
        
        return batch
    
    def compute_metrics(self, pred):
        """
        Compute evaluation metrics
        
        Args:
            pred: Predictions
            
        Returns:
            Dictionary with metrics
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        evaluation_strategy: str = "steps",
        eval_steps: int = 500,
        save_steps: int = 500,
        logging_steps: int = 100,
        fp16: bool = True,
    ):
        """
        Fine-tune the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            evaluation_strategy: Evaluation strategy
            eval_steps: Evaluation frequency
            save_steps: Model save frequency
            logging_steps: Logging frequency
            fp16: Use mixed precision training
        """
        logger.info("Starting fine-tuning...")
        
        # Prepare datasets
        logger.info("Preparing training dataset...")
        train_dataset = train_dataset.map(
            self.prepare_dataset, 
            remove_columns=train_dataset.column_names,
            num_proc=1
        )
        
        if eval_dataset:
            logger.info("Preparing evaluation dataset...")
            eval_dataset = eval_dataset.map(
                self.prepare_dataset,
                remove_columns=eval_dataset.column_names,
                num_proc=1
            )
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=fp16 and torch.cuda.is_available(),
            per_device_eval_batch_size=per_device_train_batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="wer" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
            push_to_hub=False,
            report_to=["tensorboard"],
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        logger.info("Training started...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        
        logger.info("Training completed!")
        
        return trainer


def main():
    """Main fine-tuning workflow"""
    
    # Configuration
    preprocessed_data_path = "./preprocessed_data"
    model_name = "openai/whisper-small"
    output_dir = "./whisper-small-hi-finetuned"
    
    # Check if preprocessed data exists
    if not os.path.exists(preprocessed_data_path):
        logger.error(f"Preprocessed data not found at {preprocessed_data_path}")
        logger.error("Please run preprocess_data.py first")
        return
    
    # Load preprocessed dataset
    logger.info(f"Loading preprocessed data from {preprocessed_data_path}")
    dataset = load_from_disk(preprocessed_data_path)
    
    # For demonstration, we'll use a small subset
    # In production, you would use the full dataset and split properly
    if len(dataset) > 100:
        dataset = dataset.select(range(100))
        logger.info(f"Using subset of 100 samples for demonstration")
    
    # Cast audio column to Audio feature
    # Note: This assumes audio files are accessible at the URLs
    # In production, you might need to download and cache them first
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split into train and validation
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Initialize fine-tuner
    fine_tuner = WhisperFineTuner(
        model_name=model_name,
        language="hi",
        task="transcribe",
        output_dir=output_dir,
    )
    
    # Train model
    fine_tuner.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=25,
        fp16=True,
    )
    
    logger.info(f"Model fine-tuned and saved to {output_dir}")


if __name__ == "__main__":
    main()
