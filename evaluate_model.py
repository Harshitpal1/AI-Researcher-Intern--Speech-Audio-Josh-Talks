"""
Evaluation Script for Whisper Models on FLEURS Hindi Test Dataset

This script evaluates both pretrained Whisper-small and fine-tuned model
on the Hindi portion of FLEURS test dataset and reports WER.
"""

import os
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import evaluate
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperEvaluator:
    """Evaluator for Whisper models on FLEURS dataset"""
    
    def __init__(self, device: str = None):
        """
        Initialize the evaluator
        
        Args:
            device: Device to run evaluation on (cuda/cpu)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load WER metric
        self.wer_metric = evaluate.load("wer")
        
    def load_fleurs_dataset(self, language: str = "hi_in", split: str = "test"):
        """
        Load FLEURS dataset for Hindi
        
        Args:
            language: Language code (hi_in for Hindi)
            split: Dataset split (train/validation/test)
            
        Returns:
            FLEURS dataset
        """
        logger.info(f"Loading FLEURS dataset: {language} - {split}")
        
        try:
            dataset = load_dataset("google/fleurs", language, split=split)
            logger.info(f"Loaded {len(dataset)} samples from FLEURS {split} set")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load FLEURS dataset: {e}")
            return None
    
    def evaluate_model(
        self, 
        model_name_or_path: str, 
        dataset,
        batch_size: int = 8,
        max_samples: int = None
    ) -> Dict[str, float]:
        """
        Evaluate a Whisper model on the dataset
        
        Args:
            model_name_or_path: Model name or path to model
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate (for testing)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name_or_path}")
        
        # Load model and processor
        try:
            processor = WhisperProcessor.from_pretrained(model_name_or_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
            model.to(self.device)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"wer": -1, "error": str(e)}
        
        # Create ASR pipeline
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if self.device == "cuda" else -1,
        )
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited evaluation to {max_samples} samples")
        
        # Evaluate
        predictions = []
        references = []
        
        logger.info("Running inference...")
        for sample in tqdm(dataset, desc="Evaluating"):
            try:
                # Get audio
                audio = sample["audio"]["array"]
                sampling_rate = sample["audio"]["sampling_rate"]
                
                # Get reference transcription
                reference = sample["transcription"]
                
                # Run inference
                result = asr_pipeline(
                    {"array": audio, "sampling_rate": sampling_rate},
                    generate_kwargs={"language": "hi", "task": "transcribe"},
                )
                
                prediction = result["text"]
                
                predictions.append(prediction)
                references.append(reference)
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        # Calculate WER
        if len(predictions) > 0:
            wer = self.wer_metric.compute(predictions=predictions, references=references)
            wer_percentage = wer * 100
            
            logger.info(f"Processed {len(predictions)} samples")
            logger.info(f"WER: {wer_percentage:.2f}%")
            
            return {
                "wer": wer_percentage,
                "samples_evaluated": len(predictions),
                "predictions": predictions[:5],  # Store first 5 for inspection
                "references": references[:5]
            }
        else:
            logger.error("No samples successfully processed")
            return {"wer": -1, "error": "No samples processed"}
    
    def compare_models(
        self,
        pretrained_model: str,
        finetuned_model: str,
        dataset,
        batch_size: int = 8,
        max_samples: int = None
    ) -> pd.DataFrame:
        """
        Compare pretrained and fine-tuned models
        
        Args:
            pretrained_model: Name of pretrained model
            finetuned_model: Path to fine-tuned model
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("=" * 60)
        logger.info("Comparing Pretrained vs Fine-tuned Models")
        logger.info("=" * 60)
        
        # Evaluate pretrained model
        logger.info("\n1. Evaluating pretrained model...")
        pretrained_results = self.evaluate_model(
            pretrained_model, 
            dataset, 
            batch_size, 
            max_samples
        )
        
        # Evaluate fine-tuned model
        logger.info("\n2. Evaluating fine-tuned model...")
        finetuned_results = self.evaluate_model(
            finetuned_model, 
            dataset, 
            batch_size, 
            max_samples
        )
        
        # Create comparison table
        comparison_data = {
            "Model": [
                "Whisper-small (Pretrained)",
                "Whisper-small (Fine-tuned)"
            ],
            "WER (%)": [
                f"{pretrained_results['wer']:.2f}" if pretrained_results['wer'] >= 0 else "Error",
                f"{finetuned_results['wer']:.2f}" if finetuned_results['wer'] >= 0 else "Error"
            ],
            "Samples Evaluated": [
                pretrained_results.get('samples_evaluated', 0),
                finetuned_results.get('samples_evaluated', 0)
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate improvement
        if pretrained_results['wer'] >= 0 and finetuned_results['wer'] >= 0:
            improvement = pretrained_results['wer'] - finetuned_results['wer']
            improvement_pct = (improvement / pretrained_results['wer']) * 100 if pretrained_results['wer'] > 0 else 0
            
            logger.info("\n" + "=" * 60)
            logger.info("Evaluation Results Summary")
            logger.info("=" * 60)
            logger.info(f"Pretrained WER: {pretrained_results['wer']:.2f}%")
            logger.info(f"Fine-tuned WER: {finetuned_results['wer']:.2f}%")
            logger.info(f"Absolute Improvement: {improvement:.2f} percentage points")
            logger.info(f"Relative Improvement: {improvement_pct:.2f}%")
            logger.info("=" * 60)
            
            # Add improvement to dataframe
            comparison_data["Absolute Improvement (pp)"] = ["-", f"{improvement:.2f}"]
            comparison_data["Relative Improvement (%)"] = ["-", f"{improvement_pct:.2f}"]
            df = pd.DataFrame(comparison_data)
        
        return df, pretrained_results, finetuned_results


def main():
    """Main evaluation workflow"""
    
    # Configuration
    pretrained_model = "openai/whisper-small"
    finetuned_model = "./whisper-small-hi-finetuned"
    
    # Set to None to evaluate on full test set, or specify a number for quick testing
    max_samples = None
    
    # Check if fine-tuned model exists
    if not os.path.exists(finetuned_model):
        logger.warning(f"Fine-tuned model not found at {finetuned_model}")
        logger.warning("Will only evaluate pretrained model")
        evaluate_finetuned = False
    else:
        evaluate_finetuned = True
    
    # Initialize evaluator
    evaluator = WhisperEvaluator()
    
    # Load FLEURS Hindi test dataset
    test_dataset = evaluator.load_fleurs_dataset(language="hi_in", split="test")
    
    if test_dataset is None:
        logger.error("Failed to load test dataset")
        return
    
    if evaluate_finetuned:
        # Compare both models
        comparison_df, pretrained_results, finetuned_results = evaluator.compare_models(
            pretrained_model=pretrained_model,
            finetuned_model=finetuned_model,
            dataset=test_dataset,
            batch_size=8,
            max_samples=max_samples
        )
        
        # Print comparison table
        print("\n" + "=" * 60)
        print("WER Comparison Table")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        print("=" * 60)
        
        # Save results to CSV
        output_file = "evaluation_results.csv"
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")
        
        # Show example predictions
        if pretrained_results.get('predictions'):
            print("\n" + "=" * 60)
            print("Example Predictions (Pretrained Model)")
            print("=" * 60)
            for i in range(min(3, len(pretrained_results['predictions']))):
                print(f"\nExample {i+1}:")
                print(f"Reference:  {pretrained_results['references'][i]}")
                print(f"Prediction: {pretrained_results['predictions'][i]}")
        
        if finetuned_results.get('predictions'):
            print("\n" + "=" * 60)
            print("Example Predictions (Fine-tuned Model)")
            print("=" * 60)
            for i in range(min(3, len(finetuned_results['predictions']))):
                print(f"\nExample {i+1}:")
                print(f"Reference:  {finetuned_results['references'][i]}")
                print(f"Prediction: {finetuned_results['predictions'][i]}")
    else:
        # Evaluate only pretrained model
        logger.info("Evaluating pretrained model only...")
        pretrained_results = evaluator.evaluate_model(
            pretrained_model,
            test_dataset,
            batch_size=8,
            max_samples=max_samples
        )
        
        # Create simple table
        comparison_data = {
            "Model": ["Whisper-small (Pretrained)"],
            "WER (%)": [f"{pretrained_results['wer']:.2f}" if pretrained_results['wer'] >= 0 else "Error"],
            "Samples Evaluated": [pretrained_results.get('samples_evaluated', 0)]
        }
        df = pd.DataFrame(comparison_data)
        
        print("\n" + "=" * 60)
        print("WER Results")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        
        # Save results
        output_file = "evaluation_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
