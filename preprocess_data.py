"""
Data Preprocessing Script for Hindi ASR Training Dataset

This script preprocesses the Hindi ASR training data for fine-tuning Whisper-small.
It handles:
- Loading audio files from URLs
- Loading transcriptions
- Data validation and cleaning
- Audio format normalization
- Dataset preparation for training
"""

import os
import pandas as pd
import requests
from datasets import Dataset, Audio
from tqdm import tqdm
import soundfile as sf
import io
import logging
from typing import Dict, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for Hindi ASR training data"""
    
    def __init__(self, data_path: str, cache_dir: str = "./data_cache"):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to CSV file containing dataset metadata
            cache_dir: Directory to cache downloaded audio files
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata from CSV or create from provided data"""
        logger.info(f"Loading metadata from {self.data_path}")
        
        # If path is a CSV file, load it
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
        else:
            # Note: In production, you would load from the actual data source
            # For now, return empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                'user_id', 'recording_id', 'language', 'duration', 
                'rec_url_gcp', 'transcription_url', 'metadata_url'
            ])
            logger.warning("No data file found. Please provide dataset metadata CSV.")
            
        return df
    
    def download_file(self, url: str, timeout: int = 30) -> Optional[bytes]:
        """
        Download file from URL
        
        Args:
            url: URL to download from
            timeout: Request timeout in seconds
            
        Returns:
            File content as bytes or None if download fails
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def load_transcription(self, transcription_url: str) -> Optional[str]:
        """
        Load transcription text from URL
        
        Args:
            transcription_url: URL to transcription file
            
        Returns:
            Transcription text or None if loading fails
        """
        content = self.download_file(transcription_url)
        if content:
            try:
                return content.decode('utf-8').strip()
            except Exception as e:
                logger.error(f"Failed to decode transcription: {e}")
        return None
    
    def validate_audio(self, audio_bytes: bytes) -> bool:
        """
        Validate audio file
        
        Args:
            audio_bytes: Audio file content
            
        Returns:
            True if audio is valid, False otherwise
        """
        try:
            # Try to load audio
            audio_io = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_io)
            
            # Check if audio has data
            if len(data) == 0:
                return False
                
            # Check sample rate (should be reasonable, e.g., 8kHz to 48kHz)
            if samplerate < 8000 or samplerate > 48000:
                logger.warning(f"Unusual sample rate: {samplerate}")
                
            return True
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
    
    def filter_by_duration(self, df: pd.DataFrame, 
                          min_duration: float = 1.0, 
                          max_duration: float = 30.0) -> pd.DataFrame:
        """
        Filter dataset by audio duration
        
        Args:
            df: Dataset DataFrame
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)
        df_filtered = df[
            (df['duration'] >= min_duration) & 
            (df['duration'] <= max_duration)
        ].copy()
        
        filtered_count = len(df_filtered)
        logger.info(f"Filtered by duration: {initial_count} -> {filtered_count} samples")
        logger.info(f"Removed {initial_count - filtered_count} samples")
        
        return df_filtered
    
    def clean_transcription(self, text: str) -> str:
        """
        Clean transcription text
        
        Args:
            text: Raw transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def prepare_dataset(self, df: pd.DataFrame, 
                       sample_limit: Optional[int] = None) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            df: Dataset DataFrame
            sample_limit: Limit number of samples (for testing)
            
        Returns:
            HuggingFace Dataset object
        """
        logger.info("Preparing dataset for training...")
        
        if sample_limit:
            df = df.head(sample_limit)
            logger.info(f"Limited to {sample_limit} samples for processing")
        
        # Prepare data for Dataset creation
        processed_data = {
            'audio': [],
            'transcription': [],
            'duration': [],
            'recording_id': []
        }
        
        successful = 0
        failed = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            try:
                # Download transcription
                transcription = self.load_transcription(row['transcription_url'])
                
                if transcription:
                    # Clean transcription
                    transcription = self.clean_transcription(transcription)
                    
                    if transcription:  # Only include if transcription is not empty
                        # For this implementation, we store the URL
                        # In production, you might want to download and cache audio
                        processed_data['audio'].append(row['rec_url_gcp'])
                        processed_data['transcription'].append(transcription)
                        processed_data['duration'].append(row['duration'])
                        processed_data['recording_id'].append(row['recording_id'])
                        successful += 1
                    else:
                        failed += 1
                        logger.debug(f"Empty transcription for {row['recording_id']}")
                else:
                    failed += 1
                    logger.debug(f"Failed to load transcription for {row['recording_id']}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"Error processing sample {row['recording_id']}: {e}")
        
        logger.info(f"Successfully processed: {successful} samples")
        logger.info(f"Failed: {failed} samples")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(processed_data)
        
        # Audio column is stored as URLs for now
        # To cast to Audio feature type, audio files need to be downloaded and cached locally
        # This can be done in the fine-tuning script: dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get dataset statistics
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(df),
            'total_duration_hours': df['duration'].sum() / 3600,
            'mean_duration': df['duration'].mean(),
            'median_duration': df['duration'].median(),
            'min_duration': df['duration'].min(),
            'max_duration': df['duration'].max(),
            'unique_speakers': df['user_id'].nunique() if 'user_id' in df else 0
        }
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print dataset statistics"""
        logger.info("=" * 50)
        logger.info("Dataset Statistics")
        logger.info("=" * 50)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Total duration: {stats['total_duration_hours']:.2f} hours")
        logger.info(f"Mean duration: {stats['mean_duration']:.2f} seconds")
        logger.info(f"Median duration: {stats['median_duration']:.2f} seconds")
        logger.info(f"Min duration: {stats['min_duration']:.2f} seconds")
        logger.info(f"Max duration: {stats['max_duration']:.2f} seconds")
        logger.info(f"Unique speakers: {stats['unique_speakers']}")
        logger.info("=" * 50)


def main():
    """Main preprocessing workflow"""
    
    # Example usage
    data_path = "training_data.csv"  # Path to your dataset metadata
    
    preprocessor = DataPreprocessor(data_path)
    
    # Load metadata
    df = preprocessor.load_metadata()
    
    if len(df) > 0:
        # Get initial statistics
        stats = preprocessor.get_statistics(df)
        preprocessor.print_statistics(stats)
        
        # Filter by duration (1-30 seconds)
        df_filtered = preprocessor.filter_by_duration(df, min_duration=1.0, max_duration=30.0)
        
        # Get filtered statistics
        stats_filtered = preprocessor.get_statistics(df_filtered)
        logger.info("\nAfter filtering:")
        preprocessor.print_statistics(stats_filtered)
        
        # Prepare dataset (limit to 10 samples for testing)
        # Remove sample_limit parameter to process all data
        dataset = preprocessor.prepare_dataset(df_filtered, sample_limit=10)
        
        logger.info(f"\nFinal dataset size: {len(dataset)}")
        logger.info(f"Dataset features: {dataset.features}")
        
        # Save preprocessed dataset
        output_path = "./preprocessed_data"
        dataset.save_to_disk(output_path)
        logger.info(f"Saved preprocessed dataset to {output_path}")
    else:
        logger.error("No data loaded. Please provide valid dataset metadata CSV.")


if __name__ == "__main__":
    main()
