import pandas as pd
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reduce_dataset(input_file: str, output_file: str, sample_size: int = 10000):
    """Reduce dataset size by sampling a subset of passwords."""
    try:
        # Read the CSV file
        logger.info(f"Reading {input_file}")
        df = pd.read_csv(input_file)
        
        # Sample the data
        df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Save the sampled data
        df_sampled.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df_sampled)} passwords to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")

def main():
    # Create backup of original data
    data_dir = Path('data/raw')
    backup_dir = Path('data/raw_backup')
    
    if not backup_dir.exists():
        logger.info("Creating backup of original data...")
        shutil.copytree(data_dir, backup_dir)
    
    # Define sample sizes for each category
    sample_sizes = {
        'very_weak': 5000,
        'weak': 5000,
        'average': 5000,
        'strong': 5000,
        'very_strong': 5000
    }
    
    # Process each file
    for strength, size in sample_sizes.items():
        input_file = data_dir / f'pwlds_{strength}.csv'
        output_file = data_dir / f'pwlds_{strength}_reduced.csv'
        
        if input_file.exists():
            reduce_dataset(str(input_file), str(output_file), size)
            
            # Replace original file with reduced version
            output_file.replace(input_file)
            logger.info(f"Replaced {input_file} with reduced version")

if __name__ == "__main__":
    main() 