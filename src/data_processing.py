import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional
import logging
from pathlib import Path
import secrets
import string
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PasswordProcessor:
    def __init__(self):
        self.common_patterns = [
            r'123456',
            r'password',
            r'qwerty',
            r'admin',
            r'welcome',
            r'abc123',
            r'letmein',
            r'monkey',
            r'dragon',
            r'baseball'
        ]
        
    def calculate_entropy(self, password: str) -> float:
        """Calculate Shannon entropy of a password."""
        if not password:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for char in password:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0
        for count in freq.values():
            probability = count / len(password)
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def extract_features(self, password: str) -> dict:
        """Extract features from a password."""
        features = {
            'length': len(password),
            'lowercase': sum(1 for c in password if c.islower()),
            'uppercase': sum(1 for c in password if c.isupper()),
            'digits': sum(1 for c in password if c.isdigit()),
            'special': sum(1 for c in password if not c.isalnum()),
            'entropy': self.calculate_entropy(password),
            'has_common_pattern': any(re.search(pattern, password.lower()) 
                                    for pattern in self.common_patterns)
        }
        return features
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the password dataset."""
        try:
            initial_count = len(df)
            logger.info(f"Initial dataset size: {initial_count}")
            
            # Convert passwords to strings and remove any non-string values
            df['password'] = df['password'].astype(str)
            
            # Remove passwords shorter than 6 characters
            df = df[df['password'].str.len() >= 6]
            logger.info(f"After removing short passwords: {len(df)}")
            
            # Remove passwords longer than 50 characters
            df = df[df['password'].str.len() <= 50]
            logger.info(f"After removing long passwords: {len(df)}")
            
            # Remove non-printable characters but keep the password if it's still valid
            df['password'] = df['password'].apply(
                lambda x: ''.join(c for c in str(x) if c.isprintable())
            )
            df = df[df['password'].str.len() > 0]
            logger.info(f"After removing non-printable characters: {len(df)}")
            
            # Keep only ASCII characters but preserve the password if it's still valid
            df['password'] = df['password'].apply(
                lambda x: ''.join(c for c in str(x) if ord(c) < 128)
            )
            df = df[df['password'].str.len() > 0]
            logger.info(f"After ASCII filtering: {len(df)}")
            
            # Log some example passwords from each step
            if len(df) > 0:
                logger.info("Sample passwords after cleaning:")
                for pwd in df['password'].head(5):
                    logger.info(f"  - {pwd}")
            
            logger.info(f"Final cleaned dataset shape: {df.shape}")
            logger.info(f"Removed {initial_count - len(df)} passwords during cleaning")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in clean_data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        try:
            # Extract features for each password
            features_list = []
            for password in df['password']:
                features = self.extract_features(password)
                features_list.append(features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Add original password and label
            features_df['password'] = df['password']
            features_df['label'] = df['label']
            
            # Remove any rows with NaN values
            features_df = features_df.dropna()
            
            # Separate features and target
            X = features_df.drop(['password', 'label'], axis=1)
            y = features_df['label']
            
            # Ensure all values are finite
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y[X.index]  # Align y with X after dropping NaN values
            
            logger.info(f"Prepared features shape: {X.shape}")
            logger.info(f"Label distribution after cleaning:")
            for label, count in y.value_counts().items():
                strength = ['Weak', 'Medium', 'Strong'][int(label)]
                logger.info(f"  - {strength}: {count} passwords")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            raise
    
    def generate_synthetic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate a synthetic dataset of passwords."""
        try:
            passwords = []
            labels = []
            
            # Generate weak passwords (0)
            weak_patterns = ['123456', 'password', 'qwerty', 'admin', 'welcome']
            for _ in range(n_samples // 3):
                base = secrets.choice(weak_patterns)
                password = base + str(secrets.randbelow(100))
                passwords.append(password)
                labels.append(0)
            
            # Generate medium passwords (1)
            for _ in range(n_samples // 3):
                length = secrets.randbelow(5) + 8  # 8-12 characters
                password = ''.join(secrets.choice(string.ascii_letters + string.digits)
                                 for _ in range(length))
                passwords.append(password)
                labels.append(1)
            
            # Generate strong passwords (2)
            for _ in range(n_samples // 3):
                length = secrets.randbelow(5) + 12  # 12-16 characters
                password = ''.join(secrets.choice(string.ascii_letters + string.digits + string.punctuation)
                                 for _ in range(length))
                passwords.append(password)
                labels.append(2)
            
            # Create DataFrame
            df = pd.DataFrame({
                'password': passwords,
                'label': labels
            })
            
            # Shuffle the dataset
            df = df.sample(frac=1).reset_index(drop=True)
            
            logger.info(f"Generated synthetic dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error in generate_synthetic_dataset: {str(e)}")
            raise
    
    def load_external_dataset(self, file_path: str, has_header: bool = True) -> Optional[pd.DataFrame]:
        """Load and validate external password dataset.
        
        Args:
            file_path: Path to the dataset file
            has_header: Whether the file has a header row
            
        Returns:
            DataFrame with processed passwords or None if validation fails
        """
        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"Dataset file not found: {file_path}")
                return None
            
            # Read dataset
            df = pd.read_csv(file_path, header=0 if has_header else None)
            
            # Validate dataset structure
            if len(df.columns) < 1:
                logger.error("Dataset must contain at least one column")
                return None
            
            # Assume first column contains passwords
            password_col = df.columns[0]
            
            # Basic validation
            if df[password_col].isnull().any():
                logger.warning("Dataset contains null values, removing them")
                df = df.dropna(subset=[password_col])
            
            # Remove duplicates
            df = df.drop_duplicates(subset=[password_col])
            
            # Add labels based on password strength
            df['label'] = df[password_col].apply(self._assign_strength_label)
            
            logger.info(f"Loaded dataset with {len(df)} unique passwords")
            return df
            
        except Exception as e:
            logger.error(f"Error loading external dataset: {str(e)}")
            return None
    
    def load_multiple_datasets(self, data_dir: str = 'data/raw') -> pd.DataFrame:
        """Load and combine multiple password datasets.
        
        Args:
            data_dir: Directory containing password datasets
            
        Returns:
            Combined DataFrame with processed passwords
        """
        try:
            data_path = Path(data_dir)
            datasets = {
                'very_weak': data_path / 'pwlds_very_weak.csv',
                'weak': data_path / 'pwlds_weak.csv',
                'average': data_path / 'pwlds_average.csv',
                'strong': data_path / 'pwlds_strong.csv',
                'very_strong': data_path / 'pwlds_very_strong.csv'
            }
            
            all_passwords = []
            
            for strength, file_path in datasets.items():
                if not file_path.exists():
                    logger.warning(f"Dataset file not found: {file_path}")
                    continue
                
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Verify required columns
                    if 'Password' not in df.columns or 'Strength_Level' not in df.columns:
                        logger.error(f"Invalid columns in {file_path}")
                        continue
                    
                    # Add source information
                    df['source'] = file_path.stem
                    
                    # Rename columns to match our format
                    df = df.rename(columns={
                        'Password': 'password',
                        'Strength_Level': 'label'
                    })
                    
                    # Convert strength levels to our format (0: weak, 1: medium, 2: strong)
                    df['label'] = df['label'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
                    
                    all_passwords.append(df)
                    logger.info(f"Loaded {len(df)} passwords from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
            
            if not all_passwords:
                logger.error("No valid datasets found")
                return pd.DataFrame()
            
            # Combine all datasets
            df = pd.concat(all_passwords, ignore_index=True)
            
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=['password'])
            logger.info(f"Removed {initial_count - len(df)} duplicate passwords")
            
            # Log some sample passwords
            logger.info("Sample passwords from dataset:")
            for pwd in df['password'].head(5):
                logger.info(f"  - {pwd}")
            
            # Clean the data
            df = self.clean_data(df)
            if df.empty:
                return pd.DataFrame()
            
            # Log label distribution
            label_counts = df['label'].value_counts()
            logger.info("Label distribution:")
            for label, count in label_counts.items():
                strength = ['Weak', 'Medium', 'Strong'][int(label)]
                logger.info(f"  - {strength}: {count} passwords")
            
            logger.info(f"Final dataset: {len(df)} unique passwords")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()
    
    def _assign_strength_label(self, password: str) -> int:
        """Assign strength label (0: weak, 1: medium, 2: strong) to a password."""
        features = self.extract_features(password)
        
        # Calculate strength score based on stricter criteria
        score = 0
        
        # Length contribution (max 2 points)
        if features['length'] >= 12:
            score += 2
        elif features['length'] >= 8:
            score += 1
        
        # Character type contribution (max 2 points)
        char_types = 0
        if features['lowercase'] > 0:
            char_types += 1
        if features['uppercase'] > 0:
            char_types += 1
        if features['digits'] > 0:
            char_types += 1
        if features['special'] > 0:
            char_types += 1
        
        score += min(char_types / 2, 1)  # 0.5 points per character type, max 2 points
        
        # Entropy contribution (max 1 point)
        if features['entropy'] >= 3.5:  # Higher entropy threshold
            score += 1
        
        # Pattern penalty
        if features['has_common_pattern']:
            score -= 1
        
        # Assign label based on stricter thresholds
        if score < 1.5:
            return 0  # Weak
        elif score < 3:
            return 1  # Medium
        else:
            return 2  # Strong
    
    def anonymize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize dataset by hashing passwords."""
        try:
            # Create a copy to avoid modifying original
            df_anon = df.copy()
            
            # Hash passwords using SHA-256
            df_anon['password'] = df_anon['password'].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()
            )
            
            logger.info("Dataset anonymized successfully")
            return df_anon
            
        except Exception as e:
            logger.error(f"Error anonymizing dataset: {str(e)}")
            return df

def main():
    """Main function to demonstrate usage."""
    processor = PasswordProcessor()
    
    # Generate synthetic dataset
    df = processor.generate_synthetic_dataset()
    
    # Clean data
    df_cleaned = processor.clean_data(df)
    
    # Prepare features
    X, y = processor.prepare_features(df_cleaned)
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_cleaned.to_csv(output_dir / 'cleaned_passwords.csv', index=False)
    logger.info("Data processing completed successfully")

if __name__ == "__main__":
    main() 