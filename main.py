import logging
from pathlib import Path
from src.data_processing import PasswordProcessor
from src.model_training import PasswordModelTrainer
from src.password_generator import PasswordGenerator
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the password security system pipeline."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Password Security System')
        parser.add_argument('--use-synthetic', action='store_true', help='Use synthetic data instead of real datasets')
        parser.add_argument('--synthetic-size', type=int, default=1000, help='Number of synthetic samples to generate')
        args = parser.parse_args()
        
        # Create necessary directories
        for dir_path in ['data/raw', 'data/processed', 'models', 'results']:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        processor = PasswordProcessor()
        trainer = PasswordModelTrainer()
        generator = PasswordGenerator()
        
        # Load or generate dataset
        if args.use_synthetic:
            logger.info(f"Generating synthetic dataset with {args.synthetic_size} samples")
            df = processor.generate_synthetic_dataset(n_samples=args.synthetic_size)
        else:
            logger.info("Loading and combining password datasets...")
            df = processor.load_multiple_datasets()
            if df.empty:
                logger.error("Failed to load datasets, falling back to synthetic data")
                df = processor.generate_synthetic_dataset(n_samples=args.synthetic_size)
        
        # Save processed data
        df.to_csv('data/processed/cleaned_passwords.csv', index=False)
        logger.info("Saved processed data to data/processed/cleaned_passwords.csv")
        
        # Prepare features
        logger.info("Preparing features...")
        X, y = processor.prepare_features(df)
        
        # Split data for training
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
        
        # Train models
        logger.info("Training Random Forest model...")
        rf_model = trainer.train_random_forest(X_train, y_train)
        rf_results = trainer.evaluate_model(rf_model, X_test, y_test)
        
        logger.info("Training XGBoost model...")
        xgb_model = trainer.train_xgboost(X_train, y_train)
        xgb_results = trainer.evaluate_model(xgb_model, X_test, y_test)
        
        # Save best model
        if trainer.best_model is not None:
            trainer.save_model(trainer.best_model, 'password_classifier')
            logger.info("Saved best model to models/password_classifier.pkl")
        
        # Generate example passwords
        logger.info("\nGenerating example passwords:")
        
        # Random password
        password, entropy = generator.generate_random_password()
        logger.info(f"\nRandom Password: {password}")
        logger.info(f"Entropy: {entropy:.2f} bits")
        logger.info(f"NIST Compliant: {generator.meets_nist_requirements(password)}")
        logger.info(f"Estimated Cracking Time: {generator.estimate_cracking_time(entropy)}")
        
        # Passphrase
        passphrase, entropy = generator.generate_passphrase()
        logger.info(f"\nPassphrase: {passphrase}")
        logger.info(f"Entropy: {entropy:.2f} bits")
        logger.info(f"NIST Compliant: {generator.meets_nist_requirements(passphrase)}")
        logger.info(f"Estimated Cracking Time: {generator.estimate_cracking_time(entropy)}")
        
        logger.info("\nPipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 