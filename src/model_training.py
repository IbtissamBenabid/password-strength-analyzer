import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PasswordModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets."""
        try:
            # Ensure no NaN values
            if X.isna().any().any() or y.isna().any():
                logger.warning("Found NaN values in data, removing them...")
                mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[mask]
                y = y[mask]
            
            # Ensure all values are finite
            X = X.replace([np.inf, -np.inf], np.nan)
            mask = ~X.isna().any(axis=1)
            X = X[mask]
            y = y[mask]
            
            # Log data distribution
            logger.info(f"Data distribution before splitting:")
            for label, count in y.value_counts().items():
                strength = ['Weak', 'Medium', 'Strong'][int(label)]
                logger.info(f"  - {strength}: {count} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train Random Forest model with SMOTE for handling class imbalance."""
        try:
            # Create pipeline with SMOTE and Random Forest
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            # Define parameter grid
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__class_weight': ['balanced', 'balanced_subsample']
            }
            
            # Create grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Log best parameters
            logger.info("Best parameters found:")
            for param, value in grid_search.best_params_.items():
                logger.info(f"  {param}: {value}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error in train_random_forest: {str(e)}")
            raise
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning."""
        try:
            # Create pipeline with SMOTE and StandardScaler
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42, sampling_strategy={
                    0: min(1000, sum(y_train == 0) * 5),  # Weak
                    1: min(1000, sum(y_train == 1) * 5),  # Medium
                    2: min(1000, sum(y_train == 2) * 5)   # Strong
                })),
                ('classifier', xgb.XGBClassifier(random_state=42))
            ])
            
            # Define parameter grid
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.1, 0.2],
                'classifier__subsample': [0.8, 1.0],
                'classifier__scale_pos_weight': [1]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1_weighted'
            )
            
            # Fit model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_xgb = grid_search.best_estimator_
            self.models['xgboost'] = best_xgb
            
            logger.info(f"XGBoost best parameters: {grid_search.best_params_}")
            return best_xgb
            
        except Exception as e:
            logger.error(f"Error in train_xgboost: {str(e)}")
            raise
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store results
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }
            
            # Log detailed metrics
            logger.info("\nModel Evaluation Metrics:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    strength = ['Weak', 'Medium', 'Strong'][int(label)]
                    logger.info(f"\n{strength} class:")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1-score: {metrics['f1-score']:.4f}")
            
            # Update best model if current model performs better
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluate_model: {str(e)}")
            raise
    
    def save_model(self, model: Any, model_name: str):
        """Save trained model to disk."""
        try:
            # Create models directory if it doesn't exist
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = models_dir / f'{model_name}.pkl'
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error in save_model: {str(e)}")
            raise

def main():
    """Main function to demonstrate usage."""
    # Load processed data
    data_path = Path('data/processed/cleaned_passwords.csv')
    df = pd.read_csv(data_path)
    
    # Prepare features
    X = df.drop(['password', 'label'], axis=1)
    y = df['label']
    
    # Initialize trainer
    trainer = PasswordModelTrainer()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Train models
    rf_model = trainer.train_random_forest(X_train, y_train)
    xgb_model = trainer.train_xgboost(X_train, y_train)
    
    # Evaluate models
    rf_results = trainer.evaluate_model(rf_model, X_test, y_test)
    xgb_results = trainer.evaluate_model(xgb_model, X_test, y_test)
    
    # Save best model
    if trainer.best_model is not None:
        trainer.save_model(trainer.best_model, 'password_classifier')
        logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 