import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
import logging
from pathlib import Path
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    try:
        # Try to get the directory containing the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # If we're in the notebooks directory, go up one level
        if os.path.basename(script_dir) == 'notebooks':
            return os.path.dirname(script_dir)
        return script_dir
    except NameError:
        # If running in notebook, use current directory
        current_dir = os.getcwd()
        # If we're in the notebooks directory, go up one level
        if os.path.basename(current_dir) == 'notebooks':
            return os.path.dirname(current_dir)
        return current_dir

def calculate_entropy(password):
    """Calculate Shannon entropy of a password."""
    if not password:
        return 0.0
    
    freq = {}
    for char in password:
        freq[char] = freq.get(char, 0) + 1
    
    entropy = 0
    for count in freq.values():
        probability = count / len(password)
        entropy -= probability * np.log2(probability)
    
    return entropy

def extract_features(password):
    """Extract features from a password."""
    features = {
        'length': len(password),
        'lowercase': sum(1 for c in password if c.islower()),
        'uppercase': sum(1 for c in password if c.isupper()),
        'digits': sum(1 for c in password if c.isdigit()),
        'special': sum(1 for c in password if not c.isalnum()),
        'entropy': calculate_entropy(password)
    }
    return features

def load_and_process_data():
    """Load and process all password datasets."""
    # Get the project root directory
    project_root = get_project_root()
    logger.info(f"Project root directory: {project_root}")
    
    # Load all datasets
    datasets = {
        'very_weak': os.path.join(project_root, 'data', 'raw', 'pwlds_very_weak.csv'),
        'weak': os.path.join(project_root, 'data', 'raw', 'pwlds_weak.csv'),
        'average': os.path.join(project_root, 'data', 'raw', 'pwlds_average.csv'),
        'strong': os.path.join(project_root, 'data', 'raw', 'pwlds_strong.csv'),
        'very_strong': os.path.join(project_root, 'data', 'raw', 'pwlds_very_strong.csv')
    }
    
    all_data = []
    for strength, file_path in datasets.items():
        try:
            logger.info(f"Attempting to load: {file_path}")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Rename columns to match our expected format
                df = df.rename(columns={'Password': 'password', 'Strength_Level': 'strength_level'})
                df['strength'] = strength
                all_data.append(df)
                logger.info(f"Loaded {len(df)} passwords from {file_path}")
            else:
                logger.error(f"File does not exist: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        logger.error("No datasets were loaded successfully")
        return None
    
    # Combine all datasets
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total dataset size: {len(df)} passwords")
    
    # Extract features
    features_list = []
    for password in df['password']:
        features = extract_features(password)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['strength'] = df['strength']
    
    logger.info(f"Extracted features for {len(features_df)} passwords")
    return features_df

def save_model_metrics(model, X_test, y_test, feature_names):
    """Save model performance metrics."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Get feature importance
        feature_importance = model.feature_importances_
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix,
            'feature_names': feature_names,
            'feature_importance': feature_importance
        }
        
        # Save metrics
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        metrics_path = os.path.join(models_dir, 'model_metrics.joblib')
        joblib.dump(metrics, metrics_path)
        logging.info(f"Model metrics saved to {metrics_path}")
        
    except Exception as e:
        logging.error(f"Error saving model metrics: {str(e)}")

def train_model():
    """Train the password strength model."""
    try:
        # Load and process data
        features_df = load_and_process_data()
        if features_df is None:
            return None
            
        # Prepare data for training
        X = features_df.drop('strength', axis=1)
        y = features_df['strength']
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Get feature importance
        feature_names = X.columns.tolist()
        feature_importance = model.feature_importances_
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix,
            'feature_names': feature_names,
            'feature_importance': feature_importance
        }
        
        # Save the model
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'password_strength_model.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(models_dir, 'model_metrics.joblib')
        joblib.dump(metrics, metrics_path)
        logger.info(f"Model metrics saved to {metrics_path}")
        
        # Print metrics
        logger.info("\nModel Performance Metrics:")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info("\nConfusion Matrix:")
        logger.info(conf_matrix)
        logger.info("\nFeature Importance:")
        for name, importance in zip(feature_names, feature_importance):
            logger.info(f"{name}: {importance:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

def predict_password_strength(model, password):
    """Predict the strength of a password using the trained model."""
    # Extract features for the password
    features = extract_features(password)
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    # Get probability for the predicted class
    confidence = probabilities[model.classes_.tolist().index(prediction)]
    
    return {
        'password': password,
        'predicted_strength': prediction,
        'confidence': confidence,
        'features': features
    }

def main():
    """Main function to run the password analysis pipeline."""
    try:
        # Load and process data
        features_df = load_and_process_data()
        if features_df is None:
            return
        
        # Train model
        model = train_model()
        
        # Example predictions
        test_passwords = [
            "password123",  # Should be weak
            "P@ssw0rd!",    # Should be average
            "qwerty123",    # Should be weak
            "Tr0ub4d0r&3",  # Should be strong
            "correct horse battery staple",  # Should be very strong
            "123456",       # Should be very weak
            "Admin123!",    # Should be average
            "P@ssw0rd2024!", # Should be strong
        ]
        
        print("\nPassword Strength Predictions:")
        print("-" * 50)
        for password in test_passwords:
            result = predict_password_strength(model, password)
            print(f"\nPassword: {result['password']}")
            print(f"Predicted Strength: {result['predicted_strength']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("Features:")
            for feature, value in result['features'].items():
                print(f"  - {feature}: {value}")
        
        logger.info("Password analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")

if __name__ == "__main__":
    main() 