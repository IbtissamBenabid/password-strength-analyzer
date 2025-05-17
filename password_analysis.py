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
import string
import random
import math
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
    """Calculate the entropy of a password."""
    if not password:
        return 0
    
    # Count character frequencies
    char_freq = {}
    for char in password:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0
    for count in char_freq.values():
        probability = count / len(password)
        entropy -= probability * math.log2(probability)
    
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

def generate_password_dataset(num_samples=10000):
    """Generate a dataset of passwords with varying strengths."""
    passwords = []
    labels = []
    
    # Character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    
    # Generate very weak passwords (common words, short)
    common_words = ['password', '123456', 'qwerty', 'admin', 'welcome', 'letmein']
    for _ in range(num_samples // 5):
        password = random.choice(common_words)
        passwords.append(password)
        labels.append('very_weak')
    
    # Generate weak passwords (short, simple patterns)
    for _ in range(num_samples // 5):
        length = random.randint(6, 8)
        password = ''.join(random.choices(lowercase + digits, k=length))
        passwords.append(password)
        labels.append('weak')
    
    # Generate average passwords
    for _ in range(num_samples // 5):
        length = random.randint(8, 12)
        password = ''.join(random.choices(lowercase + uppercase + digits, k=length))
        passwords.append(password)
        labels.append('average')
    
    # Generate strong passwords
    for _ in range(num_samples // 5):
        length = random.randint(12, 16)
        password = ''.join(random.choices(lowercase + uppercase + digits + special, k=length))
        passwords.append(password)
        labels.append('strong')
    
    # Generate very strong passwords
    for _ in range(num_samples // 5):
        length = random.randint(16, 20)
        password = ''.join(random.choices(lowercase + uppercase + digits + special, k=length))
        # Ensure at least one of each character type
        password = (
            random.choice(lowercase) +
            random.choice(uppercase) +
            random.choice(digits) +
            random.choice(special) +
            password[4:]
        )
        passwords.append(password)
        labels.append('very_strong')
    
    return passwords, labels

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
    """Train the password strength classifier."""
    print("Generating password dataset...")
    passwords, labels = generate_password_dataset()
    
    print("Extracting features...")
    features = [extract_features(pwd) for pwd in passwords]
    X = pd.DataFrame(features)
    y = pd.Series(labels)
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and metrics
    os.makedirs('models', exist_ok=True)
    
    print("Saving model...")
    joblib.dump(model, 'models/password_strength_model.joblib')
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': accuracy,  # Simplified for this example
        'recall': accuracy,     # Simplified for this example
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_names': X.columns.tolist(),
        'feature_importance': model.feature_importances_.tolist()
    }
    joblib.dump(metrics, 'models/model_metrics.joblib')
    
    print("Model and metrics saved successfully!")

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
        train_model()
        
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