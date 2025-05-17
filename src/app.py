import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from password_generator import PasswordGenerator
from data_processing import PasswordProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
generator = PasswordGenerator()
processor = PasswordProcessor()

def load_model():
    """Load the trained model."""
    try:
        model_path = Path('models/password_classifier.pkl')
        if model_path.exists():
            return joblib.load(model_path)
        else:
            st.error("Model file not found. Please train the model first.")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Error loading model. Please check the logs.")
        return None

def analyze_password(password: str):
    """Analyze password strength and provide recommendations."""
    try:
        # Extract features
        features = processor.extract_features(password)
        
        # Calculate metrics
        entropy = features['entropy']
        nist_compliant = generator.meets_nist_requirements(password)
        cracking_time = generator.estimate_cracking_time(entropy)
        
        # Load model and predict
        model = load_model()
        if model is not None:
            # Prepare features for prediction
            X = pd.DataFrame([features])
            prediction = model.predict(X)[0]
            strength = ['Weak', 'Medium', 'Strong'][prediction]
        else:
            strength = "Unknown"
        
        return {
            'strength': strength,
            'entropy': entropy,
            'nist_compliant': nist_compliant,
            'cracking_time': cracking_time,
            'features': features
        }
    except Exception as e:
        logger.error(f"Error analyzing password: {str(e)}")
        st.error("Error analyzing password. Please try again.")
        return None

def main():
    st.set_page_config(
        page_title="Password Security Analyzer",
        page_icon="🔒",
        layout="wide"
    )
    
    st.title("🔒 Password Security Analyzer")
    st.markdown("""
    This tool helps you analyze password strength and generate secure alternatives.
    Enter a password below to get started.
    """)
    
    # Password input
    password = st.text_input("Enter a password to analyze:", type="password")
    
    if password:
        # Analyze password
        results = analyze_password(password)
        
        if results:
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Strength", results['strength'])
                st.metric("Entropy", f"{results['entropy']:.2f} bits")
            
            with col2:
                st.metric("NIST Compliant", "✅" if results['nist_compliant'] else "❌")
                st.metric("Estimated Cracking Time", results['cracking_time'])
            
            with col3:
                st.metric("Length", results['features']['length'])
                st.metric("Character Types", sum([
                    results['features']['lowercase'] > 0,
                    results['features']['uppercase'] > 0,
                    results['features']['digits'] > 0,
                    results['features']['special'] > 0
                ]))
            
            # Generate alternatives if password is weak or medium
            if results['strength'] in ['Weak', 'Medium']:
                st.subheader("🔐 Secure Alternatives")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Random Password")
                    if st.button("Generate Random Password"):
                        new_password, new_entropy = generator.generate_random_password()
                        st.code(new_password)
                        st.info(f"Entropy: {new_entropy:.2f} bits")
                
                with col2:
                    st.subheader("Memorable Passphrase")
                    if st.button("Generate Passphrase"):
                        new_passphrase, new_entropy = generator.generate_passphrase()
                        st.code(new_passphrase)
                        st.info(f"Entropy: {new_entropy:.2f} bits")
            
            # Display feature details
            with st.expander("View Detailed Analysis"):
                st.write("### Password Features")
                st.json({
                    "Length": results['features']['length'],
                    "Lowercase Letters": results['features']['lowercase'],
                    "Uppercase Letters": results['features']['uppercase'],
                    "Digits": results['features']['digits'],
                    "Special Characters": results['features']['special'],
                    "Has Common Pattern": results['features']['has_common_pattern']
                })
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    ### Security Notes
    - This tool analyzes passwords locally in your browser
    - No passwords are stored or transmitted
    - Generated passwords meet NIST security guidelines
    - Use a password manager to store your passwords securely
    """)

if __name__ == "__main__":
    main() 