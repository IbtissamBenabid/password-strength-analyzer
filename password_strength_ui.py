import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import random
import string
import seaborn as sns
import matplotlib.pyplot as plt
from password_analysis import extract_features, calculate_entropy
from password_breach_checker import PasswordBreachChecker
from sklearn.metrics import confusion_matrix, classification_report
import time

# Initialize breach checker
breach_checker = PasswordBreachChecker()

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Password Strength Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette
COLORS = {
    'jet': '#333333',          # Primary color
    'turquoise': '#48e5c2',    # Accent color
    'seasalt': '#fcfaf9',      # Background
    'desert_sand': '#f3d3bd',  # Warm accent
    'davys_gray': '#5e5e5e'    # Text color
}

# Custom CSS with new color scheme
st.markdown(f"""
    <style>
    /* Main container styling */
    .main {{
        padding: 2rem;
        background-color: {COLORS['seasalt']};
    }}
    
    /* Input field styling */
    .stTextInput > div > div > input {{
        font-size: 1.2rem;
        border-radius: 8px;
        border: 2px solid {COLORS['davys_gray']};
        padding: 0.5rem;
        background-color: {COLORS['seasalt']};
        color: {COLORS['jet']};
    }}
    
    /* Strength box styling */
    .strength-box {{
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    /* Strength level colors */
    .very-weak {{ 
        background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
        color: {COLORS['seasalt']};
    }}
    .weak {{ 
        background: linear-gradient(135deg, #ffa726, #ffb74d);
        color: {COLORS['jet']};
    }}
    .average {{ 
        background: linear-gradient(135deg, #ffd600, #ffeb3b);
        color: {COLORS['jet']};
    }}
    .strong {{ 
        background: linear-gradient(135deg, #66bb6a, #81c784);
        color: {COLORS['jet']};
    }}
    .very-strong {{ 
        background: linear-gradient(135deg, #7e57c2, #9575cd);
        color: {COLORS['seasalt']};
    }}
    
    /* Metric card styling */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['jet']}, {COLORS['davys_gray']});
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border: 1px solid {COLORS['turquoise']}40;
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        border-color: {COLORS['turquoise']};
        background: linear-gradient(135deg, {COLORS['davys_gray']}, {COLORS['jet']});
    }}
    
    .metric-card h4 {{
        color: {COLORS['seasalt']};
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }}
    
    .metric-card h3 {{
        color: {COLORS['turquoise']};
        margin: 0;
        font-size: 1.5rem;
    }}
    
    /* Title styling */
    h1 {{
        color: {COLORS['jet']};
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }}
    
    h2 {{
        color: {COLORS['jet']};
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
    }}
    
    h3 {{
        color: {COLORS['jet']};
        font-size: 1.4rem !important;
    }}
    
    /* Info box styling */
    .stInfo {{
        background-color: {COLORS['turquoise']}20;
        border-left: 4px solid {COLORS['turquoise']};
    }}
    
    /* Success message styling */
    .stSuccess {{
        background-color: {COLORS['turquoise']}20;
        border-left: 4px solid {COLORS['turquoise']};
    }}
    
    /* Error message styling */
    .stError {{
        background-color: {COLORS['desert_sand']}40;
        border-left: 4px solid {COLORS['desert_sand']};
    }}
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the trained model."""
    try:
        # Get the current directory (where the UI script is)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # The models directory is in the same directory as the script
        model_path = os.path.join(current_dir, 'models', 'password_strength_model.joblib')
        
        if not os.path.exists(model_path):
            st.error("Model file not found. Please run 'python password_analysis.py' first to train and save the model.")
            return None
            
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_strength_color(strength):
    """Get color class based on password strength."""
    colors = {
        'very_weak': 'very-weak',
        'weak': 'weak',
        'average': 'average',
        'strong': 'strong',
        'very_strong': 'very-strong'
    }
    return colors.get(strength, 'average')

def create_character_distribution_chart(features):
    """Create a pie chart showing character distribution."""
    labels = ['Lowercase', 'Uppercase', 'Digits', 'Special']
    values = [features['lowercase'], features['uppercase'], 
              features['digits'], features['special']]
    
    colors = [COLORS['turquoise'], COLORS['desert_sand'], 
              COLORS['davys_gray'], COLORS['jet']]
    
    fig = px.pie(values=values, names=labels, 
                 title='Character Distribution',
                 color_discrete_sequence=colors)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_strength_probability_chart(probabilities, classes):
    """Create a bar chart showing strength probabilities."""
    colors = ['#ff4b4b', '#ffa726', '#ffd600', '#66bb6a', '#7e57c2']
    
    fig = go.Figure(data=[
        go.Bar(x=classes, y=probabilities,
               marker_color=colors)
    ])
    fig.update_layout(
        title='Password Strength Probabilities',
        xaxis_title='Strength Level',
        yaxis_title='Probability',
        yaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_x=0.5,
        title_font_size=20,
        showlegend=False,
        font=dict(color=COLORS['jet'])
    )
    return fig

def create_entropy_gauge(features):
    """Create a gauge chart for password entropy."""
    entropy = features['entropy']
    max_entropy = 8  # Maximum reasonable entropy for visualization
    
    colors = ['#ff4b4b', '#ffa726', '#ffd600', '#66bb6a', '#7e57c2']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=entropy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Password Entropy", 'font': {'size': 20, 'color': COLORS['jet']}},
        gauge={
            'axis': {'range': [0, max_entropy], 'tickcolor': COLORS['jet']},
            'bar': {'color': COLORS['turquoise']},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, 1.6], 'color': colors[0]},  # Very weak
                {'range': [1.6, 3.2], 'color': colors[1]},  # Weak
                {'range': [3.2, 4.8], 'color': colors[2]},  # Average
                {'range': [4.8, 6.4], 'color': colors[3]},  # Strong
                {'range': [6.4, 8], 'color': colors[4]}  # Very strong
            ],
            'threshold': {
                'line': {'color': COLORS['desert_sand'], 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['jet']}
    )
    return fig

def generate_password_recommendations(features, current_password, prediction, confidence):
    """Generate password recommendations based on model predictions and analysis."""
    recommendations = []
    
    # Skip recommendations for very strong passwords with high confidence
    if prediction == 'very_strong' and confidence > 0.8:
        return [], []
    
    # Base recommendations on model prediction and features
    if prediction in ['very_weak', 'weak']:
        # Length recommendations
        if features['length'] < 12:
            recommendations.append({
                'type': 'length',
                'message': 'Increase password length to at least 12 characters',
                'example': current_password + ''.join(random.choices(string.ascii_letters + string.digits, k=12 - len(current_password)))
            })
        
        # Character type recommendations based on what's missing
        if features['lowercase'] == 0:
            recommendations.append({
                'type': 'lowercase',
                'message': 'Add lowercase letters for better complexity',
                'example': current_password + ''.join(random.choices(string.ascii_lowercase, k=3))
            })
        
        if features['uppercase'] == 0:
            recommendations.append({
                'type': 'uppercase',
                'message': 'Add uppercase letters for better complexity',
                'example': current_password + ''.join(random.choices(string.ascii_uppercase, k=2))
            })
        
        if features['digits'] == 0:
            recommendations.append({
                'type': 'digits',
                'message': 'Add numbers for better complexity',
                'example': current_password + ''.join(random.choices(string.digits, k=2))
            })
        
        if features['special'] == 0:
            recommendations.append({
                'type': 'special',
                'message': 'Add special characters for better complexity',
                'example': current_password + ''.join(random.choices('!@#$%^&*()_+-=[]{}|;:,.<>?', k=2))
            })
    
    elif prediction == 'average':
        # For average passwords, suggest improvements based on weakest aspects
        if features['entropy'] < 3:
            recommendations.append({
                'type': 'entropy',
                'message': 'Increase password complexity by using more diverse characters',
                'example': current_password + ''.join(random.choices(string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?', k=3))
            })
        
        if features['length'] < 12:
            recommendations.append({
                'type': 'length',
                'message': 'Consider increasing password length',
                'example': current_password + ''.join(random.choices(string.ascii_letters + string.digits, k=4))
            })
    
    # Generate alternative strong passwords only for weak and average passwords
    strong_passwords = []
    if prediction in ['very_weak', 'weak', 'average']:
        for _ in range(3):
            # Generate a password with all character types
            password = ''.join([
                ''.join(random.choices(string.ascii_lowercase, k=4)),
                ''.join(random.choices(string.ascii_uppercase, k=2)),
                ''.join(random.choices(string.digits, k=2)),
                ''.join(random.choices('!@#$%^&*()_+-=[]{}|;:,.<>?', k=2))
            ])
            # Shuffle the characters
            password_list = list(password)
            random.shuffle(password_list)
            strong_passwords.append(''.join(password_list))
    
    return recommendations, strong_passwords

def display_recommendations(recommendations, strong_passwords, prediction, confidence):
    """Display password recommendations in a styled format."""
    if prediction == 'very_strong' and confidence > 0.8:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #7e57c2, #9575cd); 
                        padding: 1.2rem; 
                        border-radius: 12px; 
                        border: 1px solid #48e5c240;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
            <h3 style="color: #fcfaf9; margin: 0;">Your password is very strong! 🎉</h3>
            <p style="color: #fcfaf9; margin: 0.5rem 0 0 0;">
                No recommendations needed. Keep up the good security practices!
            </p>
            </div>
        """, unsafe_allow_html=True)
        return

    st.subheader("Password Recommendations")
    
    # Display improvement suggestions
    if recommendations:
        st.markdown("### How to Improve Your Password")
        for rec in recommendations:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>🔹 {rec['message']}</h4>
                    <p style="color: {COLORS['turquoise']}; font-family: monospace; font-size: 1.1rem;">
                        Example: {rec['example']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Display strong password suggestions only if needed
    if strong_passwords:
        st.markdown("### Strong Password Suggestions")
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {COLORS['jet']}, {COLORS['davys_gray']}); 
                        padding: 1.2rem; 
                        border-radius: 12px; 
                        border: 1px solid {COLORS['turquoise']}40;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
            <p style="color: {COLORS['seasalt']}; margin-bottom: 0.5rem;">
                Here are some strong password suggestions that include a mix of:
            </p>
            <ul style="color: {COLORS['seasalt']}; margin-bottom: 1rem;">
                <li>Uppercase and lowercase letters</li>
                <li>Numbers</li>
                <li>Special characters</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)
        
        for i, password in enumerate(strong_passwords, 1):
            st.markdown(f"""
                <div class="metric-card" style="cursor: pointer;" 
                     onclick="navigator.clipboard.writeText('{password}')">
                    <h4>Option {i}</h4>
                    <p style="color: {COLORS['turquoise']}; 
                              font-family: monospace; 
                              font-size: 1.2rem; 
                              margin: 0;">
                        {password}
                    </p>
                    <small style="color: {COLORS['seasalt']}80;">
                        Click to copy
                    </small>
                </div>
            """, unsafe_allow_html=True)

def load_model_metrics():
    """Load model performance metrics."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_path = os.path.join(current_dir, 'models', 'model_metrics.joblib')
        if os.path.exists(metrics_path):
            return joblib.load(metrics_path)
        return None
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")
        return None

def display_model_info(metrics):
    """Display model information and performance metrics."""
    st.markdown("## Model Information")
    
    if metrics:
        # Performance Metrics Section
        st.markdown("### Model Performance Metrics")
        
        # Create a grid of metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Accuracy</h4>
                    <h3>{metrics['accuracy']:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Precision</h4>
                    <h3>{metrics['precision']:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Recall</h4>
                    <h3>{metrics['recall']:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            st.markdown(f"""
                <div class="metric-card">
                    <h4>F1 Score</h4>
                    <h3>{f1_score:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrix Section
        st.markdown("### Confusion Matrix")
        st.markdown("""
            The confusion matrix shows how well the model predicts each password strength level.
            The rows represent actual values, and the columns represent predicted values.
        """)
        
        # Create a more visually appealing confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong'],
                   yticklabels=['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong'])
        plt.title('Confusion Matrix', pad=20, fontsize=14)
        plt.xlabel('Predicted', labelpad=10)
        plt.ylabel('Actual', labelpad=10)
        st.pyplot(fig)
        
        # Feature Importance Section
        st.markdown("### Feature Importance")
        st.markdown("""
            This chart shows how much each feature contributes to the model's predictions.
            Higher values indicate more important features.
        """)
        
        # Create an enhanced feature importance visualization
        fig = px.bar(x=metrics['feature_names'], 
                    y=metrics['feature_importance'],
                    title='Feature Importance in Password Strength Prediction',
                    labels={'x': 'Features', 'y': 'Importance'},
                    color=metrics['feature_importance'],
                    color_continuous_scale=[COLORS['jet'], COLORS['turquoise']])
        
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Importance",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['jet']),
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Per-Class Performance
        st.markdown("### Per-Class Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            for i, class_name in enumerate(['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong']):
                true_positives = metrics['confusion_matrix'][i][i]
                total_actual = sum(metrics['confusion_matrix'][i])
                total_predicted = sum(metrics['confusion_matrix'][:, i])
                
                precision = true_positives / total_predicted if total_predicted > 0 else 0
                recall = true_positives / total_actual if total_actual > 0 else 0
                
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>{class_name}</h4>
                        <p>Precision: {precision:.2%}</p>
                        <p>Recall: {recall:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Add feature importance table
            st.markdown("#### Feature Importance Details")
            importance_data = {
                'Feature': metrics['feature_names'],
                'Importance': metrics['feature_importance']
            }
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}))
    
    else:
        st.warning("Model metrics not found. Please run the training script first.")
        st.markdown("""
            To generate the model metrics, run:
            ```bash
            python password_analysis.py
            ```
        """)

def generate_password_by_criteria(criteria):
    """Generate password based on user-selected criteria."""
    password = []
    
    if criteria['length'] < 8:
        st.warning("Password length should be at least 8 characters")
        return None
    
    # Character pools
    char_pools = {
        'lowercase': string.ascii_lowercase,
        'uppercase': string.ascii_uppercase,
        'digits': string.digits,
        'special': '!@#$%^&*()_+-=[]{}|;:,.<>?',
        'similar': 'iIlL1oO0',
        'ambiguous': '{}[]()/\\\'"`~,;:.<>'
    }
    
    # Build initial character pool based on selected types
    char_pool = ''
    if criteria['lowercase']:
        char_pool += char_pools['lowercase']
    if criteria['uppercase']:
        char_pool += char_pools['uppercase']
    if criteria['digits']:
        char_pool += char_pools['digits']
    if criteria['special']:
        char_pool += char_pools['special']
    
    if not char_pool:
        st.error("Please select at least one character type")
        return None
    
    # Apply exclusions to character pool
    if criteria.get('exclude_similar', False):
        char_pool = ''.join(c for c in char_pool if c not in char_pools['similar'])
    if criteria.get('exclude_ambiguous', False):
        char_pool = ''.join(c for c in char_pool if c not in char_pools['ambiguous'])
    
    # Add required character types
    if criteria['lowercase']:
        password.extend(random.choices(char_pools['lowercase'], k=max(2, criteria['length'] // 4)))
    if criteria['uppercase']:
        password.extend(random.choices(char_pools['uppercase'], k=max(2, criteria['length'] // 4)))
    if criteria['digits']:
        password.extend(random.choices(char_pools['digits'], k=max(2, criteria['length'] // 4)))
    if criteria['special']:
        password.extend(random.choices(char_pools['special'], k=max(2, criteria['length'] // 4)))
    
    # Fill remaining length with random characters from the filtered pool
    remaining_length = criteria['length'] - len(password)
    if remaining_length > 0:
        password.extend(random.choices(char_pool, k=remaining_length))
    
    # Check for unique characters requirement
    if criteria.get('require_unique', False):
        while len(set(password)) != len(password):
            # Regenerate the entire password
            password = []
            if criteria['lowercase']:
                password.extend(random.choices(char_pools['lowercase'], k=max(2, criteria['length'] // 4)))
            if criteria['uppercase']:
                password.extend(random.choices(char_pools['uppercase'], k=max(2, criteria['length'] // 4)))
            if criteria['digits']:
                password.extend(random.choices(char_pools['digits'], k=max(2, criteria['length'] // 4)))
            if criteria['special']:
                password.extend(random.choices(char_pools['special'], k=max(2, criteria['length'] // 4)))
            
            remaining_length = criteria['length'] - len(password)
            if remaining_length > 0:
                password.extend(random.choices(char_pool, k=remaining_length))
    
    # Shuffle the password
    random.shuffle(password)
    return ''.join(password)

def display_breach_check_results(breach_details):
    """Display breach check results in the UI."""
    st.markdown("### 🔍 Breach Check Results")
    
    if breach_details["error"]:
        st.error(breach_details["message"])
        return
    
    # Create a styled container for breach results
    if breach_details["is_breached"]:
        # Determine color based on risk level
        color = {
            "high": "#ff4b4b",
            "medium": "#ffa726",
            "low": "#ffd600"
        }.get(breach_details["risk_level"], "#ff4b4b")
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}, {color}80); 
                        padding: 1.2rem; 
                        border-radius: 12px; 
                        border: 1px solid {color}40;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
                <h3 style="color: #fcfaf9; margin: 0;">⚠️ Password Found in Data Breaches</h3>
                <p style="color: #fcfaf9; margin: 0.5rem 0 0 0;">
                    {breach_details["message"]}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #66bb6a, #81c784); 
                        padding: 1.2rem; 
                        border-radius: 12px; 
                        border: 1px solid #48e5c240;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
                <h3 style="color: #fcfaf9; margin: 0;">✅ Password Not Found in Breaches</h3>
                <p style="color: #fcfaf9; margin: 0.5rem 0 0 0;">
                    {breach_details["message"]}
                </p>
            </div>
        """, unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("Password Analyzer")
    mode = st.sidebar.radio(
        "Choose Mode",
        ["Test Password", "Generate Password", "Model Information"]
    )
    
    if mode == "Test Password":
        st.title("🔒 Password Strength Analyzer")
        st.markdown("Enter a password to analyze its strength and get detailed metrics.")
        
        # Load the model
        model = load_model()
        if model is None:
            st.error("Could not load the model. Please make sure the model file exists.")
            return

        # Password input
        password = st.text_input("Enter your password:", type="password")
        
        if password:
            # Extract features
            features = extract_features(password)
            
            # Make prediction
            features_df = pd.DataFrame([features])
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            confidence = probabilities[model.classes_.tolist().index(prediction)]
            
            # Check for breaches
            with st.spinner("Checking for password breaches..."):
                breach_details = breach_checker.get_breach_details(password)
            
            # Display breach check results
            display_breach_check_results(breach_details)
            
            # Display results in three columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Password Strength")
                strength_color = get_strength_color(prediction)
                st.markdown(f"""
                    <div class="strength-box {strength_color}">
                        <h3>Predicted Strength: {prediction.replace('_', ' ').title()}</h3>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display strength probability chart
                st.plotly_chart(create_strength_probability_chart(
                    probabilities, model.classes_),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Password Metrics")
                metrics = {
                    "Length": features['length'],
                    "Lowercase Characters": features['lowercase'],
                    "Uppercase Characters": features['uppercase'],
                    "Digits": features['digits'],
                    "Special Characters": features['special'],
                    "Entropy": f"{features['entropy']:.2f}"
                }
                
                for metric, value in metrics.items():
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>{metric}</h4>
                            <h3>{value}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display character distribution chart
                st.plotly_chart(create_character_distribution_chart(features),
                              use_container_width=True)
            
            with col3:
                st.subheader("Entropy Analysis")
                st.plotly_chart(create_entropy_gauge(features),
                              use_container_width=True)
                
                # Password strength tips
                st.subheader("Tips to Improve Password Strength")
                tips = []
                
                if features['length'] < 12:
                    tips.append("🔹 Increase password length to at least 12 characters")
                if features['lowercase'] == 0:
                    tips.append("🔹 Add lowercase letters")
                if features['uppercase'] == 0:
                    tips.append("🔹 Add uppercase letters")
                if features['digits'] == 0:
                    tips.append("🔹 Add numbers")
                if features['special'] == 0:
                    tips.append("🔹 Add special characters")
                if features['entropy'] < 3:
                    tips.append("🔹 Use more diverse characters to increase entropy")
                
                if tips:
                    for tip in tips:
                        st.write(tip)
                else:
                    st.success("Your password follows all the best practices! 🎉")
            
            # Generate and display recommendations
            recommendations, strong_passwords = generate_password_recommendations(
                features, password, prediction, confidence)
            display_recommendations(recommendations, strong_passwords, prediction, confidence)

    elif mode == "Generate Password":
        st.title("🔑 Password Generator")
        st.markdown("Generate a strong password based on your criteria.")
        
        # Password criteria
        st.subheader("Password Criteria")
        col1, col2 = st.columns(2)
        
        with col1:
            length = st.slider("Password Length", 8, 32, 12)
            use_lowercase = st.checkbox("Include Lowercase Letters", value=True)
            use_uppercase = st.checkbox("Include Uppercase Letters", value=True)
            use_digits = st.checkbox("Include Numbers", value=True)
            use_special = st.checkbox("Include Special Characters", value=True)
        
        with col2:
            exclude_similar = st.checkbox("Exclude Similar Characters (i, l, 1, o, 0)", value=False)
            exclude_ambiguous = st.checkbox("Exclude Ambiguous Characters ({, }, [, ], etc.)", value=False)
            require_unique = st.checkbox("Require All Characters to be Unique", value=False)
            num_passwords = st.number_input("Number of Passwords to Generate", 1, 10, 1)
        
        criteria = {
            'length': length,
            'lowercase': use_lowercase,
            'uppercase': use_uppercase,
            'digits': use_digits,
            'special': use_special,
            'exclude_similar': exclude_similar,
            'exclude_ambiguous': exclude_ambiguous,
            'require_unique': require_unique
        }
        
        if st.button("Generate Password"):
            passwords = []
            for _ in range(num_passwords):
                password = generate_password_by_criteria(criteria)
                if password:
                    if require_unique and len(set(password)) != len(password):
                        # Regenerate if unique characters are required
                        continue
                    passwords.append(password)
            
            if passwords:
                st.markdown("### Generated Passwords")
                for i, password in enumerate(passwords, 1):
                    st.markdown(f"""
                        <div class="metric-card" style="cursor: pointer;" 
                             onclick="navigator.clipboard.writeText('{password}')">
                            <h4>Password {i}</h4>
                            <p style="color: {COLORS['turquoise']}; 
                                      font-family: monospace; 
                                      font-size: 1.2rem; 
                                      margin: 0;">
                                {password}
                            </p>
                            <small style="color: {COLORS['seasalt']}80;">
                                Click to copy
                            </small>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Analyze the first generated password
                features = extract_features(passwords[0])
                features_df = pd.DataFrame([features])
                model = load_model()
                if model:
                    prediction = model.predict(features_df)[0]
                    probabilities = model.predict_proba(features_df)[0]
                    confidence = probabilities[model.classes_.tolist().index(prediction)]
                    
                    st.markdown("### Password Strength Analysis")
                    strength_color = get_strength_color(prediction)
                    st.markdown(f"""
                        <div class="strength-box {strength_color}">
                            <h3>Predicted Strength: {prediction.replace('_', ' ').title()}</h3>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Check for breaches
                    with st.spinner("Checking for password breaches..."):
                        breach_details = breach_checker.get_breach_details(passwords[0])
                    
                    # Display breach check results
                    display_breach_check_results(breach_details)
                    
                    # Display character distribution
                    st.plotly_chart(create_character_distribution_chart(features),
                                  use_container_width=True)

    else:  # Model Information
        metrics = load_model_metrics()
        display_model_info(metrics)

if __name__ == "__main__":
    main() 