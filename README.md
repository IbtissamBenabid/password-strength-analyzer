# Password Strength Analyzer

A machine learning-based password strength analyzer that evaluates password security and provides detailed feedback and recommendations.

## Features

- **Password Strength Analysis**: Evaluates passwords using machine learning and provides detailed metrics
- **Password Generation**: Generates strong passwords based on user-defined criteria
- **Visual Analytics**: Interactive charts and visualizations for password metrics
- **Strength Recommendations**: Detailed suggestions for improving password strength

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/password-strength-analyzer.git
cd password-strength-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run password_strength_ui.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the application:
   - Test Password: Analyze the strength of existing passwords
   - Generate Password: Create new strong passwords
   - Model Information: View model performance metrics

## Project Structure

```
password-strength-analyzer/
├── password_strength_ui.py    # Main Streamlit application
├── password_analysis.py       # Core password analysis functionality
├── requirements.txt           # Project dependencies
├── models/                    # Trained model files
│   └── password_strength_model.joblib
└── README.md                 # Project documentation
```

## Features in Detail

### Password Strength Analysis
- Character distribution analysis
- Entropy calculation
- Machine learning-based strength prediction
- Confidence scores
- Visual strength indicators

### Password Generation
- Customizable length
- Character type selection
- Special character options
- Similar character exclusion
- Unique character requirement

### Model Information
- Model performance metrics
- Feature importance visualization
- Confusion matrix
- Per-class performance analysis

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web application framework
- scikit-learn for machine learning capabilities
- Plotly for interactive visualizations 