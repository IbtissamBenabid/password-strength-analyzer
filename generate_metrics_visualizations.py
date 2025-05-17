import os
import joblib
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require a GUI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

def load_metrics():
    """Load model metrics from the saved file."""
    try:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        metrics_path = os.path.join(models_dir, 'model_metrics.joblib')
        if os.path.exists(metrics_path):
            return joblib.load(metrics_path)
        return None
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")
        return None

def generate_confusion_matrix(metrics):
    """Generate and save confusion matrix visualization."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong'],
                yticklabels=['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong'])
    plt.title('Confusion Matrix', pad=20, fontsize=14)
    plt.xlabel('Predicted', labelpad=10)
    plt.ylabel('Actual', labelpad=10)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def generate_feature_importance(metrics):
    """Generate and save feature importance visualization."""
    # Create DataFrame for feature importance
    importance_data = {
        'Feature': metrics['feature_names'],
        'Importance': metrics['feature_importance']
    }
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance in Password Strength Prediction', pad=20, fontsize=14)
    plt.xlabel('Importance', labelpad=10)
    plt.ylabel('Features', labelpad=10)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def generate_performance_metrics(metrics):
    """Generate and save performance metrics visualization."""
    # Calculate F1 score
    f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    
    # Create metrics DataFrame
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], f1_score]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics_df['Metric'], metrics_df['Value'])
    plt.title('Model Performance Metrics', pad=20, fontsize=14)
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'performance_metrics.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def generate_per_class_metrics(metrics):
    """Generate and save per-class performance metrics."""
    class_names = ['Very Weak', 'Weak', 'Average', 'Strong', 'Very Strong']
    precision_scores = []
    recall_scores = []
    
    for i in range(len(class_names)):
        true_positives = metrics['confusion_matrix'][i][i]
        total_actual = sum(metrics['confusion_matrix'][i])
        total_predicted = sum(metrics['confusion_matrix'][:, i])
        
        precision = true_positives / total_predicted if total_predicted > 0 else 0
        recall = true_positives / total_actual if total_actual > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Create DataFrame
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision_scores,
        'Recall': recall_scores
    })
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, class_metrics['Precision'], width, label='Precision')
    plt.bar(x + width/2, class_metrics['Recall'], width, label='Recall')
    
    plt.title('Per-Class Performance Metrics', pad=20, fontsize=14)
    plt.xlabel('Password Strength Class', labelpad=10)
    plt.ylabel('Score', labelpad=10)
    plt.xticks(x, class_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'per_class_metrics.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def main():
    # Load metrics
    metrics = load_metrics()
    if metrics is None:
        print("No metrics found. Please run the training script first.")
        return
    
    # Generate all visualizations
    print("Generating visualizations...")
    generate_confusion_matrix(metrics)
    generate_feature_importance(metrics)
    generate_performance_metrics(metrics)
    generate_per_class_metrics(metrics)
    print(f"Visualizations saved to {results_dir}")

if __name__ == "__main__":
    main() 