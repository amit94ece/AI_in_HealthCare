import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class RandomBinaryClassifier:
    """
    Random baseline binary classifier.
    """
    def __init__(self):
        self.prob_positive = None
        
    def fit(self, X, y):
        self.prob_positive = np.mean(y)
        return self
        
    def predict(self, X):
        return np.random.choice([0, 1], size=len(X), 
                              p=[1-self.prob_positive, self.prob_positive])
    
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1 - self.prob_positive
        probs[:, 1] = self.prob_positive
        return probs

def prepare_data_splits(features, target_col='hospital_expire_flag', 
                       test_size=0.15, val_size=0.15):
    """
    Rebalance and split data into train, validation, and test sets.
    """
    # Extract features and target
    id_cols = [col for col in ['subject_id', 'hadm_id', 'icustay_id'] 
              if col in features.columns]
    X = features.drop(columns=id_cols + [target_col])
    y = features[target_col]
    
    # Rebalance data by downsampling majority class
    pos_indices = y[y == 1].index
    neg_indices = y[y == 0].index
    
    # Determine minority class
    if len(pos_indices) < len(neg_indices):
        minority_indices = pos_indices
        majority_indices = neg_indices
    else:
        minority_indices = neg_indices
        majority_indices = pos_indices
    
    # Randomly sample from majority class to match minority class size
    np.random.seed(42)
    sampled_majority_indices = np.random.choice(
        majority_indices, len(minority_indices), replace=False
    )
    
    # Combine minority and sampled majority indices
    balanced_indices = np.concatenate([minority_indices, sampled_majority_indices])
    
    # Get balanced dataset
    X_balanced = X.loc[balanced_indices]
    y_balanced = y.loc[balanced_indices]
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_balanced, test_size=(test_size + val_size), random_state=42
    )
    
    # Split temp set into validation and test sets
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X.columns

def evaluate_sklearn_models(X_train, X_test, y_train, y_test):
    """
    Evaluate scikit-learn binary classifiers.
    """
    # Initialize models
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'LinearSVC': LinearSVC(random_state=42, max_iter=1000),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'RandomBaselineClassifier': RandomBinaryClassifier()
    }
    
    results = {}
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        logging.info(f"Training and evaluating {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
            # Scale to [0, 1]
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (
                y_pred_proba.max() - y_pred_proba.min()
            )
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = y_pred
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
    
    return results, trained_models

def visualize_results(results, metric='accuracy'):
    """
    Visualize results for a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    # Extract metric values and sort
    metric_values = {name: metrics[metric] for name, metrics in results.items()}
    sorted_items = sorted(metric_values.items(), key=lambda x: x[1])
    
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create horizontal bar plot
    colors = ['skyblue' for _ in names]
    plt.barh(names, values, color=colors)
    
    # Add labels and title
    plt.xlabel(f'{metric.capitalize()} Score')
    plt.title(f'{metric.capitalize()} score comparison of binary classification models')
    plt.xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{metric}.png')
    plt.close()

def visualize_feature_importance(model_name, feature_importance, top_n=20):
    """
    Visualize feature importance for a given model.
    """
    # Sort and get top N features
    top_features = feature_importance.nlargest(top_n, 'importance')
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features - {model_name}')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_feature_importance.png')
    plt.close()

def compare_feature_importance(lr_importance, rf_importance, top_n=20):
    """
    Create a comparison plot of feature importance between Logistic Regression and Random Forest.
    """
    # Merge the two feature importance dataframes
    comparison = pd.merge(
        lr_importance.rename(columns={'importance': 'Logistic Regression'}),
        rf_importance.rename(columns={'importance': 'Random Forest'}),
        on='feature'
    )
    
    # Calculate mean importance and sort
    comparison['mean_importance'] = comparison[['Logistic Regression', 'Random Forest']].mean(axis=1)
    comparison = comparison.nlargest(top_n, 'mean_importance')
    
    # Create comparison plot
    plt.figure(figsize=(12, 10))
    x = np.arange(len(comparison))
    width = 0.35
    
    plt.barh(x + width/2, comparison['Random Forest'], width, 
            label='Random Forest', color='lightcoral')
    plt.barh(x - width/2, comparison['Logistic Regression'], width,
            label='Logistic Regression', color='lightblue')
    
    plt.yticks(x, comparison['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    plt.close()

def visualize_model_results(results_df):
    """
    Create a comprehensive visualization of model results across all metrics.
    """
    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    
    plt.figure(figsize=(15, 10))
    x = np.arange(len(results_df))
    width = 0.15
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        plt.bar(x + offset, results_df[metric], width, label=metric)
        multiplier += 1
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison Across Metrics')
    plt.xticks(x + width * 2, results_df.index, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i in x:
        for j, metric in enumerate(metrics):
            plt.text(i + width * j, results_df.iloc[i][metric],
                    f'{results_df.iloc[i][metric]:.3f}',
                    ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig('model_results_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load processed data
        logging.info("Loading processed data...")
        processed_data = pd.read_csv('processed_chf_data.csv')
        logging.info(f"Loaded data shape: {processed_data.shape}")
        
        # Prepare data splits
        logging.info("Preparing data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(
            processed_data
        )
        
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Validation set shape: {X_val.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # Train and evaluate models
        logging.info("Training and evaluating models...")
        results, trained_models = evaluate_sklearn_models(X_train, X_test, y_train, y_test)
        
        # Log results
        logging.info("\nModel Performance:")
        for model_name, metrics in results.items():
            logging.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                logging.info(f"{metric_name}: {value:.3f}")
        
        # Visualize results for different metrics
        for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            logging.info(f"Creating visualization for {metric}...")
            visualize_results(results, metric=metric)
        
        # Visualize feature importance for individual models
        feature_importance_dict = {}
        for model_name in ['LogisticRegression', 'RandomForestClassifier']:
            if model_name in trained_models:
                model = trained_models[model_name]
                if hasattr(model, 'coef_'):
                    importances = model.coef_[0]
                elif hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    continue
                    
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(importances)
                }).sort_values('importance', ascending=False)
                
                feature_importance_dict[model_name] = feature_importance
                
                # Visualize individual model feature importance
                visualize_feature_importance(model_name, feature_importance)
                
                # Save feature importance to CSV
                feature_importance.to_csv(f'{model_name}_feature_importance.csv')
        
        # Compare feature importance between models
        if len(feature_importance_dict) == 2:
            compare_feature_importance(
                feature_importance_dict['LogisticRegression'],
                feature_importance_dict['RandomForestClassifier']
            )
        
        # Visualize model results comparison
        results_df = pd.DataFrame(results).T
        visualize_model_results(results_df)
        
        # Display final results table
        logging.info("\nFinal Model Results:")
        logging.info("\n" + str(results_df.round(3)))
        
        # Save results
        results_df.to_csv('model_results.csv')
        logging.info("Saved model results to model_results.csv")
        
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Stack trace:", exc_info=True)
        raise
