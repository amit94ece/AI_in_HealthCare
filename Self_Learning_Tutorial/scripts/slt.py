import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


# Load dataset
df = pd.read_csv('../data/healthcare_dataset.csv')

# Display basic information
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Calculate length of stay from admission to discharge date
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], format='%m/%d/%y')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], format='%m/%d/%y')
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Basic statistics of length of stay
print(df['Length of Stay'].describe())


# Set up the visualization style
# plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# # Distribution of Length of Stay
# sns.histplot(df['Length of Stay'], kde=True, ax=axes[0, 0])
# axes[0, 0].set_title('Distribution of Length of Stay')
# axes[0, 0].set_xlabel('Days')

# # Length of Stay by Medical Condition
# sns.boxplot(x='Medical Condition', y='Length of Stay', data=df, ax=axes[0, 1])
# axes[0, 1].set_title('Length of Stay by Medical Condition')
# axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# # Length of Stay by Age Groups
# df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], 
#                          labels=['<18', '18-35', '36-50', '51-65', '>65'])
# sns.boxplot(x='Age Group', y='Length of Stay', data=df, ax=axes[1, 0])
# axes[1, 0].set_title('Length of Stay by Age Group')

# # Length of Stay by Admission Type
# sns.boxplot(x='Admission Type', y='Length of Stay', data=df, ax=axes[1, 1])
# axes[1, 1].set_title('Length of Stay by Admission Type')

# plt.tight_layout()
# plt.show()

# # Correlation between Age and Length of Stay
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Age', y='Length of Stay', hue='Medical Condition', data=df)
# plt.title('Age vs Length of Stay by Medical Condition')
# plt.show()

# # Length of Stay by Gender and Medical Condition
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Medical Condition', y='Length of Stay', hue='Gender', data=df)
# plt.title('Length of Stay by Gender and Medical Condition')
# plt.xticks(rotation=45)
# plt.show()

# Distribution of Length of Stay
sns.histplot(df['Length of Stay'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Length of Stay')
axes[0, 0].set_xlabel('Days')

# Length of Stay by Medical Condition
sns.boxplot(x='Medical Condition', y='Length of Stay', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Length of Stay by Medical Condition')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# Length of Stay by Age Groups
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], 
                         labels=['<18', '18-35', '36-50', '51-65', '>65'])
sns.boxplot(x='Age Group', y='Length of Stay', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Length of Stay by Age Group')

# Length of Stay by Admission Type
sns.boxplot(x='Admission Type', y='Length of Stay', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Length of Stay by Admission Type')

plt.tight_layout()
plt.show()

# Correlation between Age and Length of Stay
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Length of Stay', hue='Medical Condition', data=df)
plt.title('Age vs Length of Stay by Medical Condition')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Length of Stay by Gender and Medical Condition
plt.figure(figsize=(12, 6))
sns.boxplot(x='Medical Condition', y='Length of Stay', hue='Gender', data=df)
plt.title('Length of Stay by Gender and Medical Condition')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 1. Feature Engineering with proper handling of negative/invalid values
def create_features(df):
    features_df = df.copy()
    
    # Handle negative or zero values before log transform
    features_df['Billing_Log'] = np.log1p(np.maximum(features_df['Billing Amount'], 0))
    
    # Create other features safely
    features_df['Age_Squared'] = features_df['Age'] ** 2
    features_df['Age_Medical'] = features_df['Age'] * pd.factorize(features_df['Medical Condition'])[0]
    features_df['Age_Admission'] = features_df['Age'] * pd.factorize(features_df['Admission Type'])[0]
    features_df['Admission_Month'] = features_df['Date of Admission'].dt.month
    features_df['Admission_DayOfWeek'] = features_df['Date of Admission'].dt.dayofweek
    
    return features_df

# 2. Update preprocessing pipeline
categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type',
                   'Insurance Provider', 'Test Results', 'Medication']
numerical_cols = ['Age', 'Age_Squared', 'Billing_Log', 'Age_Medical', 'Age_Admission',
                 'Admission_Month', 'Admission_DayOfWeek']

# Updated preprocessor with sparse_output=False
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])

# 3. Create and prepare the feature matrix
X = create_features(df)
X = X[numerical_cols + categorical_cols]
y = df['Length of Stay']

# 4. Handle outliers
def remove_outliers(X, y, n_sigmas=3):
    z_scores = np.abs((y - y.mean()) / y.std())
    mask = z_scores < n_sigmas
    return X[mask], y[mask]

X, y = remove_outliers(X, y)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Define models with updated parameters
models = {
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ]),
    
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        ))
    ]),
    
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        ))
    ])
}

# 7. Train and evaluate models with error handling
results = {}
for name, model in models.items():
    try:
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name}:")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        print()
        
        # Feature importance for tree-based models
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': numerical_cols + [f"{col}_{val}" for col in categorical_cols 
                                          for val in model.named_steps['preprocessor']
                                          .named_transformers_['cat'].categories_[
                                              categorical_cols.index(col)][1:]],
                'importance': model.named_steps['regressor'].feature_importances_
            })
            print(f"\nTop 10 important features for {name}:")
            print(feature_importance.sort_values('importance', ascending=False).head(10))
            print()
            
    except Exception as e:
        print(f"Error training {name}: {str(e)}")
        continue

# Hyperparameter tuning for the best performing model
def tune_model(X_train, y_train, X_test, y_test):
    # Define parameter grids for each model type
    param_grids = {
        'Random Forest': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 15, 20, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__subsample': [0.8, 0.9, 1.0]
        },
        'XGBoost': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_child_weight': [1, 3, 5],
            'regressor__subsample': [0.8, 0.9, 1.0],
            'regressor__colsample_bytree': [0.8, 0.9, 1.0]
        }
    }

    # Get the best performing model from initial results
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = models[best_model_name]
    
    print(f"\nTuning {best_model_name}...")
    
    # Create and fit RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_grids[best_model_name],
        n_iter=20,  # Number of parameter settings sampled
        cv=5,       # 5-fold cross-validation
        scoring=['r2', 'neg_mean_squared_error'],
        refit='r2', # Use R2 to select the best model
        n_jobs=-1,  # Use all available cores
        verbose=2,  # Provide more information about the tuning process
        random_state=42
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nBest Parameters:", random_search.best_params_)
    print(f"Best Cross-validation R2 Score: {random_search.best_score_:.4f}")
    print("\nTest Set Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    # Compare with original model
    print("\nImprovement over original model:")
    original_r2 = results[best_model_name]['R2']
    improvement = (r2 - original_r2) / abs(original_r2) * 100
    print(f"R2 Score improvement: {improvement:.2f}%")
    
    return best_model, random_search.best_params_

# Execute the tuning
try:
    best_tuned_model, best_params = tune_model(X_train, y_train, X_test, y_test)
    
    # Feature importance for the tuned model
    if hasattr(best_tuned_model.named_steps['regressor'], 'feature_importances_'):
        feature_names = (numerical_cols + 
                        [f"{col}_{val}" for col, vals in 
                         zip(categorical_cols, 
                             best_tuned_model.named_steps['preprocessor']
                             .named_transformers_['cat']
                             .categories_) 
                         for val in vals[1:]])
        
        importances = best_tuned_model.named_steps['regressor'].feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_imp.head(15))
        plt.title('Top 15 Most Important Features (Tuned Model)')
        plt.tight_layout()
        plt.show()
        
        print("\nTop 10 Most Important Features:")
        print(feature_imp.head(10))

except Exception as e:
    print(f"Error during model tuning: {str(e)}")
    print("Traceback:")
    import traceback
    traceback.print_exc()

def plot_model_comparison(original_results, tuned_results, model_name):
    """Plot before and after R2 scores"""
    plt.figure(figsize=(10, 6))
    scores = [original_results[model_name]['R2'], tuned_results['R2']]
    labels = ['Original Model', 'Tuned Model']
    
    bars = plt.bar(labels, scores, color=['lightblue', 'darkblue'])
    plt.title(f'R2 Score Comparison for {model_name}')
    plt.ylabel('R2 Score')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Plot actual vs predicted values with regression line"""
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    line_min = min(min(y_test), min(y_pred))
    line_max = max(max(y_test), max(y_pred))
    plt.plot([line_min, line_max], [line_min, line_max], 'r--', label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "b-", alpha=0.8, label='Regression Line')
    
    plt.xlabel('Actual Length of Stay')
    plt.ylabel('Predicted Length of Stay')
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.legend()
    
    # Add R2 score to plot
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred, model_name):
    """Plot residuals analysis"""
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True, alpha=0.3)
    
    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Residuals Analysis - {model_name}')
    plt.tight_layout()
    plt.show()

# Store original predictions and results
original_predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    original_predictions[name] = model.predict(X_test)

# Find the best performing model based on R2 score
best_model_name = max(results, key=lambda k: results[k]['R2'])
best_model = models[best_model_name]

print(f"\nBest performing model: {best_model_name}")
print(f"Original R2 score: {results[best_model_name]['R2']:.3f}")


# After tuning, get the best model predictions
best_predictions = best_tuned_model.predict(X_test)

# Create visualizations
print("\nModel Performance Comparison:")
plot_model_comparison(results, 
                     {'R2': r2_score(y_test, best_predictions)},
                     best_model_name)

print("\nOriginal Model - Actual vs Predicted:")
plot_actual_vs_predicted(y_test, original_predictions[best_model_name], 
                        f"Original {best_model_name}")

print("\nTuned Model - Actual vs Predicted:")
plot_actual_vs_predicted(y_test, best_predictions, 
                        f"Tuned {best_model_name}")

print("\nResiduals Analysis - Original Model:")
plot_residuals(y_test, original_predictions[best_model_name], 
              f"Original {best_model_name}")

print("\nResiduals Analysis - Tuned Model:")
plot_residuals(y_test, best_predictions, 
              f"Tuned {best_model_name}")

# Print detailed metrics comparison
print("\nDetailed Metrics Comparison:")
print(f"\n{best_model_name} Performance:")
print("Original Model:")
print(f"R2 Score: {results[best_model_name]['R2']:.3f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.3f}")
print(f"MAE: {results[best_model_name]['MAE']:.3f}")

print("\nTuned Model:")
print(f"R2 Score: {r2_score(y_test, best_predictions):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, best_predictions)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, best_predictions):.3f}")

# Calculate prediction intervals for Random Forest
def prediction_interval(model, X, percentile=95):
    """
    Calculate prediction intervals using Random Forest's individual tree predictions
    """
    if not isinstance(model.named_steps['regressor'], RandomForestRegressor):
        raise ValueError("Model must be a Random Forest")
    
    # Get predictions from all trees
    predictions = []
    rf_model = model.named_steps['regressor']
    X_processed = model.named_steps['preprocessor'].transform(X)
    
    for estimator in rf_model.estimators_:
        predictions.append(estimator.predict(X_processed))
    
    predictions = np.array(predictions)
    
    # Calculate intervals
    lower = np.percentile(predictions, (100 - percentile) / 2, axis=0)
    upper = np.percentile(predictions, 100 - (100 - percentile) / 2, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    
    return mean_pred, lower, upper

# Plot predictions with intervals
if best_model_name == 'Random Forest':
    try:
        plt.figure(figsize=(12, 6))
        
        # Sort for better visualization
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test.iloc[sort_idx]
        
        # Calculate prediction intervals
        mean_pred, lower, upper = prediction_interval(best_tuned_model, X_test)
        
        # Sort predictions and intervals
        mean_pred_sorted = mean_pred[sort_idx]
        lower_sorted = lower[sort_idx]
        upper_sorted = upper[sort_idx]
        
        # Plot actual values
        plt.scatter(range(len(y_test)), y_test_sorted, 
                   alpha=0.5, label='Actual', color='blue')
        
        # Plot predicted values
        plt.plot(range(len(y_test)), mean_pred_sorted, 
                'r-', label='Predicted', alpha=0.7)
        
        # Plot prediction intervals
        plt.fill_between(range(len(y_test)), 
                        lower_sorted, upper_sorted,
                        alpha=0.2, color='red',
                        label='95% Prediction Interval')
        
        plt.xlabel('Sample Index (Sorted by Actual Values)')
        plt.ylabel('Length of Stay')
        plt.title('Random Forest Predictions with 95% Prediction Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R2 score to plot
        r2 = r2_score(y_test, mean_pred)
        plt.text(0.02, 0.98, f'R² = {r2:.3f}', 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
        # Print interval statistics
        interval_width = np.mean(upper - lower)
        coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
        
        print("\nPrediction Interval Statistics:")
        print(f"Average interval width: {interval_width:.2f} days")
        print(f"Empirical coverage: {coverage:.1%}")
        
        # Additional analysis of predictions
        errors = y_test.values - mean_pred
        mae = mean_absolute_error(y_test, mean_pred)
        rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
        
        print("\nPrediction Error Statistics:")
        print(f"Mean Absolute Error: {mae:.2f} days")
        print(f"Root Mean Square Error: {rmse:.2f} days")
        print(f"Error Standard Deviation: {np.std(errors):.2f} days")
        
        # Plot error distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(errors, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Count')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating prediction interval plot: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
else:
    print(f"\nPrediction intervals are only available for Random Forest models. Current model: {best_model_name}")
