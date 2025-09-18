import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix, r2_score, mean_absolute_error
from sklearn.metrics import precision_recall_curve, average_precision_score
import openml
from datasets import load_dataset
from scipy import stats

# Set page config
st.set_page_config(page_title="ML Playground", layout="wide")

# Title
st.title("🚀 Machine Learning Playground")

# Clear session state button for debugging
if st.button("🔄 Clear Session State"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
st.markdown("**Advanced ML with Comprehensive EDA & Model Evaluation**")

# Dataset loading functions
def load_openml_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # Handle different data types and ensure proper DataFrame creation
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        
        if isinstance(y, pd.Series):
            df[dataset.default_target_attribute] = y
        else:
            df[dataset.default_target_attribute] = pd.Series(y)
        
        # Clean column names (remove special characters)
        df.columns = [str(col).replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load OpenML dataset {dataset_id}: {str(e)}")
        return None

def load_huggingface_dataset(name):
    try:
        dataset = load_dataset(name)
        df = dataset['train'].to_pandas()
        return df.head(2000)
    except Exception as e:
        st.error(f"Failed to load Hugging Face dataset {name}: {str(e)}")
        return None

# Preprocessing function
def preprocess_data(df, target_col):
    """Enhanced preprocessing with better error handling"""
    try:
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Remove rows where target is NaN
        df_processed = df_processed.dropna(subset=[target_col])
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle missing values in features
        # For numeric columns, use median imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
        
        # For categorical columns, use most frequent imputation
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        # Encode categorical features
        for col in categorical_cols:
            if col in X.columns:  # Check if column still exists after imputation
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Ensure all data is numeric
        X = X.astype(float)
        y = y.astype(float)
        
        # Remove any remaining NaN values
        X = X.fillna(X.median())
        
        # Check if we have valid data
        if X.empty or len(X) == 0:
            raise ValueError("No valid data after preprocessing")
        
        # Ensure we have at least 2 samples for train/test split
        if len(X) < 2:
            raise ValueError("Not enough data for train/test split")
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
        
    except Exception as e:
        # Fallback to simple preprocessing
        st.warning(f"Advanced preprocessing failed, using fallback: {e}")
        
        # Simple fallback
        df_processed = df.dropna()
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Simple encoding
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        
        # Ensure numeric types
        X = X.astype(float)
        y = y.astype(float)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

# Model training function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}
    is_classification = len(set(y_train)) < 20

    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            acc = accuracy_score(y_test, preds)
            
            results[name] = {
                "accuracy": acc,
                "predictions": preds,
                "probabilities": pred_proba,
                "model": model
            }
            trained_models[name] = model
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            results[name] = {
                "MSE": mse, 
                "RMSE": np.sqrt(mse),
                "MAE": mae,
                "R2": r2,
                "predictions": preds,
                "model": model
            }
            trained_models[name] = model
    
    return results, trained_models

# Visualization functions
def plot_correlation_matrix(df):
    """📊 Correlation Heatmap for numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                square=True, cbar_kws={"shrink": .8})
    plt.title("📊 Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    """📈 Feature Distribution plots (histograms for numeric, bar plots for categorical)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Calculate subplot dimensions
    total_cols = len(numeric_cols) + len(categorical_cols)
    if total_cols == 0:
        return None
    
    n_cols = min(3, total_cols)
    n_rows = (total_cols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot numeric features as histograms
    for col in numeric_cols:
        if plot_idx < len(axes):
            axes[plot_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[plot_idx].set_title(f"📊 {col} (Numeric)", fontweight='bold')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1
    
    # Plot categorical features as bar plots
    for col in categorical_cols:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values)
            axes[plot_idx].set_title(f"📊 {col} (Categorical)", fontweight='bold')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("📈 Feature Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_boxplots_numeric_vs_target(df, target_col):
    """📉 Boxplots for numeric features vs target (if target is numeric)"""
    if target_col not in df.columns:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) == 0 or df[target_col].dtype not in [np.number]:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[i])
            axes[i].set_title(f"📉 {col} vs {target_col}", fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"📉 Numeric Features vs Target ({target_col})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """✅ ROC Curve for binary classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='steelblue', lw=3, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.7, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """✅ Confusion Matrix for classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true),
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name="Model", top_n=10):
    """✅ Feature Importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    n_features = len(importances)
    
    # Adjust top_n if there are fewer features
    actual_top_n = min(top_n, n_features)
    indices = np.argsort(importances)[::-1][:actual_top_n]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Pastel1(np.linspace(0, 1, actual_top_n))
    bars = ax.bar(range(actual_top_n), importances[indices], color=colors, alpha=0.8)
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(actual_top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, model_name="Model"):
    """✅ Residual Plot for regression"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'✅ Residuals vs Predicted - {model_name}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot of Residuals - {model_name}', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_prediction_probabilities(y_pred_proba, model_name="Model"):
    """📊 Probability Distribution for Classification Predictions"""
    if y_pred_proba.ndim == 1:
        # Binary classification
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_pred_proba, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'📊 Probability Distribution - {model_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        # Multi-class classification
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(y_pred_proba.shape[1]):
            ax.hist(y_pred_proba[:, i], bins=30, alpha=0.6, label=f'Class {i}')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'📊 Probability Distribution - {model_name}', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """📊 Scatter plot of actual vs predicted for regression"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue', s=50)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicted vs Actual - {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_regression_metrics_comparison(results, model_names):
    """📊 Bar chart comparing regression metrics across models"""
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Calculate metrics for each model
    rmse_values = []
    mae_values = []
    r2_values = []
    
    for model_name in model_names:
        if 'RMSE' in results[model_name]:
            rmse_values.append(results[model_name]['RMSE'])
        else:
            rmse_values.append(0)
            
        if 'MAE' in results[model_name]:
            mae_values.append(results[model_name]['MAE'])
        else:
            mae_values.append(0)
            
        if 'R2' in results[model_name]:
            r2_values.append(results[model_name]['R2'])
        else:
            r2_values.append(0)
    
    # Create bars with pastel colors
    bars1 = ax.bar(x - width, rmse_values, width, label='RMSE', alpha=0.8, color='lightcoral')
    bars2 = ax.bar(x, mae_values, width, label='MAE', alpha=0.8, color='lightblue')
    bars3 = ax.bar(x + width, r2_values, width, label='R²', alpha=0.8, color='lightgreen')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Values', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_residuals_histogram(y_true, y_pred, model_name="Model"):
    """📊 Residuals distribution histogram for regression"""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Histogram of residuals
    n, bins, patches = ax.hist(residuals, bins=30, alpha=0.6, edgecolor='black', color='lightblue', density=True)
    
    # Add KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, kde(x), 'r-', linewidth=2, label='KDE')
    
    # Add normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal_curve, 'g--', linewidth=2, label=f'Normal (μ={mu:.3f})')
    
    ax.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'Residuals Distribution - {model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    """📊 Precision-Recall curve for classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color='forestgreen', lw=3, 
            label=f'PR curve (AP = {avg_precision:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_classification_report_table(y_true, y_pred, model_name="Model"):
    """📊 Classification report as a formatted table"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create DataFrame for better display
    df_report = pd.DataFrame(report).transpose()
    
    # Remove 'support' column for the table (we'll show it separately)
    if 'support' in df_report.columns:
        support_values = df_report['support']
        df_report = df_report.drop('support', axis=1)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    headers = ['Class', 'Precision', 'Recall', 'F1-Score']
    
    for class_name, metrics in df_report.iterrows():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            table_data.append([
                class_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}"
            ])
    
    # Add macro and weighted averages
    table_data.append([
        'Macro Avg',
        f"{df_report.loc['macro avg', 'precision']:.3f}",
        f"{df_report.loc['macro avg', 'recall']:.3f}",
        f"{df_report.loc['macro avg', 'f1-score']:.3f}"
    ])
    table_data.append([
        'Weighted Avg',
        f"{df_report.loc['weighted avg', 'precision']:.3f}",
        f"{df_report.loc['weighted avg', 'recall']:.3f}",
        f"{df_report.loc['weighted avg', 'f1-score']:.3f}"
    ])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title(f'📊 Classification Report - {model_name}', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# Main App Interface
# Dataset selection
st.sidebar.header("📂 Dataset Options")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Upload CSV", "OpenML", "Hugging Face"])

# Add info about dataset sources
with st.sidebar.expander("ℹ️ Dataset Info"):
    st.write("""
    **📁 Upload CSV**: Upload your own dataset
    
    **🔬 OpenML**: 1000+ datasets for ML research
    - Classification: Iris, Wine, Breast Cancer, etc.
    - Regression: Boston Housing, Auto MPG, etc.
    
    **🤗 Hugging Face**: NLP and text datasets
    - Sentiment: IMDB, Amazon, Yelp reviews
    - Classification: News, DBpedia
    - GLUE benchmark tasks
    """)

df = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif dataset_source == "OpenML":
    st.sidebar.subheader("📋 Popular Datasets")
    
    # Popular OpenML datasets
    popular_datasets = {
        "Iris (Classification)": "61",
        "Wine (Classification)": "187", 
        "Breast Cancer (Classification)": "13",
        "Diabetes (Classification)": "37",
        "Heart Disease (Classification)": "45",
        "Boston Housing (Regression)": "529",
        "Auto MPG (Regression)": "9",
        "Abalone (Regression)": "183",
        "CPU Performance (Regression)": "562",
        "Servo (Regression)": "871",
        "Glass Identification (Classification)": "40",
        "Sonar (Classification)": "151",
        "Vehicle (Classification)": "54",
        "Segment (Classification)": "36",
        "Waveform (Classification)": "60"
    }
    
    # Dataset selection method
    selection_method = st.sidebar.radio("Choose selection method:", ["Popular Datasets", "Custom ID"])
    
    if selection_method == "Popular Datasets":
        selected_dataset = st.sidebar.selectbox(
            "Select a dataset:",
            options=list(popular_datasets.keys()),
            index=0
        )
        openml_id = popular_datasets[selected_dataset]
        st.sidebar.caption(f"Dataset ID: {openml_id}")
    else:
        openml_id = st.sidebar.text_input("Enter OpenML dataset ID", "61")
        st.sidebar.caption("👉 Example: 61 = Iris dataset")
    
    if st.sidebar.button("Load from OpenML"):
        with st.spinner(f"🔄 Loading dataset {openml_id}..."):
            df = load_openml_dataset(openml_id)
            if df is None:
                st.error("❌ Failed to load dataset from OpenML")
            else:
                st.success(f"✅ Successfully loaded dataset {openml_id}")

elif dataset_source == "Hugging Face":
    st.sidebar.subheader("📋 Popular NLP Datasets")
    
    # Popular Hugging Face datasets
    popular_hf_datasets = {
        "IMDB Reviews (Sentiment)": "imdb",
        "Amazon Reviews (Sentiment)": "amazon_polarity",
        "Yelp Reviews (Sentiment)": "yelp_review_full",
        "AG News (Classification)": "ag_news",
        "DBpedia (Classification)": "dbpedia_14",
        "20 Newsgroups (Classification)": "newsgroup",
        "SQuAD (QA)": "squad",
        "CoLA (Grammar)": "glue",
        "SST-2 (Sentiment)": "glue",
        "MRPC (Paraphrase)": "glue",
        "QQP (Paraphrase)": "glue",
        "MNLI (NLI)": "glue",
        "QNLI (NLI)": "glue",
        "RTE (NLI)": "glue",
        "WNLI (NLI)": "glue"
    }
    
    # Dataset selection method
    hf_selection_method = st.sidebar.radio("Choose selection method:", ["Popular Datasets", "Custom Name"])
    
    if hf_selection_method == "Popular Datasets":
        selected_hf_dataset = st.sidebar.selectbox(
            "Select a dataset:",
            options=list(popular_hf_datasets.keys()),
            index=0
        )
        hf_name = popular_hf_datasets[selected_hf_dataset]
        st.sidebar.caption(f"Dataset: {hf_name}")
    else:
        hf_name = st.sidebar.text_input("Enter Hugging Face dataset name", "imdb")
        st.sidebar.caption("👉 Example: imdb = sentiment analysis dataset")
    
    if st.sidebar.button("Load from Hugging Face"):
        with st.spinner(f"🔄 Loading dataset {hf_name}..."):
            df = load_huggingface_dataset(hf_name)
            if df is None:
                st.error("❌ Failed to load dataset from Hugging Face")
            else:
                st.success(f"✅ Successfully loaded dataset {hf_name}")

if df is not None and not df.empty:
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head(10))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(df.columns))
    with col2:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
    with col3:
        st.metric("Categorical Features", len(df.select_dtypes(include=['object', 'category']).columns))

    target_col = st.selectbox("🎯 Select target column", df.columns)
    
    if target_col:
        # Determine problem type
        unique_targets = df[target_col].nunique()
        is_classification = unique_targets < 20
        
        st.info(f"**Problem Type:** {'Classification' if is_classification else 'Regression'} ({unique_targets} unique values)")
        
        # Preprocessing
        try:
            with st.spinner("🔄 Preprocessing data..."):
                X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
            
            # Check if preprocessing was successful
            if X_train is None or len(X_train) == 0:
                st.error("❌ Preprocessing failed: No valid data after preprocessing")
                st.stop()
            
            st.success(f"✅ Data preprocessed successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.info("💡 Try selecting a different target column or check your data for issues")
            st.stop()
        
        # Model Training
        try:
            with st.spinner("🤖 Training models..."):
                results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
            
            if not results or not trained_models:
                st.error("❌ Model training failed")
                st.stop()
            
            # Store results in session state for access in other tabs
            st.session_state['results'] = results
            st.session_state['trained_models'] = trained_models
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['task_type'] = 'classification' if is_classification else 'regression'
                
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.info("💡 This might be due to data type issues or insufficient data")
            st.stop()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model Results", "📈 Predictions", "ℹ️ Dataset Info"])
        
        with tab1:
            st.header("📊 Exploratory Data Analysis (EDA)")
            
            # Correlation Matrix
            st.subheader("📊 Correlation Heatmap")
            corr_plot = plot_correlation_matrix(df)
            if corr_plot:
                st.pyplot(corr_plot)
            else:
                st.warning("Not enough numeric features for correlation analysis")
            
            # Feature Distributions
            st.subheader("📈 Feature Distributions")
            dist_plot = plot_feature_distributions(df)
            if dist_plot:
                st.pyplot(dist_plot)
            else:
                st.warning("No features available for distribution analysis")
            
            # Boxplots for numeric vs target
            if is_classification:
                st.subheader("📉 Numeric Features vs Target")
                box_plot = plot_boxplots_numeric_vs_target(df, target_col)
                if box_plot:
                    st.pyplot(box_plot)
                else:
                    st.warning("Target column is not numeric or no numeric features available")
        
        with tab2:
            st.subheader("📊 Model Results & Evaluation")
            
            if 'results' in st.session_state and st.session_state['results']:
                results = st.session_state['results']
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
                is_classification = st.session_state.get('task_type', 'classification') == 'classification'
                
                if is_classification:
                    # CLASSIFICATION DASHBOARD FLOW
                    
                    # 1. Metrics Table
                    st.subheader("📊 Classification Metrics")
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                    
                    metrics_data = []
                    for model_name, metrics in results.items():
                        preds = metrics['predictions']
                        acc = metrics['accuracy']
                        
                        # Calculate precision, recall, f1 (macro average for multi-class)
                        precision = precision_score(y_test, preds, average='macro', zero_division=0)
                        recall = recall_score(y_test, preds, average='macro', zero_division=0)
                        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                        
                        # Calculate ROC-AUC for binary classification
                        roc_auc = None
                        if len(np.unique(y_test)) == 2 and 'probabilities' in metrics:
                            try:
                                roc_auc = roc_auc_score(y_test, metrics['probabilities'][:, 1])
                            except:
                                roc_auc = None
                        
                        metrics_data.append({
                            'Model': model_name,
                            'Accuracy': round(acc, 2),
                            'Precision': round(precision, 2),
                            'Recall': round(recall, 2),
                            'F1': round(f1, 2),
                            'ROC-AUC': round(roc_auc, 2) if roc_auc is not None else 'N/A'
                        })
                    
                    # Find best model based on F1 score
                    best_model_name = max(metrics_data, key=lambda x: x['F1'])['Model']
                    best_f1_score = max(metrics_data, key=lambda x: x['F1'])['F1']
                    
                    # Display classification metrics table
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # 2. Best Model Announcement
                    st.success(f"🏆 **{best_model_name}** selected as best model with F1-score of {best_f1_score:.2f} - chosen for balanced precision-recall performance")
                    
                    # 3. Best Model Visuals (side-by-side, medium size)
                    st.subheader("📈 Best Model Analysis")
                    best_model_metrics = results[best_model_name]
                    
                    # Normalized Confusion Matrix
                    col1, col2 = st.columns(2)
                    with col1:
                        # Create normalized confusion matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test, best_model_metrics['predictions'])
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test),
                                    cbar_kws={'shrink': 0.8})
                        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
                        ax.set_title(f'Normalized Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.caption("*Normalized Confusion Matrix*")
                        st.info("**💡 Insight:** Values show proportion of correct predictions per class - diagonal values closer to 1.0 indicate better performance.")
                    
                    # Per-class Precision/Recall/F1 bar chart
                    with col2:
                        from sklearn.metrics import classification_report
                        report = classification_report(y_test, best_model_metrics['predictions'], output_dict=True)
                        
                        # Extract per-class metrics
                        classes = []
                        precision_vals = []
                        recall_vals = []
                        f1_vals = []
                        
                        for class_name, metrics in report.items():
                            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                classes.append(str(class_name))
                                precision_vals.append(metrics['precision'])
                                recall_vals.append(metrics['recall'])
                                f1_vals.append(metrics['f1-score'])
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(6, 4))
                        x = np.arange(len(classes))
                        width = 0.25
                        
                        bars1 = ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color='lightcoral')
                        bars2 = ax.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='lightblue')
                        bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='lightgreen')
                        
                        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                        ax.set_title(f'Per-class Metrics - {best_model_name}', fontsize=14, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(classes, rotation=45, ha='right')
                        ax.legend()
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.set_ylim(0, 1.1)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.caption("*Per-class Precision/Recall/F1*")
                        st.info("**💡 Insight:** Higher bars indicate better performance for each class - balanced performance across classes is ideal.")
                    
                    # 4. Class distribution (only if imbalance exists)
                    class_counts = pd.Series(y_test).value_counts()
                    max_count = class_counts.max()
                    min_count = class_counts.min()
                    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                    
                    if imbalance_ratio > 2:  # Show if there's significant imbalance
                        st.subheader("📊 Class Distribution Analysis")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            bars = ax.bar(class_counts.index.astype(str), class_counts.values, 
                                        color='lightcoral', alpha=0.7)
                            ax.set_xlabel('Class')
                            ax.set_ylabel('Count')
                            ax.set_title('Class Distribution in Test Set')
                            ax.tick_params(axis='x', rotation=45)
                            
                            # Add value labels on bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                       f'{int(height)}', ha='center', va='bottom')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.caption("*Class Distribution*")
                            st.info(f"**💡 Insight:** Imbalance ratio of {imbalance_ratio:.1f}:1 detected - consider class balancing techniques for better performance.")
                    
                    # 5. Feature Importance (if available)
                    if hasattr(best_model_metrics['model'], 'feature_importances_'):
                        st.subheader("🌳 Feature Importance")
                        col5, col6 = st.columns(2)
                        
                        with col5:
                            feat_imp_plot = plot_feature_importance(best_model_metrics['model'], X_train.columns, best_model_name)
                            if feat_imp_plot:
                                st.pyplot(feat_imp_plot)
                            st.caption("*Feature Importance*")
                            importances = best_model_metrics['model'].feature_importances_
                            top_features = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)[:3]
                            st.info(f"**💡 Insight:** Top features: {', '.join([f'{feat} ({imp:.3f})' for feat, imp in top_features])}")
                    
                    # 6. Key Understanding & Conclusion
                    st.subheader("🎯 Key Understanding & Conclusion")
                    
                    # Calculate key metrics for conclusion
                    from sklearn.metrics import classification_report
                    report = classification_report(y_test, best_model_metrics['predictions'], output_dict=True)
                    overall_accuracy = report['accuracy']
                    macro_f1 = report['macro avg']['f1-score']
                    
                    if len(np.unique(y_test)) == 2:
                        roc_auc = roc_auc_score(y_test, best_model_metrics['probabilities'][:, 1])
                        avg_precision = average_precision_score(y_test, best_model_metrics['probabilities'][:, 1])
                        
                        st.success(f"""
                        **🏆 Model Performance Summary:**
                        
                        The **{best_model_name}** model demonstrates {'excellent' if overall_accuracy > 0.9 else 'good' if overall_accuracy > 0.8 else 'moderate'} performance with:
                        - **Overall Accuracy:** {overall_accuracy:.2f} ({overall_accuracy*100:.1f}% correct predictions)
                        - **F1-Score:** {macro_f1:.2f} (balanced precision-recall performance)
                        - **ROC-AUC:** {roc_auc:.2f} ({'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'fair'} discriminative ability)
                        - **Average Precision:** {avg_precision:.2f} (precision-recall balance)
                        
                        **Key Insights:**
                        - The model shows {'strong' if macro_f1 > 0.8 else 'moderate'} ability to distinguish between classes
                        - {'High' if roc_auc > 0.8 else 'Moderate'} confidence in predictions with good separation between classes
                        - {'Well-balanced' if abs(avg_precision - macro_f1) < 0.1 else 'Imbalanced'} precision-recall trade-off
                        """)
                    else:
                        st.success(f"""
                        **🏆 Model Performance Summary:**
                        
                        The **{best_model_name}** model demonstrates {'excellent' if overall_accuracy > 0.9 else 'good' if overall_accuracy > 0.8 else 'moderate'} performance with:
                        - **Overall Accuracy:** {overall_accuracy:.2f} ({overall_accuracy*100:.1f}% correct predictions)
                        - **Macro F1-Score:** {macro_f1:.2f} (average performance across all classes)
                        
                        **Key Insights:**
                        - The model shows {'strong' if macro_f1 > 0.8 else 'moderate'} ability to distinguish between multiple classes
                        - {'Good' if overall_accuracy > 0.8 else 'Moderate'} generalization across different class categories
                        - Confusion matrix reveals {'excellent' if overall_accuracy > 0.9 else 'good' if overall_accuracy > 0.8 else 'moderate'} class separation
                        """)
                
                else:
                    # REGRESSION DASHBOARD FLOW
                    
                    # 1. Metrics Table
                    st.subheader("📊 Regression Metrics")
                    metrics_data = []
                    for model_name, metrics in results.items():
                        metrics_data.append({
                            'Model': model_name,
                            'R²': round(metrics['R2'], 3),
                            'MAE': round(metrics['MAE'], 3),
                            'MSE': round(metrics['MSE'], 3),
                            'RMSE': round(metrics['RMSE'], 3)
                        })
                    
                    # Find best model based on RMSE (lower is better)
                    best_model_name = min(metrics_data, key=lambda x: x['RMSE'])['Model']
                    best_rmse_score = min(metrics_data, key=lambda x: x['RMSE'])['RMSE']
                    
                    # Display regression metrics table
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # 2. Best Model Announcement
                    st.success(f"🏆 **{best_model_name}** selected as best model with RMSE of {best_rmse_score:.3f}")
                    
                    # 3. Best Model Visuals
                    st.subheader("📈 Best Model Analysis")
                    best_model_metrics = results[best_model_name]
                    
                    # Residual Distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        resid_hist_plot = plot_residuals_histogram(y_test, best_model_metrics['predictions'], best_model_name)
                        if resid_hist_plot:
                            st.pyplot(resid_hist_plot)
                        st.caption("*Residual Distribution*")
                        st.info("**💡 Insight:** Residuals centered around 0 with normal distribution indicate good model fit.")
                    
                    # Predicted vs Actual
                    with col2:
                        actual_pred_plot = plot_actual_vs_predicted(y_test, best_model_metrics['predictions'], best_model_name)
                        if actual_pred_plot:
                            st.pyplot(actual_pred_plot)
                        st.caption("*Predicted vs Actual*")
                        st.info("**💡 Insight:** Points close to diagonal line indicate strong predictive power.")
                    
                    # Residuals vs Predicted
                    col3, col4 = st.columns(2)
                    with col3:
                        # Create residuals vs predicted plot
                        residuals = y_test - best_model_metrics['predictions']
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.scatter(best_model_metrics['predictions'], residuals, alpha=0.6, color='steelblue')
                        ax.axhline(y=0, color='red', linestyle='--')
                        ax.set_xlabel('Predicted Values')
                        ax.set_ylabel('Residuals')
                        ax.set_title(f'Residuals vs Predicted - {best_model_name}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.caption("*Residuals vs Predicted*")
                        st.info("**💡 Insight:** Random scatter around zero indicates good model assumptions.")
                    
                    # 4. Model Comparison Bar Chart
                    with col4:
                        model_names = list(results.keys())
                        metrics_comparison_plot = plot_regression_metrics_comparison(results, model_names)
                        if metrics_comparison_plot:
                            st.pyplot(metrics_comparison_plot)
                        st.caption("*Model Comparison*")
                        st.info("**💡 Insight:** Lower RMSE/MAE and higher R² indicate better performance.")
                    
                    # 5. Feature Importance (Optional)
                    if hasattr(best_model_metrics['model'], 'feature_importances_'):
                        st.subheader("🌳 Feature Importance")
                        col5, col6 = st.columns(2)
                        with col5:
                            feat_imp_plot = plot_feature_importance(best_model_metrics['model'], X_train.columns, best_model_name)
                            if feat_imp_plot:
                                st.pyplot(feat_imp_plot)
                            st.caption("*Feature Importance*")
                            importances = best_model_metrics['model'].feature_importances_
                            top_features = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)[:3]
                            st.info(f"**💡 Insight:** Top features: {', '.join([f'{feat} ({imp:.3f})' for feat, imp in top_features])}")
                    else:
                        st.subheader("🌳 Feature Importance")
                        st.info(f"**Note:** {best_model_name} does not provide feature importance scores. This is common for linear models like Linear Regression, where coefficients can be interpreted as feature importance.")
                    
                    # Key Understanding & Conclusion for Regression
                    st.subheader("🎯 Key Understanding & Conclusion")
                    
                    # Calculate key metrics for conclusion
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    mse = mean_squared_error(y_test, best_model_metrics['predictions'])
                    mae = mean_absolute_error(y_test, best_model_metrics['predictions'])
                    r2 = r2_score(y_test, best_model_metrics['predictions'])
                    rmse = np.sqrt(mse)
                    
                    st.success(f"""
                    **🏆 Model Performance Summary:**
                    
                    The **{best_model_name}** model demonstrates {'excellent' if r2 > 0.9 else 'good' if r2 > 0.8 else 'moderate'} performance with:
                    - **R² Score:** {r2:.3f} ({'excellent' if r2 > 0.9 else 'good' if r2 > 0.8 else 'moderate'} variance explained)
                    - **RMSE:** {rmse:.3f} ({'low' if rmse < 0.1 else 'moderate' if rmse < 0.5 else 'high'} prediction error)
                    - **MAE:** {mae:.3f} ({'low' if mae < 0.1 else 'moderate' if mae < 0.5 else 'high'} average error)
                    
                    **Key Insights:**
                    - The model explains {'most' if r2 > 0.8 else 'some' if r2 > 0.5 else 'little'} of the variance in the target variable
                    - {'Low' if rmse < 0.1 else 'Moderate' if rmse < 0.5 else 'High'} prediction errors suggest {'excellent' if rmse < 0.1 else 'good' if rmse < 0.5 else 'moderate'} model fit
                    - Residual analysis shows {'well-distributed' if abs(mae - rmse/2) < 0.1 else 'some bias'} in predictions
                    """)
            
            else:
                st.info("Please train models first to see results and evaluation metrics.")
        
        with tab3:
            st.header("📈 Predictions")
            
            if 'trained_models' in st.session_state and st.session_state['trained_models']:
                trained_models = st.session_state['trained_models']
                X_train = st.session_state['X_train']
                is_classification = st.session_state.get('task_type', 'classification') == 'classification'
            
                # Model Selection for Predictions
                model_names = list(trained_models.keys())
                selected_model_name = st.selectbox("Select model for predictions:", model_names)
                selected_model = trained_models[selected_model_name]
                
                # Prediction Type Selection
                pred_type = st.radio("Choose prediction type:", ["Manual Input", "Batch Upload"])
                
                if pred_type == "Manual Input":
                    st.subheader("🔮 Manual Predictions")
                    st.write("Enter feature values for prediction:")
                    
                    # Create input form
                    input_data = {}
                    feature_cols = X_train.columns
                    
                    cols = st.columns(min(3, len(feature_cols)))
                    for i, col in enumerate(feature_cols):
                        with cols[i % len(cols)]:
                            if X_train[col].dtype in ['int64', 'float64']:
                                # Numeric input
                                min_val = float(X_train[col].min())
                                max_val = float(X_train[col].max())
                                input_data[col] = st.number_input(
                                    f"{col}", 
                                    min_value=min_val, 
                                    max_value=max_val, 
                                    value=float(X_train[col].median())
                                )
                            else:
                                # Categorical input
                                unique_vals = X_train[col].unique()
                                input_data[col] = st.selectbox(f"{col}", unique_vals)
                    
                    if st.button("🔮 Make Prediction", type="primary"):
                        # Convert to DataFrame
                        input_df = pd.DataFrame([input_data])
                        
                        # Make prediction
                        if is_classification:
                            pred = selected_model.predict(input_df)[0]
                            pred_proba = selected_model.predict_proba(input_df)[0]
                            classes = selected_model.classes_
                            
                            # Store prediction results in session state
                            st.session_state['prediction_result'] = {
                                'prediction': pred,
                                'probabilities': pred_proba,
                                'classes': classes,
                                'model_name': selected_model_name,
                                'input_data': input_data,
                                'is_classification': True
                            }
                        else:
                            pred = selected_model.predict(input_df)[0]
                            
                            # Store prediction results in session state
                            st.session_state['prediction_result'] = {
                                'prediction': pred,
                                'model_name': selected_model_name,
                                'input_data': input_data,
                                'is_classification': False
                            }
                
                else:  # Batch Upload
                    st.subheader("📁 Batch Predictions")
                    uploaded_pred_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
                    
                    if uploaded_pred_file is not None:
                        pred_df = pd.read_csv(uploaded_pred_file)
                        st.write("**Uploaded Data Preview:**")
                        st.dataframe(pred_df.head())
                        
                        if st.button("🔮 Make Batch Predictions", type="primary"):
                            # Ensure same features as training data
                            missing_cols = set(X_train.columns) - set(pred_df.columns)
                            if missing_cols:
                                st.error(f"Missing columns: {missing_cols}")
                            else:
                                # Make predictions
                                pred_df_subset = pred_df[X_train.columns]
                                predictions = selected_model.predict(pred_df_subset)
                                
                                # Add predictions to dataframe
                                result_df = pred_df.copy()
                                result_df['Prediction'] = predictions
                                
                                if is_classification:
                                    probabilities = selected_model.predict_proba(pred_df_subset)
                                    for i, class_name in enumerate(selected_model.classes_):
                                        result_df[f'Probability_{class_name}'] = probabilities[:, i]
                                
                                # Store batch results in session state
                                st.session_state['batch_prediction_result'] = {
                                    'results_df': result_df,
                                    'model_name': selected_model_name,
                                    'is_classification': is_classification,
                                    'classes': selected_model.classes_ if is_classification else None
                                }
                
                # Display Prediction Results
                if 'prediction_result' in st.session_state:
                    result = st.session_state['prediction_result']
                    
                    st.subheader("🎯 Prediction Results")
                    
                    # Create highlighted prediction card
                    if result['is_classification']:
                        # Classification Results
                        pred = result['prediction']
                        pred_proba = result['probabilities']
                        classes = result['classes']
                        
                        # Main prediction card
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                                padding: 20px;
                                border-radius: 10px;
                                color: white;
                                margin: 10px 0;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            ">
                                <h3 style="margin: 0; color: white;">🎯 Predicted Class</h3>
                                <h1 style="margin: 10px 0; color: white; font-size: 2.5em;">{pred}</h1>
                                <p style="margin: 0; opacity: 0.9;">Using {result['model_name']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Confidence score
                            max_prob = max(pred_proba)
                            confidence_color = "green" if max_prob > 0.8 else "orange" if max_prob > 0.6 else "red"
                            st.markdown(f"""
                            <div style="
                                background: {confidence_color};
                                padding: 15px;
                                border-radius: 10px;
                                color: white;
                                margin: 10px 0;
                                text-align: center;
                            ">
                                <h4 style="margin: 0; color: white;">Confidence</h4>
                                <h2 style="margin: 5px 0; color: white;">{max_prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Class Probabilities Visualization
                        st.subheader("📊 Class Probabilities")
                        
                        # Create probability bar chart
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': pred_proba
                        }).sort_values('Probability', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#ff6b6b' if x == pred else '#4ecdc4' for x in prob_df['Class']]
                        bars = ax.barh(prob_df['Class'], prob_df['Probability'], color=colors, alpha=0.8)
                        
                        # Add value labels on bars
                        for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
                            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{prob:.3f}', va='center', fontweight='bold')
                        
                        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Class', fontsize=12, fontweight='bold')
                        ax.set_title(f'Class Probabilities - {result["model_name"]}', fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 1)
                        ax.grid(True, alpha=0.3, axis='x')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Detailed probability table
                        st.subheader("📋 Detailed Probabilities")
                        prob_table = pd.DataFrame({
                            'Class': classes,
                            'Probability': pred_proba,
                            'Percentage': [f"{p:.1%}" for p in pred_proba]
                        }).sort_values('Probability', ascending=False)
                        st.dataframe(prob_table, use_container_width=True)
                        
                    else:
                        # Regression Results
                        pred = result['prediction']
                        
                        # Main prediction card
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            margin: 15px 0;
                            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                            text-align: center;
                        ">
                            <h3 style="margin: 0; color: white;">🎯 Predicted Value</h3>
                            <h1 style="margin: 15px 0; color: white; font-size: 3em; font-weight: bold;">{pred:.3f}</h1>
                            <p style="margin: 0; opacity: 0.9; font-size: 1.1em;">Using {result['model_name']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Description for regression
                        st.subheader("📝 Prediction Description")
                        st.info(f"""
                        **Model Prediction:** The {result['model_name']} model predicts a value of **{pred:.3f}** for the given input features.
                        
                        This prediction is based on the patterns learned from the training data. The model has analyzed the relationships between your input features and the target variable to make this estimate.
                        """)
                
                # Display Batch Prediction Results
                if 'batch_prediction_result' in st.session_state:
                    batch_result = st.session_state['batch_prediction_result']
                    
                    st.subheader("📊 Batch Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(batch_result['results_df']))
                    with col2:
                        if batch_result['is_classification']:
                            unique_preds = batch_result['results_df']['Prediction'].nunique()
                            st.metric("Unique Classes", unique_preds)
                        else:
                            mean_pred = batch_result['results_df']['Prediction'].mean()
                            st.metric("Average Prediction", f"{mean_pred:.3f}")
                    with col3:
                        if batch_result['is_classification']:
                            most_common = batch_result['results_df']['Prediction'].mode()[0]
                            st.metric("Most Common Class", most_common)
                        else:
                            std_pred = batch_result['results_df']['Prediction'].std()
                            st.metric("Std Deviation", f"{std_pred:.3f}")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(batch_result['results_df'], use_container_width=True)
                    
                    # Download results
                    csv = batch_result['results_df'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Predictions",
                        data=csv,
                        file_name=f"batch_predictions_{batch_result['model_name'].replace(' ', '_')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Batch prediction visualizations
                    if batch_result['is_classification']:
                        # Class distribution
                        st.subheader("📊 Prediction Distribution")
                        class_counts = batch_result['results_df']['Prediction'].value_counts()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(class_counts.index.astype(str), class_counts.values, 
                                    color='skyblue', alpha=0.8, edgecolor='black')
                        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                        ax.set_title('Distribution of Predicted Classes', fontsize=14, fontweight='bold')
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        # Prediction distribution histogram
                        st.subheader("📊 Prediction Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(batch_result['results_df']['Prediction'], bins=20, 
                               color='lightgreen', alpha=0.7, edgecolor='black')
                        ax.set_xlabel('Predicted Value', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                        ax.set_title('Distribution of Predicted Values', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
            
            else:
                st.info("👆 Please train models first to make predictions!")
        
        with tab4:
            st.header("ℹ️ Dataset Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("📈 Statistical Summary")
                st.dataframe(df.describe())
            
            st.subheader("🔍 Missing Values")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                st.bar_chart(missing_data)
            else:
                st.success("✅ No missing values found!")
            
            st.subheader("🎯 Target Variable Analysis")
            st.write(f"**Target Column:** {target_col}")
            st.write(f"**Unique Values:** {df[target_col].nunique()}")
            st.write(f"**Data Type:** {df[target_col].dtype}")
            
            if is_classification:
                st.write("**Class Distribution:**")
                class_counts = df[target_col].value_counts()
                st.bar_chart(class_counts)
            else:
                st.write("**Target Statistics:**")
                st.write(df[target_col].describe())



