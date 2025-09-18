import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc, confusion_matrix, r2_score, mean_absolute_error
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report

# Set page config
st.set_page_config(page_title="ML Playground", layout="wide")

# Title
st.title("🚀 Machine Learning Playground")
st.markdown("**Simple ML App with Visualizations**")

# Sample data generation function
def create_sample_data():
    """Create sample datasets for demonstration"""
    np.random.seed(42)
    
    # Sample classification data (Iris-like)
    n_samples = 150
    data = {
        'sepal_length': np.random.normal(5.8, 0.8, n_samples),
        'sepal_width': np.random.normal(3.0, 0.4, n_samples),
        'petal_length': np.random.normal(3.8, 1.8, n_samples),
        'petal_width': np.random.normal(1.2, 0.8, n_samples),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
    }
    
    return pd.DataFrame(data)

# Visualization functions
def plot_correlation_matrix(df):
    """Correlation Heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    """Feature Distribution plots"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    total_cols = len(numeric_cols) + len(categorical_cols)
    if total_cols == 0:
        return None
    
    n_cols = min(3, total_cols)
    n_rows = (total_cols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot numeric features
    for col in numeric_cols:
        if plot_idx < len(axes):
            axes[plot_idx].hist(df[col].dropna(), bins=20, alpha=0.7)
            axes[plot_idx].set_title(f"{col} (Numeric)")
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Frequency")
            plot_idx += 1
    
    # Plot categorical features
    for col in categorical_cols:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts()
            axes[plot_idx].bar(range(len(value_counts)), value_counts.values)
            axes[plot_idx].set_title(f"{col} (Categorical)")
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].set_xticks(range(len(value_counts)))
            axes[plot_idx].set_xticklabels(value_counts.index, rotation=45)
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """ROC Curve for binary classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name="Model"):
    """Feature Importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    n_features = len(importances)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_features), importances)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """Actual vs Predicted for regression"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Actual vs Predicted - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_regression_metrics_comparison(results, model_names):
    """Bar chart comparing regression metrics across models"""
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
    
    # Create bars
    bars1 = ax.bar(x - width, rmse_values, width, label='RMSE', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, mae_values, width, label='MAE', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + width, r2_values, width, label='R²', alpha=0.8, color='lightgreen')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Values', fontsize=12, fontweight='bold')
    ax.set_title('Regression Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_residuals_histogram(y_true, y_pred, model_name="Model"):
    """Residuals distribution histogram for regression"""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of residuals
    n, bins, patches = ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    
    # Add normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = len(residuals) * (bins[1] - bins[0]) * (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
    
    ax.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Residuals Distribution - {model_name}', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    from scipy import stats
    ax.text(0.02, 0.98, f'Mean: {mu:.3f}\nStd: {sigma:.3f}\nSkewness: {stats.skew(residuals):.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    """Precision-Recall curve for classification"""
    if len(np.unique(y_true)) != 2:
        return None
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2, 
            label=f'PR curve (AP = {avg_precision:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Preprocessing function
def preprocess_data(df, target_col):
    """Simple preprocessing"""
    df_processed = df.dropna()
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))
    
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

# Main App Interface
st.sidebar.header("📂 Dataset Options")

# Dataset selection
dataset_option = st.sidebar.radio("Choose dataset:", ["Sample Data", "Upload CSV"])

df = None

if dataset_option == "Sample Data":
    if st.sidebar.button("Generate Sample Data"):
        df = create_sample_data()
        st.sidebar.success("✅ Sample data generated!")

elif dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

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
        st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))

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
            
            st.success(f"✅ Data preprocessed successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {str(e)}")
            st.stop()
        
        # Model Training
        try:
            with st.spinner("🤖 Training models..."):
                results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
                
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.stop()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model Results", "📈 Predictions"])
        
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
        
        with tab2:
            st.header("🤖 Model Results & Evaluation")
            
            # Model Results - Metrics Table
            st.subheader("📊 Model Performance Metrics")
            
            if is_classification:
                # Classification Metrics Table
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                # Calculate additional metrics for classification
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
                        'Accuracy': round(acc, 3),
                        'Precision': round(precision, 3),
                        'Recall': round(recall, 3),
                        'F1 Score': round(f1, 3),
                        'ROC-AUC': round(roc_auc, 3) if roc_auc is not None else 'N/A'
                    })
                
                # Find best model based on F1 score
                best_model_name = max(metrics_data, key=lambda x: x['F1 Score'])['Model']
                best_f1_score = max(metrics_data, key=lambda x: x['F1 Score'])['F1 Score']
                
                # Display classification metrics table with highlighting
                metrics_df = pd.DataFrame(metrics_data)
                
                # Highlight best model row
                def highlight_best_model(row):
                    if row['Model'] == best_model_name:
                        return ['background-color: #e6f3ff'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(metrics_df.style.apply(highlight_best_model, axis=1), use_container_width=True)
                
                # Best Model Announcement
                st.subheader("🏆 Best Model Selection")
                st.success(f"**{best_model_name}** was selected as the best model because it achieved the highest F1-score of **{best_f1_score}** compared to other models.")
                
                # Get best model metrics
                best_model_metrics = results[best_model_name]
                
            else:
                # Regression Metrics Table
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
                
                # Display regression metrics table with highlighting
                metrics_df = pd.DataFrame(metrics_data)
                
                # Highlight best model row
                def highlight_best_model(row):
                    if row['Model'] == best_model_name:
                        return ['background-color: #e6f3ff'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(metrics_df.style.apply(highlight_best_model, axis=1), use_container_width=True)
                
                # Best Model Announcement
                st.subheader("🏆 Best Model Selection")
                st.success(f"**{best_model_name}** was selected as the best model because it achieved the lowest RMSE of **{best_rmse_score}** compared to other models.")
                
                # Get best model metrics
                best_model_metrics = results[best_model_name]
            
            # Evaluation Visualizations
            st.subheader("📈 Detailed Analysis of Best Model")
            
            if is_classification:
                # Classification Visualizations for Best Model Only
                
                # Confusion Matrix
                st.write("#### Confusion Matrix Analysis")
                conf_plot = plot_confusion_matrix(y_test, best_model_metrics['predictions'], best_model_name)
                if conf_plot:
                    st.pyplot(conf_plot)
                    st.caption(f"*Figure 1: Confusion Matrix for {best_model_name}*")
                
                # Confusion Matrix Explanation
                cm = confusion_matrix(y_test, best_model_metrics['predictions'])
                if len(cm) == 2:  # Binary classification
                    tn, fp, fn, tp = cm.ravel()
                    st.info(f"**Analysis:** The model correctly predicted {tp} positive cases and {tn} negative cases. It made {fp} false positive errors (predicted positive when actual was negative) and {fn} false negative errors (predicted negative when actual was positive).")
                else:  # Multi-class
                    st.info(f"**Analysis:** The diagonal elements show correct predictions for each class. Off-diagonal elements indicate misclassifications between different classes.")
                
                # ROC Curve (for binary classification)
                if len(np.unique(y_test)) == 2 and 'probabilities' in best_model_metrics:
                    st.write("#### ROC Curve Analysis")
                    roc_plot = plot_roc_curve(y_test, best_model_metrics['probabilities'][:, 1], best_model_name)
                    if roc_plot:
                        st.pyplot(roc_plot)
                        st.caption(f"*Figure 2: ROC Curve for {best_model_name}*")
                    
                    # ROC Curve Explanation
                    roc_auc = roc_auc_score(y_test, best_model_metrics['probabilities'][:, 1])
                    st.info(f"**Analysis:** The ROC curve shows the trade-off between True Positive Rate (sensitivity) and False Positive Rate (1-specificity). An AUC of {roc_auc:.3f} indicates {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'fair' if roc_auc > 0.7 else 'poor'} discriminative ability.")
                
                # Precision-Recall Curve
                if 'probabilities' in best_model_metrics:
                    st.write("#### Precision-Recall Curve Analysis")
                    pr_plot = plot_precision_recall_curve(y_test, best_model_metrics['probabilities'][:, 1], best_model_name)
                    if pr_plot:
                        st.pyplot(pr_plot)
                        st.caption(f"*Figure 3: Precision-Recall Curve for {best_model_name}*")
                    
                    # PR Curve Explanation
                    precision, recall, _ = precision_recall_curve(y_test, best_model_metrics['probabilities'][:, 1])
                    avg_precision = average_precision_score(y_test, best_model_metrics['probabilities'][:, 1])
                    st.info(f"**Analysis:** The Precision-Recall curve shows the trade-off between precision and recall. An Average Precision of {avg_precision:.3f} indicates how well the model balances precision and recall across different thresholds.")
            
            else:
                # Regression Visualizations for Best Model Only
                
                # Model Comparison Bar Chart
                st.write("#### Model Performance Comparison")
                model_names = list(results.keys())
                metrics_comparison_plot = plot_regression_metrics_comparison(results, model_names)
                if metrics_comparison_plot:
                    st.pyplot(metrics_comparison_plot)
                    st.caption("*Figure 1: Model Performance Comparison - Lower values are better for MAE, MSE, RMSE; Higher values are better for R²*")
                
                st.info(f"**Analysis:** This comparison clearly shows why {best_model_name} was selected as the best model, with superior performance across multiple metrics.")
                
                # Residuals Distribution
                st.write("#### Residuals Distribution Analysis")
                resid_hist_plot = plot_residuals_histogram(y_test, best_model_metrics['predictions'], best_model_name)
                if resid_hist_plot:
                    st.pyplot(resid_hist_plot)
                    st.caption(f"*Figure 2: Residuals Distribution for {best_model_name}*")
                
                st.info("**Analysis:** Ideally, residuals should be centered around 0 with a normal distribution. This plot shows how well the model's errors are distributed - if residuals are spread evenly, errors are random, meaning the model generalizes well.")
                
                # Actual vs Predicted
                st.write("#### Predicted vs Actual Values Analysis")
                actual_pred_plot = plot_actual_vs_predicted(y_test, best_model_metrics['predictions'], best_model_name)
                if actual_pred_plot:
                    st.pyplot(actual_pred_plot)
                    st.caption(f"*Figure 3: Predicted vs Actual Values for {best_model_name}*")
                
                st.info("**Analysis:** Points close to the diagonal line (y=x) indicate strong predictive power. The closer the points are to this line, the better the model's predictions match the actual values.")
            
            # Feature Importance Analysis (for tree-based models)
            if hasattr(best_model_metrics['model'], 'feature_importances_'):
                st.subheader("🌳 Feature Importance Analysis")
                feat_imp_plot = plot_feature_importance(
                    best_model_metrics['model'], X_train.columns, best_model_name
                )
                if feat_imp_plot:
                    st.pyplot(feat_imp_plot)
                    st.caption(f"*Figure 4: Feature Importance for {best_model_name}*")
                
                # Get top 3 most important features
                importances = best_model_metrics['model'].feature_importances_
                feature_names = X_train.columns
                top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
                
                st.info(f"**Analysis:** The most important features for {best_model_name} are: {', '.join([f'{feat} ({imp:.3f})' for feat, imp in top_features])}. These features contribute most significantly to the model's decision-making process.")
            else:
                st.subheader("🌳 Feature Importance Analysis")
                st.info(f"**Note:** {best_model_name} does not provide feature importance scores. This is common for linear models like Logistic Regression, where coefficients can be interpreted as feature importance.")
        
        with tab3:
            st.header("📈 Predictions")
            
            # Model Selection for Predictions
            model_names = list(trained_models.keys())
            selected_model_name = st.selectbox("Select model for predictions:", model_names)
            selected_model = trained_models[selected_model_name]
            
            # Manual Predictions
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
            
            if st.button("🔮 Make Prediction"):
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                if is_classification:
                    pred = selected_model.predict(input_df)[0]
                    pred_proba = selected_model.predict_proba(input_df)[0]
                    
                    st.success(f"**Prediction:** {pred}")
                    st.write("**Class Probabilities:**")
                    classes = selected_model.classes_
                    for i, prob in enumerate(pred_proba):
                        st.write(f"  {classes[i]}: {prob:.3f}")
                else:
                    pred = selected_model.predict(input_df)[0]
                    st.success(f"**Prediction:** {pred:.3f}")

else:
    st.info("👆 Please select a dataset option from the sidebar to get started!")



