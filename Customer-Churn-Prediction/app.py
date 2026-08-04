"""
Streamlit Business Dashboard.

This module provides an interactive web-based dashboard for the Financial Transaction
Risk & Anomaly Engine. Business users can view dataset profiling, model performance,
upload new transaction files, run predictions, and inspect anomaly visualizations.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib and seaborn styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.alpha": 0.3
})

# Add repository root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.predict import predict_dataframe, ValidationError

st.set_page_config(
    page_title="Financial Transaction Anomaly Engine Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Cached ML Pipeline Loading
# -------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    """Load model and preprocessor artifacts, caching them for performance."""
    # Find best model path
    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.RF_TUNED_MODEL_PATH
        if not os.path.exists(model_path):
            model_path = config.RF_MODEL_PATH
            
    preprocessor_path = config.PREPROCESSOR_PATH
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None, None
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor, os.path.basename(model_path)

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.title("🛡️ Risk Engine")
st.sidebar.markdown("Monitoring financial transactions for anomaly signatures and business risk levels.")

page = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard Overview", "Batch Prediction Engine"]
)

# Load pipeline info
model, preprocessor, model_name = load_ml_pipeline()

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if model is not None:
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.text(f"Active: {model_name}")
else:
    st.sidebar.error("❌ Model Missing")
    st.sidebar.text("Please run training pipeline.")

# -------------------------------------------------------------
# Page 1: Dashboard Overview
# -------------------------------------------------------------
if page == "Dashboard Overview":
    st.title("📊 Anomaly & Risk Engine Overview")
    st.markdown("""
    Welcome to the **Financial Transaction Risk & Anomaly Engine Dashboard**. 
    This system utilizes advanced machine learning classifiers and business rules to analyze 
    incoming transaction events, flag high-risk anomalies, and assign actionable risk levels.
    """)
    
    tab_desc, tab_data, tab_model = st.tabs([
        "🛡️ Project Overview & Threat Signatures",
        "📂 Dataset Profiling Summary",
        "📈 Model Comparison & Metrics"
    ])
    
    with tab_desc:
        st.subheader("Overview")
        st.markdown("""
        The engine identifies fraudulent behaviors and anomalies using a hybrid model:
        1. **Supervised Machine Learning**: Scans transactions using trained estimators optimized for severe class imbalance.
        2. **Risk Scoring Engine**: Converts continuous probability estimates into a business risk rating scale (0-100) and risk levels.
        """)
        
        st.subheader("Anomalous Spending Signatures")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("💰 **High Value Spikes**")
            st.markdown("Unusually large values ($5,000 to $20,000) mapped to online retail channels or money transfers.")
        with col2:
            st.info("🗺️ **Geographical Anomalies**")
            st.markdown("Transactions originating from international locations that are inconsistent with typical user spend profiles.")
        with col3:
            st.info("⏰ **Odd Hours Behavior**")
            st.markdown("Large values processed during card-present swipes or wire transfers between 2:00 AM and 4:00 AM local time.")

    with tab_data:
        st.subheader("Historical Training Dataset Profile")
        
        # Load processed database to show stats
        clean_data_path = config.PROCESSED_DATA_PATH
        if os.path.exists(clean_data_path):
            df_clean = pd.read_csv(clean_data_path)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Records", f"{len(df_clean):,}")
            with c2:
                anomaly_rate = df_clean[config.TARGET_COLUMN].mean() if config.TARGET_COLUMN in df_clean.columns else 0.015
                st.metric("Base Anomaly Rate", f"{anomaly_rate:.2%}")
            with c3:
                st.metric("Unique Customers", f"{df_clean['customer_id'].nunique():,}")
            with c4:
                st.metric("Average Transaction Amount", f"${df_clean['amount'].mean():.2f}")
                
            st.markdown("### Sample Historical Clean Data")
            st.dataframe(df_clean.head(10), use_container_width=True)
        else:
            st.warning(f"Clean database file not found at: {clean_data_path}. Please run training first.")

    with tab_model:
        st.subheader("Trained Model Performance Report")
        
        comparison_csv_path = config.COMPARISON_METRICS_THREE_WAY_PATH
        if os.path.exists(comparison_csv_path):
            df_comp = pd.read_csv(comparison_csv_path)
            st.dataframe(
                df_comp.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                    "ROC-AUC": "{:.4f}"
                }),
                use_container_width=True
            )
            
            # Show active model description
            st.success(f"**Active Model Details:** The currently active model is **{model_name}**.")
            
            # Load and show confusion matrix plot
            # Let's find confusion matrix from metrics JSON
            metrics_path = config.RF_TUNED_METRICS_PATH if "tuned" in model_name else config.RF_METRICS_PATH
            if not os.path.exists(metrics_path):
                metrics_path = config.BASELINE_METRICS_PATH
                
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, "r") as f:
                    metrics_dict = json.load(f)
                
                if "confusion_matrix" in metrics_dict:
                    cm = np.array(metrics_dict["confusion_matrix"])
                    
                    st.subheader("Validation Confusion Matrix Heatmap")
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Normal", "Anomaly"],
                        yticklabels=["Normal", "Anomaly"],
                        annot_kws={"weight": "bold", "size": 10},
                        ax=ax
                    )
                    ax.set_title(f"Confusion Matrix ({model_name})")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        else:
            st.warning("Performance comparison metrics CSV file not found. Please run training pipeline.")

# -------------------------------------------------------------
# Page 2: Batch Prediction Engine
# -------------------------------------------------------------
elif page == "Batch Prediction Engine":
    st.title("🛡️ Batch Risk Assessment & Prediction Engine")
    st.markdown("""
    Upload a batch CSV file containing transactions to evaluate them for anomalies and risk scores.
    """)
    
    if model is None or preprocessor is None:
        st.error("🚨 Inference pipeline artifacts are missing! Please run model training first before attempting predictions.")
    else:
        uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            # Read CSV
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Transaction Dataset Preview")
                st.dataframe(uploaded_df.head(10), use_container_width=True)
                
                # Predict button
                if st.button("🚀 Run Risk Assessment & Flag Anomalies", type="primary"):
                    with st.spinner("Executing inference and risk grading pipeline..."):
                        try:
                            # Run prediction logic using refactored predict_dataframe function
                            results_df = predict_dataframe(
                                df_input=uploaded_df,
                                model=model,
                                preprocessor=preprocessor
                            )
                            
                            st.success("✅ Batch Anomaly Detection Completed Successfully!")
                            
                            # KPI Metrics
                            total_txns = len(results_df)
                            anomalies = int(results_df["prediction"].sum())
                            anomaly_pct = anomalies / total_txns
                            
                            high_risk = int((results_df["risk_level"] == "High Risk").sum())
                            high_risk_pct = high_risk / total_txns
                            
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Processed Transactions", f"{total_txns:,}")
                            with m2:
                                st.metric(
                                    "Model Flagged Anomalies", 
                                    f"{anomalies:,}", 
                                    f"{anomaly_pct:.2%} of total", 
                                    delta_color="inverse"
                                )
                            with m3:
                                st.metric(
                                    "High Risk Transactions", 
                                    f"{high_risk:,}", 
                                    f"{high_risk_pct:.2%} of total", 
                                    delta_color="inverse"
                                )
                                
                            # Download prediction results
                            csv_data = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Risk Assessment CSV",
                                data=csv_data,
                                file_name="batch_transaction_risk_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Prediction Table Details
                            st.subheader("Risk Assessment Results Table")
                            # Highlight flagged transactions
                            st.dataframe(
                                results_df.head(100),
                                use_container_width=True
                            )
                            if len(results_df) > 100:
                                st.caption("Showing first 100 rows. Download the complete dataset to view all predictions.")
                                
                            # Predictions Visualizations Tab
                            st.subheader("🔍 Prediction Visualizations")
                            viz1, viz2, viz3 = st.tabs([
                                "📊 Risk Levels Distribution",
                                "📈 Prediction Probabilities Distribution",
                                "🧬 Model Feature Drivers"
                            ])
                            
                            with viz1:
                                st.markdown("#### Proportion of Business Risk Levels")
                                counts = results_df["risk_level"].value_counts().reindex(
                                    ["Low Risk", "Medium Risk", "High Risk"], fill_value=0
                                )
                                colors = {"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}
                                
                                fig, ax = plt.subplots(figsize=(6, 3.5))
                                sns.barplot(
                                    x=counts.index, 
                                    y=counts.values, 
                                    palette=colors.values(),
                                    hue=counts.index,
                                    legend=False,
                                    ax=ax
                                )
                                ax.set_ylabel("Count")
                                ax.set_title("Risk Category Counts")
                                for i, v in enumerate(counts.values):
                                    ax.text(i, v + (max(counts.values)*0.01), f"{v}\n({v/total_txns:.1%})", ha="center", va="bottom", fontsize=8, fontweight="bold")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                            with viz2:
                                st.markdown("#### Distribution of Model Anomaly Probability (0.0 to 1.0)")
                                fig, ax = plt.subplots(figsize=(6, 3.5))
                                sns.histplot(
                                    results_df["probability_score"],
                                    bins=20,
                                    kde=True,
                                    color="#4F46E5",
                                    ax=ax
                                )
                                ax.set_xlabel("Anomaly Probability Score")
                                ax.set_ylabel("Frequency")
                                ax.set_title("Prediction Probability Distribution")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                            with viz3:
                                st.markdown("#### Feature Drivers Contribution")
                                # Try extracting feature names
                                try:
                                    cat_encoder = preprocessor.named_transformers_["cat"]
                                    cat_feature_names = cat_encoder.get_feature_names_out(config.ENGINEERED_CATEGORICAL_FEATURES).tolist()
                                    all_feature_names = config.ENGINEERED_NUMERIC_FEATURES + cat_feature_names
                                    
                                    # Plot Feature Importance or Coefficients
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    if hasattr(model, "feature_importances_"):
                                        importances = model.feature_importances_
                                        feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(10)
                                        sns.barplot(
                                            x=feat_imp.values,
                                            y=feat_imp.index,
                                            palette="viridis",
                                            hue=feat_imp.index,
                                            legend=False,
                                            ax=ax
                                        )
                                        ax.set_title("Top 10 Feature Importances")
                                        ax.set_xlabel("Importance Score")
                                    elif hasattr(model, "coef_"):
                                        coefs = model.coef_[0]
                                        feat_coef = pd.Series(coefs, index=all_feature_names)
                                        # Sort by absolute coefficient weight
                                        feat_coef_abs = feat_coef.abs().sort_values(ascending=False).head(10)
                                        feat_imp = feat_coef.loc[feat_coef_abs.index]
                                        sns.barplot(
                                            x=feat_imp.values,
                                            y=feat_imp.index,
                                            palette="coolwarm",
                                            hue=feat_imp.index,
                                            legend=False,
                                            ax=ax
                                        )
                                        ax.set_title("Top 10 Linear Feature Weights (Coefficients)")
                                        ax.set_xlabel("Coefficient Weight")
                                    else:
                                        st.warning("Feature importance plotting is not supported for this model class.")
                                        
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                except Exception as e:
                                    st.warning(f"Could not compute model feature importance: {e}")
                                    
                        except ValidationError as ve:
                            st.error(f"❌ Input Data Validation Failed: {ve}")
                        except Exception as e:
                            st.error(f"❌ Prediction Execution Failed: {e}")
                            
            except Exception as e:
                st.error(f"Could not read uploaded CSV file: {e}")
