"""
Report Generator Module.

This module provides functions to automatically generate a business-friendly PDF report
following a transaction prediction run. The report includes dataset profiling summaries,
model metrics, risk distributions, charts, and actionable business recommendations.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from src import config

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render 'Page X of Y' footers
    and standard professional headers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Header (Only on page 2 and later)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Financial Transaction Risk Engine — Risk Assessment Report")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer (On all pages)
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 35, text)
        self.drawString(54, 35, "CONFIDENTIAL — FOR INTERNAL BUSINESS USE ONLY")
        self.setStrokeColor(colors.HexColor("#D1D5DB"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        
        self.restoreState()


def generate_recommendations(pct_high_risk: float, avg_risk_score: float, avg_confidence_anomalies: float) -> list:
    """
    Generate tailored business recommendations based on run statistics.
    
    Args:
        pct_high_risk (float): Percentage of High Risk transactions in the run.
        avg_risk_score (float): Average risk score (0-100) across all transactions.
        avg_confidence_anomalies (float): Average confidence score (probability) of flagged anomalies.
        
    Returns:
        list: List of dicts containing recommendation level, title, and action details.
    """
    recommendations = []
    
    # 1. Action recommendation based on Percentage of High Risk transactions
    if pct_high_risk > 5.0:
        recommendations.append({
            "level": "High Priority",
            "title": "Immediate Operational Hold & MFA Enforcement",
            "description": f"With {pct_high_risk:.2f}% of transactions flagged as High Risk (above the 5.0% critical threshold), there is an elevated threat level. We recommend: (1) Placing an immediate operational hold on all flagged high-risk transactions. (2) Prompting users for Multi-Factor Authentication (MFA) to clear holds. (3) Routing cases directly to the Tier 1 fraud investigation queue."
        })
    elif pct_high_risk >= 2.0:
        recommendations.append({
            "level": "Medium Priority",
            "title": "Increased Monitoring & Queue Prioritization",
            "description": f"A moderate volume of transactions ({pct_high_risk:.2f}%) are flagged as High Risk. We recommend: (1) Placing temporary holds on transactions exceeding $1,000 in this segment. (2) Enforcing standard batch auditing within 12 hours. (3) Monitoring velocity trends for the affected customer accounts."
        })
    else:
        recommendations.append({
            "level": "Low Priority",
            "title": "Standard Automated Processing",
            "description": f"High Risk transactions are within normal operating limits ({pct_high_risk:.2f}%). We recommend: (1) Continuing with standard automated processing. (2) Standard daily reconciliation report audits. (3) No immediate operational interventions."
        })

    # 2. Recommendation based on Average Risk Score
    if avg_risk_score > 40.0:
        recommendations.append({
            "level": "Medium Priority",
            "title": "Systemic Risk Threshold Audit",
            "description": f"The average transaction risk score is elevated at {avg_risk_score:.1f}/100. This indicates a general shift in transaction patterns (e.g. higher amounts, international locations, odd hours). We recommend reviewing current policy thresholds in config/risk_scoring_config.json to ensure they align with business risk tolerance."
        })
    else:
        recommendations.append({
            "level": "Low Priority",
            "title": "Systemic Risk Stable",
            "description": f"The average transaction risk score is stable at {avg_risk_score:.1f}/100, indicating healthy transactional behavior across the network."
        })

    # 3. Recommendation based on Model Confidence
    if avg_confidence_anomalies < 0.85 and avg_confidence_anomalies > 0:
        recommendations.append({
            "level": "Medium Priority",
            "title": "Uncertain Prediction Audit & Model Retraining",
            "description": f"The model is flagging anomalies with relatively low confidence (average anomaly confidence of {avg_confidence_anomalies*100:.1f}% is below the 85.0% threshold). This indicates the model may be encountering novel transaction patterns. We recommend: (1) Manually auditing borderline flags (risk scores 50-70). (2) Scheduling a model retraining pipeline run with recent verified transaction labels to reduce ambiguity."
        })
    elif avg_confidence_anomalies >= 0.85:
        recommendations.append({
            "level": "Low Priority",
            "title": "High Confidence Detections",
            "description": f"The machine learning model is operating with high certainty (average anomaly confidence of {avg_confidence_anomalies*100:.1f}%). The automated decisions and risk scoring can be highly trusted for system actions."
        })

    return recommendations


def _generate_report_charts(df_predictions: pd.DataFrame, output_dir: str) -> tuple:
    """
    Generate and save charts specific to the current prediction run.
    
    Returns:
        tuple: (path_to_risk_dist_chart, path_to_risk_hist_chart)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Risk Level Distribution Plot
    dist_path = os.path.join(output_dir, "temp_run_risk_distribution.png")
    plt.figure(figsize=(6, 3))
    counts = df_predictions["risk_level"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"], fill_value=0
    )
    colors_dict = {"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}
    
    ax = sns.barplot(
        x=counts.index, 
        y=counts.values, 
        palette=colors_dict.values(),
        hue=counts.index,
        legend=False
    )
    plt.title("Transaction Risk Level Distribution", fontsize=10, fontweight="bold", pad=8)
    plt.ylabel("Transaction Count", fontsize=8)
    plt.xlabel("Risk Level", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    # Annotate bars
    total = len(df_predictions)
    for i, p in enumerate(ax.patches):
        val = int(counts.values[i])
        pct = (val / total) * 100 if total > 0 else 0
        ax.annotate(f"{val} ({pct:.1f}%)", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", xytext=(0, 2), textcoords="offset points",
                    fontsize=8, fontweight="bold")
                    
    plt.tight_layout()
    plt.savefig(dist_path, dpi=200)
    plt.close()
    
    # 2. Risk Score Histogram
    hist_path = os.path.join(output_dir, "temp_run_risk_histogram.png")
    plt.figure(figsize=(6, 3))
    
    sns.histplot(
        data=df_predictions, 
        x="risk_score", 
        bins=20, 
        kde=True, 
        color="#4F46E5", 
        edgecolor="white", 
        alpha=0.7
    )
    
    # Add vertical lines for thresholds (default or from file)
    plt.axvline(20, color="#F59E0B", linestyle="--", linewidth=1)
    plt.axvline(60, color="#EF4444", linestyle="--", linewidth=1)
    
    plt.title("Transaction Risk Score Distribution (0-100 Scale)", fontsize=10, fontweight="bold", pad=8)
    plt.xlabel("Business Risk Score", fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.xlim(-2, 102)
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()
    
    return dist_path, hist_path


def load_model_metrics(model_name: str) -> dict:
    """
    Attempt to load trained performance metrics for the active model.
    """
    # Map model name to JSON file
    metrics_path = None
    if "tuned" in model_name.lower():
        metrics_path = config.RF_TUNED_METRICS_PATH
    elif "rf" in model_name.lower() or "random_forest" in model_name.lower():
        metrics_path = config.RF_METRICS_PATH
    else:
        metrics_path = config.BASELINE_METRICS_PATH
        
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics
        except Exception:
            pass
            
    # Try looking for any of them as fallback
    for path in [config.RF_TUNED_METRICS_PATH, config.RF_METRICS_PATH, config.BASELINE_METRICS_PATH]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
                
    return None


def generate_pdf_report(df_predictions: pd.DataFrame, model_name: str, output_path: str = None) -> str:
    """
    Generate a styled PDF report for a prediction run and save it to disk.
    
    Args:
        df_predictions (pd.DataFrame): Dataframe containing the predictions and risk scores.
        model_name (str): Name or path of the model used for predictions.
        output_path (str, optional): Target path for the PDF. If None, saves in config.PDF_REPORTS_DIR
                                     with a timestamped file name.
                                     
    Returns:
        str: Absolute path to the generated PDF.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Resolve output path
    if output_path is None:
        os.makedirs(config.PDF_REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(config.PDF_REPORTS_DIR, f"risk_report_{timestamp}.pdf")
        
    # 2. Reconstruct summaries & analytics
    total_txns = len(df_predictions)
    total_volume = df_predictions["amount"].sum() if "amount" in df_predictions.columns else 0.0
    avg_amount = df_predictions["amount"].mean() if "amount" in df_predictions.columns else 0.0
    
    # Handle timestamps
    min_time_str = "N/A"
    max_time_str = "N/A"
    if "timestamp" in df_predictions.columns:
        try:
            ts = pd.to_datetime(df_predictions["timestamp"])
            min_time_str = ts.min().strftime("%Y-%m-%d %H:%M:%S")
            max_time_str = ts.max().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
            
    unique_cust = df_predictions["customer_id"].nunique() if "customer_id" in df_predictions.columns else 0
    
    # Risk Distribution
    risk_counts = df_predictions["risk_level"].value_counts()
    low_count = int(risk_counts.get("Low Risk", 0))
    medium_count = int(risk_counts.get("Medium Risk", 0))
    high_count = int(risk_counts.get("High Risk", 0))
    
    pct_low = (low_count / total_txns) * 100 if total_txns > 0 else 0.0
    pct_medium = (medium_count / total_txns) * 100 if total_txns > 0 else 0.0
    pct_high = (high_count / total_txns) * 100 if total_txns > 0 else 0.0
    
    avg_risk = df_predictions["risk_score"].mean() if "risk_score" in df_predictions.columns else 0.0
    
    # Confidence metrics (probability score for anomalies)
    anomalies = df_predictions[df_predictions["prediction"] == 1]
    avg_conf_anom = anomalies["probability_score"].mean() if len(anomalies) > 0 else 0.0
    
    # Recommendations
    recs = generate_recommendations(pct_high, avg_risk, avg_conf_anom)
    
    # Model pre-trained metrics
    historical_metrics = load_model_metrics(model_name)
    
    # Run level evaluation (if actual labels exist)
    run_metrics = None
    target_col = config.TARGET_COLUMN
    actual_col = "Actual Class"
    # Find matching target column
    found_target = None
    for col in [target_col, actual_col, "actual", "is_anomaly", "is_churn"]:
        if col in df_predictions.columns:
            found_target = col
            break
            
    if found_target is not None and df_predictions[found_target].nunique() > 1:
        # We can calculate performance metrics!
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_true = df_predictions[found_target].astype(int)
        y_pred = df_predictions["prediction"].astype(int)
        run_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    # Generate charts
    temp_dir = os.path.join(config.REPORTS_DIR, "temp")
    dist_chart, hist_chart = _generate_report_charts(df_predictions, temp_dir)
    
    # 3. Build PDF Document
    # Page setup
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54, # 0.75 in
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette Definitions
    primary_color = colors.HexColor("#1E3A8A")   # Navy
    secondary_color = colors.HexColor("#4F46E5") # Slate Blue
    dark_text_color = colors.HexColor("#1F2937")  # Off-black
    bg_light_color = colors.HexColor("#F3F4F6")   # Light gray
    
    # Style definitions
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=0, # Left-aligned
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "SubSectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text_color,
        spaceAfter=8
    )
    
    body_bold_style = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    
    table_text_style = ParagraphStyle(
        "TableText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=dark_text_color
    )
    
    table_header_style = ParagraphStyle(
        "TableHeaderText",
        parent=table_text_style,
        fontName="Helvetica-Bold",
        textColor=colors.white
    )

    story = []
    
    # Title & Header
    story.append(Paragraph("RISK ASSESSMENT REPORT", title_style))
    story.append(Paragraph(f"Financial Anomaly & Risk Scoring Engine — Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Executive Summary Card
    story.append(Paragraph("Executive Summary", h1_style))
    summary_text = (
        f"A batch prediction run was executed using the <b>{os.path.basename(model_name)}</b> classifier. "
        f"A total of <b>{total_txns:,}</b> transactions were analyzed. Out of these, the engine flagged "
        f"<b>{high_count:,} ({pct_high:.2f}%)</b> transactions as <b>High Risk</b> anomalies. "
        f"The average risk score across the dataset was <b>{avg_risk:.1f}/100</b>, and the model's confidence "
        f"level on flagged anomalies averaged <b>{avg_conf_anom * 100:.1f}%</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))
    
    # KPI Stats Table Grid (Key Metrics Callout)
    kpi_data = [
        [
            Paragraph("<b>Total Transactions</b>", table_text_style), 
            Paragraph(f"{total_txns:,}", table_text_style),
            Paragraph("<b>High-Risk Flags</b>", table_text_style), 
            Paragraph(f"<font color='#EF4444'><b>{high_count:,} ({pct_high:.2f}%)</b></font>", table_text_style)
        ],
        [
            Paragraph("<b>Total Volume</b>", table_text_style), 
            Paragraph(f"${total_volume:,.2f}", table_text_style),
            Paragraph("<b>Avg Risk Score</b>", table_text_style), 
            Paragraph(f"<b>{avg_risk:.1f} / 100</b>", table_text_style)
        ],
        [
            Paragraph("<b>Unique Customers</b>", table_text_style), 
            Paragraph(f"{unique_cust:,}", table_text_style),
            Paragraph("<b>Anomaly Confidence</b>", table_text_style), 
            Paragraph(f"{avg_conf_anom * 100:.1f}%", table_text_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[120, 132, 120, 132])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 15))
    
    # Section 1: Dataset Summary & Diagnostics
    story.append(Paragraph("Dataset & Context Profile", h1_style))
    profile_data = [
        [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Value Description</b>", table_header_style)],
        [Paragraph("Earliest Transaction", table_text_style), Paragraph(min_time_str, table_text_style)],
        [Paragraph("Latest Transaction", table_text_style), Paragraph(max_time_str, table_text_style)],
        [Paragraph("Average Transaction Size", table_text_style), Paragraph(f"${avg_amount:,.2f}", table_text_style)],
        [Paragraph("Unique Customers Active", table_text_style), Paragraph(f"{unique_cust:,} customer profiles evaluated", table_text_style)],
    ]
    profile_table = Table(profile_data, colWidths=[150, 354])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), primary_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light_color])
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 15))
    
    # Section 2: Model Performance Context
    story.append(Paragraph("Model Performance Profile", h1_style))
    model_meta_text = f"The active classification model is <b>{os.path.basename(model_name)}</b>. "
    if historical_metrics:
        model_meta_text += (
            f"During offline validation, this model demonstrated an overall Accuracy of "
            f"<b>{historical_metrics.get('accuracy', 0.0):.2%}</b>, Precision of "
            f"<b>{historical_metrics.get('precision', 0.0):.2%}</b> (low false alarm rate), "
            f"and Recall of <b>{historical_metrics.get('recall', 0.0):.2%}</b> (anomaly catch rate)."
        )
    else:
        model_meta_text += "Historical performance metrics are currently unavailable."
        
    story.append(Paragraph(model_meta_text, body_style))
    story.append(Spacer(1, 6))
    
    # Model evaluation tables
    if run_metrics:
        # Run has actual labels
        metrics_data = [
            [Paragraph("<b>Evaluation Metric</b>", table_header_style), Paragraph("<b>Active Model Baseline</b>", table_header_style), Paragraph("<b>Current Run Actual</b>", table_header_style)],
            [Paragraph("Accuracy", table_text_style), Paragraph(f"{historical_metrics.get('accuracy', 0.0):.2%}" if historical_metrics else "N/A", table_text_style), Paragraph(f"{run_metrics['accuracy']:.2%}", table_text_style)],
            [Paragraph("Precision (False Positive Mitigation)", table_text_style), Paragraph(f"{historical_metrics.get('precision', 0.0):.2%}" if historical_metrics else "N/A", table_text_style), Paragraph(f"{run_metrics['precision']:.2%}", table_text_style)],
            [Paragraph("Recall (Fraud Anomaly Capture Rate)", table_text_style), Paragraph(f"{historical_metrics.get('recall', 0.0):.2%}" if historical_metrics else "N/A", table_text_style), Paragraph(f"{run_metrics['recall']:.2%}", table_text_style)],
            [Paragraph("F1-Score (Balanced Performance)", table_text_style), Paragraph(f"{historical_metrics.get('f1_score', 0.0):.2%}" if historical_metrics else "N/A", table_text_style), Paragraph(f"{run_metrics['f1']:.2%}", table_text_style)]
        ]
        metrics_table = Table(metrics_data, colWidths=[200, 152, 152])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (2,0), primary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light_color])
        ]))
        story.append(metrics_table)
    elif historical_metrics:
        # Show historical validation metrics
        metrics_data = [
            [Paragraph("<b>Evaluation Metric</b>", table_header_style), Paragraph("<b>Trained Baseline Score</b>", table_header_style), Paragraph("<b>Jargon-Free Description</b>", table_header_style)],
            [Paragraph("Accuracy", table_text_style), Paragraph(f"{historical_metrics.get('accuracy', 0.0):.2%}", table_text_style), Paragraph("Percentage of correct overall decisions.", table_text_style)],
            [Paragraph("Precision", table_text_style), Paragraph(f"{historical_metrics.get('precision', 0.0):.2%}", table_text_style), Paragraph("Proportion of predicted anomalies that were actual anomalies (fewer false alarms).", table_text_style)],
            [Paragraph("Recall", table_text_style), Paragraph(f"{historical_metrics.get('recall', 0.0):.2%}", table_text_style), Paragraph("Proportion of actual anomalies captured by the engine (detection coverage).", table_text_style)],
            [Paragraph("F1-Score", table_text_style), Paragraph(f"{historical_metrics.get('f1_score', 0.0):.2%}", table_text_style), Paragraph("Harmonic mean of Precision and Recall. Best overall benchmark.", table_text_style)]
        ]
        metrics_table = Table(metrics_data, colWidths=[120, 120, 264])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (2,0), primary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light_color])
        ]))
        story.append(metrics_table)
        
    story.append(Spacer(1, 15))
    
    # Page Break for clean layout (Separates tables from visual charts and recommendations)
    story.append(PageBreak())
    
    # Section 3: Risk Distributions & Charts
    story.append(Paragraph("Risk Analytics & Visualizations", h1_style))
    story.append(Paragraph(
        "Below are the analytical plots mapping the transaction risk profiles for the current batch prediction run. "
        "The left chart visualizes counts and percentages by risk tiers, while the right chart displays the score distributions.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Place images side-by-side in a table
    img_width = 3.3 * inch
    img_height = 1.65 * inch
    img_dist = Image(dist_chart, width=img_width, height=img_height)
    img_hist = Image(hist_chart, width=img_width, height=img_height)
    
    charts_table = Table([[img_dist, img_hist]], colWidths=[252, 252])
    charts_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(charts_table)
    story.append(Spacer(1, 10))
    
    # Risk Categories Detail Table
    risk_detail_data = [
        [
            Paragraph("<b>Risk Category</b>", table_header_style), 
            Paragraph("<b>Risk Score Range</b>", table_header_style), 
            Paragraph("<b>Transaction Count</b>", table_header_style),
            Paragraph("<b>Proportion</b>", table_header_style)
        ],
        [
            Paragraph("<font color='#10B981'><b>Low Risk</b></font>", table_text_style), 
            Paragraph("0.0 - 20.0", table_text_style), 
            Paragraph(f"{low_count:,}", table_text_style), 
            Paragraph(f"{pct_low:.2f}%", table_text_style)
        ],
        [
            Paragraph("<font color='#F59E0B'><b>Medium Risk</b></font>", table_text_style), 
            Paragraph("20.1 - 60.0", table_text_style), 
            Paragraph(f"{medium_count:,}", table_text_style), 
            Paragraph(f"{pct_medium:.2f}%", table_text_style)
        ],
        [
            Paragraph("<font color='#EF4444'><b>High Risk</b></font>", table_text_style), 
            Paragraph("60.1 - 100.0", table_text_style), 
            Paragraph(f"{high_count:,}", table_text_style), 
            Paragraph(f"{pct_high:.2f}%", table_text_style)
        ]
    ]
    risk_detail_table = Table(risk_detail_data, colWidths=[126, 126, 126, 126])
    risk_detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (3,0), primary_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light_color])
    ]))
    story.append(risk_detail_table)
    story.append(Spacer(1, 15))
    
    # Section 4: Business Recommendations
    story.append(Paragraph("Business Insights & Operational Recommendations", h1_style))
    story.append(Paragraph(
        "Based on the analysis of risk levels, average risk score, and model prediction confidence, "
        "the system has generated the following operational recommendations for risk mitigation:",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    # Add recommendations inside Callout / Tables
    for r in recs:
        # Determine border color based on level
        if r["level"] == "High Priority":
            border_color = colors.HexColor("#EF4444")  # Crimson
            title_p = Paragraph(f"<font color='#EF4444'><b>[CRITICAL] {r['title']}</b></font>", body_bold_style)
        elif r["level"] == "Medium Priority":
            border_color = colors.HexColor("#F59E0B")  # Amber
            title_p = Paragraph(f"<font color='#D97706'><b>[WARNING] {r['title']}</b></font>", body_bold_style)
        else:
            border_color = colors.HexColor("#10B981")  # Emerald
            title_p = Paragraph(f"<font color='#059669'><b>[INFO] {r['title']}</b></font>", body_bold_style)
            
        desc_p = Paragraph(r["description"], body_style)
        
        rec_box = Table([[title_p], [desc_p]], colWidths=[500])
        rec_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9FAFB")),
            ('BOX', (0,0), (-1,-1), 1, border_color),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('BOTTOMPADDING', (0,0), (-1,0), 2),
            ('TOPPADDING', (0,1), (-1,1), 2),
            ('BOTTOMPADDING', (0,1), (-1,1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        
        story.append(rec_box)
        story.append(Spacer(1, 10))
        
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Clean up temp charts
    try:
        if os.path.exists(dist_chart):
            os.remove(dist_chart)
        if os.path.exists(hist_chart):
            os.remove(hist_chart)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass
        
    return os.path.abspath(output_path)
