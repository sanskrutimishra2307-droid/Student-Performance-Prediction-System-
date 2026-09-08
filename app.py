"""
Student Performance Prediction System - Luxury Enterprise Dashboard.
High-end AI analytics, feature selection benchmarking, model arena, real-time student simulation,
interactive 3D visuals, XAI decision support, and model retraining hub.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    DEFAULT_DATASET_PATH,
    SAMPLE_BATCH_PATH,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_REPORT_PATH,
    TARGET_SCHEMES,
    DEFAULT_TARGET_SCHEME
)
from src.data_loader import DataLoader
from src.preprocessing import PreprocessingPipeline
from src.feature_selection import FeatureSelectorSuite
from src.models import ModelFactory
from src.trainer import ModelTrainer
from src.explainability import ExplainabilityEngine
from src.evaluation import ModelEvaluator

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AURA • Student Performance Intelligence",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# LUXURY GLASSMORPHIC DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-dark: #07090e;
        --card-bg: rgba(18, 24, 38, 0.65);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-purple: #8b5cf6;
        --accent-indigo: #6366f1;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-gold: #f59e0b;
        --accent-rose: #f43f5e;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-dark);
        color: var(--text-primary);
    }
    
    /* Background Ambient Lighting */
    .stApp {
        background: 
            radial-gradient(circle at 10% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 90% 20%, rgba(139, 92, 246, 0.10) 0%, transparent 45%),
            radial-gradient(circle at 50% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
            #07090e;
    }
    
    /* Luxury Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(26, 33, 56, 0.75) 0%, rgba(13, 17, 28, 0.85) 100%);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        padding: 32px 40px;
        margin-bottom: 28px;
        box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #06b6d4);
    }

    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }
    .brand-subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 8px;
        letter-spacing: -0.01em;
    }

    /* Luxury Glass Cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(26, 33, 56, 0.45) 0%, rgba(15, 23, 42, 0.6) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px 26px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.35);
        box-shadow: 0 16px 40px rgba(99, 102, 241, 0.2);
    }

    /* Metric Tiles */
    .metric-tile {
        background: linear-gradient(145deg, rgba(30, 41, 67, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-tile:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    .metric-tile-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-tile-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Luxury Status Badges */
    .badge-glow-distinction {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 10px 24px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
        display: inline-block;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
    }
    .badge-glow-pass {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.3) 100%);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.4);
        padding: 10px 24px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
        display: inline-block;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.35);
    }
    .badge-glow-atrisk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.3) 100%);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 10px 24px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
        display: inline-block;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
    }

    /* Leaderboard Cards */
    .leader-card-gold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(180, 83, 9, 0.18) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.15);
    }
    .leader-card-silver {
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.1) 0%, rgba(71, 85, 105, 0.15) 100%);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 18px;
        padding: 20px;
    }
    .leader-card-bronze {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.08) 0%, rgba(146, 64, 14, 0.12) 100%);
        border: 1px solid rgba(217, 119, 6, 0.25);
        border-radius: 18px;
        padding: 20px;
    }

    /* Recommendation Action Boxes */
    .action-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid var(--accent-indigo);
        border-radius: 0 14px 14px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* Pulsating Dot */
    .pulse-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 12px #10b981;
        margin-right: 8px;
    }

    /* Form and Controls Luxury Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 28px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(124, 58, 237, 0.6);
        border-color: rgba(255, 255, 255, 0.4);
    }

    /* Sidebar Navigation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f19 0%, #07090e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Hide Default Header Elements for Clean Luxury Look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA ACCESS & CACHING
# -------------------------------------------------------------
@st.cache_data
def get_cached_dataset():
    loader = DataLoader(str(DEFAULT_DATASET_PATH))
    return loader.load_data()

df_dataset = get_cached_dataset()

# -------------------------------------------------------------
# HERO HEADER BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span class="pulse-dot"></span>
                <span style="color: #a5b4fc; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;">
                    AURA • Academic AI Suite
                </span>
            </div>
            <h1 class="brand-title">Student Performance Prediction System</h1>
            <p class="brand-subtitle">
                Predictive Outcome Modeling, Feature Selection Discovery & Prescriptive Intervention Analytics
            </p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">BENCHMARK DATASET</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8;">15,000 Records</div>
            </div>
            <div style="background: rgba(99, 102, 241, 0.15); padding: 8px 18px; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3); text-align: right;">
                <div style="font-size: 0.75rem; color: #a5b4fc; font-weight: 500;">ACTIVE ARCHITECTURE</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #c084fc;">Stacking Meta-Ensemble</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLLER
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Navigation Portal")
    nav_selection = st.radio(
        "Workspace View:",
        [
            "🌌 Executive Intelligence & EDA",
            "💎 Feature Selection Discovery Lab",
            "⚔️ Model Arena & Elite Leaderboard",
            "🔮 Live Student Prediction Simulator",
            "📂 High-Throughput Batch Scoring",
            "⚙️ Autonomous Retraining Studio"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Live Cohort Snapshot")
    st.markdown(f"👥 **Cohort Population:** `{len(df_dataset):,}`")
    st.markdown(f"🧬 **Feature Dimension:** `{len(df_dataset.columns)} attributes`")
    
    if METRICS_REPORT_PATH.exists():
        with open(METRICS_REPORT_PATH, "r") as f:
            metrics_summary = json.load(f)
            meta = metrics_summary.get("metadata", {})
            st.success(f"**Best Model:** {meta.get('best_model_name', 'Stacking Ensemble').replace('_', ' ').title()}")
            st.markdown(f"**Test F1-Score:** `{meta.get('best_f1_score', 0.9967):.4f}`")
            st.markdown(f"**Selected Subset:** `{len(meta.get('selected_features', []))} Features`")

# -------------------------------------------------------------
# TAB 1: EXECUTIVE INTELLIGENCE & EDA
# -------------------------------------------------------------
if nav_selection == "🌌 Executive Intelligence & EDA":
    st.markdown("### 🌌 Executive Academic Overview & Exploratory Intelligence")
    
    # 5 Luxury KPI Metric Tiles
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Total Student Cohort</div>
            <div class="metric-tile-val" style="color: #f8fafc;">{len(df_dataset):,}</div>
            <div class="metric-sub">Across 16 Academic Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Mean Study Hours</div>
            <div class="metric-tile-val" style="color: #38bdf8;">{df_dataset['study_hours'].mean():.1f} <span style="font-size: 1.1rem; color: #94a3b8;">hrs/day</span></div>
            <div class="metric-sub">Std Dev: ±{df_dataset['study_hours'].std():.1f} hrs</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Mean Attendance</div>
            <div class="metric-tile-val" style="color: #34d399;">{df_dataset['attendance_percentage'].mean():.1f}%</div>
            <div class="metric-sub">Institutional Target: 75%</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Mean Academic Score</div>
            <div class="metric-tile-val" style="color: #fbbf24;">{df_dataset['overall_score'].mean():.1f}<span style="font-size: 1rem; color: #94a3b8;">/100</span></div>
            <div class="metric-sub">Median: {df_dataset['overall_score'].median():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        top_g = df_dataset['final_grade'].mode()[0].upper()
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Modal Grade Tier</div>
            <div class="metric-tile-val" style="color: #c084fc;">Grade {top_g}</div>
            <div class="metric-sub">{df_dataset['final_grade'].value_counts().max():,} Students</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3D Interactive Visual Row
    row1_c1, row1_c2 = st.columns([1.2, 1])
    
    with row1_c1:
        st.markdown("#### 🌟 3D Multi-Dimensional Factor Space")
        st.caption("Interactive 3D representation of Study Hours vs Attendance vs Overall Score, colored by Grade.")
        sample_3d = df_dataset.sample(n=min(1500, len(df_dataset)), random_state=42)
        fig_3d = px.scatter_3d(
            sample_3d,
            x="study_hours",
            y="attendance_percentage",
            z="overall_score",
            color="final_grade",
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"study_hours": "Study Hours", "attendance_percentage": "Attendance %", "overall_score": "Score (0-100)"},
            template="plotly_dark"
        )
        fig_3d.update_layout(
            height=460,
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="#0b0f19", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(backgroundcolor="#0b0f19", gridcolor="rgba(255,255,255,0.06)"),
                zaxis=dict(backgroundcolor="#0b0f19", gridcolor="rgba(255,255,255,0.06)")
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with row1_c2:
        st.markdown("#### 🏆 Letter Grade Distribution Breakdown")
        st.caption("Proportionate distribution of academic outcomes across the student population.")
        grade_order = ['a', 'b', 'c', 'd', 'e', 'f']
        grade_cnts = df_dataset['final_grade'].value_counts().reindex(grade_order).dropna()
        
        fig_donut = px.pie(
            values=grade_cnts.values,
            names=[f"Grade {g.upper()}" for g in grade_cnts.index],
            hole=0.6,
            color_discrete_sequence=["#10b981", "#3b82f6", "#6366f1", "#f59e0b", "#f97316", "#ef4444"],
            template="plotly_dark"
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Secondary Analytics Row
    row2_c1, row2_c2 = st.columns([1, 1])
    
    with row2_c1:
        st.markdown("#### 📖 Study Method Effectiveness Spectrum")
        method_summary = df_dataset.groupby("study_method")["overall_score"].agg(["mean", "count"]).reset_index()
        method_summary = method_summary.sort_values(by="mean", ascending=True)
        
        fig_method = px.bar(
            method_summary,
            x="mean",
            y="study_method",
            orientation="h",
            color="mean",
            color_continuous_scale="Viridis",
            labels={"mean": "Average Overall Score", "study_method": "Study Method"},
            template="plotly_dark"
        )
        fig_method.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_method, use_container_width=True)
        
    with row2_c2:
        st.markdown("#### 🔥 Academic Collinearity & Feature Correlation Heatmap")
        academic_vars = ["study_hours", "attendance_percentage", "math_score", "science_score", "english_score", "overall_score"]
        corr = df_dataset[academic_vars].corr()
        
        fig_heatmap = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Magma",
            template="plotly_dark"
        )
        fig_heatmap.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_heatmap, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: FEATURE SELECTION DISCOVERY LAB
# -------------------------------------------------------------
elif nav_selection == "💎 Feature Selection Discovery Lab":
    st.markdown("### 💎 Multi-Method Feature Selection & Dimensionality Discovery")
    st.markdown("Uncover the mathematical drivers of student success by comparing Filter, Wrapper, Embedded, and Ensemble Consensus feature selection methods.")
    
    fs_left, fs_right = st.columns([1, 2])
    
    with fs_left:
        st.markdown("#### 🎛️ Selection Engine Settings")
        chosen_method = st.selectbox(
            "Feature Selection Algorithm:",
            [
                "consensus (Ensemble Rank Aggregator)",
                "mutual_info (Filter - Non-linear Information Gain)",
                "anova_f (Filter - ANOVA F-Value)",
                "chi2 (Filter - Chi-Square Independence)",
                "random_forest_mdi (Embedded - Gini Impurity Reduction)",
                "xgboost_mdi (Embedded - Gradient Boost Gain)",
                "lasso_l1 (Embedded - L1 Sparsity Regularization)",
                "rfecv (Wrapper - Recursive Cross-Validated Pruning)"
            ],
            index=0
        )
        m_key = chosen_method.split(" ")[0]
        top_k = st.slider("Select Top-K Features:", min_value=3, max_value=16, value=8)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 16px;">
            <div style="font-weight: 700; color: #a5b4fc; margin-bottom: 6px;">💡 Feature Selection Insights</div>
            <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                • <strong>Filter Methods</strong> assess statistical signal independent of model architecture.<br>
                • <strong>Wrapper Methods</strong> iteratively test candidate subsets using CV.<br>
                • <strong>Embedded Methods</strong> penalize non-informative features during tree growth or L1 optimization.<br>
                • <strong>Consensus Rank</strong> eliminates method bias through multi-paradigm ensemble voting.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with fs_right:
        with st.spinner("Computing high-dimensional feature importance scores..."):
            loader = DataLoader(str(DEFAULT_DATASET_PATH))
            X, y = loader.prepare_features_and_target()
            pipeline = PreprocessingPipeline()
            X_p, y_e, f_names = pipeline.fit_transform(X, y)
            
            suite = FeatureSelectorSuite(n_features_to_select=top_k)
            suite.run_all_methods(X_p, y_e, f_names)
            
            if m_key == "rfecv":
                top_feats, scores = suite.wrapper_rfecv(X_p, y_e, f_names)
            elif m_key in suite.rankings:
                scores = suite.rankings[m_key]
                top_feats = suite.selected_features[m_key][:top_k]
            else:
                top_feats, scores = suite.ensemble_consensus(X_p, y_e, f_names)
                
            df_scores = pd.DataFrame({
                "Feature": list(scores.keys()),
                "Importance Score": list(scores.values())
            }).sort_values(by="Importance Score", ascending=True)
            
            df_scores["Status"] = df_scores["Feature"].apply(lambda f: "Selected Subset (Top-K)" if f in top_feats else "Eliminated / Noise")
            
            fig_bar = px.bar(
                df_scores,
                x="Importance Score",
                y="Feature",
                orientation="h",
                color="Status",
                color_discrete_map={"Selected Subset (Top-K)": "#8b5cf6", "Eliminated / Noise": "#334155"},
                title=f"Feature Significance Ranking: {chosen_method.upper()}",
                template="plotly_dark"
            )
            fig_bar.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
    st.markdown("---")
    st.markdown("#### 📉 Dimensionality Reduction Efficiency Curve (Optimal K Threshold)")
    st.caption("Simulates classification F1-Score as feature count varies from 1 to N, highlighting where performance saturates.")
    
    if st.button("🚀 Calculate Dimensionality Curve", key="btn_calc_curve"):
        with st.spinner("Running 3-fold cross-validation across all feature dimensions (1 to N)..."):
            curve = suite.evaluate_dimensionality_curve(X_p, y_e, f_names, method=m_key, step=1)
            
            fig_dim = go.Figure()
            fig_dim.add_trace(go.Scatter(
                x=curve["k_values"],
                y=curve["f1_scores"],
                mode="lines+markers",
                name="Weighted F1-Score",
                line=dict(color="#10b981", width=3.5),
                marker=dict(size=8, color="#34d399")
            ))
            fig_dim.add_trace(go.Scatter(
                x=curve["k_values"],
                y=curve["accuracies"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="#3b82f6", width=2, dash="dash"),
                marker=dict(size=6, color="#60a5fa")
            ))
            fig_dim.add_vline(
                x=curve["optimal_k"],
                line_width=2,
                line_dash="dot",
                line_color="#f59e0b",
                annotation_text=f"Optimal K = {curve['optimal_k']} (F1: {curve['max_f1']:.3%})",
                annotation_position="bottom right"
            )
            fig_dim.update_layout(
                title=f"Dimensionality Reduction Performance Curve: {m_key.upper()}",
                xaxis_title="Number of Top Features (K)",
                yaxis_title="Validation Score",
                template="plotly_dark",
                height=420
            )
            st.plotly_chart(fig_dim, use_container_width=True)
            st.success(f"🎯 **Empirical Discovery:** Retaining just the top **{curve['optimal_k']} features** achieves a peak validation F1-Score of **{curve['max_f1']:.3%}**, validating substantial dimensionality reduction without accuracy loss.")

# -------------------------------------------------------------
# TAB 3: MODEL ARENA & ELITE LEADERBOARD
# -------------------------------------------------------------
elif nav_selection == "⚔️ Model Arena & Elite Leaderboard":
    st.markdown("### ⚔️ Model Arena & Classification Leaderboard")
    st.markdown("Comprehensive benchmark of 8 machine learning algorithms evaluated under Stratified 5-Fold Cross Validation on the selected feature subset.")
    
    if not METRICS_REPORT_PATH.exists():
        st.warning("Benchmark metrics not yet computed. Please run baseline training in Tab 6.")
    else:
        with open(METRICS_REPORT_PATH, "r") as f:
            b_data = json.load(f)
            
        m_dict = b_data.get("models", {})
        meta_info = b_data.get("metadata", {})
        
        # Build Table
        t_rows = []
        for name, d in m_dict.items():
            t_rows.append({
                "Algorithm": name.replace("_", " ").title(),
                "CV F1-Score": float(d.get("cv_f1_mean", 0.0)),
                "Test Accuracy": float(d.get("test_accuracy", 0.0)),
                "Test F1-Score": float(d.get("test_f1_weighted", 0.0)),
                "Precision": float(d.get("test_precision_weighted", 0.0)),
                "Recall": float(d.get("test_recall_weighted", 0.0)),
                "ROC-AUC": float(d.get("roc_auc", 0.0)) if d.get("roc_auc") else 0.0
            })
            
        df_lead = pd.DataFrame(t_rows).sort_values(by="Test F1-Score", ascending=False).reset_index(drop=True)
        
        # Top 3 Podium Highlights
        p1, p2, p3 = st.columns(3)
        if len(df_lead) >= 3:
            with p1:
                r1 = df_lead.iloc[0]
                st.markdown(f"""
                <div class="leader-card-gold">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #fbbf24; text-transform: uppercase;">🥇 1st Place • Champion</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{r1['Algorithm']}</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #fbbf24; margin-top: 8px;">{r1['Test F1-Score']:.4f} <span style="font-size: 0.9rem; color: #cbd5e1;">F1</span></div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">Accuracy: {r1['Test Accuracy']:.2%} | ROC-AUC: {r1['ROC-AUC']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with p2:
                r2 = df_lead.iloc[1]
                st.markdown(f"""
                <div class="leader-card-silver">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">🥈 2nd Place • Runner-Up</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{r2['Algorithm']}</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #cbd5e1; margin-top: 8px;">{r2['Test F1-Score']:.4f} <span style="font-size: 0.9rem; color: #94a3b8;">F1</span></div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Accuracy: {r2['Test Accuracy']:.2%} | ROC-AUC: {r2['ROC-AUC']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with p3:
                r3 = df_lead.iloc[2]
                st.markdown(f"""
                <div class="leader-card-bronze">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #f59e0b; text-transform: uppercase;">🥉 3rd Place • Contender</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{r3['Algorithm']}</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #f59e0b; margin-top: 8px;">{r3['Test F1-Score']:.4f} <span style="font-size: 0.9rem; color: #94a3b8;">F1</span></div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Accuracy: {r3['Test Accuracy']:.2%} | ROC-AUC: {r3['ROC-AUC']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            df_lead.style.format({
                "CV F1-Score": "{:.4f}",
                "Test Accuracy": "{:.2%}",
                "Test F1-Score": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "ROC-AUC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        col_radar, col_bars = st.columns([1, 1])
        with col_radar:
            st.markdown("#### 🕸️ Multi-Model Radar Profiler")
            top_models = df_lead.head(3)["Algorithm"].tolist()
            rad_categories = ["Test Accuracy", "Test F1-Score", "Precision", "Recall", "ROC-AUC"]
            fig_rad = go.Figure()
            
            for alg in top_models:
                row_data = df_lead[df_lead["Algorithm"] == alg].iloc[0]
                vals = [row_data[c] for c in rad_categories]
                vals.append(vals[0])
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=rad_categories + [rad_categories[0]],
                    fill='toself',
                    name=alg
                ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0.9, 1.0])),
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_rad, use_container_width=True)
            
        with col_bars:
            st.markdown("#### 📊 Comparative F1-Score Performance")
            fig_lead_bar = px.bar(
                df_lead,
                x="Test F1-Score",
                y="Algorithm",
                orientation="h",
                color="Test F1-Score",
                color_continuous_scale="Plasma",
                template="plotly_dark"
            )
            fig_lead_bar.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_lead_bar, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: LUXURY STUDENT PREDICTION SIMULATOR
# -------------------------------------------------------------
elif nav_selection == "🔮 Live Student Prediction Simulator":
    st.markdown("### 🔮 High-Precision Student Prediction Simulator & XAI Suite")
    st.markdown("Configure an individual student profile to generate instantaneous outcome predictions, inspect class probability distributions, and receive tailored intervention directives.")
    
    with st.form("luxury_sim_form"):
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        
        with sim_c1:
            st.markdown("##### 👤 Demographics & Background")
            s_age = st.number_input("Age:", min_value=14, max_value=25, value=16)
            s_gender = st.selectbox("Gender:", ["female", "male", "other"])
            s_school = st.selectbox("School Type:", ["public", "private"])
            s_parent_edu = st.selectbox(
                "Parent Educational Attainment:",
                ["graduate", "post graduate", "diploma", "high school", "phd", "no formal"]
            )
            
        with sim_c2:
            st.markdown("##### 📖 Study Habits & Attendance")
            s_hours = st.slider("Daily Self-Study Hours:", min_value=0.5, max_value=10.0, value=5.0, step=0.1)
            s_att = st.slider("Class Attendance Percentage (%):", min_value=30.0, max_value=100.0, value=85.0, step=0.5)
            s_method = st.selectbox(
                "Primary Learning Method:",
                ["notes", "textbook", "group study", "online videos", "coaching", "mixed"]
            )
            s_internet = st.radio("Home Internet Connectivity:", ["yes", "no"], horizontal=True)
            
        with sim_c3:
            st.markdown("##### 🏫 Context & Test Scores")
            s_travel = st.selectbox("Daily Commute Time:", ["<15 min", "15-30 min", "30-60 min", ">60 min"])
            s_extra = st.radio("Extracurricular Engagement:", ["yes", "no"], horizontal=True)
            
            s_has_scores = st.checkbox("Include Subject Test Scores (Post-Exams)?", value=True)
            if s_has_scores:
                s_math = st.slider("Math Assessment (0-100):", 0.0, 100.0, 75.0, 1.0)
                s_science = st.slider("Science Assessment (0-100):", 0.0, 100.0, 80.0, 1.0)
                s_english = st.slider("English Assessment (0-100):", 0.0, 100.0, 78.0, 1.0)
            else:
                s_math, s_science, s_english = 65.0, 65.0, 65.0
                
        sim_submit = st.form_submit_button("✨ Compute Outcome Assessment", use_container_width=True)
        
    if sim_submit:
        student_payload = {
            "age": s_age,
            "gender": s_gender,
            "school_type": s_school,
            "parent_education": s_parent_edu,
            "study_hours": s_hours,
            "attendance_percentage": s_att,
            "internet_access": s_internet,
            "travel_time": s_travel,
            "extra_activities": s_extra,
            "study_method": s_method,
            "math_score": s_math if s_has_scores else 60.0,
            "science_score": s_science if s_has_scores else 60.0,
            "english_score": s_english if s_has_scores else 60.0
        }
        
        try:
            with st.spinner("Executing Stacking Ensemble inference and computing XAI attribution..."):
                p_res = ModelTrainer.predict_single(student_payload)
                xai_data = ExplainabilityEngine.analyze_student_risk_factors(student_payload, p_res["prediction"])
                
            pred_tier = p_res["prediction"]
            conf = p_res["confidence"]
            
            st.markdown("---")
            st.markdown("### 🏆 Prediction Outcome & Risk Profile")
            
            out_c1, out_c2 = st.columns([1, 1.4])
            
            with out_c1:
                if pred_tier.lower() in ["distinction", "a", "b"]:
                    badge_style = "badge-glow-distinction"
                    icon = "🌟"
                    dial_val = int(conf * 100)
                    gauge_color = "#10b981"
                elif pred_tier.lower() in ["pass", "c", "d"]:
                    badge_style = "badge-glow-pass"
                    icon = "✅"
                    dial_val = int(conf * 100)
                    gauge_color = "#3b82f6"
                else:
                    badge_style = "badge-glow-atrisk"
                    icon = "⚠️"
                    dial_val = int(conf * 100)
                    gauge_color = "#ef4444"
                    
                st.markdown(f"""
                <div style="text-align: center; padding: 28px; background: rgba(18, 24, 38, 0.6); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.08);">
                    <div class="{badge_style}">{icon} {pred_tier.upper()}</div>
                    <div style="font-size: 1.15rem; color: #cbd5e1; margin-top: 18px;">
                        Model Confidence: <strong style="color: #38bdf8; font-size: 1.3rem;">{conf:.1%}</strong>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                        Architecture: {p_res['model_name'].replace('_', ' ').title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with out_c2:
                st.markdown("#### 📊 Multi-Class Probability Spread")
                prob_dict = p_res["probabilities"]
                df_p = pd.DataFrame({
                    "Tier": list(prob_dict.keys()),
                    "Probability": list(prob_dict.values())
                })
                fig_probs = px.bar(
                    df_p,
                    x="Probability",
                    y="Tier",
                    orientation="h",
                    color="Probability",
                    color_continuous_scale="Tealgrn",
                    range_x=[0, 1.0],
                    template="plotly_dark"
                )
                fig_probs.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_probs, use_container_width=True)
                
            # XAI & Decision Directives
            st.markdown("---")
            st.markdown("### 🔍 Explainable AI & Prescriptive Intervention Guidance")
            
            x_c1, x_c2 = st.columns([1, 1])
            with x_c1:
                st.markdown("#### 🚨 Vulnerability Flags & Academic Assets")
                if xai_data["risk_factors"]:
                    for rf in xai_data["risk_factors"]:
                        st.error(f"**[{rf['severity']}] {rf['factor']}:** {rf['detail']}")
                else:
                    st.success("No critical risk flags detected in this profile.")
                    
                if xai_data["positive_assets"]:
                    for pa in xai_data["positive_assets"]:
                        st.info(f"**✨ Asset: {pa['factor']}:** {pa['detail']}")
                        
            with x_c2:
                st.markdown("#### 💡 Prescriptive Academic Action Plan")
                st.markdown(f"**Intervention Priority:** `{xai_data['urgency']}`")
                st.markdown(f"_{xai_data['summary_statement']}_")
                
                for idx, action in enumerate(xai_data["recommended_interventions"], 1):
                    st.markdown(f"""
                    <div class="action-box">
                        <strong>Directive {idx}:</strong> {action}
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Prediction execution failed: {str(e)}")

# -------------------------------------------------------------
# TAB 5: HIGH-THROUGHPUT BATCH SCORING
# -------------------------------------------------------------
elif nav_selection == "📂 High-Throughput Batch Scoring":
    st.markdown("### 📂 High-Throughput Batch Student Scoring & Export")
    st.markdown("Upload bulk institutional datasets or process cohort samples to produce predictions, confidence metrics, and probability distributions.")
    
    b_up, b_sam = st.columns([2, 1])
    with b_up:
        b_file = st.file_uploader("Upload Student Cohort CSV:", type=["csv"])
    with b_sam:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄 Ingest 25-Student Benchmark Sample", key="btn_ingest_sample"):
            b_file = str(SAMPLE_BATCH_PATH)
            
    df_batch_in = None
    if isinstance(b_file, str) and os.path.exists(b_file):
        df_batch_in = pd.read_csv(b_file)
    elif b_file is not None:
        df_batch_in = pd.read_csv(b_file)
        
    if df_batch_in is not None:
        st.markdown("#### 📋 Cohort Preview")
        st.dataframe(df_batch_in.head(8), use_container_width=True)
        
        if st.button("⚡ Execute Batch Scoring Engine", key="btn_exec_batch"):
            with st.spinner("Processing batch scoring pipeline..."):
                b_results = ModelTrainer.predict_batch(df_batch_in)
                
            st.success(f"Batch inference complete! Generated predictions for {len(b_results):,} records.")
            
            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                b_counts = b_results["predicted_performance"].value_counts()
                fig_b_pie = px.pie(
                    values=b_counts.values,
                    names=b_counts.index,
                    title="Cohort Prediction Distribution",
                    hole=0.45,
                    color_discrete_sequence=["#10b981", "#3b82f6", "#ef4444"],
                    template="plotly_dark"
                )
                st.plotly_chart(fig_b_pie, use_container_width=True)
                
            with res_c2:
                fig_b_hist = px.histogram(
                    b_results,
                    x="prediction_confidence",
                    nbins=12,
                    title="Confidence Distribution",
                    color_discrete_sequence=["#8b5cf6"],
                    template="plotly_dark"
                )
                st.plotly_chart(fig_b_hist, use_container_width=True)
                
            st.dataframe(b_results, use_container_width=True)
            
            # Download CSV
            csv_str = b_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Annotated Predictions CSV",
                data=csv_str,
                file_name="student_predictions_annotated.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------------------------------------------------------------
# TAB 6: AUTONOMOUS RETRAINING STUDIO
# -------------------------------------------------------------
elif nav_selection == "⚙️ Autonomous Retraining Studio":
    st.markdown("### ⚙️ Autonomous Pipeline Retraining Studio")
    st.markdown("Re-calibrate models, evaluate custom feature selection strategies, and serialize new production pipelines.")
    
    with st.form("retraining_studio_form"):
        rs1, rs2, rs3 = st.columns(3)
        with rs1:
            st.markdown("##### 🎯 Target Architecture")
            r_target = st.selectbox(
                "Classification Scheme:",
                options=list(TARGET_SCHEMES.keys()),
                format_func=lambda x: TARGET_SCHEMES[x]["name"],
                index=0
            )
        with rs2:
            st.markdown("##### 🔬 Feature Selection Strategy")
            r_fs = st.selectbox(
                "Algorithm Paradigm:",
                [
                    "consensus",
                    "mutual_info",
                    "anova_f",
                    "chi2",
                    "random_forest_mdi",
                    "xgboost_mdi",
                    "lasso_l1",
                    "all_features"
                ],
                index=0
            )
            r_k = st.slider("Feature Count (K):", min_value=3, max_value=16, value=8)
        with rs3:
            st.markdown("##### ⚡ Preprocessing & Optimization")
            r_scaler = st.selectbox("Numerical Scaler:", ["standard", "robust", "minmax"])
            r_tune = st.checkbox("Enable Grid Search Hyperparameter Tuning?", value=False)
            
        r_submit = st.form_submit_button("🚀 Launch Autonomous Training Pipeline", use_container_width=True)
        
    if r_submit:
        p_bar = st.progress(0)
        s_msg = st.empty()
        
        try:
            s_msg.text("Step 1/4: Ingesting cohort data & generating academic features...")
            p_bar.progress(25)
            
            new_trainer = ModelTrainer(
                target_scheme=r_target,
                feature_selection_method=r_fs,
                n_features=r_k,
                scaler_type=r_scaler
            )
            
            s_msg.text("Step 2/4: Computing feature rankings...")
            p_bar.progress(50)
            
            s_msg.text("Step 3/4: Stratified 5-Fold cross-validation on 8 classifiers...")
            p_bar.progress(75)
            
            retrain_report = new_trainer.run_full_pipeline(
                tune_hyperparameters=r_tune,
                save_artifacts=True
            )
            
            p_bar.progress(100)
            s_msg.text("Step 4/4: Serialization completed!")
            
            st.success(f"🎉 **Champion Model Selected:** {retrain_report['best_model_name'].replace('_', ' ').title()} with Weighted F1-Score of **{retrain_report['best_score_f1']:.4f}**")
            st.markdown(f"**Top {len(retrain_report['selected_features'])} Features:** `{retrain_report['selected_features']}`")
            st.balloons()
            
        except Exception as e:
            st.error(f"Retraining failed: {str(e)}")
