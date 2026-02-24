"""
GasLambda - Gas Thermal Conductivity Predictor
Streamlit Web Application

A QSPR/ML tool for predicting gas-phase thermal conductivity
of organic compounds from molecular structure (SMILES).

© 2025 Alysophil SAS - AlChemAI Technology
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import ThermalConductivityPredictor
from src.descriptors import compute_descriptors, FEATURE_NAMES

# ---- Page config ----
st.set_page_config(
    page_title="GasLambda - Thermal Conductivity Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .domain-ok {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 0.8rem;
        color: #155724;
    }
    .domain-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem;
        color: #856404;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load model (cached)."""
    return ThermalConductivityPredictor(model_dir="models")


@st.cache_resource
def load_metrics():
    """Load model metrics."""
    with open("models/metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_feature_importance():
    """Load feature importance."""
    with open("models/feature_importance.json") as f:
        return json.load(f)


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🔬 GasLambda")
    st.markdown("**Gas Thermal Conductivity Predictor**")
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    QSPR/ML model predicting gas-phase thermal conductivity (λ) 
    of organic compounds from molecular structure.
    
    **Model**: XGBoost + RDKit descriptors  
    **Data**: DIPPR 801 / Yaws / NIST  
    **Range**: 250-600 K, 1 atm  
    """)
    
    # Load and display metrics
    try:
        metrics = load_metrics()
        st.markdown("### Model Performance (5-fold CV)")
        st.metric("R²", f"{metrics['R2']:.4f}")
        st.metric("MAPE", f"{metrics['MAPE_percent']:.1f}%")
        st.metric("MAE", f"{metrics['MAE_W_mK']:.5f} W/(m·K)")
        st.metric("Training samples", metrics["n_samples"])
    except:
        st.warning("Model not trained yet. Run `python src/train.py` first.")
    
    st.markdown("---")
    st.markdown("""
    ### References
    - Poling et al., *Properties of Gases & Liquids*, 5th ed.
    - Gharagheizi et al., *Ind. Eng. Chem. Res.*, 2013
    - DIPPR 801 Database
    """)
    
    st.markdown("---")
    st.markdown("*© 2025 Alysophil SAS*")


# ---- Main content ----
st.markdown('<div class="main-header">🔬 GasLambda</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Predict gas-phase thermal conductivity of organic compounds using ML/QSPR'
    '</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction",
    "📊 Temperature Sweep",
    "📋 Batch Prediction",
    "🔍 Model Insights",
])


# ==== TAB 1: Single Prediction ====
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input")
        
        # Common compounds for quick selection
        common_compounds = {
            "-- Custom SMILES --": "",
            "Methane": "C",
            "Ethane": "CC",
            "Propane": "CCC",
            "n-Butane": "CCCC",
            "n-Hexane": "CCCCCC",
            "Ethylene": "C=C",
            "Acetylene": "C#C",
            "Benzene": "c1ccccc1",
            "Toluene": "Cc1ccccc1",
            "Methanol": "CO",
            "Ethanol": "CCO",
            "Acetone": "CC(=O)C",
            "Acetic acid": "CC(=O)O",
            "Diethyl ether": "CCOCC",
            "Dimethyl ether": "COC",
            "Chloroform": "ClC(Cl)Cl",
            "Dichloromethane": "ClCCl",
            "Acetonitrile": "CC#N",
            "Methylamine": "CN",
            "Dimethyl sulfide": "CSC",
            "Pyridine": "c1ccncc1",
            "THF": "C1CCOC1",
            "DMF": "CN(C)C=O",
            "Ethyl acetate": "CCOC(=O)C",
        }
        
        selected = st.selectbox(
            "Quick select a compound:",
            options=list(common_compounds.keys()),
        )
        
        if selected == "-- Custom SMILES --":
            smiles_input = st.text_input(
                "Enter SMILES:",
                value="CCO",
                help="Enter a valid SMILES string for the organic compound",
            )
        else:
            smiles_input = common_compounds[selected]
            st.code(f"SMILES: {smiles_input}", language=None)
        
        temperature = st.slider(
            "Temperature (K):",
            min_value=200,
            max_value=700,
            value=300,
            step=5,
            help="Gas temperature at 1 atm",
        )
        
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn and smiles_input:
            try:
                predictor = load_predictor()
                result = predictor.predict(smiles_input, float(temperature))
                
                st.markdown("### Result")
                
                # Main result
                lam = result["thermal_conductivity_W_mK"]
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Thermal Conductivity λ</div>'
                    f'<div class="metric-value">{lam:.4f} W/(m·K)</div>'
                    f'<div class="metric-label">{lam*1000:.2f} mW/(m·K) at {temperature} K</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                
                st.markdown("")
                
                # Domain check
                if result["in_domain"]:
                    st.markdown(
                        '<div class="domain-ok">'
                        '✅ <b>Within applicability domain</b> — prediction is reliable'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="domain-warning">'
                        f'⚠️ <b>Outside applicability domain</b><br>'
                        f'{result["domain_warning"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                
                st.markdown(
                    f"**Estimated uncertainty**: ±{result['model_uncertainty_pct']:.1f}%"
                )
                
                # Show molecular descriptors
                with st.expander("📐 Molecular Descriptors"):
                    desc = compute_descriptors(smiles_input, float(temperature))
                    desc_df = pd.DataFrame([
                        {"Descriptor": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                        for k, v in desc.items()
                    ])
                    st.dataframe(desc_df, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        elif predict_btn:
            st.warning("Please enter a SMILES string.")


# ==== TAB 2: Temperature Sweep ====
with tab2:
    st.markdown("### Temperature Dependence of λ")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        smiles_sweep = st.text_input(
            "SMILES for sweep:",
            value="CCO",
            key="sweep_smiles",
        )
        
        T_min = st.number_input("T min (K)", value=250, min_value=200, max_value=600)
        T_max = st.number_input("T max (K)", value=600, min_value=300, max_value=800)
        n_points = st.slider("Number of points", 10, 100, 50)
        
        # Compare multiple compounds
        add_compounds = st.text_area(
            "Compare with (one SMILES per line, optional):",
            value="",
            height=100,
            help="Add SMILES to overlay on the same plot",
        )
        
        sweep_btn = st.button("📈 Generate", type="primary", key="sweep_btn")
    
    with col2:
        if sweep_btn and smiles_sweep:
            try:
                predictor = load_predictor()
                
                fig = go.Figure()
                
                # Main compound
                sweep_df = predictor.predict_temperature_sweep(
                    smiles_sweep, T_min, T_max, n_points
                )
                fig.add_trace(go.Scatter(
                    x=sweep_df["temperature_K"],
                    y=sweep_df["thermal_conductivity_W_mK"] * 1000,
                    mode="lines",
                    name=smiles_sweep,
                    line=dict(width=3),
                ))
                
                # Additional compounds
                if add_compounds.strip():
                    for smi in add_compounds.strip().split("\n"):
                        smi = smi.strip()
                        if smi:
                            try:
                                df_add = predictor.predict_temperature_sweep(
                                    smi, T_min, T_max, n_points
                                )
                                fig.add_trace(go.Scatter(
                                    x=df_add["temperature_K"],
                                    y=df_add["thermal_conductivity_W_mK"] * 1000,
                                    mode="lines",
                                    name=smi,
                                    line=dict(width=2),
                                ))
                            except:
                                st.warning(f"Could not process: {smi}")
                
                fig.update_layout(
                    title="Gas Thermal Conductivity vs Temperature",
                    xaxis_title="Temperature (K)",
                    yaxis_title="λ (mW/(m·K))",
                    template="plotly_white",
                    height=500,
                    legend=dict(
                        yanchor="top", y=0.99,
                        xanchor="left", x=0.01,
                    ),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                with st.expander("📊 Data Table"):
                    display_df = sweep_df[["temperature_K", "thermal_conductivity_W_mK", "in_domain"]].copy()
                    display_df["λ (mW/(m·K))"] = display_df["thermal_conductivity_W_mK"] * 1000
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Download
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "thermal_conductivity_sweep.csv",
                        "text/csv",
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ==== TAB 3: Batch Prediction ====
with tab3:
    st.markdown("### Batch Prediction")
    st.markdown("Enter multiple compounds (one per line): `SMILES, Temperature_K`")
    
    batch_input = st.text_area(
        "Input (SMILES, T_K):",
        value="C, 300\nCC, 300\nCCC, 300\nCCCC, 300\nCCCCC, 300\nCCCCCC, 300\nc1ccccc1, 400\nCCO, 350\nCC(=O)C, 400",
        height=200,
    )
    
    batch_btn = st.button("🔮 Predict Batch", type="primary", key="batch_btn")
    
    if batch_btn and batch_input:
        try:
            predictor = load_predictor()
            
            smiles_list = []
            temps_list = []
            
            for line in batch_input.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                smiles_list.append(parts[0])
                temps_list.append(float(parts[1]) if len(parts) > 1 else 300.0)
            
            results_df = predictor.predict_batch(smiles_list, temps_list)
            
            # Display
            display_cols = [
                "smiles", "temperature_K", "thermal_conductivity_W_mK",
                "in_domain", "model_uncertainty_pct",
            ]
            display_df = results_df[display_cols].copy()
            display_df["λ (mW/(m·K))"] = display_df["thermal_conductivity_W_mK"] * 1000
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Visualization
            valid = display_df.dropna(subset=["thermal_conductivity_W_mK"])
            if len(valid) > 1:
                fig = px.bar(
                    valid,
                    x="smiles",
                    y="λ (mW/(m·K))",
                    color="in_domain",
                    color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
                    title="Predicted Thermal Conductivity",
                )
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "batch_predictions.csv",
                "text/csv",
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ==== TAB 4: Model Insights ====
with tab4:
    st.markdown("### Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Importance")
        try:
            feat_imp = load_feature_importance()
            fi_df = pd.DataFrame(feat_imp[:15])
            
            fig = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 15 Most Important Features",
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                template="plotly_white",
                height=500,
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Train the model first to see feature importance.")
    
    with col2:
        st.markdown("#### Cross-Validation: Predicted vs Actual")
        try:
            cv_df = pd.read_csv("models/cv_predictions.csv")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cv_df["thermal_conductivity_W_mK"] * 1000,
                y=cv_df["predicted_W_mK"] * 1000,
                mode="markers",
                marker=dict(
                    size=6,
                    color=cv_df["error_pct"],
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="Error (%)"),
                    line=dict(width=0.5, color="white"),
                ),
                text=cv_df["name"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Actual: %{x:.2f} mW/(m·K)<br>"
                    "Predicted: %{y:.2f} mW/(m·K)<br>"
                    "<extra></extra>"
                ),
            ))
            
            # Perfect prediction line
            min_val = cv_df["thermal_conductivity_W_mK"].min() * 1000
            max_val = cv_df["thermal_conductivity_W_mK"].max() * 1000
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=1),
                showlegend=False,
            ))
            
            fig.update_layout(
                title="Parity Plot (5-fold CV)",
                xaxis_title="Actual λ (mW/(m·K))",
                yaxis_title="Predicted λ (mW/(m·K))",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Train the model first to see CV results.")
    
    # Error distribution
    try:
        cv_df = pd.read_csv("models/cv_predictions.csv")
        
        st.markdown("#### Error Distribution")
        fig = px.histogram(
            cv_df,
            x="error_pct",
            nbins=30,
            title="Distribution of Prediction Errors (%)",
            labels={"error_pct": "Absolute Error (%)"},
            color_discrete_sequence=["#667eea"],
        )
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Worst predictions
        st.markdown("#### Compounds with Highest Errors")
        worst = cv_df.nlargest(10, "error_pct")[
            ["name", "smiles", "temperature_K", "thermal_conductivity_W_mK", 
             "predicted_W_mK", "error_pct"]
        ]
        st.dataframe(worst, use_container_width=True, hide_index=True)
    except:
        pass
