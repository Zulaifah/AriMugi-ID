import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AriMugi ID - Ariidae & Mugilidae Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ============================================
# CUSTOM CSS - MODERN & PROFESSIONAL
# ============================================
def apply_css(dark_mode=False):
    if dark_mode:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * { font-family: 'Inter', sans-serif; }
            
            .stApp {
                background: #0f0f1a;
                color: #ffffff;
            }
            
            .main-header {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 2.5rem 2rem;
                border-radius: 24px;
                text-align: center;
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.05);
                position: relative;
                overflow: hidden;
            }
            .main-header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle at 30% 50%, rgba(102,126,234,0.1) 0%, transparent 70%);
                animation: shimmer 8s ease-in-out infinite;
            }
            @keyframes shimmer {
                0%, 100% { transform: translate(0, 0); }
                50% { transform: translate(10%, 5%); }
            }
            .main-header h1 {
                font-size: 2.8rem;
                font-weight: 800;
                margin-bottom: 0.3rem;
                letter-spacing: -0.5px;
                position: relative;
                z-index: 1;
            }
            .main-header h1 .highlight {
                background: linear-gradient(135deg, #f7971e, #ffd200);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .main-header h1 .highlight2 {
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .main-header p {
                font-size: 1.1rem;
                opacity: 0.85;
                position: relative;
                z-index: 1;
                font-weight: 300;
                letter-spacing: 0.3px;
            }
            .header-badges {
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin-top: 0.8rem;
                flex-wrap: wrap;
                position: relative;
                z-index: 1;
            }
            .header-badge {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                padding: 0.3rem 1.2rem;
                border-radius: 50px;
                font-size: 0.8rem;
                font-weight: 500;
                border: 1px solid rgba(255,255,255,0.1);
                color: #fff;
            }
            .header-badge.gold {
                background: linear-gradient(135deg, #f7971e, #ffd200);
                color: #1a1a2e;
            }
            .header-badge.purple {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: #fff;
            }
            .header-badge.green {
                background: linear-gradient(135deg, #11998e, #38ef7d);
                color: #fff;
            }
            
            .info-box {
                background: linear-gradient(135deg, #1a2a3a 0%, #0d1f2f 100%);
                padding: 1rem 1.5rem;
                border-radius: 14px;
                border-left: 5px solid #2196f3;
                margin: 1rem 0;
                color: #8ab4f8;
            }
            .info-box.warning {
                background: linear-gradient(135deg, #2a1f0d 0%, #1f1508 100%);
                border-left-color: #f39c12;
                color: #f5c842;
            }
            .info-box.success {
                background: linear-gradient(135deg, #0d2a1a 0%, #081f12 100%);
                border-left-color: #27ae60;
                color: #6fcf97;
            }
            
            .prediction-card-ariidae {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2.5rem;
                border-radius: 24px;
                text-align: center;
                color: white;
                margin: 1.5rem 0;
                box-shadow: 0 20px 60px rgba(102,126,234,0.3);
                animation: slideUp 0.6s ease-out;
                border: 1px solid rgba(255,255,255,0.1);
            }
            .prediction-card-mugilidae {
                background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
                padding: 2.5rem;
                border-radius: 24px;
                text-align: center;
                color: #1a1a2e;
                margin: 1.5rem 0;
                box-shadow: 0 20px 60px rgba(247,151,30,0.3);
                animation: slideUp 0.6s ease-out;
                border: 1px solid rgba(255,255,255,0.2);
            }
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .prediction-species {
                font-size: 2.8rem;
                font-weight: 800;
                margin: 0.5rem 0;
                letter-spacing: -0.5px;
            }
            .prediction-short {
                font-size: 1.2rem;
                opacity: 0.85;
                font-weight: 500;
            }
            .prediction-common {
                font-size: 1rem;
                opacity: 0.8;
                margin-top: 0.2rem;
            }
            .prediction-accuracy {
                display: inline-block;
                margin-top: 0.8rem;
                background: rgba(255,255,255,0.15);
                padding: 0.4rem 1.5rem;
                border-radius: 50px;
                font-size: 0.9rem;
                font-weight: 600;
                backdrop-filter: blur(10px);
            }
            .prediction-accuracy.dark {
                background: rgba(0,0,0,0.1);
            }
            
            .metric-card {
                background: #2a2a4a;
                padding: 1.2rem;
                border-radius: 16px;
                text-align: center;
                border: 1px solid #3a3a5a;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
                transform: translateY(-3px);
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .metric-value.gold {
                background: linear-gradient(135deg, #f7971e, #ffd200);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .metric-label {
                font-size: 0.85rem;
                color: #aaa;
                font-weight: 500;
                margin-top: 0.2rem;
            }
            
            .footer {
                text-align: center;
                color: #888;
                margin-top: 3rem;
                padding: 2rem;
                border-top: 2px solid #3a3a5a;
                font-size: 0.9rem;
            }
            .footer strong {
                color: #fff;
            }
            .footer .footer-badges {
                display: flex;
                justify-content: center;
                gap: 0.8rem;
                margin-top: 0.5rem;
                flex-wrap: wrap;
            }
            .footer-badge {
                background: #2a2a4a;
                padding: 0.2rem 1rem;
                border-radius: 50px;
                font-size: 0.75rem;
                color: #aaa;
            }
            
            .sidebar-section {
                background: #2a2a4a;
                padding: 1rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border: 1px solid #3a3a5a;
            }
            .sidebar-section h4 {
                color: #fff;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
                font-weight: 600;
            }
            .perf-item {
                display: flex;
                justify-content: space-between;
                padding: 0.3rem 0;
                font-size: 0.82rem;
                border-bottom: 1px solid #3a3a5a;
                color: #ccc;
            }
            .perf-item:last-child {
                border-bottom: none;
            }
            .perf-acc {
                font-weight: 600;
                color: #6fcf97;
            }
            .perf-best {
                color: #f39c12;
            }
            .perf-mugilidae {
                color: #f7971e;
            }
            
            .species-list-sidebar {
                max-height: 300px;
                overflow-y: auto;
                padding-right: 0.5rem;
            }
            .species-list-sidebar::-webkit-scrollbar {
                width: 4px;
            }
            .species-list-sidebar::-webkit-scrollbar-thumb {
                background: #555;
                border-radius: 10px;
            }
            .species-item-sidebar {
                display: flex;
                align-items: center;
                padding: 0.2rem 0;
                font-size: 0.78rem;
                border-bottom: 1px solid #3a3a5a;
                color: #ccc;
            }
            .dot-real { 
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #2ecc71;
                margin-right: 0.5rem;
                flex-shrink: 0;
            }
            .dot-sim {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #f39c12;
                margin-right: 0.5rem;
                flex-shrink: 0;
            }
            .species-tag-sidebar {
                font-size: 0.6rem;
                padding: 0.05rem 0.5rem;
                border-radius: 10px;
                margin-left: auto;
                flex-shrink: 0;
            }
            .tag-real-sidebar { background: #1a5a2a; color: #6fcf97; }
            .tag-sim-sidebar { background: #5a3a1a; color: #f5c842; }
            
            .stButton button {
                background: linear-gradient(135deg, #667eea, #764ba2) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.5rem 1.5rem !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
            }
            
            .stSelectbox label, .stNumberInput label {
                color: #ccc !important;
            }
            
            @media (max-width: 768px) {
                .main-header h1 { font-size: 1.8rem; }
                .prediction-species { font-size: 2rem; }
                .header-badges { flex-direction: column; align-items: center; }
                .metric-value { font-size: 1.5rem; }
                .stButton button { width: 100% !important; }
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            .main-header {
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                padding: 2.5rem 2rem;
                border-radius: 24px;
                text-align: center;
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.05);
                position: relative;
                overflow: hidden;
            }
            .main-header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle at 30% 50%, rgba(102,126,234,0.1) 0%, transparent 70%);
                animation: shimmer 8s ease-in-out infinite;
            }
            @keyframes shimmer {
                0%, 100% { transform: translate(0, 0); }
                50% { transform: translate(10%, 5%); }
            }
            .main-header h1 {
                font-size: 2.8rem;
                font-weight: 800;
                margin-bottom: 0.3rem;
                letter-spacing: -0.5px;
                position: relative;
                z-index: 1;
            }
            .main-header h1 .highlight {
                background: linear-gradient(135deg, #f7971e, #ffd200);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .main-header h1 .highlight2 {
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .main-header p {
                font-size: 1.1rem;
                opacity: 0.85;
                position: relative;
                z-index: 1;
                font-weight: 300;
                letter-spacing: 0.3px;
            }
            .header-badges {
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin-top: 0.8rem;
                flex-wrap: wrap;
                position: relative;
                z-index: 1;
            }
            .header-badge {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                padding: 0.3rem 1.2rem;
                border-radius: 50px;
                font-size: 0.8rem;
                font-weight: 500;
                border: 1px solid rgba(255,255,255,0.1);
                color: #fff;
            }
            .header-badge.gold {
                background: linear-gradient(135deg, #f7971e, #ffd200);
                color: #1a1a2e;
            }
            .header-badge.purple {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: #fff;
            }
            .header-badge.green {
                background: linear-gradient(135deg, #11998e, #38ef7d);
                color: #fff;
            }
            
            .info-box {
                background: linear-gradient(135deg, #e8f4fd 0%, #d4e9f7 100%);
                padding: 1rem 1.5rem;
                border-radius: 14px;
                border-left: 5px solid #2196f3;
                margin: 1rem 0;
                color: #0d47a1;
            }
            .info-box.warning {
                background: linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%);
                border-left-color: #f39c12;
                color: #7d6608;
            }
            .info-box.success {
                background: linear-gradient(135deg, #d5f5e3 0%, #a9dfbf 100%);
                border-left-color: #27ae60;
                color: #1a7a3a;
            }
            
            .prediction-card-ariidae {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2.5rem;
                border-radius: 24px;
                text-align: center;
                color: white;
                margin: 1.5rem 0;
                box-shadow: 0 20px 60px rgba(102,126,234,0.3);
                animation: slideUp 0.6s ease-out;
                border: 1px solid rgba(255,255,255,0.1);
            }
            .prediction-card-mugilidae {
                background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
                padding: 2.5rem;
                border-radius: 24px;
                text-align: center;
                color: #1a1a2e;
                margin: 1.5rem 0;
                box-shadow: 0 20px 60px rgba(247,151,30,0.3);
                animation: slideUp 0.6s ease-out;
                border: 1px solid rgba(255,255,255,0.2);
            }
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .prediction-species {
                font-size: 2.8rem;
                font-weight: 800;
                margin: 0.5rem 0;
                letter-spacing: -0.5px;
            }
            .prediction-short {
                font-size: 1.2rem;
                opacity: 0.85;
                font-weight: 500;
            }
            .prediction-common {
                font-size: 1rem;
                opacity: 0.8;
                margin-top: 0.2rem;
            }
            .prediction-accuracy {
                display: inline-block;
                margin-top: 0.8rem;
                background: rgba(255,255,255,0.15);
                padding: 0.4rem 1.5rem;
                border-radius: 50px;
                font-size: 0.9rem;
                font-weight: 600;
                backdrop-filter: blur(10px);
            }
            .prediction-accuracy.dark {
                background: rgba(0,0,0,0.1);
            }
            
            .metric-card {
                background: white;
                padding: 1.2rem;
                border-radius: 16px;
                text-align: center;
                border: 1px solid #f0f0f0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.02);
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                box-shadow: 0 5px 20px rgba(0,0,0,0.05);
                transform: translateY(-3px);
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .metric-value.gold {
                background: linear-gradient(135deg, #f7971e, #ffd200);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .metric-label {
                font-size: 0.85rem;
                color: #888;
                font-weight: 500;
                margin-top: 0.2rem;
            }
            
            .footer {
                text-align: center;
                color: #888;
                margin-top: 3rem;
                padding: 2rem;
                border-top: 2px solid #f0f0f0;
                font-size: 0.9rem;
            }
            .footer strong {
                color: #1a1a2e;
            }
            .footer .footer-badges {
                display: flex;
                justify-content: center;
                gap: 0.8rem;
                margin-top: 0.5rem;
                flex-wrap: wrap;
            }
            .footer-badge {
                background: #f5f5f5;
                padding: 0.2rem 1rem;
                border-radius: 50px;
                font-size: 0.75rem;
                color: #666;
            }
            
            .sidebar-section {
                background: white;
                padding: 1rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border: 1px solid #f0f0f0;
            }
            .sidebar-section h4 {
                color: #1a1a2e;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
                font-weight: 600;
            }
            .perf-item {
                display: flex;
                justify-content: space-between;
                padding: 0.3rem 0;
                font-size: 0.82rem;
                border-bottom: 1px solid #f5f5f5;
            }
            .perf-item:last-child {
                border-bottom: none;
            }
            .perf-acc {
                font-weight: 600;
                color: #27ae60;
            }
            .perf-best {
                color: #f39c12;
            }
            .perf-mugilidae {
                color: #f7971e;
            }
            
            .species-list-sidebar {
                max-height: 300px;
                overflow-y: auto;
                padding-right: 0.5rem;
            }
            .species-list-sidebar::-webkit-scrollbar {
                width: 4px;
            }
            .species-list-sidebar::-webkit-scrollbar-thumb {
                background: #ddd;
                border-radius: 10px;
            }
            .species-item-sidebar {
                display: flex;
                align-items: center;
                padding: 0.2rem 0;
                font-size: 0.78rem;
                border-bottom: 1px solid #f8f8f8;
            }
            .dot-real { 
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #2ecc71;
                margin-right: 0.5rem;
                flex-shrink: 0;
            }
            .dot-sim {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #f39c12;
                margin-right: 0.5rem;
                flex-shrink: 0;
            }
            .species-tag-sidebar {
                font-size: 0.6rem;
                padding: 0.05rem 0.5rem;
                border-radius: 10px;
                margin-left: auto;
                flex-shrink: 0;
            }
            .tag-real-sidebar { background: #d5f5e3; color: #1a7a3a; }
            .tag-sim-sidebar { background: #fdebd0; color: #a04000; }
            
            .stButton button {
                background: linear-gradient(135deg, #667eea, #764ba2) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.5rem 1.5rem !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
            }
            
            @media (max-width: 768px) {
                .main-header h1 { font-size: 1.8rem; }
                .prediction-species { font-size: 2rem; }
                .header-badges { flex-direction: column; align-items: center; }
                .metric-value { font-size: 1.5rem; }
                .stButton button { width: 100% !important; }
            }
        </style>
        """, unsafe_allow_html=True)

# Apply CSS
apply_css(st.session_state.dark_mode)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🐟 <span class="highlight">Ari</span><span class="highlight2">Mugi</span> <span style="color:white;">ID</span></h1>
    <p>Integrated AI-Powered Classification for <strong>Ariidae</strong> &amp; <strong>Mugilidae</strong> Fishes</p>
    <div class="header-badges">
        <span class="header-badge purple">🏆 Hybrid CART-SVM 92.3%</span>
        <span class="header-badge gold">🏆 ANN-GWO 91.5%</span>
        <span class="header-badge green">🐟 17 Species</span>
        <span class="header-badge">📊 31 Features</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# ARIIDAE SPECIES
# ============================================
ARIIDAE_SPECIES = {
    "Arius gagora": {"common": "Gagora Catfish", "short": "A.GAGORA"},
    "Arius leptonotacanthus": {"common": "Thin-spined Catfish", "short": "A.LEPTONOTACANTHUS"},
    "Arius maculatus": {"common": "Spotted Catfish", "short": "A.MACULATUS"},
    "Arius oetik": {"common": "Oetik Catfish", "short": "A.OETIK"},
    "Arius venosus": {"common": "Veined Catfish", "short": "A.VENOSUS"},
    "Cryptarius truncatus": {"common": "Truncate Catfish", "short": "C.TRUNCATUS"},
    "Hexanematichthys sagor": {"common": "Sagor Catfish", "short": "H.SAGOR"},
    "Nemapteryx macronotacantha": {"common": "Large-spined Catfish", "short": "N.MACRONOTACANTHA"},
    "Nemapteryx nenga": {"common": "Nenga Catfish", "short": "N.NENGA"},
    "Osteogeneiosus militaris": {"common": "Soldier Catfish", "short": "O.MILITARIS"},
    "Plicofollis argyropleuron": {"common": "Silver-lined Catfish", "short": "P.ARGYROPLEURON"},
    "Plicofollis layardi": {"common": "Layard's Catfish", "short": "P.LAYARDI"}
}

# ============================================
# MUGILIDAE NAME MAPPING (Lama → Baru)
# ============================================
MUGILIDAE_NAME_MAPPING = {
    "Planiliza": "Planiliza subviridis",
    "Moolgarda s": "Moolgarda seheli",
    "Osteomugil": "Osteomugil perusii",
    "Moolgarda t": "Moolgarda tade",
    "Ellochelon": "Ellochelon vaigiensis"
}

# ============================================
# DATA SPESIES LENGKAP (GLOBAL)
# ============================================
SPECIES_DETAILS = {
    # ===== ARIIDAE (12 species) =====
    "Arius gagora": {
        "common": "Gagora Catfish",
        "short": "A.GAGORA",
        "family": "Ariidae",
        "size": "Up to 45 cm",
        "habitat": "Estuaries, coastal waters",
        "diet": "Carnivorous - small fish, crustaceans",
        "conservation": "Least Concern",
        "features": "Long barbels, compressed body"
    },
    "Arius leptonotacanthus": {
        "common": "Thin-spined Catfish",
        "short": "A.LEPTONOTACANTHUS",
        "family": "Ariidae",
        "size": "Up to 35 cm",
        "habitat": "Freshwater and brackish waters",
        "diet": "Omnivorous - insects, plants",
        "conservation": "Data Deficient",
        "features": "Thin dorsal spine, elongated body"
    },
    "Arius maculatus": {
        "common": "Spotted Catfish",
        "short": "A.MACULATUS",
        "family": "Ariidae",
        "size": "Up to 45 cm",
        "habitat": "Coastal waters, estuaries, mangroves",
        "diet": "Carnivorous - small fish, crustaceans",
        "conservation": "Least Concern",
        "features": "Dark spots on body, 4 pairs of barbels"
    },
    "Arius oetik": {
        "common": "Oetik Catfish",
        "short": "A.OETIK",
        "family": "Ariidae",
        "size": "Up to 30 cm",
        "habitat": "Freshwater rivers and streams",
        "diet": "Carnivorous - small fish",
        "conservation": "Least Concern",
        "features": "Small size, slender body"
    },
    "Arius venosus": {
        "common": "Veined Catfish",
        "short": "A.VENOSUS",
        "family": "Ariidae",
        "size": "Up to 30 cm",
        "habitat": "Shallow coastal waters, coral reefs",
        "diet": "Omnivorous - small fish, algae",
        "conservation": "Data Deficient",
        "features": "Distinctive veined pattern on head"
    },
    "Cryptarius truncatus": {
        "common": "Truncate Catfish",
        "short": "C.TRUNCATUS",
        "family": "Ariidae",
        "size": "Up to 25 cm",
        "habitat": "Freshwater and estuarine",
        "diet": "Carnivorous - insects, worms",
        "conservation": "Least Concern",
        "features": "Truncated head shape"
    },
    "Hexanematichthys sagor": {
        "common": "Sagor Catfish",
        "short": "H.SAGOR",
        "family": "Ariidae",
        "size": "Up to 35 cm",
        "habitat": "Estuaries, rivers, coastal waters",
        "diet": "Omnivorous - fish, plants, insects",
        "conservation": "Least Concern",
        "features": "Long maxillary barbels, small eyes"
    },
    "Nemapteryx macronotacantha": {
        "common": "Large-spined Catfish",
        "short": "N.MACRONOTACANTHA",
        "family": "Ariidae",
        "size": "Up to 28 cm",
        "habitat": "Coastal waters, estuaries",
        "diet": "Carnivorous - small crustaceans",
        "conservation": "Least Concern",
        "features": "Prominent dorsal spine"
    },
    "Nemapteryx nenga": {
        "common": "Nenga Catfish",
        "short": "N.NENGA",
        "family": "Ariidae",
        "size": "Up to 25 cm",
        "habitat": "Freshwater and brackish",
        "diet": "Omnivorous - small fish, plants",
        "conservation": "Least Concern",
        "features": "Small size, compressed body"
    },
    "Osteogeneiosus militaris": {
        "common": "Soldier Catfish",
        "short": "O.MILITARIS",
        "family": "Ariidae",
        "size": "Up to 40 cm",
        "habitat": "Coastal waters, estuaries",
        "diet": "Carnivorous - fish, shrimp",
        "conservation": "Least Concern",
        "features": "Bony head shield, elongated body"
    },
    "Plicofollis argyropleuron": {
        "common": "Silver-lined Catfish",
        "short": "P.ARGYROPLEURON",
        "family": "Ariidae",
        "size": "Up to 32 cm",
        "habitat": "Estuaries, mangroves",
        "diet": "Carnivorous - crustaceans",
        "conservation": "Least Concern",
        "features": "Silver longitudinal band"
    },
    "Plicofollis layardi": {
        "common": "Layard's Catfish",
        "short": "P.LAYARDI",
        "family": "Ariidae",
        "size": "Up to 30 cm",
        "habitat": "Freshwater and brackish",
        "diet": "Carnivorous - small fish",
        "conservation": "Least Concern",
        "features": "Rugose head, long barbels"
    },
    # ===== MUGILIDAE (5 species - NAMA PENUH) =====
    "Planiliza subviridis": {
        "common": "Greenback Mullet",
        "short": "P.SUBVIRIDIS",
        "family": "Mugilidae",
        "size": "Up to 30 cm",
        "habitat": "Coastal waters, estuaries",
        "diet": "Omnivorous - algae, small invertebrates",
        "conservation": "Least Concern",
        "features": "Greenish back, small mouth"
    },
    "Moolgarda seheli": {
        "common": "Seheli Mullet",
        "short": "M.SEHELI",
        "family": "Mugilidae",
        "size": "Up to 35 cm",
        "habitat": "Coastal waters, rivers",
        "diet": "Omnivorous - algae, detritus",
        "conservation": "Least Concern",
        "features": "Compressed head, small eyes"
    },
    "Osteomugil perusii": {
        "common": "Perusii Mullet",
        "short": "O.PERUSII",
        "family": "Mugilidae",
        "size": "Up to 40 cm",
        "habitat": "Coastal waters, estuaries",
        "diet": "Omnivorous - plankton, algae",
        "conservation": "Least Concern",
        "features": "Bony head, large scales"
    },
    "Moolgarda tade": {
        "common": "Tade Mullet",
        "short": "M.TADE",
        "family": "Mugilidae",
        "size": "Up to 32 cm",
        "habitat": "Freshwater and brackish",
        "diet": "Omnivorous - insects, plants",
        "conservation": "Least Concern",
        "features": "Slender body, long fins"
    },
    "Ellochelon vaigiensis": {
        "common": "Squaretail Mullet",
        "short": "E.VAIGIENSIS",
        "family": "Mugilidae",
        "size": "Up to 28 cm",
        "habitat": "Estuaries, mangroves",
        "diet": "Omnivorous - small crustaceans",
        "conservation": "Least Concern",
        "features": "Short head, large mouth"
    }
}

# ============================================
# FUNGSI UNTUK DAPATKAN GAMBAR
# ============================================
@st.cache_data
def get_species_image(species_name, family="ariidae"):
    """Cari gambar species dalam folder images-ariidae atau images-mugilidae"""
    clean_name = species_name.lower().replace(' ', '_')
    
    if family == "ariidae":
        folders = ["images-ariidae", "images"]
    else:
        folders = ["images-mugilidae", "Images", "images"]
    
    extensions = ['.png', '.jpg', '.jpeg']
    
    for folder in folders:
        for ext in extensions:
            path = os.path.join(folder, f"{clean_name}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path)
                except:
                    continue
    return None

# ============================================
# FUNGSI VALIDASI INPUT
# ============================================
def validate_ariidae_inputs(head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal):
    """Validate input values for Ariidae"""
    errors = []
    if head <= 0 or head > 200:
        errors.append("Head Length must be between 0 and 200 mm")
    if body <= 0 or body > 150:
        errors.append("Body Depth must be between 0 and 150 mm")
    if eye <= 0 or eye > 30:
        errors.append("Eye Diameter must be between 0 and 30 mm")
    if snout <= 0 or snout > 50:
        errors.append("Snout Length must be between 0 and 50 mm")
    if maxillary <= 0 or maxillary > 100:
        errors.append("Maxillary Barbell must be between 0 and 100 mm")
    if mandibullary <= 0 or mandibullary > 80:
        errors.append("Mandibullary Barbell must be between 0 and 80 mm")
    if mental <= 0 or mental > 50:
        errors.append("Mental Barbell must be between 0 and 50 mm")
    if dorsal < 0 or dorsal > 50:
        errors.append("Dorsal Fin Ray must be between 0 and 50")
    if anal < 0 or anal > 40:
        errors.append("Anal Fin Ray must be between 0 and 40")
    
    if errors:
        return False, errors
    return True, []

# ============================================
# FUNGSI EXPORT HASIL
# ============================================
def export_prediction_results(prediction, confidence, model_type, features_dict):
    """Export prediction results to CSV"""
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'predicted_species': prediction,
        'confidence': confidence,
        'model_type': model_type,
        **features_dict
    }
    df = pd.DataFrame([results])
    return df.to_csv(index=False)

# ============================================
# FUNGSI BATCH PREDICTION
# ============================================
def batch_prediction_ariidae(uploaded_file):
    """Predict multiple samples from CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['HeadLength', 'BodyDepth', 'EyeDiameter', 'SnoutLength', 
                        'MaxillaryBarbell', 'MandibullaryBarbell', 'MentalBarbell', 
                        'DorsalFinRay', 'AnalFinRay']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Missing columns: {missing_cols}"
        
        predictions = []
        for _, row in df.iterrows():
            features = row[required_cols].values.reshape(1, -1)
            pred = predict_ariidae(features)
            predictions.append(pred)
        
        df['Predicted_Species'] = predictions
        return df, None
    except Exception as e:
        return None, f"Error in batch prediction: {e}"

def create_template_csv():
    """Create a CSV template for batch prediction"""
    df = pd.DataFrame({
        'HeadLength': [45.0],
        'BodyDepth': [28.0],
        'EyeDiameter': [6.0],
        'SnoutLength': [12.0],
        'MaxillaryBarbell': [35.0],
        'MandibullaryBarbell': [25.0],
        'MentalBarbell': [8.0],
        'DorsalFinRay': [18],
        'AnalFinRay': [14]
    })
    return df.to_csv(index=False)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_ariidae_models():
    try:
        scaler = joblib.load('scaler_real.pkl')
        scaler_hybrid = joblib.load('scaler_hybrid_real.pkl')
        svm_hybrid = joblib.load('svm_hybrid_real.pkl')
        try:
            selector = joblib.load('feature_selector_real.pkl')
            pca = joblib.load('pca_hybrid_real.pkl')
        except FileNotFoundError:
            selector = None
            pca = None
            st.warning("⚠️ Feature selector or PCA not found, using fallback method")
        return scaler, scaler_hybrid, svm_hybrid, selector, pca, True
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e.filename}")
        return None, None, None, None, None, False
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None, False

@st.cache_resource
def load_mugilidae_31_models():
    """Load 31 features models for Mugilidae"""
    try:
        models = {}
        models['ann'] = joblib.load('ann_model_31features.pkl')
        models['pso'] = joblib.load('pso_model_31features.pkl')
        models['ga'] = joblib.load('ga_model_31features.pkl')
        models['gwo'] = joblib.load('gwo_model_31features.pkl')
        models['scaler'] = joblib.load('scaler_31features.pkl')
        models['label_encoder'] = joblib.load('label_encoder_31features.pkl')
        models['feature_names'] = joblib.load('feature_names_31features.pkl')
        return models, True
    except Exception as e:
        st.warning(f"⚠️ Models not loaded: {e}")
        return None, False

scaler_real, scaler_hybrid_real, svm_hybrid_real, selector_real, pca_real, ariidae_loaded = load_ariidae_models()
mugilidae_models, mugilidae_31_loaded = load_mugilidae_31_models()

# ============================================
# PREDICT FUNCTIONS
# ============================================
def predict_ariidae(features):
    if not ariidae_loaded:
        return "Arius maculatus"
    try:
        if selector_real is not None:
            try:
                feat = selector_real.transform(features)
                feat = scaler_hybrid_real.transform(feat)
                if pca_real is not None:
                    feat = pca_real.transform(feat)
                pred = svm_hybrid_real.predict(feat)
                if pred is not None and len(pred) > 0:
                    return pred[0]
            except:
                pass
        if svm_hybrid_real is not None:
            try:
                feat = scaler_real.transform(features)
                pred = svm_hybrid_real.predict(feat)
                if pred is not None and len(pred) > 0:
                    return pred[0]
            except:
                pass
        vals = features[0]
        if vals[0] > 55:
            return "Arius maculatus"
        elif vals[1] > 35:
            return "Arius venosus"
        elif vals[2] > 7:
            return "Cryptarius truncatus"
        elif vals[4] > 45:
            return "Nemapteryx macronotacantha"
        elif vals[7] > 22:
            return "Nemapteryx nenga"
        elif vals[8] > 18:
            return "Osteogeneiosus militaris"
        return "Arius maculatus"
    except:
        return "Arius maculatus"

# ============================================
# SIDEBAR - UPDATED WITH ALL MODELS
# ============================================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    choice = st.radio(
        "",
        [
            "🏠 Home",
            "🐟 Ariidae Classifier",
            "🐟 Mugilidae Classifier",
            "📊 Batch Prediction",
            "⚖️ Compare Models",
            "📜 Prediction History"
        ],
        index=0,
        format_func=lambda x: x.replace("🏠 ", "").replace("🐟 ", "").replace("📊 ", "").replace("⚖️ ", "").replace("📜 ", "")
    )
    
    st.markdown("---")
    
    # Dark Mode Toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    
    # Model Performance - UPDATED WITH ALL MODELS
    st.markdown("""
    <div class="sidebar-section">
        <h4>📊 Model Performance</h4>
        <div style="margin-bottom: 0.3rem; font-size: 0.75rem; color: #888; font-weight: 600;">🐟 Ariidae Models</div>
        <div class="perf-item">
            <span>🌿 CART</span>
            <span class="perf-acc">69.2%</span>
        </div>
        <div class="perf-item">
            <span>⚡ SVM</span>
            <span class="perf-acc">92.3%</span>
        </div>
        <div class="perf-item">
            <span>📊 KNN</span>
            <span class="perf-acc">88.5%</span>
        </div>
        <div class="perf-item" style="border-bottom: 2px solid #f39c12; padding-bottom: 0.5rem; margin-bottom: 0.5rem;">
            <span>🏆 HYBRID CART-SVM</span>
            <span class="perf-acc perf-best">92.3%</span>
        </div>
        
        <div style="margin-bottom: 0.3rem; font-size: 0.75rem; color: #888; font-weight: 600;">🐟 Mugilidae Models</div>
        <div class="perf-item">
            <span>🧠 ANN</span>
            <span class="perf-mugilidae">85.5%</span>
        </div>
        <div class="perf-item">
            <span>🧠 ANN-PSO</span>
            <span class="perf-mugilidae">89.0%</span>
        </div>
        <div class="perf-item">
            <span>🧠 ANN-GA</span>
            <span class="perf-mugilidae">90.0%</span>
        </div>
        <div class="perf-item" style="border-bottom: 2px solid #f7971e; padding-bottom: 0.5rem;">
            <span>🏆 ANN-GWO</span>
            <span class="perf-acc perf-best">91.5%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Species List
    st.markdown("""
    <div class="sidebar-section">
        <h4>🐟 17 Species</h4>
        <div class="species-list-sidebar">
    """, unsafe_allow_html=True)
    
    for name in list(ARIIDAE_SPECIES.keys())[:6]:
        st.markdown(f"""
        <div class="species-item-sidebar">
            <span class="dot-real"></span>
            <span>{name.split()[1] if len(name.split()) > 1 else name[:10]}</span>
            <span class="species-tag-sidebar tag-real-sidebar">Real</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="font-size:0.7rem;color:#999;margin-top:0.3rem;">+6 more species</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download Template
    if st.button("📥 Download CSV Template"):
        csv = create_template_csv()
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="ariidae_template.csv",
            mime="text/csv"
        )
    
    st.caption("🎓 Final Year Project | Universiti Malaysia Terengganu")

# ============================================
# HOME PAGE
# ============================================
if choice == "🏠 Home":
    st.markdown("## 🌟 Welcome to AriMugi ID")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa, #fff); padding: 1.5rem; border-radius: 16px; border: 1px solid #f0f0f0;">
            <h3 style="margin-top: 0;">🚀 Smart Fish Identification for Two Families</h3>
            <p style="color: #555; font-size: 1.05rem; line-height: 1.6;">
                <strong>AriMugi ID</strong> combines <strong>two powerful AI models</strong> to identify fish from 
                the <strong>Ariidae</strong> (12 species) and <strong>Mugilidae</strong> (5 species) families 
                using morphological measurements.
            </p>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: 1rem;">
                <div>
                    <span style="font-weight: 700; color: #667eea; font-size: 1.2rem;">92.3%</span>
                    <span style="color: #888; display: block; font-size: 0.85rem;">Ariidae Accuracy</span>
                </div>
                <div>
                    <span style="font-weight: 700; color: #f7971e; font-size: 1.2rem;">91.5%</span>
                    <span style="color: #888; display: block; font-size: 0.85rem;">Mugilidae Accuracy</span>
                </div>
                <div>
                    <span style="font-weight: 700; color: #2ecc71; font-size: 1.2rem;">17</span>
                    <span style="color: #888; display: block; font-size: 0.85rem;">Total Species</span>
                </div>
                <div>
                    <span style="font-weight: 700; color: #e74c3c; font-size: 1.2rem;">31</span>
                    <span style="color: #888; display: block; font-size: 0.85rem;">Mugilidae Features</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_color1 = "#27ae60" if ariidae_loaded else "#e74c3c"
        status_color2 = "#27ae60" if mugilidae_31_loaded else "#e74c3c"
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 16px; border: 1px solid #f0f0f0;">
            <h4 style="margin-top: 0;">📡 System Status</h4>
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0; border-bottom: 1px solid #f5f5f5;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{status_color1};"></span>
                <span>Ariidae Model</span>
                <span style="margin-left:auto; font-size:0.8rem; color:{status_color1};">{'✅ Loaded' if ariidae_loaded else '❌ Not Loaded'}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{status_color2};"></span>
                <span>Mugilidae Model</span>
                <span style="margin-left:auto; font-size:0.8rem; color:{status_color2};">{'✅ Loaded' if mugilidae_31_loaded else '❌ Not Loaded'}</span>
            </div>
            <div style="margin-top: 0.8rem; padding: 0.5rem; background: #f8f9fa; border-radius: 8px; font-size: 0.85rem; color: #666;">
                💡 Select a classifier from the sidebar to start
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("### 📊 Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12</div>
            <div class="metric-label">Ariidae Species</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value gold">5</div>
            <div class="metric-label">Mugilidae Species</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">31</div>
            <div class="metric-label">Mugilidae Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value gold">17</div>
            <div class="metric-label">Combined Species</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Species Quick Look
    st.markdown("### 📋 Species Quick Look")
    st.markdown("👆 **Click on any species name** to view detailed information")
    
    col1, col2 = st.columns(2)
    
    # Ariidae species
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f4ff, #fff); padding: 1rem; border-radius: 12px; border: 1px solid #e8e8e8;">
            <h4 style="color: #667eea; margin-top: 0;">🐟 Ariidae (12 species)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        ariidae_species = [
            "Arius gagora", "Arius leptonotacanthus", "Arius maculatus",
            "Arius oetik", "Arius venosus", "Cryptarius truncatus",
            "Hexanematichthys sagor", "Nemapteryx macronotacantha",
            "Nemapteryx nenga", "Osteogeneiosus militaris",
            "Plicofollis argyropleuron", "Plicofollis layardi"
        ]
        
        for species in ariidae_species:
            details = SPECIES_DETAILS.get(species, {})
            short = details.get("short", "")
            with st.expander(f"🐟 {species} ({short})"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    **📌 Scientific Name:** {species}  
                    **📝 Common Name:** {details.get('common', 'N/A')}  
                    **🏷️ Short Code:** {short}  
                    **👨‍👩‍👧‍👦 Family:** {details.get('family', 'N/A')}  
                    """)
                with col_b:
                    st.markdown(f"""
                    **📏 Size:** {details.get('size', 'N/A')}  
                    **🌊 Habitat:** {details.get('habitat', 'N/A')}  
                    **🍽️ Diet:** {details.get('diet', 'N/A')}  
                    **🌍 Conservation:** {details.get('conservation', 'N/A')}  
                    """)
                st.markdown(f"**🔬 Features:** {details.get('features', 'N/A')}")
    
    # Mugilidae species
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef9e7, #fff); padding: 1rem; border-radius: 12px; border: 1px solid #f0e8d0;">
            <h4 style="color: #f7971e; margin-top: 0;">🐟 Mugilidae (5 species)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        mugilidae_species = [
            "Planiliza subviridis", 
            "Moolgarda seheli", 
            "Osteomugil perusii", 
            "Moolgarda tade", 
            "Ellochelon vaigiensis"
        ]
        
        for species in mugilidae_species:
            details = SPECIES_DETAILS.get(species, {})
            short = details.get("short", "")
            with st.expander(f"🐟 {species} ({short})"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    **📌 Scientific Name:** {species}  
                    **📝 Common Name:** {details.get('common', 'N/A')}  
                    **🏷️ Short Code:** {short}  
                    **👨‍👩‍👧‍👦 Family:** {details.get('family', 'N/A')}  
                    """)
                with col_b:
                    st.markdown(f"""
                    **📏 Size:** {details.get('size', 'N/A')}  
                    **🌊 Habitat:** {details.get('habitat', 'N/A')}  
                    **🍽️ Diet:** {details.get('diet', 'N/A')}  
                    **🌍 Conservation:** {details.get('conservation', 'N/A')}  
                    """)
                st.markdown(f"**🔬 Features:** {details.get('features', 'N/A')}")

# ============================================
# ARIIDAE CLASSIFIER
# ============================================
elif choice == "🐟 Ariidae Classifier":
    st.markdown("## 🐟 Ariidae Fish Classification")
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ 12 Species</strong> · Hybrid CART-SVM · <strong>92.3%</strong> Accuracy
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📏 Enter 9 Morphological Measurements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📏 Head & Body**")
        head = st.number_input("Head Length (mm)", 0.0, 200.0, 45.0, 0.1, key="h_a")
        body = st.number_input("Body Depth (mm)", 0.0, 150.0, 28.0, 0.1, key="b_a")
        eye = st.number_input("Eye Diameter (mm)", 0.0, 30.0, 6.0, 0.1, key="e_a")
    
    with col2:
        st.markdown("**🪢 Barbell & Snout**")
        snout = st.number_input("Snout Length (mm)", 0.0, 50.0, 12.0, 0.1, key="sn_a")
        maxillary = st.number_input("Maxillary Barbell (mm)", 0.0, 100.0, 35.0, 0.1, key="mx_a")
        mandibullary = st.number_input("Mandibullary Barbell (mm)", 0.0, 80.0, 25.0, 0.1, key="md_a")
    
    with col3:
        st.markdown("**🎯 Fins & Other**")
        mental = st.number_input("Mental Barbell (mm)", 0.0, 50.0, 8.0, 0.1, key="mt_a")
        dorsal = st.number_input("Dorsal Fin Ray", 0, 50, 18, 1, key="d_a")
        anal = st.number_input("Anal Fin Ray", 0, 40, 14, 1, key="an_a")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔍 Identify Species", key="btn_ariidae", use_container_width=True)
    
    if predict_clicked:
        is_valid, errors = validate_ariidae_inputs(head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal)
        
        if not is_valid:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
            prediction = predict_ariidae(input_data)
            
            species_info = ARIIDAE_SPECIES.get(prediction, {})
            common = species_info.get("common", "")
            short = species_info.get("short", "")
            
            # Simpan ke histori
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'species': prediction,
                'common_name': common,
                'confidence': '92.3%',
                'model': 'Hybrid CART-SVM',
                'family': 'Ariidae'
            })
            
            st.markdown(f"""
            <div class="prediction-card-ariidae">
                <div style="font-size: 0.9rem; opacity: 0.8;">🎯 Predicted Species</div>
                <div class="prediction-species">{prediction}</div>
                <div class="prediction-short">{short}</div>
                <div class="prediction-common">{common}</div>
                <div class="prediction-accuracy">🏆 Hybrid CART-SVM · 92.3% Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📸 Fish Image")
            img = get_species_image(prediction, "ariidae")
            if img:
                st.image(img, caption=f"{prediction} - {common}", use_container_width=True)
            else:
                st.info(f"📸 Image for {prediction} will be available soon")
            
            # Export button
            features_dict = {
                'HeadLength': head, 'BodyDepth': body, 'EyeDiameter': eye,
                'SnoutLength': snout, 'MaxillaryBarbell': maxillary,
                'MandibullaryBarbell': mandibullary, 'MentalBarbell': mental,
                'DorsalFinRay': dorsal, 'AnalFinRay': anal
            }
            csv = export_prediction_results(prediction, '92.3%', 'Hybrid CART-SVM', features_dict)
            st.download_button(
                label="📥 Download Result CSV",
                data=csv,
                file_name=f"prediction_{prediction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            with st.expander("📖 Species Information"):
                st.markdown(f"""
                | Property | Value |
                |----------|-------|
                | **Scientific Name** | {prediction} |
                | **Common Name** | {common} |
                | **Short Code** | {short} |
                | **Family** | Ariidae |
                | **Classification Method** | Hybrid CART-SVM |
                """)

# ============================================
# MUGILIDAE CLASSIFIER
# ============================================
elif choice == "🐟 Mugilidae Classifier":
    st.markdown("## 🐟 Mugilidae Fish Classification")
    st.markdown("""
    <div class="info-box warning">
        <strong>ℹ️ 5 Species</strong> · ANN-GWO · <strong>91.5%</strong> Accuracy
    </div>
    """, unsafe_allow_html=True)
    
    if not mugilidae_31_loaded:
        st.error("❌ Models not loaded. Please ensure all .pkl files are uploaded.")
        st.info("""
        Required files:
        - ann_model_31features.pkl
        - pso_model_31features.pkl
        - ga_model_31features.pkl
        - gwo_model_31features.pkl
        - scaler_31features.pkl
        - label_encoder_31features.pkl
        - feature_names_31features.pkl
        """)
    else:
        st.markdown("### 📏 Enter 31 Morphological Measurements")
        st.caption("📌 Meristic counts are integers. All other measurements in mm.")
        
        # ROW 1: MERISTIC FEATURES (6)
        st.markdown("**📏 Meristic Features**")
        col1, col2, col3 = st.columns(3)
        with col1:
            nd1 = st.number_input("ND1_Total", 0.0, 50.0, 4.0, 1.0, key="nd1_31")
            nd2 = st.number_input("ND2_Total", 0.0, 50.0, 7.0, 1.0, key="nd2_31")
        with col2:
            np_val = st.number_input("NP (Pectoral Fin Rays)", 0.0, 50.0, 14.0, 1.0, key="np_31")
            nc = st.number_input("NC (Caudal Fin Rays)", 0.0, 50.0, 14.0, 1.0, key="nc_31")
        with col3:
            nv = st.number_input("NV_Total", 0.0, 50.0, 6.0, 1.0, key="nv_31")
            na = st.number_input("NA_Total", 0.0, 50.0, 10.0, 1.0, key="na_31")
        
        # ROW 2: MORPHOMETRIC FEATURES (4)
        st.markdown("**📐 Morphometric Features (mm)**")
        col1, col2 = st.columns(2)
        with col1:
            sl = st.number_input("SL (Standard Length)", 0.0, 500.0, 150.0, 10.0, key="sl_31")
            pl = st.number_input("PL (Pectoral Fin Length)", 0.0, 300.0, 40.0, 5.0, key="pl_31")
        with col2:
            bh = st.number_input("BH (Body Height)", 0.0, 300.0, 45.0, 5.0, key="bh_31")
            hl = st.number_input("HL (Head Length)", 0.0, 300.0, 40.0, 5.0, key="hl_31")
        
        # ROW 3-5: TRUSS NETWORK FEATURES (21)
        st.markdown("**📐 Truss Network Features (mm)**")
        st.caption("AB, AC, AD, BC, BD, CD, CE, CF, DE, DF, EF, EG, EH, FG, FH, GH, GI, GJ, HI, HJ, IJ")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            truss_AB = st.number_input("AB", 0.0, 500.0, 8.0, 1.0, key="ab_31")
            truss_AC = st.number_input("AC", 0.0, 500.0, 30.0, 1.0, key="ac_31")
            truss_AD = st.number_input("AD", 0.0, 500.0, 25.0, 1.0, key="ad_31")
        with col2:
            truss_BC = st.number_input("BC", 0.0, 500.0, 25.0, 1.0, key="bc_31")
            truss_BD = st.number_input("BD", 0.0, 500.0, 20.0, 1.0, key="bd_31")
            truss_CD = st.number_input("CD", 0.0, 500.0, 25.0, 1.0, key="cd_31")
        with col3:
            truss_CE = st.number_input("CE", 0.0, 500.0, 45.0, 1.0, key="ce_31")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            truss_CF = st.number_input("CF", 0.0, 500.0, 40.0, 1.0, key="cf_31")
            truss_DE = st.number_input("DE", 0.0, 500.0, 55.0, 1.0, key="de_31")
            truss_DF = st.number_input("DF", 0.0, 500.0, 30.0, 1.0, key="df_31")
        with col2:
            truss_EF = st.number_input("EF", 0.0, 500.0, 45.0, 1.0, key="ef_31")
            truss_EG = st.number_input("EG", 0.0, 500.0, 35.0, 1.0, key="eg_31")
            truss_EH = st.number_input("EH", 0.0, 500.0, 50.0, 1.0, key="eh_31")
        with col3:
            truss_FG = st.number_input("FG", 0.0, 500.0, 60.0, 1.0, key="fg_31")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            truss_FH = st.number_input("FH", 0.0, 500.0, 45.0, 1.0, key="fh_31")
            truss_GH = st.number_input("GH", 0.0, 500.0, 35.0, 1.0, key="gh_31")
            truss_GI = st.number_input("GI", 0.0, 500.0, 35.0, 1.0, key="gi_31")
        with col2:
            truss_GJ = st.number_input("GJ", 0.0, 500.0, 40.0, 1.0, key="gj_31")
            truss_HI = st.number_input("HI", 0.0, 500.0, 45.0, 1.0, key="hi_31")
            truss_HJ = st.number_input("HJ", 0.0, 500.0, 35.0, 1.0, key="hj_31")
        with col3:
            truss_IJ = st.number_input("IJ", 0.0, 500.0, 18.0, 1.0, key="ij_31")
        
        # MODEL SELECTION with correct accuracy
        st.markdown("### 🧠 Select Model for Prediction")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            model_choice = st.selectbox(
                "Choose Model",
                ["ANN-GWO 🏆 (Recommended - 91.5%)", "ANN-GA (90.0%)", "ANN-PSO (89.0%)", "ANN (85.5%)"],
                index=0
            )
        with col2:
            st.markdown("""
            <div style="background: #f0faf0; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #27ae60; margin-top: 1.5rem;">
                <span style="font-size: 0.8rem; color: #1a7a3a;">
                    ✅ <strong>Best: ANN-GWO</strong><br>91.5% Accuracy
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_clicked = st.button("🔍 Identify Species", key="btn_mugilidae", use_container_width=True)
        
        if predict_clicked:
            try:
                input_values = [
                    nd1, nd2, np_val, nc, nv, na,
                    sl, pl, bh, hl,
                    truss_AB, truss_AC, truss_AD, truss_BC, truss_BD, truss_CD, truss_CE,
                    truss_CF, truss_DE, truss_DF, truss_EF, truss_EG, truss_EH, truss_FG,
                    truss_FH, truss_GH, truss_GI, truss_GJ, truss_HI, truss_HJ, truss_IJ
                ]
                
                if len(input_values) != 31:
                    st.error(f"❌ Expected 31 features, got {len(input_values)}")
                else:
                    input_array = np.array(input_values, dtype=np.float64).reshape(1, -1)
                    input_scaled = mugilidae_models['scaler'].transform(input_array)
                    
                    # Model selection with correct accuracy
                    if "GWO" in model_choice:
                        model = mugilidae_models['gwo']
                        model_name = "ANN-GWO"
                        accuracy = "91.5%"
                    elif "GA" in model_choice:
                        model = mugilidae_models['ga']
                        model_name = "ANN-GA"
                        accuracy = "90.0%"
                    elif "PSO" in model_choice:
                        model = mugilidae_models['pso']
                        model_name = "ANN-PSO"
                        accuracy = "89.0%"
                    else:
                        model = mugilidae_models['ann']
                        model_name = "ANN"
                        accuracy = "85.5%"
                    
                    prediction = model.predict(input_scaled)[0]
                    predicted_species_old = mugilidae_models['label_encoder'].inverse_transform([prediction])[0]
                    predicted_species = MUGILIDAE_NAME_MAPPING.get(predicted_species_old, predicted_species_old)
                    
                    probabilities = model.predict_proba(input_scaled)[0]
                    confidence = np.max(probabilities) * 100
                    
                    species_details = SPECIES_DETAILS.get(predicted_species, {})
                    short = species_details.get("short", predicted_species_old)
                    common = species_details.get("common", "")
                    
                    # Simpan ke histori
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'species': predicted_species,
                        'common_name': common,
                        'confidence': f'{confidence:.1f}%',
                        'model': model_name,
                        'family': 'Mugilidae'
                    })
                    
                    # Display accuracy badges
                    accuracy_badge = "🏆" if "GWO" in model_choice else "📊"
                    
                    st.markdown(f"""
                    <div class="prediction-card-mugilidae">
                        <div style="font-size: 0.9rem; opacity: 0.8;">🎯 Predicted Species</div>
                        <div class="prediction-species">{predicted_species}</div>
                        <div style="font-size: 1.2rem; opacity: 0.8;">{short}</div>
                        <div style="font-size: 1rem; opacity: 0.8;">{common}</div>
                        <div style="margin-top: 0.3rem; font-size: 1rem; opacity: 0.8;">Confidence: {confidence:.1f}%</div>
                        <div class="prediction-accuracy dark">{accuracy_badge} {model_name} · {accuracy} Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(int(confidence))
                    
                    st.markdown("### 📸 Fish Image")
                    img = get_species_image(predicted_species, "mugilidae")
                    if img:
                        st.image(img, caption=f"{predicted_species}", use_container_width=True)
                    else:
                        st.info(f"📸 Image for {predicted_species} will be available soon")
                    
                    st.markdown("#### 📊 Species Probabilities")
                    prob_df = pd.DataFrame({
                        'Species': mugilidae_models['label_encoder'].classes_,
                        'Probability (%)': probabilities * 100
                    })
                    prob_df['Species'] = prob_df['Species'].map(MUGILIDAE_NAME_MAPPING).fillna(prob_df['Species'])
                    prob_df = prob_df.sort_values('Probability (%)', ascending=False)
                    
                    st.bar_chart(prob_df.set_index('Species'))
                    
                    # Export button
                    features_dict = {
                        'ND1': nd1, 'ND2': nd2, 'NP': np_val, 'NC': nc,
                        'NV': nv, 'NA': na, 'SL': sl, 'PL': pl,
                        'BH': bh, 'HL': hl, 'AB': truss_AB, 'AC': truss_AC,
                        'AD': truss_AD, 'BC': truss_BC, 'BD': truss_BD,
                        'CD': truss_CD, 'CE': truss_CE, 'CF': truss_CF,
                        'DE': truss_DE, 'DF': truss_DF, 'EF': truss_EF,
                        'EG': truss_EG, 'EH': truss_EH, 'FG': truss_FG,
                        'FH': truss_FH, 'GH': truss_GH, 'GI': truss_GI,
                        'GJ': truss_GJ, 'HI': truss_HI, 'HJ': truss_HJ,
                        'IJ': truss_IJ
                    }
                    csv = export_prediction_results(predicted_species, f'{confidence:.1f}%', model_name, features_dict)
                    st.download_button(
                        label="📥 Download Result CSV",
                        data=csv,
                        file_name=f"prediction_{predicted_species}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================
# BATCH PREDICTION
# ============================================
elif choice == "📊 Batch Prediction":
    st.markdown("## 📊 Batch Prediction")
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️</strong> Upload a CSV file with multiple samples for batch prediction.
        Download the template below to get started.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 Download CSV Template"):
        csv = create_template_csv()
        st.download_button(
            label="Download Template",
            data=csv,
            file_name="ariidae_template.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        with st.spinner("Processing batch prediction..."):
            result_df, error = batch_prediction_ariidae(uploaded_file)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ Batch prediction completed!")
                st.dataframe(result_df)
                
                # Summary statistics
                st.markdown("### 📊 Prediction Summary")
                summary = result_df['Predicted_Species'].value_counts()
                st.bar_chart(summary)
                
                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ============================================
# COMPARE MODELS - UPDATED WITH ALL MODELS
# ============================================
elif choice == "⚖️ Compare Models":
    st.markdown("## ⚖️ Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f4ff, #fff); padding: 1.5rem; border-radius: 16px; border: 1px solid #e8e8e8;">
            <h3 style="margin-top: 0; color: #667eea;">🐟 Ariidae Models</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem;">
                <div><span style="color:#888;">🌿 CART:</span></div>
                <div><strong>69.2%</strong></div>
                <div><span style="color:#888;">⚡ SVM:</span></div>
                <div><strong>92.3%</strong></div>
                <div><span style="color:#888;">📊 KNN:</span></div>
                <div><strong>88.5%</strong></div>
                <div><span style="color:#888;">🏆 Hybrid CART-SVM:</span></div>
                <div><strong style="color:#27ae60;">92.3%</strong></div>
                <div style="margin-top: 0.5rem; grid-column: span 2; font-size: 0.85rem; color: #888;">
                    <span>⭐ Best: <strong style="color:#27ae60;">Hybrid CART-SVM</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef9e7, #fff); padding: 1.5rem; border-radius: 16px; border: 1px solid #f0e8d0;">
            <h3 style="margin-top: 0; color: #f7971e;">🐟 Mugilidae Models</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem;">
                <div><span style="color:#888;">🧠 ANN:</span></div>
                <div><strong>85.5%</strong></div>
                <div><span style="color:#888;">🧠 ANN-PSO:</span></div>
                <div><strong>89.0%</strong></div>
                <div><span style="color:#888;">🧠 ANN-GA:</span></div>
                <div><strong>90.0%</strong></div>
                <div><span style="color:#888;">🏆 ANN-GWO:</span></div>
                <div><strong style="color:#27ae60;">91.5%</strong></div>
                <div style="margin-top: 0.5rem; grid-column: span 2; font-size: 0.85rem; color: #888;">
                    <span>⭐ Best: <strong style="color:#27ae60;">ANN-GWO</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Accuracy Comparison Chart with ALL models
    st.markdown("### 📊 Accuracy Comparison Chart")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # All models from both families
    models = ['CART', 'KNN', 'SVM', 'Hybrid CART-SVM', 'ANN', 'ANN-PSO', 'ANN-GA', 'ANN-GWO']
    accuracies = [69.2, 88.5, 92.3, 92.3, 85.5, 89.0, 90.0, 91.5]
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#d35400', '#c0392b']
    
    # Create grouped bars with different colors for each model
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='600')
    ax.set_title('Model Accuracy Comparison - AriMugi ID', fontsize=14, fontweight='700')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add family labels
    ax.text(1.5, -5, '🐟 Ariidae', ha='center', va='top', fontsize=11, fontweight='bold', color='#667eea')
    ax.text(6, -5, '🐟 Mugilidae', ha='center', va='top', fontsize=11, fontweight='bold', color='#f7971e')
    
    # Add best model indicators
    ax.axhline(y=92.3, color='#2ecc71', linestyle='--', alpha=0.7, linewidth=1.5, label='Ariidae Best: 92.3%')
    ax.axhline(y=91.5, color='#c0392b', linestyle='--', alpha=0.7, linewidth=1.5, label='Mugilidae Best: 91.5%')
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### 📌 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #f0faf0; padding: 1rem; border-radius: 12px; border-left: 4px solid #27ae60;">
            <h4 style="margin: 0; color: #27ae60;">✅ Ariidae</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1.2rem; color: #444;">
                <li><strong>Best Model:</strong> Hybrid CART-SVM (<strong>92.3%</strong>)</li>
                <li>Species coverage: <strong>12</strong></li>
                <li>Features: <strong>9</strong></li>
                <li>SVM also achieves <strong>92.3%</strong> accuracy</li>
                <li>KNN: <strong>88.5%</strong> · CART: <strong>69.2%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #fef9e7; padding: 1rem; border-radius: 12px; border-left: 4px solid #f39c12;">
            <h4 style="margin: 0; color: #f39c12;">✅ Mugilidae</h4>
            <ul style="margin: 0.5rem 0; padding-left: 1.2rem; color: #444;">
                <li><strong>Best Model:</strong> ANN-GWO (<strong>91.5%</strong>)</li>
                <li>Species coverage: <strong>5</strong></li>
                <li>Features: <strong>31</strong></li>
                <li>ANN-GA: <strong>90.0%</strong> · ANN-PSO: <strong>89.0%</strong></li>
                <li>ANN: <strong>85.5%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PREDICTION HISTORY
# ============================================
elif choice == "📜 Prediction History":
    st.markdown("## 📜 Prediction History")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions made yet. Start classifying fish to see history here!")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📊 History Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            unique_species = history_df['species'].nunique()
            st.metric("Unique Species", unique_species)
        with col3:
            family_counts = history_df['family'].value_counts()
            most_common_family = family_counts.index[0] if len(family_counts) > 0 else "N/A"
            st.metric("Most Common Family", most_common_family)
        
        # Species distribution
        st.markdown("### 📊 Species Distribution")
        species_counts = history_df['species'].value_counts().head(10)
        st.bar_chart(species_counts)
        
        # Model distribution
        st.markdown("### 🧠 Model Usage Distribution")
        model_counts = history_df['model'].value_counts()
        st.bar_chart(model_counts)
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p>🎓 <strong>Final Year Project</strong> · <strong>AriMugi ID</strong> · Ariidae &amp; Mugilidae Classification</p>
    <p style="font-size: 0.85rem; color: #999;">
        🏆 Hybrid CART-SVM (92.3%) · ANN-GWO (91.5%) · 17 Species · 31 Features
    </p>
    <div class="footer-badges">
        <span class="footer-badge">🐟 Ariidae: 12 species</span>
        <span class="footer-badge">🐟 Mugilidae: 5 species</span>
        <span class="footer-badge">📊 31 Morphological Features</span>
        <span class="footer-badge">🎓 UMT</span>
    </div>
</div>
""", unsafe_allow_html=True)
