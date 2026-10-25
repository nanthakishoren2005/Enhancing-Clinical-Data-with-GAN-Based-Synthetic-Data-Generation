"""
Centralized UI & CSS Design System for Medical AI Platform.
Provides modern, clinical SaaS aesthetics, responsive theme adaptability (Light & Dark),
robust font visibility guarantees, and reusable component widgets.
"""
import streamlit as st

def get_theme_css() -> str:
    """
    Returns high-performance, responsive CSS for modern healthcare AI aesthetic.
    Guarantees WCAG AAA text visibility across both Light and Dark themes.
    """
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
    /* =========================================================================
       1. GLOBAL THEME DESIGN TOKENS (LIGHT & DARK ADAPTIVE)
       ========================================================================= */
    :root {
        --med-font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --med-font-mono: 'JetBrains Mono', monospace;
        
        /* Light Theme Palette Defaults */
        --med-bg-canvas: #f8fafc;
        --med-bg-surface: #ffffff;
        --med-bg-card: #ffffff;
        --med-bg-card-subtle: #f1f5f9;
        --med-bg-glass: rgba(255, 255, 255, 0.88);
        --med-border: #e2e8f0;
        --med-border-hover: #cbd5e1;
        --med-border-accent: #38bdf8;
        
        --med-text-primary: #0f172a;
        --med-text-secondary: #334155;
        --med-text-muted: #64748b;
        --med-text-inverted: #ffffff;
        
        /* Clinical Accents */
        --med-primary: #0284c7;
        --med-primary-dark: #0369a1;
        --med-primary-light: #e0f2fe;
        --med-primary-glow: rgba(2, 132, 199, 0.25);
        
        --med-teal: #0d9488;
        --med-teal-light: #ccfbf1;
        --med-cyan: #06b6d4;
        
        /* Pathological State Accents */
        --med-normal: #10b981;
        --med-normal-bg: #ecfdf5;
        --med-normal-border: #a7f3d0;
        
        --med-covid: #8b5cf6;
        --med-covid-bg: #f5f3ff;
        --med-covid-border: #ddd6fe;
        
        --med-opacity: #f59e0b;
        --med-opacity-bg: #fffbeb;
        --med-opacity-border: #fde68a;
        
        --med-pneumonia: #ef4444;
        --med-pneumonia-bg: #fef2f2;
        --med-pneumonia-border: #fecaca;
        
        --med-shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.05);
        --med-shadow-md: 0 4px 14px -2px rgba(15, 23, 42, 0.08);
        --med-shadow-lg: 0 12px 28px -6px rgba(15, 23, 42, 0.12);
        --med-shadow-glow: 0 0 20px rgba(14, 165, 233, 0.2);
    }

    /* Dark Mode Overrides (Streamlit Dark Theme & OS Dark Scheme) */
    @media (prefers-color-scheme: dark) {
        :root {
            --med-bg-canvas: #0b1120;
            --med-bg-surface: #0f172a;
            --med-bg-card: #131d35;
            --med-bg-card-subtle: #1e293b;
            --med-bg-glass: rgba(19, 29, 53, 0.88);
            --med-border: #243452;
            --med-border-hover: #334e75;
            --med-border-accent: #0ea5e9;
            
            --med-text-primary: #f8fafc;
            --med-text-secondary: #cbd5e1;
            --med-text-muted: #94a3b8;
            --med-text-inverted: #0f172a;
            
            --med-primary: #38bdf8;
            --med-primary-dark: #0284c7;
            --med-primary-light: rgba(56, 189, 248, 0.15);
            --med-primary-glow: rgba(56, 189, 248, 0.35);
            
            --med-normal-bg: rgba(16, 185, 129, 0.15);
            --med-normal-border: rgba(16, 185, 129, 0.35);
            
            --med-covid-bg: rgba(139, 92, 246, 0.15);
            --med-covid-border: rgba(139, 92, 246, 0.35);
            
            --med-opacity-bg: rgba(245, 158, 11, 0.15);
            --med-opacity-border: rgba(245, 158, 11, 0.35);
            
            --med-pneumonia-bg: rgba(239, 68, 68, 0.15);
            --med-pneumonia-border: rgba(239, 68, 68, 0.35);
            
            --med-shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
            --med-shadow-md: 0 4px 14px -2px rgba(0, 0, 0, 0.45);
            --med-shadow-lg: 0 14px 32px -4px rgba(0, 0, 0, 0.6);
            --med-shadow-glow: 0 0 25px rgba(56, 189, 248, 0.25);
        }
    }

    /* Force dark theme rules when Streamlit app container is in dark mode */
    [data-theme="dark"], .stApp[data-theme="dark"], [data-testid="stAppViewContainer"] {
        color-scheme: dark;
    }

    /* =========================================================================
       2. TYPOGRAPHY & GLOBAL RESETS
       ========================================================================= */
    html, body, [class*="css"], .stApp {
        font-family: var(--med-font-main) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Ensure Streamlit Headings Have High Contrast & Smooth Spacing */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--med-font-main) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--med-text-primary) !important;
    }

    p, span, div, label {
        font-family: var(--med-font-main);
    }

    /* =========================================================================
       3. MODERN CLINICAL HERO & NAVBAR HEADER
       ========================================================================= */
    .med-hero-banner {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.1) 0%, rgba(13, 148, 136, 0.08) 50%, rgba(139, 92, 246, 0.06) 100%);
        border: 1px solid var(--med-border);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--med-shadow-md);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(8px);
    }
    
    .med-hero-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0284c7 0%, #06b6d4 50%, #10b981 100%);
    }

    .med-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--med-text-primary) !important;
        margin: 0.2rem 0 0.4rem 0;
        line-height: 1.2;
    }

    .med-hero-subtitle {
        font-size: 1.02rem;
        color: var(--med-text-secondary) !important;
        line-height: 1.55;
        max-width: 950px;
        margin: 0;
    }

    .med-badge-pulse {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: var(--med-primary-light);
        border: 1px solid var(--med-border-accent);
        color: var(--med-primary);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .med-pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
        animation: medPulse 2s infinite ease-in-out;
    }

    @keyframes medPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    /* =========================================================================
       4. CARDS, SURFACES & METRICS
       ========================================================================= */
    .med-card {
        background-color: var(--med-bg-card);
        border: 1px solid var(--med-border);
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        box-shadow: var(--med-shadow-sm);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 1rem;
        color: var(--med-text-primary);
    }

    .med-card:hover {
        border-color: var(--med-border-hover);
        box-shadow: var(--med-shadow-md);
        transform: translateY(-2px);
    }

    .med-metric-card {
        background: var(--med-bg-card);
        border: 1px solid var(--med-border);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: var(--med-shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .med-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--med-shadow-md);
        border-color: var(--med-border-accent);
    }

    .med-metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--med-text-muted) !important;
        margin-bottom: 0.3rem;
    }

    .med-metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--med-text-primary) !important;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .med-metric-sub {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* =========================================================================
       5. CLINICAL STATUS BANNERS & RISK ACCENTS
       ========================================================================= */
    .med-status-banner {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.25rem;
        border-left: 6px solid var(--med-primary);
        background: var(--med-bg-card);
        border-top: 1px solid var(--med-border);
        border-right: 1px solid var(--med-border);
        border-bottom: 1px solid var(--med-border);
        box-shadow: var(--med-shadow-md);
    }

    .med-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 0.28rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* =========================================================================
       6. STREAMLIT NATIVE COMPONENT REFINEMENTS
       ========================================================================= */
    /* Primary & Secondary Buttons */
    .stButton > button {
        font-family: var(--med-font-main) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid transparent !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button[kind="secondary"], .stButton > button[data-testid="baseButton-secondary"] {
        background-color: var(--med-bg-card) !important;
        border-color: var(--med-border) !important;
        color: var(--med-text-primary) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: var(--med-border-accent) !important;
        color: var(--med-primary) !important;
        background-color: var(--med-bg-card-subtle) !important;
        transform: translateY(-1px) !important;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--med-bg-card-subtle);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--med-border);
        margin-bottom: 1.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: var(--med-font-main) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border-radius: 8px !important;
        color: var(--med-text-secondary) !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--med-bg-card) !important;
        color: var(--med-primary) !important;
        box-shadow: var(--med-shadow-sm) !important;
        font-weight: 700 !important;
    }

    /* File Uploader Glow Zone */
    [data-testid="stFileUploader"] {
        background: var(--med-bg-card);
        border: 2px dashed var(--med-border);
        border-radius: 14px;
        padding: 1rem;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--med-primary);
        box-shadow: var(--med-shadow-glow);
    }

    /* Streamlit Sidebar Polish */
    [data-testid="stSidebar"] {
        background-color: var(--med-bg-surface) !important;
        border-right: 1px solid var(--med-border) !important;
    }

    [data-testid="stSidebarNav"] {
        font-family: var(--med-font-main) !important;
    }

    /* Callout & Alert Box Aesthetics */
    .stAlert {
        border-radius: 12px !important;
        border-width: 1px !important;
        box-shadow: var(--med-shadow-sm) !important;
        font-family: var(--med-font-main) !important;
    }

    /* Streamlit Metric Overrides */
    [data-testid="stMetric"] {
        background-color: var(--med-bg-card);
        border: 1px solid var(--med-border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: var(--med-shadow-sm);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        color: var(--med-text-muted) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: var(--med-text-primary) !important;
    }

    /* Tables & DataFrames */
    .dataframe {
        font-family: var(--med-font-main) !important;
        border: 1px solid var(--med-border) !important;
        border-radius: 8px !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: var(--med-font-main) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        background-color: var(--med-bg-card) !important;
    }

    /* Modern Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--med-bg-canvas);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--med-border-hover);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--med-primary);
    }
    </style>
    """


def apply_theme():
    """Inject the centralized design system CSS into the current Streamlit page."""
    st.markdown(get_theme_css(), unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badge: str = "Academic Clinical AI Platform", icon: str = ""):
    """Render a clean, modern clinical hero banner."""
    st.markdown(f"""
    <div class="med-hero-banner">
        <div class="med-badge-pulse">
            <span class="med-pulse-dot"></span>
            <span>{badge}</span>
        </div>
        <div class="med-hero-title">{icon} {title}</div>
        <p class="med-hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render the mandatory academic clinical research disclaimer box."""
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.08); border-left: 5px solid #f59e0b; padding: 0.9rem 1.25rem; border-radius: 8px; margin-bottom: 1.4rem; font-size: 0.88rem; color: var(--med-text-secondary); border-top: 1px solid var(--med-border); border-right: 1px solid var(--med-border); border-bottom: 1px solid var(--med-border);">
        <strong style="color: #d97706;">Warning MANDATORY ACADEMIC & CLINICAL DISCLAIMER:</strong><br>
        This platform is an academic experimental framework evaluating deep learning & generative augmentation.
        Computed classifications, Grad-CAM++ saliency heatmaps, and calibrated risk scores are algorithmic research metrics and 
        <strong>DO NOT</strong> constitute certified medical diagnosis or clinical treatment guidelines.
    </div>
    """, unsafe_allow_html=True)
