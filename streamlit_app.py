import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #040d0a;
    color: #e8f5f0;
}

/* Hide Streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #081210 !important;
    border-right: 1px solid #1a3528;
}
[data-testid="stSidebar"] * { color: #e8f5f0 !important; }
[data-testid="stSidebar"] .stMultiSelect > div {
    background: #0d1f1a !important;
    border: 1px solid #2a5540 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0d1f1a !important;
    border: 1px solid #2a5540 !important;
}

/* ── Cards ── */
.medi-card {
    background: #0d1f1a;
    border: 1px solid #1a3528;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color .2s;
}
.medi-card:hover { border-color: #2a5540; }

.top-card {
    background: linear-gradient(135deg, #0d1f1a 0%, #0a1a14 100%);
    border: 1px solid #00b85e;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 0 30px rgba(0,232,122,0.08);
}

/* ── Disease name ── */
.disease-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    color: #e8f5f0;
    margin: 0 0 6px 0;
}
.disease-desc {
    font-size: 13px;
    color: #5a8070;
    line-height: 1.6;
    margin: 0 0 16px 0;
}

/* ── Confidence badge ── */
.conf-badge-gold {
    display: inline-block;
    background: rgba(245,166,35,0.15);
    border: 1px solid rgba(245,166,35,0.4);
    color: #f5a623;
    padding: 4px 14px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    font-weight: 500;
}
.conf-badge-blue {
    display: inline-block;
    background: rgba(84,160,255,0.12);
    border: 1px solid rgba(84,160,255,0.3);
    color: #54a0ff;
    padding: 4px 14px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}
.conf-badge-gray {
    display: inline-block;
    background: rgba(90,128,112,0.15);
    border: 1px solid #1a3528;
    color: #5a8070;
    padding: 4px 14px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
}

/* ── Info blocks ── */
.info-block {
    background: #081210;
    border: 1px solid #1a3528;
    border-radius: 12px;
    padding: 16px;
    height: 100%;
}
.info-block-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a8070;
    margin-bottom: 10px;
}
.info-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 13px;
    color: #e8f5f0;
    margin-bottom: 7px;
    line-height: 1.5;
}
.info-bullet { color: #00e87a; font-weight: 700; flex-shrink: 0; }

/* ── Hero ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 42px;
    line-height: 1.1;
    margin-bottom: 10px;
}
.hero-title span { color: #00e87a; }
.hero-sub {
    font-size: 15px;
    color: #5a8070;
    font-weight: 300;
    margin-bottom: 24px;
    line-height: 1.7;
}

/* ── Stat chips ── */
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.stat-chip {
    background: #0d1f1a;
    border: 1px solid #1a3528;
    border-radius: 10px;
    padding: 12px 20px;
    text-align: center;
    min-width: 100px;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #00e87a;
    display: block;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #5a8070;
    text-transform: uppercase;
    letter-spacing: .06em;
}

/* ── Rank badge ── */
.rank-1 { color: #f5a623; font-family: 'Syne',sans-serif; font-weight:800; font-size:28px; }
.rank-2 { color: #54a0ff; font-family: 'Syne',sans-serif; font-weight:800; font-size:28px; }
.rank-3 { color: #5a8070; font-family: 'Syne',sans-serif; font-weight:800; font-size:28px; }

/* ── Tag pill ── */
.sym-tag {
    display: inline-block;
    background: rgba(0,232,122,0.08);
    border: 1px solid #2a5540;
    color: #00e87a;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    margin: 2px;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,71,87,0.06);
    border: 1px solid rgba(255,71,87,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 12px;
    color: #5a8070;
    line-height: 1.6;
    margin-top: 20px;
}

/* ── Divider ── */
hr { border-color: #1a3528 !important; margin: 20px 0 !important; }

/* ── Progress bar override ── */
.stProgress > div > div > div { background: #00e87a !important; }

/* ── Button ── */
.stButton > button {
    background: #00e87a !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: #00ff88 !important;
    box-shadow: 0 0 24px rgba(0,232,122,0.4) !important;
}

/* Multiselect tags */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(0,232,122,0.15) !important;
    border: 1px solid #2a5540 !important;
    color: #00e87a !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    BASE = os.path.dirname(__file__)
    with open(f'{BASE}/models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open(f'{BASE}/models/gb_model.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open(f'{BASE}/models/symptoms.json') as f:
        symptoms = json.load(f)
    with open(f'{BASE}/models/disease_info.json') as f:
        disease_info = json.load(f)
    return rf, gb, symptoms, disease_info

rf_model, gb_model, ALL_SYMPTOMS, DISEASE_INFO = load_models()

# ─── Prediction Function ─────────────────────────────────────────────────────
def predict_disease(selected_symptoms, model_choice='rf'):
    model = rf_model if model_choice == 'rf' else gb_model
    features = {s: (1 if s in selected_symptoms else 0) for s in ALL_SYMPTOMS}
    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[::-1][:3]
    results = []
    for i in top_indices:
        disease = classes[i]
        conf = round(float(proba[i]) * 100, 1)
        if conf < 0.5:
            continue
        info = DISEASE_INFO.get(disease, {
            "description": "Information not available.",
            "precautions": [], "medications": [], "diet": [], "workout": []
        })
        results.append({
            "disease": disease,
            "confidence": conf,
            **info
        })
    return results

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px'>
        <div style='font-family:Syne,sans-serif;font-weight:800;font-size:22px;'>
            ⚕️ Medi<span style='color:#00e87a'>Predict</span>
        </div>
        <div style='font-size:11px;font-family:DM Mono,monospace;color:#3a6050;margin-top:4px;'>
            ML-POWERED DISEASE ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 Select Algorithm")
    model_choice = st.selectbox(
        "ML Model",
        options=["rf", "gb"],
        format_func=lambda x: "🌲 Random Forest (99.9%)" if x == "rf" else "⚡ Gradient Boosting (99.2%)",
        label_visibility="collapsed"
    )

    st.markdown("### 🩺 Select Symptoms")
    symptom_display = [s.replace('_', ' ').title() for s in ALL_SYMPTOMS]
    symptom_map = {s.replace('_', ' ').title(): s for s in ALL_SYMPTOMS}

    selected_display = st.multiselect(
        "Search and select your symptoms",
        options=symptom_display,
        placeholder="Type to search symptoms...",
        label_visibility="collapsed"
    )
    selected_symptoms = [symptom_map[s] for s in selected_display]

    st.markdown(f"""
    <div style='margin:12px 0;padding:10px 14px;background:#0d1f1a;border:1px solid #1a3528;border-radius:10px;'>
        <span style='font-family:DM Mono,monospace;font-size:12px;color:#5a8070;'>SELECTED</span>
        <span style='font-family:Syne,sans-serif;font-weight:700;font-size:20px;color:#00e87a;margin-left:10px;'>{len(selected_symptoms)}</span>
        <span style='font-size:12px;color:#3a6050;'> symptoms</span>
    </div>
    """, unsafe_allow_html=True)

    analyze_clicked = st.button("🔍 Analyze Symptoms", disabled=len(selected_symptoms) == 0)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <strong style='color:#e8f5f0'>Educational use only.</strong>
        Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:10px;color:#3a6050;'>
        DISEASES: 41 &nbsp;|&nbsp; SYMPTOMS: {len(ALL_SYMPTOMS)}<br>
        MODELS: RF + GB &nbsp;|&nbsp; Flask + Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Symptom-Based<br/><span>Disease Prediction</span></div>
<div class='hero-sub'>Select your symptoms from the sidebar and let our ensemble ML models<br/>analyze patterns to predict potential diseases with full health guidance.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='stat-row'>
    <div class='stat-chip'><span class='stat-num'>99.9%</span><span class='stat-lbl'>Accuracy</span></div>
    <div class='stat-chip'><span class='stat-num'>41</span><span class='stat-lbl'>Diseases</span></div>
    <div class='stat-chip'><span class='stat-num'>137</span><span class='stat-lbl'>Symptoms</span></div>
    <div class='stat-chip'><span class='stat-num'>2</span><span class='stat-lbl'>ML Models</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Results ─────────────────────────────────────────────────────────────────
if analyze_clicked and selected_symptoms:
    with st.spinner("🧬 Running ML inference..."):
        predictions = predict_disease(selected_symptoms, model_choice)

    if not predictions:
        st.warning("No strong predictions found. Try adding more symptoms.")
    else:
        model_name = "Random Forest" if model_choice == "rf" else "Gradient Boosting"

        # Symptoms used
        tags_html = "".join([f"<span class='sym-tag'>{s.replace('_',' ')}</span>" for s in selected_symptoms])
        st.markdown(f"""
        <div class='medi-card' style='margin-bottom:24px;'>
            <div style='font-family:DM Mono,monospace;font-size:11px;color:#5a8070;margin-bottom:8px;'>
                ANALYSIS · {model_name} · {len(selected_symptoms)} symptoms
            </div>
            <div>{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

        rank_colors = ['conf-badge-gold', 'conf-badge-blue', 'conf-badge-gray']
        rank_labels = ['#1', '#2', '#3']
        rank_css    = ['rank-1', 'rank-2', 'rank-3']
        card_css    = ['top-card', 'medi-card', 'medi-card']

        for i, pred in enumerate(predictions):
            with st.container():
                # Header row
                col_rank, col_info, col_conf = st.columns([0.8, 5, 2])
                with col_rank:
                    st.markdown(f"<div class='{rank_css[i]}'>{rank_labels[i]}</div>", unsafe_allow_html=True)
                with col_info:
                    st.markdown(f"""
                    <div class='disease-title'>{pred['disease']}</div>
                    <div class='disease-desc'>{pred['description']}</div>
                    """, unsafe_allow_html=True)
                with col_conf:
                    st.markdown(f"<div class='{rank_colors[i]}'>{pred['confidence']}% confidence</div>", unsafe_allow_html=True)
                    st.progress(int(pred['confidence']))

                # 4 info blocks
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    items = "".join([f"<div class='info-item'><span class='info-bullet'>›</span>{item}</div>" for item in pred['precautions']])
                    st.markdown(f"""
                    <div class='info-block'>
                        <div class='info-block-title'>🛡️ Precautions</div>
                        {items}
                    </div>""", unsafe_allow_html=True)

                with c2:
                    items = "".join([f"<div class='info-item'><span class='info-bullet'>›</span>{item}</div>" for item in pred['medications']])
                    st.markdown(f"""
                    <div class='info-block'>
                        <div class='info-block-title'>💊 Medications</div>
                        {items}
                    </div>""", unsafe_allow_html=True)

                with c3:
                    items = "".join([f"<div class='info-item'><span class='info-bullet'>›</span>{item}</div>" for item in pred['diet']])
                    st.markdown(f"""
                    <div class='info-block'>
                        <div class='info-block-title'>🥗 Diet Plan</div>
                        {items}
                    </div>""", unsafe_allow_html=True)

                with c4:
                    items = "".join([f"<div class='info-item'><span class='info-bullet'>›</span>{item}</div>" for item in pred['workout']])
                    st.markdown(f"""
                    <div class='info-block'>
                        <div class='info-block-title'>🏃 Workout</div>
                        {items}
                    </div>""", unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

elif not analyze_clicked:
    # Empty state
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;'>
        <div style='font-size:64px;margin-bottom:20px;'>🧬</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;font-size:20px;color:#3a6050;margin-bottom:10px;'>
            No Analysis Yet
        </div>
        <div style='font-size:14px;color:#3a6050;line-height:1.7;'>
            Select symptoms from the sidebar on the left<br/>
            and click <strong style='color:#5a8070'>"Analyze Symptoms"</strong> to get predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)
