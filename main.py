import streamlit as st
import requests
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, clean CSS with improved styling
st.markdown("""
    <style>
    /* Clean background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #dbeafe 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Aggressively hide all Streamlit alert/info/warning boxes */
    .stAlert, .stWarning, .stInfo, .stSuccess, .stError {
        display: none !important;
    }
    
    /* Hide empty divs and containers */
    div.element-container:empty {
        display: none !important;
    }
    
    /* Hide stale empty blocks */
    .stMarkdown:empty {
        display: none !important;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Main header with modern design */
    .main-header {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px rgba(5, 150, 105, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: #d1fae5;
        font-size: 1.2rem;
        margin: 0;
        font-weight: 500;
    }
    
    /* Mode selection card */
    .mode-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 2px solid #f0f9ff;
    }
    
    .mode-card h3 {
        color: #059669;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 2px solid #f0f9ff;
    }
    
    .upload-section h3 {
        color: #1f2937;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    /* Result card - more compact */
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border: 2px solid #f0f9ff;
    }
    
    /* Hide any empty containers within result box */
    .result-box > div:empty,
    .result-box .element-container:empty,
    .result-box .stMarkdown:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Plant name with icon */
    .plant-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #059669;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #d1fae5;
    }
    
    /* Status badges - modern design */
    .status {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .status:hover {
        transform: translateY(-2px);
    }
    
    .status.healthy {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
    }
    
    .status.sick {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
    }
    
    .status.info {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        color: #1e40af;
    }
    
    /* Section headers - modern style */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #059669;
        margin: 2rem 0 1rem 0 !important;
        padding: 0.8rem 1.2rem;
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-radius: 12px;
        border-left: 5px solid #059669;
    }
    
    /* Force no white space between sections */
    .section-header + div,
    .section-header + .simple-list,
    .section-header + .info-box {
        margin-top: 0.5rem !important;
    }
    
    /* Info box - cleaner design */
    .info-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .info-box p {
        margin: 0.5rem 0;
        color: #374151;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    
    .info-box strong {
        color: #059669;
    }
    
    /* Simple list - improved spacing */
    .simple-list {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .simple-list ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .simple-list li {
        margin: 1rem 0;
        color: #374151;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    
    .simple-list li strong {
        color: #059669;
    }
    
    /* Button style - modern gradient */
    .stButton > button {
        background: linear-gradient(135deg, #059669, #10b981) !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 0.9rem 2.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #047857, #059669) !important;
        box-shadow: 0 6px 16px rgba(5, 150, 105, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Radio buttons - cleaner */
    .stRadio > div {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .stRadio label {
        font-size: 1.05rem;
        font-weight: 500;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #10b981;
    }
    
    /* Success message */
    .stSuccess {
        background: #d1fae5;
        color: #065f46;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Warning styling */
    .stWarning {
        border-radius: 12px;
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Footer tips */
    .tips-footer {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .tips-footer strong {
        color: #059669;
        font-size: 1.1rem;
    }
    
    .tips-footer p {
        color: #6b7280;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #d1fae5, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🌱 Plant Doctor</h1>
        <p>AI-Powered Plant Health Analysis • Get Instant Care Recommendations</p>
    </div>
""", unsafe_allow_html=True)

# API configuration
api_url = "http://localhost:8000"
fallback_url = "http://leaf-diseases-detect.vercel.app"


def display_complete_diagnosis(result):
    """Display full diagnosis from /diagnose endpoint"""
    
    # Plant name
    plant_name = result.get('plant_name', 'Your Plant')
    
    # Create main result container with plant name inside
    st.markdown(f'''
        <div class="result-box">
            <div class="plant-name">🌿 {plant_name}</div>
    ''', unsafe_allow_html=True)
    
    # Check if pipeline was successful
    if not result.get('pipeline_success', False):
        st.markdown("""
            <div class="info-box" style="background: linear-gradient(135deg, #fef3c7, #fde68a); border-left-color: #f59e0b;">
                <p>⚠️ Analysis completed with limited information</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Plant name
    #plant_name = result.get('plant_name', 'Your Plant')
    #st.markdown(f'<div class="plant-name">🌿 {plant_name}</div>', unsafe_allow_html=True)
    
    # Health status
    health_status = result.get('health_status', 'unknown')
    
    if health_status == 'healthy':
        st.markdown('<div style="text-align: center;"><span class="status healthy">✅ Healthy Plant</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><p>🎉 Excellent! Your plant is in great health. Continue with your current care routine!</p></div>', unsafe_allow_html=True)
    
    elif health_status == 'unhealthy':
        st.markdown('<div style="text-align: center;"><span class="status sick">⚠️ Plant Needs Care</span></div>', unsafe_allow_html=True)
        
        # Disease info
        disease_info = result.get('disease_info', {})
        if disease_info.get('disease_detected'):
            disease_name = disease_info.get('disease_name', 'Unknown issue')
            severity = disease_info.get('severity', 'Unknown')
            
            st.markdown(f'<div class="info-box"><p><strong>Detected Issue:</strong> {disease_name}</p><p><strong>Severity Level:</strong> {severity}</p></div>', unsafe_allow_html=True)
            
            # Symptoms
            symptoms = disease_info.get('symptoms', [])
            if symptoms:
                symptoms_html = '<div class="section-header">🔍 Observed Symptoms</div><div class="simple-list"><ul>'
                for symptom in symptoms[:4]:
                    symptoms_html += f'<li>{symptom}</li>'
                symptoms_html += '</ul></div>'
                st.markdown(symptoms_html, unsafe_allow_html=True)
    
    else:
        st.markdown('<div style="text-align: center;"><span class="status info">ℹ️ Analysis Complete</span></div>', unsafe_allow_html=True)
    
    # Confidence score
    confidence = result.get('confidence', {})
    overall_conf = confidence.get('overall', 0)
    if overall_conf > 0:
        conf_percent = int(overall_conf)
        st.markdown(f'<div style="text-align: center; margin: 1.5rem 0;"><span class="status info">🎯 Analysis Confidence: {conf_percent}%</span></div>', unsafe_allow_html=True)
    
    # Treatment recommendations
    treatments = result.get('treatments', {})
    combined_treatments = treatments.get('combined_treatments', [])
    if combined_treatments:
        treatment_html = '<div class="section-header">💊 Treatment Plan</div><div class="simple-list"><ul>'
        for i, treatment in enumerate(combined_treatments[:6], 1):
            treatment_html += f'<li><strong>Step {i}:</strong> {treatment}</li>'
        treatment_html += '</ul></div>'
        st.markdown(treatment_html, unsafe_allow_html=True)
    
    # Care tips from knowledge base
    kb_advice = result.get('kb_advice', {})
    if kb_advice.get('plant_found_in_kb'):
        general_care = kb_advice.get('general_care', '')
        if general_care:
            st.markdown('<div class="section-header">🌟 General Care Guidelines</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-box"><p>{general_care}</p></div>', unsafe_allow_html=True)
        
        # Prevention tips
        prevention_tips = kb_advice.get('prevention_tips', [])
        if prevention_tips:
            prevention_html = '<div class="section-header">🛡️ Prevention Tips</div><div class="simple-list"><ul>'
            for tip in prevention_tips[:4]:
                prevention_html += f'<li>{tip}</li>'
            prevention_html += '</ul></div>'
            st.markdown(prevention_html, unsafe_allow_html=True)
    
    # Timestamp
    timestamp = result.get('timestamp', '')
    if timestamp:
        st.markdown(f'<div style="text-align: center; color: #9ca3af; font-size: 0.9rem; margin-top: 2rem;">📅 Analysis completed: {timestamp}</div>', unsafe_allow_html=True)
    
    # Close result box
    st.markdown('</div>', unsafe_allow_html=True)


def display_disease_detection(result):
    """Display disease detection results"""
    
    # Create main result container
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if result.get("disease_type") == "invalid_image":
        st.markdown("""
            <div class="info-box" style="background: linear-gradient(135deg, #fef2f2, #fee2e2); border-left-color: #dc2626;">
                <p><strong>⚠️ Invalid Image</strong></p>
                <p>Please upload a clear photo showing plant leaves or affected areas.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if result.get("disease_detected"):
        disease_name = result.get('disease_name', 'Unknown Disease')
        st.markdown(f'<div class="plant-name">🦠 {disease_name}</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><span class="status sick">⚠️ Disease Detected</span></div>', unsafe_allow_html=True)
        
        # Symptoms
        symptoms = result.get("symptoms", [])
        if symptoms:
            symptoms_html = '<div class="section-header">🔍 Identified Symptoms</div><div class="simple-list"><ul>'
            for symptom in symptoms[:3]:
                symptoms_html += f'<li>{symptom}</li>'
            symptoms_html += '</ul></div>'
            st.markdown(symptoms_html, unsafe_allow_html=True)
        
        # Treatment
        treatments = result.get("treatment", [])
        if treatments:
            treatment_html = '<div class="section-header">💊 Recommended Treatment</div><div class="simple-list"><ul>'
            for i, treatment in enumerate(treatments[:5], 1):
                treatment_html += f'<li><strong>Step {i}:</strong> {treatment}</li>'
            treatment_html += '</ul></div>'
            st.markdown(treatment_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="plant-name">🌿 Healthy Plant</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><span class="status healthy">✅ No Issues Found</span></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="info-box">
                <p>🎉 Great news! Your plant appears healthy with no visible diseases detected.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Close result box
    st.markdown('</div>', unsafe_allow_html=True)


# Mode selection
st.markdown('<div class="mode-card">', unsafe_allow_html=True)
st.markdown("<h3>🔬 Select Analysis Mode</h3>", unsafe_allow_html=True)
mode = st.radio(
    "",
    ["🌱 Full Diagnosis (Plant ID + Health Check)", "🔍 Quick Scan (Disease Detection Only)"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("<h3>📸 Upload Plant Photo</h3>", unsafe_allow_html=True)
st.markdown("*For best results, use natural lighting and ensure the plant is clearly visible*")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Set button text and endpoint based on mode
        if "Full Diagnosis" in mode:
            button_text = "🌱 Analyze Plant"
            endpoint = "/diagnose"
            spinner_text = "🔬 Analyzing plant health..."
        else:
            button_text = "🔍 Scan for Diseases"
            endpoint = "/disease-detection-file"
            spinner_text = "🔬 Scanning for diseases..."
        
        if st.button(button_text, use_container_width=True):
            with st.spinner(spinner_text):
                success = False
                result = None
                
                # Try local API first, then fallback
                for url in [api_url, fallback_url]:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{url}{endpoint}", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            success = True
                            if url == api_url:
                                st.markdown("""
                                    <div style="background: #d1fae5; color: #065f46; padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem; font-weight: 600; text-align: center;">
                                        ✅ Connected to local server
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div style="background: #dbeafe; color: #1e40af; padding: 0.8rem; border-radius: 10px; margin-bottom: 1rem; font-weight: 600; text-align: center;">
                                        ☁️ Connected to cloud server
                                    </div>
                                """, unsafe_allow_html=True)
                            break
                            
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                        continue
                    except Exception:
                        continue
                
                # Display results ONLY if successful
                if success and result:
                    
                    st.markdown('</div>', unsafe_allow_html=True)  # Close upload section
                    
                    # Display results without extra wrapper
                    if "Full Diagnosis" in mode:
                        display_complete_diagnosis(result)
                    else:
                        display_disease_detection(result)
                else:
                    st.markdown("""
                        <div style="background: #fee2e2; color: #991b1b; padding: 1rem; border-radius: 10px; margin-top: 1rem; font-weight: 600; text-align: center;">
                            ❌ Unable to connect to server. Please ensure the API is running.
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="background: #dbeafe; color: #1e40af; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600;">
                👆 Upload an image to begin analysis
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Only show upload section closing tag if file wasn't uploaded
if uploaded_file is None:
    pass  # Already closed above
elif uploaded_file is not None and not st.session_state.get('button_clicked'):
    st.markdown('</div>', unsafe_allow_html=True)

# Footer tips
st.markdown("---")
st.markdown("""
    <div class="tips-footer">
        <p><strong>💡 Pro Tips for Accurate Results</strong></p>
        <p>✓ Use natural daylight for photography</p>
        <p>✓ Keep the camera focused on affected areas</p>
        <p>✓ Include both healthy and affected parts when possible</p>
        <p>✓ Avoid shadows and overexposure</p>
    </div>
""", unsafe_allow_html=True)