"""
Phishing Detection System - Streamlit Web Interface
A user-friendly web app to detect phishing, malware, defacement, and benign URLs
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the UI directory to the path
ui_dir = Path(__file__).parent
sys.path.insert(0, str(ui_dir))

from model_utils import PhishingDetector
from explainer import URLExplainer


# Page configuration
st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .benign-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .phishing-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .malware-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .defacement-box {
        background-color: #ffe5cc;
        border: 2px solid #fd7e14;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the model and vectorizer (cached for performance)"""
    # Define paths - Get the project root (go up from UI folder to PhisingDetectionSystem)
    ui_folder = Path(__file__).parent.resolve()  # UI folder
    project_root = ui_folder.parent  # PhisingDetectionSystem folder
    
    # Model path - Updated to 2.2M balanced dataset with Tranco URLs
    model_path = project_root / "results_2mil238k_dataset" / "best_model.keras"
    
    # Character mapping path (for character-level encoding used in training)
    char_mapping_path = project_root / "results_2mil238k_dataset" / "char_to_idx.pkl"
    
    # Check if files exist
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("Please check the model path in app.py")
        return None
    
    if not char_mapping_path.exists():
        st.error(f"❌ Character mapping file not found at: {char_mapping_path}")
        st.warning("""
        **You need to create the character mapping first!**
        
        Run this in the UI folder:
        ```bash
        python create_char_mapping.py
        ```
        """)
        return None
    
    try:
        detector = PhishingDetector(str(model_path), str(char_mapping_path))
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Phishing Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze URLs for phishing, malware, defacement, and other threats")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This system uses a **Neural Network** trained on 1.5M+ URLs to detect:
        - 🟢 **Benign** - Safe URLs
        - 🔴 **Phishing** - Credential theft attempts
        - 🟡 **Malware** - Malicious software distribution
        - 🟠 **Defacement** - Compromised websites
        
        **Model Accuracy:** 97.0%
        """)
        
        st.header("📊 Model Info")
        detector = load_model()
        if detector:
            info = detector.get_model_info()
            st.write(f"**Classes:** {', '.join(info['class_names'])}")
            st.write(f"**Parameters:** {info['total_params']:,}")
    
    # Main content
    detector = load_model()
    
    if detector is None:
        st.error("⚠️ Could not load the model. Please check the error messages above.")
        return
    
    # URL Input
    st.markdown("---")
    url_input = st.text_input(
        "🔗 Enter a URL to analyze:",
        placeholder="https://example.com/page",
        help="Enter the complete URL including http:// or https://"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze URL", type="primary", use_container_width=True)
    
    # Process URL when button is clicked
    if analyze_button and url_input:
        with st.spinner("🔄 Analyzing URL..."):
            try:
                # Get prediction
                result = detector.predict(url_input)
                
                # Generate explanation
                explainer = URLExplainer()
                explanation = explainer.generate_explanation(
                    url_input, 
                    result['prediction'],
                    result['probabilities']
                )
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Prediction box with styling
                prediction = result['prediction']
                confidence = result['confidence'] * 100
                
                box_class = f"{prediction}-box"
                prediction_emoji = {
                    'benign': '✅',
                    'phishing': '🎣',
                    'malware': '🦠',
                    'defacement': '⚠️'
                }
                
                st.markdown(
                    f'<div class="prediction-box {box_class}">'
                    f'<h2>{prediction_emoji.get(prediction, "🔍")} Prediction: {prediction.upper()}</h2>'
                    f'<h3>Confidence: {confidence:.2f}%</h3>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Probability bars
                st.subheader("📈 Class Probabilities")
                probabilities = result['probabilities']
                
                # Sort by probability (highest first)
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                
                for class_name, prob in sorted_probs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Color based on class
                        color_map = {
                            'benign': '#28a745',
                            'phishing': '#dc3545',
                            'malware': '#ffc107',
                            'defacement': '#fd7e14'
                        }
                        color = color_map.get(class_name, '#6c757d')
                        
                        st.markdown(f"**{class_name.capitalize()}**")
                        st.progress(prob)
                    with col2:
                        st.markdown(f"**{prob*100:.2f}%**")
                
                # Explanation section
                st.markdown("---")
                st.header("🔍 Detailed Explanation")
                
                # Primary factors
                if explanation['primary_factors']:
                    st.subheader("📌 Primary Analysis")
                    for factor in explanation['primary_factors']:
                        st.write(factor)
                
                # Risk indicators
                if explanation['risk_indicators']:
                    st.subheader("⚠️ Risk Indicators Detected")
                    for risk in explanation['risk_indicators']:
                        st.markdown(f"- {risk}")
                
                # Safe indicators
                if explanation['safe_indicators']:
                    st.subheader("✅ Safety Indicators")
                    for safe in explanation['safe_indicators']:
                        st.markdown(f"- {safe}")
                
                # Technical details
                if explanation['technical_details']:
                    st.subheader("🔧 Technical Details")
                    for key, value in explanation['technical_details'].items():
                        st.write(f"**{key}:** {value}")
                
                # Confidence explanation
                if explanation['confidence_explanation']:
                    st.info(f"**Confidence Assessment:** {explanation['confidence_explanation']}")
                
                # Warning/Recommendation
                st.markdown("---")
                if prediction in ['phishing', 'malware']:
                    st.error(f"""
                    ### ⚠️ DANGER - Do Not Visit This URL!
                    This URL has been classified as **{prediction.upper()}** and may:
                    - Steal your personal information or credentials
                    - Install malicious software on your device
                    - Compromise your security
                    
                    **Recommended Action:** Avoid this URL and report it to appropriate authorities.
                    """)
                elif prediction == 'defacement':
                    st.warning("""
                    ### ⚠️ WARNING - Compromised Website
                    This appears to be a defaced/compromised website. Exercise extreme caution.
                    """)
                else:
                    st.success("""
                    ### ✅ This URL appears safe
                    However, always exercise caution when browsing the internet:
                    - Verify the domain matches the expected website
                    - Check for HTTPS encryption
                    - Be careful with personal information
                    """)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.error("Please check that the URL is valid and try again.")
    
    elif analyze_button and not url_input:
        st.warning("⚠️ Please enter a URL to analyze")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Phishing Detection System | Powered by Neural Networks & TF-IDF | 2025"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
