import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# 1️⃣ PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="BloodSense AI - Fingerprint Blood Group Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================================================================
# 2️⃣ CUSTOM STYLES
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .header-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        text-align: center;
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        text-align: center;
    }
    
    .confidence-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A24 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px 0 rgba(255, 107, 107, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(255, 107, 107, 0.6);
    }
    
    .blood-group-badge {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B, #EE5A24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-meter {
        height: 20px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
            
    /* Remove unwanted top space before sections */
    .block-container {
        padding-top: 1rem !important;
    }

    h3, h4 {
        margin-top: 0.2rem !important;
    }

    .input-section h3:first-child,
    .info-section h3:first-child {
        margin-top: 0 !important;
    }

</style>
""", unsafe_allow_html=True)

# ================================================================
# 3️⃣ MODEL LOADING
# ================================================================
@st.cache_resource
def load_model(model_path="blood_model.pth"):
    """Load the trained ResNet50 model for blood group classification."""
    class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

    model.eval()
    model.to(device)
    return model, class_names, device

# ================================================================
# 4️⃣ IMAGE TRANSFORMATION PIPELINE
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict_image(image_bytes):
    """Preprocess and predict the blood group from fingerprint image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ Invalid image format. Please upload a valid fingerprint (JPG, PNG, BMP).")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error reading image: {e}")
        st.stop()

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_label = class_names[pred.item()]
        confidence = round(conf.item() * 100, 2)

    return predicted_label, confidence, probs.cpu().numpy().flatten()

# ================================================================
# 5️⃣ USER INTERFACE
# ================================================================

# Header Section
st.markdown("""
<div class='header-container'>
    <h1 style='color: white; margin-bottom: 0.5rem;'> BloodSense AI</h1>
    <h3 style='color: rgba(255, 255, 255, 0.8); font-weight: 300; margin-top: 0;'>
    Advanced Blood Group Detection from Fingerprint Analysis</h3>
</div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    # Personal Information Section
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("### 👤 Personal Information")
    
    name = st.text_input("**Full Name**", placeholder="Enter your full name")
    
    col_a, col_b = st.columns(2)
    with col_a:
        mobile = st.text_input("**Mobile Number**", placeholder="+1 234 567 8900")
    with col_b:
        age = st.number_input("**Age**", min_value=1, max_value=120, step=1, value=25)
    
    gender = st.selectbox("**Gender**", ["Select", "Male", "Female", "Other"])
    nationality = st.text_input("**Nationality**", placeholder="Your nationality")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Fingerprint Upload Section
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("### 📸 Fingerprint Upload")
    uploaded_file = st.file_uploader(
        "**Upload your fingerprint image**",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Please upload a clear image of your fingerprint for accurate analysis"
    )
    
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Display image
            st.markdown("**📁 Uploaded Fingerprint Preview:**")
            st.image(image, caption="🖐 Your Fingerprint", width=250, use_container_width=True)
        except UnidentifiedImageError:
            st.error("❌ Invalid or corrupted image. Please upload a valid fingerprint image.")
            st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Information Section
    st.markdown("<div class='info-section'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ About BloodSense AI")
    
    st.markdown("""
    <div style='color: white; font-size: 0.95rem; line-height: 1.6;'>
    <p>🔬 <b>How it works:</b> Our advanced AI analyzes unique patterns in your fingerprint 
    to predict your blood group with high accuracy using deep learning technology.</p>
    
    <p>🩺 <b>Medical Importance:</b> Knowing your blood group is crucial for:</p>
    <ul>
        <li>Emergency transfusions</li>
        <li>Organ transplants</li>
        <li>Pregnancy care</li>
        <li>Genetic research</li>
    </ul>
    
    <p>⚡ <b>Process:</b></p>
    <ol>
        <li>Upload your fingerprint image</li>
        <li>AI analyzes unique patterns</li>
        <li>Get instant blood group results</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection Button
    st.markdown("---")
    if st.button("🔍 **ANALYZE FINGERPRINT & DETECT BLOOD GROUP**", use_container_width=True):
        if not all([name.strip(), mobile.strip(), gender != "Select", age, uploaded_file is not None]):
            st.warning("⚠️ Please fill all required fields and upload a fingerprint image before proceeding.")
        else:
            with st.spinner("🔄 Loading AI model..."):
                model, class_names, device = load_model()
            
            with st.spinner("🔬 Analyzing fingerprint patterns..."):
                predicted_label, confidence, all_probs = predict_image(image_bytes)

            st.success("🎉 Analysis Complete!")
            
            # Display Results
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            
            # Blood Group with animated badge
            st.markdown(f"""
            <div style='text-align: center;'>
                <h3 style='color: white; margin-bottom: 0.5rem;'>Predicted Blood Group</h3>
                <div class='blood-group-badge'>{predicted_label}</div>
                <p style='color: rgba(255, 255, 255, 0.8); font-size: 1.1rem;'>
                Confidence Level: <b>{confidence}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Meter
            st.markdown(f"""
            <div style='margin: 2rem 0;'>
                <p style='color: white; margin-bottom: 0.5rem;'>AI Confidence Meter:</p>
                <div class='confidence-meter'>
                    <div class='confidence-fill' style='width: {confidence}%;'></div>
                </div>
                <p style='color: rgba(255, 255, 255, 0.8); text-align: center; font-size: 0.9rem;'>
                The AI is {confidence}% confident in this prediction</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Confidence Chart
            st.markdown("<div class='confidence-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Confidence Distribution Across All Blood Groups")
            
            # Create interactive bar chart
            df = pd.DataFrame({
                'Blood Group': class_names,
                'Confidence (%)': [round(p * 100, 2) for p in all_probs]
            })
            
            fig = px.bar(df, x='Blood Group', y='Confidence (%)', 
                        color='Confidence (%)',
                        color_continuous_scale='viridis',
                        text='Confidence (%)')
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Blood Groups",
                yaxis_title="Confidence (%)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # User Details
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("### 👤 Analysis Report Summary")
            
            user_data = {
                "Field": ["Name", "Mobile", "Gender", "Age", "Nationality", "Blood Group", "Confidence"],
                "Value": [name, mobile, gender, str(age), nationality, predicted_label, f"{confidence}%"]
            }
            
            user_df = pd.DataFrame(user_data)
            st.dataframe(user_df, use_container_width=True, hide_index=True)
            
            # Download Report Button
            csv = user_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Report as CSV",
                data=csv,
                file_name=f"blood_group_report_{name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255, 255, 255, 0.7); font-size: 0.8rem;'>
    <p><b>BloodSense AI</b> - Advanced Fingerprint Blood Group Detection System</p>
</div>
""", unsafe_allow_html=True)