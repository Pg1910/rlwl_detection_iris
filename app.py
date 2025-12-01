import streamlit as st
import tempfile
import os
import json
import uuid
from datetime import datetime
import numpy as np
from io import BytesIO
import hashlib

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Aksha - Railway Object Detection",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding and styling - Clean Professional Theme
st.markdown("""
<style>
    /* ========== GLOBAL THEME ========== */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* ========== HEADER BRANDING ========== */
    .main-header {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .brand-title {
        color: #ffffff !important;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .brand-subtitle {
        color: #dfe6e9 !important;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    .brand-tagline {
        color: #fdcb6e !important;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* ========== SECTION HEADERS ========== */
    .section-header {
        background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        color: #ffffff !important;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(108,92,231,0.3);
    }
    
    /* ========== INFO BOXES ========== */
    .success-box {
        background: linear-gradient(135deg, #00b894, #00cec9);
        border: none;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #ffffff !important;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,184,148,0.3);
    }
    
    .success-box strong {
        color: #ffffff !important;
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3436 0%, #636e72 100%) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stSlider > div > div > div {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    
    /* ========== BUTTONS ========== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(108,92,231,0.4) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(108,92,231,0.5) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 1rem 2.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,184,148,0.4) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(0,184,148,0.5) !important;
    }
    
    .stButton > button {
        background: #ffffff !important;
        color: #6c5ce7 !important;
        border: 2px solid #6c5ce7 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #6c5ce7 !important;
        color: #ffffff !important;
    }
    
    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploader"] {
        background: #f8f9fa;
        border: 3px dashed #6c5ce7;
        border-radius: 15px;
        padding: 1.5rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #2d3436 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* ========== METRICS ========== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa, #dfe6e9);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #6c5ce7;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetric"] label {
        color: #636e72 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #2d3436 !important;
        font-weight: 800;
        font-size: 1.8rem;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #636e72 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe) !important;
        border-radius: 8px;
    }
    
    /* ========== TEXT & HEADINGS ========== */
    .main h1, .main h2, .main h3, .main h4 {
        color: #2d3436 !important;
        font-weight: 700;
    }
    
    .main p, .main span, .main label {
        color: #2d3436 !important;
    }
    
    .stCaption {
        color: #636e72 !important;
    }
    
    /* ========== VIDEO CONTAINER ========== */
    .video-container {
        border: 4px solid #6c5ce7;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* ========== FOOTER ========== */
    .footer {
        background: linear-gradient(135deg, #2d3436, #636e72);
        color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .footer a {
        color: #fdcb6e !important;
        font-weight: 600;
    }
    
    /* ========== LOGIN PAGE ========== */
    .login-container {
        max-width: 450px;
        margin: 3rem auto;
        padding: 2.5rem;
        background: #ffffff;
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .login-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .login-title {
        color: #2d3436 !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
    }
    
    .login-subtitle {
        color: #636e72 !important;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* ========== USER WELCOME ========== */
    .user-welcome {
        background: linear-gradient(135deg, #00b894, #00cec9);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,184,148,0.3);
    }
    
    .user-welcome span, .user-welcome strong {
        color: #ffffff !important;
    }
    
    /* ========== FORMS & INPUTS ========== */
    .stTextInput input {
        background: #f8f9fa !important;
        border: 2px solid #dfe6e9 !important;
        border-radius: 10px !important;
        color: #2d3436 !important;
        padding: 0.8rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #6c5ce7 !important;
        box-shadow: 0 0 0 3px rgba(108,92,231,0.2) !important;
    }
    
    /* ========== CODE BLOCKS ========== */
    code {
        background: #f8f9fa !important;
        color: #6c5ce7 !important;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* ========== ALERTS ========== */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        font-weight: 500;
    }
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6c5ce7, #a29bfe) !important;
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Lazy imports to avoid import errors on Streamlit Cloud
@st.cache_resource
def load_cv2():
    """Lazy load OpenCV."""
    import cv2
    return cv2

@st.cache_resource
def load_torch():
    """Lazy load PyTorch."""
    import torch
    return torch

@st.cache_resource
def load_yolo():
    """Lazy load YOLO from ultralytics."""
    from ultralytics import YOLO
    return YOLO

# Color palette for different classes (BGR format for OpenCV)
COLORS = [
    (0, 255, 0), (255, 165, 0), (139, 69, 19), (128, 128, 128),
    (255, 0, 0), (144, 238, 144), (169, 169, 169), (255, 255, 0),
    (255, 0, 255), (0, 128, 128), (128, 0, 128), (192, 192, 192),
    (0, 0, 128), (128, 128, 0), (0, 255, 255), (75, 0, 130),
    (34, 139, 34), (70, 130, 180), (255, 192, 203), (0, 191, 255),
]


# ============ Authentication Functions ============

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_auth_state():
    """Initialize authentication session state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'users_db' not in st.session_state:
        # Demo users database (in production, use a real database)
        st.session_state.users_db = {
            'admin': hash_password('admin123'),
            'demo': hash_password('demo123')
        }
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'


def authenticate_user(username, password):
    """Check if username and password are valid."""
    if username in st.session_state.users_db:
        if st.session_state.users_db[username] == hash_password(password):
            return True
    return False


def register_user(username, password):
    """Register a new user."""
    if username in st.session_state.users_db:
        return False, "Username already exists"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    st.session_state.users_db[username] = hash_password(password)
    return True, "Registration successful!"


def logout():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    # Clear detection results on logout
    st.session_state.detection_complete = False
    st.session_state.output_path = None
    st.session_state.all_detections = None


def show_login_page():
    """Display the login/signup page."""
    st.markdown("""
    <div class="login-container">
        <div class="login-logo">🚂</div>
        <h1 class="login-title">AKSHA</h1>
        <p class="login-subtitle">Railway Infrastructure Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for Login and Signup
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        if authenticate_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("⚠️ Please enter both username and password")
            
            st.markdown("---")
            st.markdown("**Demo Accounts:**")
            st.code("Username: admin | Password: admin123\nUsername: demo | Password: demo123")
        
        with tab2:
            st.markdown("### Create Account")
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username", placeholder="At least 3 characters")
                new_password = st.text_input("Choose Password", type="password", placeholder="At least 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                signup = st.form_submit_button("Sign Up", use_container_width=True)
                
                if signup:
                    if new_username and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("❌ Passwords do not match")
                        else:
                            success, message = register_user(new_username, new_password)
                            if success:
                                st.success(f"✅ {message} Please login.")
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("⚠️ Please fill in all fields")

# ============ End Authentication Functions ============


def get_color_for_class(class_id):
    """Get color for a given class ID."""
    return COLORS[class_id % len(COLORS)]


def get_device():
    """Get the best available device (CUDA GPU or CPU)."""
    torch = load_torch()
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@st.cache_resource
def load_model_local():
    """Load the local YOLO model."""
    torch = load_torch()
    YOLO = load_yolo()
    
    device = get_device()
    
    # Check if weights file exists
    weights_path = 'weights.pt'
    if not os.path.exists(weights_path):
        st.error(f"❌ Model weights not found at: {weights_path}")
        st.info("Please ensure 'weights.pt' is in the repository root.")
        return None, device, {}
    
    model = YOLO(weights_path)
    model.to(device)
    
    # Extract class names from the model itself
    class_names = model.names
    
    return model, device, class_names


def process_frame_local(model, frame, confidence_threshold, overlap_threshold, device, class_names, input_size=640):
    """Process a single frame using local GPU/CPU and return detections."""
    
    # Run inference
    results = model.predict(
        frame,
        conf=confidence_threshold,
        iou=overlap_threshold,
        device=device,
        imgsz=input_size,
        verbose=False
    )
    
    detections = []
    
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center and dimensions
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name from model's own mapping
                class_name = class_names.get(class_id, f"Class_{class_id}")
                
                detection = {
                    'x': float(x_center),
                    'y': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id
                }
                detections.append(detection)
    
    return detections


def draw_detections(cv2, frame, detections, opacity_threshold):
    """Draw bounding boxes, labels, and confidence on the frame."""
    overlay = frame.copy()
    opacity = opacity_threshold / 100.0
    
    for det in detections:
        x_center = det.get('x', 0)
        y_center = det.get('y', 0)
        width = det.get('width', 0)
        height = det.get('height', 0)
        confidence = det.get('confidence', 0)
        class_name = det.get('class', 'Unknown')
        class_id = det.get('class_id', 0)
        
        # Calculate bounding box coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Get color for this class
        color = get_color_for_class(class_id)
        
        # Draw filled rectangle with opacity
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Apply opacity blend
    frame = cv2.addWeighted(overlay, opacity * 0.3, frame, 1 - opacity * 0.3, 0)
    
    return frame


def format_predictions_json_with_timestamps(all_detections, fps):
    """Format all detections into the required JSON structure with timestamps."""
    formatted_predictions = []
    
    for frame_idx, frame_detections in enumerate(all_detections):
        # Calculate timestamp for this frame
        timestamp_seconds = frame_idx / fps if fps > 0 else 0
        minutes = int(timestamp_seconds // 60)
        seconds = timestamp_seconds % 60
        timestamp_str = f"{minutes:02d}:{seconds:05.2f}"
        
        for det in frame_detections:
            prediction = {
                "frame": frame_idx + 1,
                "timestamp": timestamp_str,
                "timestamp_seconds": round(timestamp_seconds, 3),
                "x": round(det.get('x', 0), 1),
                "y": round(det.get('y', 0), 1),
                "width": round(det.get('width', 0), 0),
                "height": round(det.get('height', 0), 0),
                "confidence": round(det.get('confidence', 0), 3),
                "class": det.get('class', 'Unknown'),
                "class_id": det.get('class_id', 0),
                "detection_id": str(uuid.uuid4())
            }
            formatted_predictions.append(prediction)
    
    return {"predictions": formatted_predictions}


def generate_pdf_report(json_output, class_counts, video_info, processing_time):
    """Generate a PDF report with detection results and timestamps."""
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, 'Aksha - Railway Object Detection Report', ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(20)
    
    # Video Information Section
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Video Information', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Resolution: {video_info["width"]}x{video_info["height"]}', ln=True)
    pdf.cell(0, 8, f'FPS: {video_info["fps"]}', ln=True)
    pdf.cell(0, 8, f'Total Frames: {video_info["frames"]}', ln=True)
    pdf.cell(0, 8, f'Duration: {video_info["duration"]:.1f} seconds', ln=True)
    pdf.cell(0, 8, f'Processing Time: {processing_time:.1f} seconds', ln=True)
    pdf.ln(10)
    
    # Summary Statistics Section
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Detection Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    
    total_detections = len(json_output.get('predictions', []))
    pdf.cell(0, 8, f'Total Detections: {total_detections}', ln=True)
    pdf.ln(5)
    
    # Detections by Class
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detections by Class:', ln=True)
    pdf.set_font('Arial', '', 11)
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pdf.cell(0, 7, f'  - {class_name}: {count}', ln=True)
    
    pdf.ln(10)
    
    # Detailed Detections Table
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Detailed Detection Log', ln=True)
    pdf.ln(5)
    
    # Table Header
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(200, 200, 200)
    col_widths = [20, 25, 45, 25, 35, 40]
    headers = ['Frame', 'Timestamp', 'Class', 'Conf.', 'Position (x,y)', 'Size (WxH)']
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, fill=True, align='C')
    pdf.ln()
    
    # Table Data
    pdf.set_font('Arial', '', 8)
    predictions = json_output.get('predictions', [])
    
    # Limit to prevent extremely large PDFs
    max_rows = 500
    for i, pred in enumerate(predictions[:max_rows]):
        if pdf.get_y() > 270:  # Check if we need a new page
            pdf.add_page()
            # Re-add header
            pdf.set_font('Arial', 'B', 9)
            for j, header in enumerate(headers):
                pdf.cell(col_widths[j], 8, header, border=1, fill=True, align='C')
            pdf.ln()
            pdf.set_font('Arial', '', 8)
        
        pdf.cell(col_widths[0], 7, str(pred.get('frame', '')), border=1, align='C')
        pdf.cell(col_widths[1], 7, pred.get('timestamp', ''), border=1, align='C')
        pdf.cell(col_widths[2], 7, pred.get('class', '')[:20], border=1, align='L')
        pdf.cell(col_widths[3], 7, f"{pred.get('confidence', 0):.2f}", border=1, align='C')
        pdf.cell(col_widths[4], 7, f"({pred.get('x', 0):.0f}, {pred.get('y', 0):.0f})", border=1, align='C')
        pdf.cell(col_widths[5], 7, f"{pred.get('width', 0):.0f} x {pred.get('height', 0):.0f}", border=1, align='C')
        pdf.ln()
    
    if len(predictions) > max_rows:
        pdf.ln(5)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 8, f'Note: Showing first {max_rows} of {len(predictions)} detections. See JSON for complete data.', ln=True)
    
    # Timeline Summary (group by timestamp)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Detection Timeline Summary', ln=True)
    pdf.ln(5)
    
    # Group detections by second
    timeline = {}
    for pred in predictions:
        second = int(pred.get('timestamp_seconds', 0))
        if second not in timeline:
            timeline[second] = {}
        class_name = pred.get('class', 'Unknown')
        timeline[second][class_name] = timeline[second].get(class_name, 0) + 1
    
    pdf.set_font('Arial', '', 10)
    for second in sorted(timeline.keys()):
        classes_at_second = timeline[second]
        classes_str = ', '.join([f"{name}: {count}" for name, count in classes_at_second.items()])
        
        minutes = second // 60
        secs = second % 60
        time_str = f"{minutes:02d}:{secs:02d}"
        
        if pdf.get_y() > 270:
            pdf.add_page()
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(25, 7, time_str, ln=False)
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 7, classes_str)
    
    # Output PDF to bytes
    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    
    return pdf_output


def process_video_local(model, device, class_names, video_path, confidence_threshold, overlap_threshold, opacity_threshold, frame_skip, input_size, progress_bar, status_text):
    """Process the entire video using local GPU/CPU and return the output path and detections."""
    cv2 = load_cv2()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video path
    output_path = os.path.join(tempfile.gettempdir(), f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    last_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Process frame only if not skipping
        if frame_count % (frame_skip + 1) == 0 or frame_count == 1:
            status_text.text(f"Processing frame {frame_count}/{total_frames} on {device.upper()}")
            
            detections = process_frame_local(
                model, frame, confidence_threshold, overlap_threshold, device, class_names, input_size
            )
            last_detections = detections
        else:
            status_text.text(f"Skipping frame {frame_count}/{total_frames} (using cached detections)")
            detections = last_detections
        
        all_detections.append(detections)
        
        # Draw detections on frame
        annotated_frame = draw_detections(cv2, frame.copy(), detections, opacity_threshold)
        
        # Write frame
        out.write(annotated_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path, all_detections


def main():
    # Initialize authentication state
    init_auth_state()
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Initialize session state for persisting results
    if 'detection_complete' not in st.session_state:
        st.session_state.detection_complete = False
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    if 'all_detections' not in st.session_state:
        st.session_state.all_detections = None
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'class_counts' not in st.session_state:
        st.session_state.class_counts = None
    if 'video_bytes' not in st.session_state:
        st.session_state.video_bytes = None
    if 'json_output' not in st.session_state:
        st.session_state.json_output = None
    if 'pdf_buffer' not in st.session_state:
        st.session_state.pdf_buffer = None
    
    # Header with branding and user welcome
    col_header, col_user = st.columns([3, 1])
    with col_header:
        st.markdown("""
        <div class="main-header">
            <h1 class="brand-title">🚂 AKSHA</h1>
            <p class="brand-subtitle">Railway Infrastructure Object Detection System</p>
            <p class="brand-tagline">Powered by Advanced AI & Computer Vision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_user:
        st.markdown(f"""
        <div class="user-welcome">
            <span>👤 <strong>{st.session_state.username}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()
    
    # Load dependencies
    try:
        cv2 = load_cv2()
        torch = load_torch()
    except Exception as e:
        st.error(f"❌ Error loading dependencies: {e}")
        st.stop()
    
    # Sidebar branding
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3a5f, #2d5a87); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">⚙️ AKSHA</h2>
        <p style="color: #a8d5ff; font-size: 0.8rem; margin: 0;">Control Panel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check device and display info
    device = get_device()
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
    else:
        st.sidebar.info("💻 Running on CPU")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Detection Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0, max_value=100, value=50,
        format="%d%%",
        help="Minimum confidence score for detections"
    )
    
    overlap_threshold = st.sidebar.slider(
        "Overlap Threshold (NMS)",
        min_value=0, max_value=100, value=50,
        format="%d%%",
        help="Maximum overlap allowed between detections"
    )
    
    opacity_threshold = st.sidebar.slider(
        "Box Opacity",
        min_value=0, max_value=100, value=75,
        format="%d%%",
        help="Opacity of bounding box fill"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Performance")
    
    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=0, max_value=10, value=0,
        help="Skip frames for faster processing (0 = process all)"
    )
    
    input_size = st.sidebar.selectbox(
        "Model Input Size",
        options=[320, 416, 512, 640, 768, 1024],
        index=3,
        help="Larger = more accurate but slower"
    )
    
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    st.sidebar.markdown("---")
    
    # Load model
    try:
        model, device, class_names = load_model_local()
        
        if model is None:
            st.warning("⚠️ Model not loaded. Please check weights.pt file.")
            st.stop()
        
        st.sidebar.subheader("📋 Detectable Classes")
        with st.sidebar.expander(f"View All {len(class_names)} Classes"):
            for class_id, class_name in class_names.items():
                st.write(f"`{class_id}` {class_name}")
        
        if debug_mode:
            st.info(f"**Model loaded successfully!** Classes: {len(class_names)}")
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">📤 Upload Video</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown('<div class="section-header">📊 Current Settings</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-container">
            <p><strong>🖥️ Device:</strong> {device.upper()}</p>
            <p><strong>🎯 Confidence:</strong> {confidence_threshold}%</p>
            <p><strong>📐 Overlap:</strong> {overlap_threshold}%</p>
            <p><strong>🎨 Opacity:</strong> {opacity_threshold}%</p>
            <p><strong>⏭️ Frame Skip:</strong> {frame_skip}</p>
            <p><strong>📏 Input Size:</strong> {input_size}px</p>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input_path = os.path.join(tempfile.gettempdir(), f"input_{uploaded_file.name}")
        with open(temp_input_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Get video info
        cap = cv2.VideoCapture(temp_input_path)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = video_frames / video_fps if video_fps > 0 else 0
        cap.release()
        
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ Video Loaded:</strong> {uploaded_file.name}<br>
            <strong>📐 Resolution:</strong> {video_width}×{video_height} | 
            <strong>🎬 FPS:</strong> {video_fps} | 
            <strong>📊 Frames:</strong> {video_frames} | 
            <strong>⏱️ Duration:</strong> {video_duration:.1f}s
        </div>
        """, unsafe_allow_html=True)
        
        # Show video preview
        st.markdown("#### 📹 Video Preview")
        st.video(temp_input_path)
        
        frames_to_process = video_frames // (frame_skip + 1)
        st.caption(f"📊 Frames to process: {frames_to_process} (with frame skip = {frame_skip})")
        
        # Store video info for PDF report
        video_info = {
            "width": video_width,
            "height": video_height,
            "fps": video_fps,
            "frames": video_frames,
            "duration": video_duration
        }
        
        # Reset results if new video uploaded
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            # Clear previous results
            st.session_state.detection_complete = False
            
            st.markdown('<div class="section-header">⏳ Processing Video...</div>', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = datetime.now()
            
            output_path, all_detections = process_video_local(
                model, device, class_names, temp_input_path,
                confidence_threshold / 100, overlap_threshold / 100,
                opacity_threshold, frame_skip, input_size,
                progress_bar, status_text
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if output_path and os.path.exists(output_path):
                # Calculate class counts
                class_counts = {}
                for frame_det in all_detections:
                    for det in frame_det:
                        class_name = det.get('class', 'Unknown')
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Read video bytes for download
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Generate JSON
                json_output = format_predictions_json_with_timestamps(all_detections, video_fps)
                
                # Generate PDF
                try:
                    pdf_buffer = generate_pdf_report(json_output, class_counts, video_info, processing_time)
                except:
                    pdf_buffer = None
                
                # Store everything in session state
                st.session_state.detection_complete = True
                st.session_state.output_path = output_path
                st.session_state.all_detections = all_detections
                st.session_state.video_info = video_info
                st.session_state.processing_time = processing_time
                st.session_state.class_counts = class_counts
                st.session_state.video_bytes = video_bytes
                st.session_state.json_output = json_output
                st.session_state.pdf_buffer = pdf_buffer
                
                status_text.text(f"✅ Processing complete on {device.upper()} in {processing_time:.1f}s!")
        
        # Display results if detection is complete (persists across reruns)
        if st.session_state.detection_complete:
            st.markdown("---")
            st.markdown('<div class="section-header">🎬 Detection Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.markdown("#### 🎥 Annotated Video")
                st.success("✅ Video processing complete! Download the annotated video below.")
                
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=st.session_state.video_bytes,
                    file_name=f"aksha_detected_{uploaded_file.name}",
                    mime="video/mp4",
                    key="download_video"
                )
            
            with col2:
                st.markdown("#### 📈 Statistics")
                
                all_detections = st.session_state.all_detections
                processing_time = st.session_state.processing_time
                class_counts = st.session_state.class_counts
                
                total_detections = sum(len(frame_det) for frame_det in all_detections)
                frames_with_detections = sum(1 for frame_det in all_detections if frame_det)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("🔍 Total Detections", f"{total_detections:,}")
                    st.metric("🎞️ Frames Processed", f"{len(all_detections):,}")
                with col_b:
                    st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
                    st.metric("⚡ Avg FPS", f"{len(all_detections) / processing_time:.1f}")
                
                if class_counts:
                    st.markdown("##### 📊 By Class")
                    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
                        percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                        st.progress(percentage / 100, text=f"{class_name}: {count}")
            
            # Debug info
            if debug_mode and all_detections:
                with st.expander("🐛 Debug Info"):
                    first_frame_with_det = next((d for d in all_detections if d), None)
                    if first_frame_with_det:
                        st.json(first_frame_with_det[:3])
            
            st.markdown("---")
            st.markdown('<div class="section-header">📄 Export Reports</div>', unsafe_allow_html=True)
            
            col_json, col_pdf = st.columns(2)
            
            json_output = st.session_state.json_output
            json_str = json.dumps(json_output, indent=2)
            
            with col_json:
                st.markdown("#### 📋 JSON Report")
                st.caption("Includes frame numbers and timestamps")
                st.write(f"📊 {len(json_output.get('predictions', []))} detections recorded")
                
                st.download_button(
                    label="📥 Download Full JSON",
                    data=json_str,
                    file_name=f"aksha_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
            
            with col_pdf:
                st.markdown("#### 📑 PDF Report")
                st.caption("Detailed report with timeline & statistics")
                
                if st.session_state.pdf_buffer:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=st.session_state.pdf_buffer,
                        file_name=f"aksha_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                else:
                    st.warning("PDF generation failed. Please ensure fpdf is installed.")
            
            # Clear results button
            st.markdown("---")
            if st.button("🔄 Clear Results & Start New Detection", use_container_width=True):
                st.session_state.detection_complete = False
                st.session_state.output_path = None
                st.session_state.all_detections = None
                st.session_state.video_bytes = None
                st.session_state.json_output = None
                st.session_state.pdf_buffer = None
                st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="margin: 0; font-size: 1.1rem;"><strong>AKSHA</strong> - Railway Infrastructure Detection System</p>
        <p style="margin: 0.3rem 0; font-size: 0.9rem;">Powered by YOLOv8 | Built with Streamlit</p>
        <p style="margin: 0; font-size: 0.8rem; color: #a8d5ff;">© 2024 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
