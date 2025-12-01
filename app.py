import streamlit as st
import tempfile
import os
import json
import uuid
from datetime import datetime
import numpy as np

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Railway Object Detection",
    page_icon="🚂",
    layout="wide"
)

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


def format_predictions_json(all_detections):
    """Format all detections into the required JSON structure."""
    formatted_predictions = []
    
    for frame_detections in all_detections:
        for det in frame_detections:
            prediction = {
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
    st.title("🚂 Railway Object Detection System")
    st.markdown("---")
    
    # Load dependencies
    try:
        cv2 = load_cv2()
        torch = load_torch()
        st.sidebar.success("✅ Dependencies loaded")
    except Exception as e:
        st.error(f"❌ Error loading dependencies: {e}")
        st.stop()
    
    # Check device and display info
    device = get_device()
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU Enabled: {gpu_name}")
    else:
        st.sidebar.info("💻 Running on CPU")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Thresholds")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold (%)",
        min_value=0, max_value=100, value=50,
        help="Minimum confidence score for detections"
    )
    
    overlap_threshold = st.sidebar.slider(
        "Overlap Threshold (%)",
        min_value=0, max_value=100, value=50,
        help="Maximum overlap allowed between detections (NMS)"
    )
    
    opacity_threshold = st.sidebar.slider(
        "Opacity Threshold (%)",
        min_value=0, max_value=100, value=75,
        help="Opacity of bounding box fill"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Performance Settings")
    
    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=0, max_value=10, value=0,
        help="Number of frames to skip between detections"
    )
    
    input_size = st.sidebar.selectbox(
        "Detection Input Size",
        options=[320, 416, 512, 640, 768, 1024],
        index=3,
        help="Larger size = more accurate but slower"
    )
    
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    st.sidebar.markdown("---")
    
    # Load model
    try:
        model, device, class_names = load_model_local()
        
        if model is None:
            st.warning("⚠️ Model not loaded. Please check weights.pt file.")
            st.stop()
        
        st.sidebar.subheader("📋 Model Class Labels")
        with st.sidebar.expander("View All Classes from Model"):
            for class_id, class_name in class_names.items():
                st.write(f"{class_id}: {class_name}")
        
        if debug_mode:
            st.info(f"**Model loaded successfully!**\n\nClasses detected: {len(class_names)}")
            st.json(dict(class_names))
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a .mp4 video file",
            type=['mp4'],
            help="Upload the video you want to analyze"
        )
    
    with col2:
        st.subheader("📊 Current Settings")
        st.info(f"""
        - **Device:** {device.upper()}
        - **Confidence:** {confidence_threshold}%
        - **Overlap:** {overlap_threshold}%
        - **Opacity:** {opacity_threshold}%
        - **Frame Skip:** {frame_skip}
        - **Input Size:** {input_size}px
        """)
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input_path = os.path.join(tempfile.gettempdir(), f"input_{uploaded_file.name}")
        with open(temp_input_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        st.success(f"✅ Video uploaded successfully: {uploaded_file.name}")
        
        # Get video info
        cap = cv2.VideoCapture(temp_input_path)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = video_frames / video_fps if video_fps > 0 else 0
        cap.release()
        
        st.info(f"**Video Info:** {video_width}x{video_height} | {video_fps} FPS | {video_frames} frames | {video_duration:.1f}s")
        
        st.subheader("📹 Original Video Preview")
        st.video(temp_input_path)
        
        frames_to_process = video_frames // (frame_skip + 1)
        st.caption(f"📊 Frames to process: {frames_to_process} (with frame skip = {frame_skip})")
        
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            st.subheader("⏳ Processing Video...")
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
                status_text.text(f"✅ Processing complete on {device.upper()} in {processing_time:.1f}s!")
                
                st.markdown("---")
                st.subheader("🎬 Detection Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Annotated Video")
                    st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=f.read(),
                            file_name=f"detected_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                
                with col2:
                    st.markdown("### Detection Statistics")
                    
                    total_detections = sum(len(frame_det) for frame_det in all_detections)
                    frames_with_detections = sum(1 for frame_det in all_detections if frame_det)
                    
                    class_counts = {}
                    for frame_det in all_detections:
                        for det in frame_det:
                            class_name = det.get('class', 'Unknown')
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    st.metric("Total Detections", total_detections)
                    st.metric("Frames with Detections", frames_with_detections)
                    st.metric("Total Frames Processed", len(all_detections))
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                    st.metric("Avg FPS", f"{len(all_detections) / processing_time:.1f}")
                    
                    if class_counts:
                        st.markdown("#### Detections by Class")
                        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• **{class_name}**: {count}")
                
                if debug_mode and all_detections:
                    st.markdown("---")
                    st.subheader("🐛 Debug Info")
                    first_frame_with_det = next((d for d in all_detections if d), None)
                    if first_frame_with_det:
                        st.write("**Sample detection:**")
                        st.json(first_frame_with_det[:3])
                
                st.markdown("---")
                st.subheader("📄 JSON Output")
                
                json_output = format_predictions_json(all_detections)
                json_str = json.dumps(json_output, indent=2)
                
                with st.expander("View Full JSON Output", expanded=False):
                    st.json(json_output)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                try:
                    os.remove(temp_input_path)
                except:
                    pass
            else:
                st.error("❌ Error processing video. Please try again.")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "<p>Railway Object Detection System | Powered by YOLOv8</p>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
