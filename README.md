# 🚂 Railway Object Detection System

A Streamlit-based web application for detecting railway-related objects in video files using a YOLOv8 model trained on Roboflow.

## Features

- **Video Upload**: Upload .mp4 video files for object detection
- **Real-time Detection**: Uses YOLOv8 model from Roboflow (railway_objects/1)
- **Configurable Thresholds**:
  - Confidence Threshold (0-100%)
  - Overlap Threshold (0-100%) for NMS
  - Opacity Threshold (0-100%) for visualization
- **Annotated Output**: Video output with bounding boxes, labels, and confidence scores
- **JSON Export**: Detection results in structured JSON format
- **Statistics Dashboard**: View detection counts by class

## Detected Classes

The model can detect 20 different railway-related objects:

| Class ID | Class Name |
|----------|------------|
| 0 | Bushes |
| 1 | Electric Poles |
| 2 | Farms |
| 3 | Foot Over Bridges (FOBs) |
| 4 | Level Crossings |
| 5 | Open Fields |
| 6 | Parking Lots |
| 7 | Power Lines |
| 8 | Railway Crossings |
| 9 | Railway Platforms |
| 10 | Railway Stations |
| 11 | Railway Tracks |
| 12 | Road Over Bridges (ROBs) |
| 13 | Roads/Highways |
| 14 | Signal Poles |
| 15 | Subways |
| 16 | Trees |
| 17 | Tunnel Entrances |
| 18 | Urban Areas |
| 19 | Water Bodies |

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd rlwl_detection_iris
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   .\venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Your Roboflow API Key

1. Go to [Roboflow](https://roboflow.com/) and sign in or create an account
2. Navigate to your workspace settings
3. Find your API key in the "API Keys" section
4. Copy the API key to use in the application

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open in browser**: The app will automatically open at `http://localhost:8501`

3. **Configure settings**:
   - Enter your Roboflow API key in the sidebar
   - Adjust detection thresholds as needed
   - Upload a .mp4 video file

4. **Run Detection**: Click "Start Detection" to process the video

## Usage

1. **Enter API Key**: Input your Roboflow API key in the sidebar
2. **Set Thresholds**:
   - **Confidence Threshold**: Minimum confidence for a detection (default: 50%)
   - **Overlap Threshold**: Maximum IoU for NMS (default: 50%)
   - **Opacity Threshold**: Visual opacity of bounding boxes (default: 75%)
3. **Upload Video**: Select a .mp4 video file
4. **Process**: Click "Start Detection" to begin processing
5. **Download Results**: 
   - Download the annotated video
   - Download the JSON detection results

## JSON Output Format

The detection results are exported in the following format:

```json
{
  "predictions": [
    {
      "x": 818.5,
      "y": 523.5,
      "width": 243,
      "height": 391,
      "confidence": 0.972,
      "class": "Trees",
      "class_id": 13,
      "detection_id": "18f4c061-7e78-4873-bf33-d632f6e47694"
    }
  ]
}
```

## Requirements

- Python 3.8+
- Roboflow API key with access to the railway_objects project
- Sufficient RAM for video processing (recommended: 8GB+)

## Troubleshooting

### Common Issues

1. **"Error loading model"**: 
   - Verify your API key is correct
   - Ensure you have access to the railway_objects project

2. **Video processing is slow**:
   - Larger videos take more time
   - Consider using shorter video clips for testing

3. **OpenCV errors**:
   - Ensure opencv-python is properly installed
   - On some systems, you may need opencv-python-headless

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify your Python version (3.8+ required)
3. Ensure your video file is a valid .mp4 format

## License

This project is for educational and research purposes.

## Acknowledgments

- Model trained using [Roboflow](https://roboflow.com/)
- Object detection powered by [YOLOv8](https://github.com/ultralytics/ultralytics)
- Web interface built with [Streamlit](https://streamlit.io/)
