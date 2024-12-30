import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2


# Fungsi untuk memuat model YOLO
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def detect_objects(model, image, conf_threshold):
    img = np.array(image)

    # Run inference with the model
    results = model(img)

    # Debugging: Print the results structure
    st.write("Results structure:", results)

    # Filter detections based on confidence threshold
    detections = []
    
    # Check if results are not empty
    if results:
        for r in results:
            # Check if boxes are present in the result
            if hasattr(r, 'boxes') and r.boxes is not None:
                boxes = r.boxes
                st.write(f"Number of boxes detected: {len(boxes.xyxy)}")  # Debugging

                for i in range(len(boxes.xyxy)):  # Iterate through each box
                    box = boxes.xyxy[i]  # Get the box coordinates
                    conf = boxes.conf[i]  # Get the confidence score
                    cls = int(boxes.cls[i])  # Get the class index

                    # Debugging: Print box details
                    st.write(f"Box: {box}, Confidence: {conf}, Class: {cls}")  # Debugging
                    if conf >= conf_threshold:
                        detections.append({
                            "Class": model.names[cls],
                            "Confidence": float(conf),
                            "Coordinates": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                        })

    # Render image with bounding box
    if detections:
        annotated_img = results[0].plot()  # Annotate the image with detected boxes
    else:
        annotated_img = img  # Fallback to original image if no results

    return detections, annotated_img


# Streamlit UI
st.title("YOLO Object Detection")
st.sidebar.title("Upload Image and Settings")

model_path = "D:\\File Kuliah\\VISKOM\\viskom-final-project\\viskom-final-project\\yolo2.pt"
model = load_model(model_path)

if model is not None:
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((640, 640))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        conf_threshold = st.sidebar.slider("Confidence Threshold", 0.00001, 1.0, 0.5, 0.00001)

        with st.spinner("Detecting objects..."):
            detections, annotated_img = detect_objects(model, image, conf_threshold)

        st.image(annotated_img, caption="Detection Results", use_column_width=True)

        if len(detections) == 0:
            st.warning("No objects detected. Try lowering the confidence threshold.")
        else:
            st.write("Detections:")
            for det in detections:
                st.write(det)
else:
    st.error("Model failed to load. Please check the model path or file.")
