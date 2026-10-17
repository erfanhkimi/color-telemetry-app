import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(page_title="Color Telemetry Studio", layout="wide")
st.title("🎬 Multimedia Color Telemetry Extractor")
st.markdown("Analyze visual color palettes, lighting trends, and visual mood from video frames.")

# 2. Sidebar Controls
st.sidebar.header("Settings")
num_colors = st.sidebar.slider("Number of Dominant Colors", min_value=3, max_value=8, value=5)
frame_sample_rate = st.sidebar.slider("Video Sampling Rate (Every Nth Frame)", min_value=10, max_value=60, value=30)

uploaded_file = st.sidebar.file_uploader("Upload Video or Image", type=["mp4", "mov", "jpg", "png"])


# Helper Function: Extract Color Palette & Metrics from a Single Image
def analyze_frame(frame_rgb, k=5):
    # Downsample for ultra-fast performance
    small_frame = cv2.resize(frame_rgb, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small_frame.reshape(-1, 3)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # Convert RGB to Hex codes
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

    # Calculate overall frame brightness & saturation
    hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HSV)
    avg_saturation = np.mean(hsv_frame[:, :, 1]) / 255.0 * 100
    avg_brightness = np.mean(hsv_frame[:, :, 2]) / 255.0 * 100

    return hex_colors, avg_brightness, avg_saturation


# 3. Main Processing Pipeline
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Option A: Processing Single Image
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hex_colors, brightness, saturation = analyze_frame(img_rgb, k=num_colors)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Dominant Palette")
            palette_fig = px.bar(
                x=hex_colors,
                y=[1] * len(hex_colors),
                color=hex_colors,
                color_discrete_map="identity",
                labels={"x": "Hex Code", "y": ""},
            )
            palette_fig.update_layout(showlegend=False, yaxis_visible=False, height=250)
            st.plotly_chart(palette_fig, use_container_width=True)

            st.metric("Avg Brightness Score", f"{brightness:.1f}%")
            st.metric("Avg Saturation Score", f"{saturation:.1f}%")

    # Option B: Processing Video Timeline
    else:
        # Save temporary video file to disk for OpenCV processing
        with open("temp_video.mp4", "wb") as f:
            f.write(file_bytes)

        cap = cv2.VideoCapture("temp_video.mp4")
        telemetry_data = []
        frame_count = 0

        progress_bar = st.progress(0)
        st.info("Extracting frame telemetry across video timeline...")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hex_colors, brightness, saturation = analyze_frame(frame_rgb, k=num_colors)
                timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)

                telemetry_data.append({
                    "Timestamp (s)": round(timestamp, 2),
                    "Brightness": round(brightness, 2),
                    "Saturation": round(saturation, 2),
                    "Primary Color": hex_colors[0],
                })

                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            frame_count += 1

        cap.release()
        progress_bar.empty()

        # Render Telemetry Dashboard
        df = pd.DataFrame(telemetry_data)

        st.subheader("📊 Video Visual Telemetry Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            fig_bright = px.line(df, x="Timestamp (s)", y="Brightness", title="Lighting & Brightness Over Time")
            st.plotly_chart(fig_bright, use_container_width=True)

        with col2:
            fig_sat = px.line(df, x="Timestamp (s)", y="Saturation", title="Color Saturation & Intensity Over Time")
            st.plotly_chart(fig_sat, use_container_width=True)

        st.subheader("Raw Frame Data")
        st.dataframe(df, use_container_width=True)