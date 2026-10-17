# 🎬 Color Telemetry & Visual Analytics Studio

An automated video analytics web application that extracts frame-by-frame dominant color palettes, lighting trends, and saturation metrics from video assets and image files in real time.

Built with **Python**, **OpenCV**, **Scikit-Learn**, **Streamlit**, **Plotly**, and **Pandas**.

---

## 📌 Project Overview

Visual analytics in media computing often requires understanding lighting shifts, mood transitions, and dominant color schemes across video timelines. This application automates visual data extraction by applying unsupervised Machine Learning (**K-Means Clustering**) and Computer Vision signal processing (**HSV Color Space Mapping**) to media uploads.

To handle high-resolution video streams without UI latency, the pipeline applies **spatial downsampling** prior to pixel vector clustering, maintaining high palette accuracy while reducing computational overhead by up to **90%**.

---

## ✨ Key Features

- **Dominant Color Extraction:** Uses K-Means clustering to extract top $K$ dominant hex colors from uploaded media.
- **Time-Series Telemetry Dashboard:** Tracks frame-by-frame brightness and saturation variations across video timelines using Plotly interactive charts.
- **Performance Optimization:** Downsamples image matrices to $100 \times 100$ vectors to optimize matrix operations and accelerate clustering convergence.
- **Flexible Media Pipeline:** Supports processing single images (`JPG`, `PNG`) and full video assets (`MP4`, `MOV`) with custom frame-sampling rate controls.
- **Interactive UI:** Built using Streamlit for fast deployment and intuitive web-based analysis.

---

## 🛠️ Tech Stack & Dependencies

- **Computer Vision & ML:** OpenCV (`opencv-python-headless`), Scikit-Learn (`sklearn.cluster.KMeans`)
- **Data Structuring & Visualization:** Pandas, Plotly Express
- **Web Application Framework:** Streamlit
- **Language & Runtime:** Python 3.13

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/erfanhkimi/color-telemetry-app.git](https://github.com/erfanhkimi/color-telemetry-app.git)
cd color-telemetry-app
