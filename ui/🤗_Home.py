import streamlit as st

st.title("Autodetectify")

st.write("""
Welcome to **Autodetectify!**, an advanced solution for testing Zero-Shot(ZS) object detection models with different 
prompts and autolabelling datasets.
This application provides a modern interface for:

- Comparing different ZS object detection models with  prompts on your uploaded images.
- Auto-labeling images based on your specified prompt and exporting the dataset in different formats.

### Features:
- **Prompt Comparison**: Upload your images, enter different prompts, and visualize side-by-side comparisons of object detection results.
- **Auto-Labeling and Export**: Automatically label your images using a specified prompt and download the labeled dataset in YOLOv5, YOLOv8, or COCO format.

### How to Use:
1. Navigate to the **Prompt Comparison** tab to compare object detection results with different prompts.
2. Go to the **Auto-Label and Export** tab to upload images, run the auto-labeling process, and download the dataset.
3. Follow the instructions on each page and utilize the various features to enhance your image labeling workflow.

Thank you for trying Autodetectify. We hope this tool helps you in your object detection projects!
""")
