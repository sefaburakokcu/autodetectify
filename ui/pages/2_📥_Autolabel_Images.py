import os
import zipfile
import shutil
import streamlit as st
import numpy as np

from PIL import Image
from ui_utils import plot_results, convert_pil_to_cv_image, save_dataset_as_zip, initialize_zero_shot_models, extract_labels, get_label_list_from_prompt

st.set_page_config(page_title="Autolabel Images", page_icon="📥")

# Constants
MAX_IMAGE_NUMBER = 50
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
ZIP_EXTENSION = ["zip"]
MODEL_NAMES = ["groundingdino"]
TEMP_DIR = "temp_images"
DATASET_EXPORT_FORMATS = ['yolov5', 'yolov8', 'coco']


def cleanup_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    st.session_state.uploaded_filenames = []
    st.session_state.file_paths = []
    st.session_state.predictions_dict = {}


def save_images_from_zip(uploaded_zip_file):
    image_files = []
    with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
        for root, _, files in os.walk(TEMP_DIR):
            for file in files:
                if file.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    image_files.append(os.path.join(root, file))
    return image_files


def cache_predictions(model, images_dict, prompt):
    with st.spinner("Extracting labels for the prompt..."):
        st.session_state.predictions_dict = extract_labels(model, images_dict, prompt)


def autolabel_and_export_tab(model):
    if "predict_button" not in st.session_state:
        st.session_state.predict_button = False

    if "images_dict" not in st.session_state:
        st.session_state.images_dict = {}

    if "predictions_dict" not in st.session_state:
        st.session_state.predictions_dict = {}

    with st.form("autolabel_form"):
        uploaded_files = st.file_uploader("Upload images or a ZIP file:", type=IMAGE_EXTENSIONS + ZIP_EXTENSION, accept_multiple_files=True)
        uploaded_image_number = len(uploaded_files)

        prompt = st.text_input("Please enter the prompt label", "pedestrian")
        label_format = st.selectbox("Please select dataset format", DATASET_EXPORT_FORMATS)
        predict_button = st.form_submit_button("Predict")

    if uploaded_image_number == 0:
        st.session_state.predict_button = False
        st.session_state.images_dict = {}
        st.session_state.predictions_dict = {}

    if predict_button:
        st.session_state.predict_button = True
        if uploaded_image_number > 0:
            cleanup_temp_dir()
            image_paths = []
            for file in uploaded_files:
                if file.name.lower().endswith(".zip"):
                    image_paths.extend(save_images_from_zip(file))
                else:
                    image_path = os.path.join(TEMP_DIR, file.name)
                    with open(image_path, "wb") as f:
                        f.write(file.getbuffer())
                    image_paths.append(image_path)

            if len(image_paths) > MAX_IMAGE_NUMBER:
                st.warning(f"Too many images uploaded. Only the first {MAX_IMAGE_NUMBER} images will be processed.")
                image_paths = image_paths[:MAX_IMAGE_NUMBER]

            st.session_state.images_dict = {os.path.basename(path): Image.open(path).convert('RGB') for path in image_paths}
            cache_predictions(model, st.session_state.images_dict, prompt)

    if st.session_state.predict_button:
        labels_list = get_label_list_from_prompt(prompt)
        if uploaded_image_number == 0:
            st.warning("Please upload at least one image.")
            cleanup_temp_dir()
        else:
            max_page_num = min(MAX_IMAGE_NUMBER, uploaded_image_number, len(st.session_state.images_dict))

            if uploaded_image_number > 1:
                selected_image_num = st.slider("Select Image", 1, max_page_num, 1)
            else:
                selected_image_num = 1

            selected_image_name = list(st.session_state.images_dict.keys())[selected_image_num - 1]
            selected_pil_image = st.session_state.images_dict[selected_image_name]

            selected_image_predictions = st.session_state.predictions_dict[selected_image_name]

            selected_cv_image = np.array(selected_pil_image)
            annotated_image = plot_results(selected_cv_image, selected_image_predictions['scores'],
                                           selected_image_predictions['labels'],
                                           selected_image_predictions['boxes'])

            st.image(annotated_image, caption=f"Annotated image for prompt: {prompt}", use_column_width=True)

            st.download_button(
                label="Download Dataset as ZIP",
                data=save_dataset_as_zip(st.session_state.images_dict, st.session_state.predictions_dict, labels_list,
                                         formats=[label_format]),
                file_name='dataset.zip',
                mime='application/zip'
            )


if __name__ == "__main__":
    title = "Autolabel Images"

    st.markdown(f"# {title}")
    st.sidebar.header(f"{title}")
    st.write(
        """This page allows you to automatically label your images using a specified prompt and export the labeled 
        dataset in various formats. Upload up to 50 images or a ZIP file containing images, run the labeling process, 
        visualize the results, and download the dataset as a ZIP file in YOLOv5, YOLOv8, or COCO format."""
    )

    model_name = st.sidebar.selectbox("Zero Shot Detection Model", MODEL_NAMES)
    model = initialize_zero_shot_models(model_name)
    autolabel_and_export_tab(model)