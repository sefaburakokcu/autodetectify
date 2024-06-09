import os
import tempfile
import numpy as np
import streamlit as st

from PIL import Image
from streamlit_image_comparison import image_comparison

from ui_utils import (plot_results, initialize_zero_shot_models, extract_labels, filter_predictions_by_score,
                      BOX_COLOR_DICT)

st.set_page_config(page_title="Compare Object Detection Prompts", page_icon="👀")

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

MODEL_NAMES = ["groundingdino"]
MAX_IMAGE_NUMBER = 5


def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return temp_dir, file_paths


def reset_cache_and_predictions():
    st.session_state.images_dict = {}
    st.session_state.predictions_dict_1 = {}
    st.session_state.predictions_dict_2 = {}
    st.session_state.predict_button = False


def cache_predictions(model, images_dict, prompt_1, prompt_2):
    with st.spinner("Extracting labels for the first prompt..."):
        st.session_state.predictions_dict_1 = extract_labels(model, images_dict, prompt_1)
    with st.spinner("Extracting labels for the second prompt..."):
        st.session_state.predictions_dict_2 = extract_labels(model, images_dict, prompt_2)


# Main Function
def prompt_compare_tab(model):
    if "predict_button" not in st.session_state:
        st.session_state.predict_button = False

    if "images_dict" not in st.session_state:
        st.session_state.images_dict = {}

    if "predictions_dict_1" not in st.session_state:
        st.session_state.predictions_dict_1 = {}

    if "predictions_dict_2" not in st.session_state:
        st.session_state.predictions_dict_2 = {}

    with st.form("compare_form"):
        uploaded_image_files = st.file_uploader("Upload images:", type=IMAGE_EXTENSIONS, accept_multiple_files=True)
        uploaded_image_number = len(uploaded_image_files)

        col1, col2 = st.columns(2)
        prompt_1 = col1.text_input("Please enter the first prompt label", "pedestrian")
        prompt_2 = col2.text_input("Please enter the second prompt label", "man")
        predict_button = st.form_submit_button("Predict")

    if uploaded_image_number == 0:
        reset_cache_and_predictions()

    if predict_button:
        st.session_state.predict_button = True
        if uploaded_image_number > 0:
            if uploaded_image_number > MAX_IMAGE_NUMBER:
                st.warning(f"{uploaded_image_number} images are uploaded. "
                           f"Only the first {MAX_IMAGE_NUMBER} images will be processed.")
                uploaded_image_files = uploaded_image_files[:MAX_IMAGE_NUMBER]

            st.session_state.images_dict = {file.name: Image.open(file).convert('RGB') for file in uploaded_image_files}
            cache_predictions(model, st.session_state.images_dict, prompt_1, prompt_2)

    if st.session_state.predict_button:
        if uploaded_image_number == 0:
            st.warning("Please upload at least one image.")
            reset_cache_and_predictions()
        else:
            with st.expander("Expand for more settings..."):
                score_threshold = st.slider("Confidence Threshold:", 0.10, 1.00, 0.10)
                image_width = st.slider("Image width:", 300, 1920, 700)
                bbox_color = BOX_COLOR_DICT[st.selectbox("Bounding Box Color:", list(BOX_COLOR_DICT.keys()))]
                bbox_thickness = st.number_input("Bounding Box Thickness:", 1, 10, 1)

            st.session_state.predictions_dict_1 = filter_predictions_by_score(st.session_state.predictions_dict_1,
                                                                              score_threshold)
            st.session_state.predictions_dict_2 = filter_predictions_by_score(st.session_state.predictions_dict_2,
                                                                              score_threshold)

            max_page_num = min(MAX_IMAGE_NUMBER, uploaded_image_number, len(st.session_state.images_dict))

            if uploaded_image_number > 1:
                selected_image_num = st.slider("Select Image", 1, max_page_num, 1)
            else:
                selected_image_num = 1

            selected_image_name = list(st.session_state.images_dict.keys())[selected_image_num - 1]
            selected_pil_image = st.session_state.images_dict[selected_image_name]

            selected_image_predictions_1 = st.session_state.predictions_dict_1[selected_image_name]
            selected_image_predictions_2 = st.session_state.predictions_dict_2[selected_image_name]

            selected_cv_image = np.array(selected_pil_image)
            annotated_image_1 = plot_results(selected_cv_image, selected_image_predictions_1['scores'],
                                             selected_image_predictions_1['labels'],
                                             selected_image_predictions_1['boxes'],
                                             color=bbox_color,
                                             thickness=bbox_thickness)

            annotated_image_2 = plot_results(selected_cv_image, selected_image_predictions_2['scores'],
                                             selected_image_predictions_2['labels'],
                                             selected_image_predictions_2['boxes'],
                                             color=bbox_color,
                                             thickness=bbox_thickness)

            image_comparison(
                img1=annotated_image_1,
                img2=annotated_image_2,
                label1=prompt_1,
                label2=prompt_2,
                width=image_width,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )


if __name__ == "__main__":
    title = "Compare Object Detection Prompts"

    st.markdown(f"# {title}")
    st.sidebar.header(f"{title}")
    st.write(
        """This page allows you to compare object detection results for two different prompts across multiple images. 
        Upload your images, enter the prompts, and visualize the predictions side by side to see how the model performs 
        for each prompt."""
    )

    model_name = st.sidebar.selectbox("Zero Shot Detection Model", MODEL_NAMES)
    model = initialize_zero_shot_models(model_name)

    prompt_compare_tab(model)
