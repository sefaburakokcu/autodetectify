import sys
import cv2
import numpy as np
import torch
import os
import re
import shutil
import tempfile
from PIL import Image
import zipfile
import os
import json
import io
import streamlit as st
from PIL import Image

sys.path.append("..")
from autodetectify.models.zero_shot_detection.models import GroundingDINOZeroShotObjectDetectionModel


def get_label_list_from_prompt(prompt):
    # Split the prompt by either ',' or '.' and filter out any empty strings
    labels = [label.strip() for label in re.split('[,.]', prompt) if label.strip()]
    # Remove duplicate labels and maintain order
    unique_labels = list(dict.fromkeys(labels))
    return unique_labels


@st.cache_resource
def initialize_zero_shot_models(model_name="groundingdino"):
    if model_name == "groundingdino":
        model = GroundingDINOZeroShotObjectDetectionModel(model_name="IDEA-Research/grounding-dino-base")
    else:
        raise ValueError(f"{model_name} is not defined.")
    return model


# @st.cache_resource#(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def extract_labels(model, images_dict, text_queries):
    predictions_dict = {}
    progress_bar = st.progress(0)
    total_images = len(images_dict)

    for i, (image_name, image) in enumerate(images_dict.items()):
        predictions = model.predict([image], preprocess_caption(text_queries), query_type='text')
        predictions = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in
                       predictions[0].items()}
        image_width, image_height = image.size
        predictions["image_width"] = image_width
        predictions["image_height"] = image_height
        predictions_dict[image_name] = predictions

        progress_bar.progress((i + 1) / total_images)

    progress_bar.empty()  # Remove the progress bar once done
    return predictions_dict


def convert_to_yolo_format(predictions_dict, label_list):
    yolo_labels_dict = {}
    for image_name, predictions in predictions_dict.items():
        image_width = predictions["image_width"]
        image_height = predictions["image_height"]

        yolo_boxes = []
        for box in predictions["boxes"]:
            xmin, ymin, xmax, ymax = box
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            yolo_boxes.append([x_center, y_center, width, height])

        integer_labels = [label_list.index(label) for label in predictions["labels"]]

        yolo_labels_dict[image_name] = {
            "labels": integer_labels,
            "boxes": yolo_boxes
        }
    return yolo_labels_dict


def save_yolo_labels(labels_dict, output_dir, version):
    os.makedirs(output_dir, exist_ok=True)
    for image_name, labels in labels_dict.items():
        txt_filename = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(txt_filename, 'w') as f:
            for label, box in zip(labels['labels'], labels['boxes']):
                x_center, y_center, width, height = box
                f.write(f"{label} {x_center} {y_center} {width} {height}\n")


def save_coco_labels(labels_dict, output_dir):
    annotations = []
    images = []
    categories = []
    category_ids = {label: i for i, label in
                    enumerate(set([label for labels in labels_dict.values() for label in labels['labels']]))}

    for image_id, (image_name, labels) in enumerate(labels_dict.items()):
        images.append({
            "id": image_id,
            "file_name": image_name,
            "width": labels['image_width'],
            "height": labels['image_height']
        })
        for label, box in zip(labels['labels'], labels['boxes']):
            x_min, y_min, width, height = box
            annotations.append({
                "id": len(annotations),
                "image_id": image_id,
                "category_id": category_ids[label],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })

    categories = [{"id": id, "name": name} for name, id in category_ids.items()]
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_format, f)


def save_dataset_as_zip(images_dict, predictions_dict, label_list, formats):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        # Save images
        for image_name, image in images_dict.items():
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)
            zip_file.writestr(f"images/{image_name}", img_buffer.read())

        for format in formats:
            if format.startswith('yolo'):
                version = format
                labels_dir = f"labels"
                os.makedirs(labels_dir, exist_ok=True)
                yolo_labels_dict = convert_to_yolo_format(predictions_dict, label_list)
                save_yolo_labels(yolo_labels_dict, labels_dir, version)
                for label_file in os.listdir(labels_dir):
                    with open(os.path.join(labels_dir, label_file), 'r') as lf:
                        zip_file.writestr(f"{labels_dir}/{label_file}", lf.read())
                shutil.rmtree(labels_dir)
                # Create dataset.yaml
                yaml_content = f"""train: images_dict
val: images
nc: {len(label_list)}
names: {label_list}"""
                zip_file.writestr(f"dataset.yaml", yaml_content)

            elif format == 'coco':
                labels_dir = "labels_coco"
                os.makedirs(labels_dir, exist_ok=True)
                save_coco_labels(predictions_dict, labels_dir)
                with open(os.path.join(labels_dir, 'annotations.json'), 'r') as lf:
                    zip_file.writestr(f"{labels_dir}/annotations.json", lf.read())
                shutil.rmtree(labels_dir)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def convert_pil_to_cv_image(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


def convert_cv_to_pil_image(cv_image):
    pil_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(pil_image)
    return pil_image


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(org_img, scores, labels, boxes):
    img = org_img.copy()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        img = cv2.rectangle(
            img=img,
            pt1=(int(xmin), int(ymin)),
            pt2=(int(xmax), int(ymax)),
            color=c,
            thickness=max(1, int(img.shape[1] / 750)),
        )
        img = cv2.putText(
            img=img,
            text=f"{label}{score:.2f}",
            org=(int(xmin), int(ymin) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=c,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return temp_dir, file_paths

