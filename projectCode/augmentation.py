import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def save_original_image(img, output_path, img_name):
    cv2.imwrite(os.path.join(output_path, f"original_{img_name}"), img)

def apply_random_rotation(img, output_path, img_name):
    angle = random.uniform(-20, 20)
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(output_path, f"rotated_{img_name}"), rotated_img)

def apply_random_translation(img, output_path, img_name):
    height, width = img.shape[:2]
    tx = random.uniform(-0.2 * width, 0.2 * width)
    ty = random.uniform(-0.2 * height, 0.2 * height)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(output_path, f"translated_{img_name}"), translated_img)

def apply_random_illumination(img, output_path, img_name):
    illumination_factor = random.uniform(0.5, 1.5)
    illuminated_img = cv2.convertScaleAbs(img, alpha=illumination_factor, beta=0)
    cv2.imwrite(os.path.join(output_path, f"illuminated_{img_name}"), illuminated_img)

def apply_random_scaling(img, output_path, img_name):
    height, width = img.shape[:2]
    scale = random.uniform(0.5, 1.5)
    scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Crop or pad to maintain original size
    if scale > 1.0:  # Crop the image
        scaled_img = scaled_img[:height, :width]
    else:  # Pad the image
        delta_h = height - scaled_img.shape[0]
        delta_w = width - scaled_img.shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        scaled_img = cv2.copyMakeBorder(scaled_img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)

    cv2.imwrite(os.path.join(output_path, f"scaled_{img_name}"), scaled_img)

def augment_images(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    # Loop through each class directory
    for class_dir in os.listdir(input_root):
        input_class_dir = os.path.join(input_root, class_dir)
        output_class_dir = os.path.join(output_root, class_dir)
        os.makedirs(output_class_dir, exist_ok=True)

        if not os.path.isdir(input_class_dir):
            continue

        for img_name in tqdm(os.listdir(input_class_dir), desc=f"Processing {class_dir}"):
            img_path = os.path.join(input_class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Apply augmentations
            save_original_image(img, output_class_dir, img_name)
            apply_random_rotation(img, output_class_dir, img_name)
            apply_random_translation(img, output_class_dir, img_name)
            apply_random_illumination(img, output_class_dir, img_name)
            apply_random_scaling(img, output_class_dir, img_name)

