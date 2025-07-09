import cv2
import numpy as np
import os
import csv
import math
import albumentations as A
from zernike import compute_zernike_moments  # ← necesitas esta función
import mahotas

def compute_zernike_moments(image, radius=21, degree=9):
    cy, cx = np.array(image.shape) // 2
    radius = min(cx, cy, radius)
    return mahotas.features.zernike_moments(image, radius=radius, degree=degree)

# Transformaciones
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.3)
])

base_path = 'dataset'
output_file = '../src/main/assets/momentos_zernike_dataset.csv'

def binarizar_y_normalizar(img):
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if np.mean(bin_img) > 127:
        bin_img = 255 - bin_img
    return bin_img


def centrar_figura(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bin_img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    figura = bin_img[y:y+h, x:x+w]
    canvas = np.zeros_like(bin_img)
    cx, cy = canvas.shape[1] // 2, canvas.shape[0] // 2
    fx, fy = figura.shape[1] // 2, figura.shape[0] // 2
    start_x = cx - fx
    start_y = cy - fy
    canvas[start_y:start_y+figura.shape[0], start_x:start_x+figura.shape[1]] = figura
    return canvas



with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['label'] + [f'z{i+1}' for i in range(15)])  # Ejemplo: 15 Zernike moments

    for label in os.listdir(base_path):
        path_label = os.path.join(base_path, label)
        if not os.path.isdir(path_label):
            continue

        for img_name in os.listdir(path_label):
            img_path = os.path.join(path_label, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            bin_img = binarizar_y_normalizar(img)

            # Original
            bin_img = binarizar_y_normalizar(img)
            bin_img = centrar_figura(bin_img)
            zernike = compute_zernike_moments(bin_img)
            if not np.any(zernike): continue
            writer.writerow([label] + list(zernike))

            # Aumentos
            for _ in range(5):
                color = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
                aug = transform(image=color)['image']
                gray_aug = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY)
                bin_aug = binarizar_y_normalizar(gray_aug)
                zernike_aug = compute_zernike_moments(bin_aug)
                writer.writerow([label] + list(zernike_aug))
