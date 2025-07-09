import cv2
import numpy as np
import os
import random

# === Configuración ===
input_dir = 'dataset'
output_dir = 'dataset_augmented'
variaciones_por_imagen = 10

os.makedirs(output_dir, exist_ok=True)

for clase in os.listdir(input_dir):
    path_clase = os.path.join(input_dir, clase)
    if not os.path.isdir(path_clase):
        continue

    os.makedirs(os.path.join(output_dir, clase), exist_ok=True)

    for nombre_archivo in os.listdir(path_clase):
        if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(path_clase, nombre_archivo)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        h, w = img.shape
        base_name = os.path.splitext(nombre_archivo)[0]

        for i in range(variaciones_por_imagen):
            aug = img.copy()

            # 🔄 Rotación aleatoria
            angle = random.uniform(0, 360)
            rot_mat = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            aug = cv2.warpAffine(aug, rot_mat, (w, h), borderValue=0)

            # 📦 Zoom aleatorio
            scale = random.uniform(0.8, 1.2)
            center = (w // 2, h // 2)
            zoom_mat = cv2.getRotationMatrix2D(center, 0, scale)
            aug = cv2.warpAffine(aug, zoom_mat, (w, h), borderValue=0)

            # ➡️ Traslación aleatoria
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            aug = cv2.warpAffine(aug, M_trans, (w, h), borderValue=0)

            # 💾 Guardar imagen aumentada
            out_name = f"{base_name}_aug{i}.png"
            cv2.imwrite(os.path.join(output_dir, clase, out_name), aug)

        # Guardar también la original
        cv2.imwrite(os.path.join(output_dir, clase, base_name + "_orig.png"), img)

print("\n✅ Aumentos generados en 'dataset_augmented/'")
