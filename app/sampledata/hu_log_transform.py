import cv2
import numpy as np
import os
import csv
import math
import random

def hu_log_transform(hu):
    return [-1 * math.copysign(1.0, h) * math.log10(abs(h)) if h != 0 else 0 for h in hu]

def calcular_shape_signature(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return []
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return [np.sqrt((p[0][0] - cx) ** 2 + (p[0][1] - cy) ** 2) for p in contour]

def calcular_fft(signature):
    signature = np.array(signature, dtype=np.float32)
    dft = cv2.dft(signature.reshape(1, -1), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    return mag.flatten().tolist()

# === Configuración ===
base_path = 'dataset'
output_file = '../src/main/assets/momentos_hu_dataset.csv'

# === Paso 1: Mapear imágenes por clase ===
imagenes_por_clase = {}
for label in os.listdir(base_path):
    path_label = os.path.join(base_path, label)
    if not os.path.isdir(path_label):
        continue
    imagenes = [os.path.join(path_label, f) for f in os.listdir(path_label) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if imagenes:
        imagenes_por_clase[label] = imagenes

# === Paso 2: Encontrar la clase con menor cantidad de imágenes ===
min_count = min(len(imgs) for imgs in imagenes_por_clase.values())
print(f"\n🔢 Se balanceará a {min_count} imágenes por clase.")

# === Paso 3: Crear CSV ===
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['label'] + [f'hu{i+1}' for i in range(7)] + [f'fft{i+1}' for i in range(64)]
    writer.writerow(header)

    for label, imagenes in imagenes_por_clase.items():
        seleccionadas = random.sample(imagenes, min_count)
        for img_path in seleccionadas:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ No se pudo leer {img_path}")
                    continue

                _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                if np.mean(bin_img) > 127:
                    bin_img = 255 - bin_img

                contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    print(f"⚠️ Sin contornos en {img_path}")
                    continue

                filled = np.zeros_like(bin_img)
                cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

                hu = cv2.HuMoments(cv2.moments(filled)).flatten()
                hu_log = hu_log_transform(hu)

                sig = calcular_shape_signature(contours[0])
                if len(sig) < 2:
                    continue

                fft_mag = calcular_fft(sig)[:64]

                if len(hu_log) == 7 and len(fft_mag) == 64:
                    writer.writerow([label] + hu_log + fft_mag)
                    print(f"✅ {label} - Descriptor guardado")
            except Exception as e:
                print(f"❌ Error en {img_path}: {e}")

print("\n✅ Generación equilibrada completada.")
