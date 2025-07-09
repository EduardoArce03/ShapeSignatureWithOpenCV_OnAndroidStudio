import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cambia la ruta si tu CSV está en otra carpeta
csv_path = '../src/main/assets/momentos_hu_dataset.csv'

# Cargar CSV
df = pd.read_csv(csv_path)

# Validar estructura esperada
expected_columns = 1 + 7 + 64
if df.shape[1] != expected_columns:
    print(f"⚠️ El CSV tiene {df.shape[1]} columnas, se esperaban {expected_columns}")
else:
    print("✅ CSV cargado correctamente.")

# Separar columnas
df.columns = ['label'] + [f'hu{i}' for i in range(1, 8)] + [f'fft{i}' for i in range(1, 65)]

# Contar clases
print("\n🔢 Distribución de clases:")
print(df['label'].value_counts())

# Visualizar Momentos de Hu (hu1 vs hu2)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='hu1', y='hu2', hue='label', s=60)
plt.title('Momentos de Hu: hu1 vs hu2')
plt.xlabel('hu1')
plt.ylabel('hu2')
plt.grid(True)
plt.legend()
plt.show()

# Visualizar FFT (fft1 vs fft2)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='fft1', y='fft2', hue='label', s=60)
plt.title('FFT: componente 1 vs 2')
plt.xlabel('fft1')
plt.ylabel('fft2')
plt.grid(True)
plt.legend()
plt.show()
