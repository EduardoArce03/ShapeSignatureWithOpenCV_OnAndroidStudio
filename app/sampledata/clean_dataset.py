import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ruta del CSV original y el archivo normalizado de salida
input_csv = "momentos_hu_dataset.csv"
output_csv = "momentos_hu_dataset.csv"

# Cargar dataset
df = pd.read_csv(input_csv)

# Separar columnas
label_col = df.columns[0]
hu_cols = df.columns[1:8]    # 7 Momentos de Hu
fft_cols = df.columns[8:]    # 64 componentes FFT

# Normalizar Momentos de Hu (ya están en escala logarítmica, se normalizan de todas formas)
scaler_hu = MinMaxScaler()
df[hu_cols] = scaler_hu.fit_transform(df[hu_cols])

# Normalizar FFT
scaler_fft = MinMaxScaler()
df[fft_cols] = scaler_fft.fit_transform(df[fft_cols])

# Guardar dataset normalizado
df.to_csv(output_csv, index=False)
print(f"✅ Dataset normalizado guardado como: {output_csv}")
