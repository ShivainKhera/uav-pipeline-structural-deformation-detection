import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load CSV skipping metadata and specify delimiter - likely tab '\t'
# Adjust 'skiprows' according to your metadata lines (6 assumed here)
try:
    df = pd.read_csv('DI-IP-TV-9Hz-sand_0.csv', skiprows=6, delimiter=',', header=None)
except Exception as e:
    print("Error loading with tab delimiter:", e)
    # Try whitespace delimiter fallback if above fails
    df = pd.read_csv('DI-IP-TV-9Hz-sand_2.csv', skiprows=6, delim_whitespace=True, header=None)

print("Data shape after loading:", df.shape)

# Step 2: Convert all values to numeric, coercing errors to NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Step 3: Handle NaNs by replacing forward then backward (if any present)
if df_numeric.isna().any().any():
    df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')

# Step 4: Convert to NumPy array for manipulation
matrix = df_numeric.values

print("Matrix shape:", matrix.shape)

# Step 5: Normalize matrix for visualization scaling [0,255]
normalized_matrix = 255 * (matrix - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix))
normalized_matrix = normalized_matrix.astype(np.uint8)

# Step 6: Visualize thermal image
plt.figure(figsize=(8, 6))
plt.imshow(normalized_matrix, cmap='inferno', aspect='auto')
plt.axis('off')
plt.title('Thermal Image Visualization')
plt.show()
