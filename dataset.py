import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv('/Users/shivasaimuthyam/Downloads/dataset_all_points.csv')

# Step 2: Explore labels (exercise types)
print("Exercise labels:", df['class'].unique())

# Step 3: Separate labels and landmark data
labels = df['class']
landmark_cols = [col for col in df.columns if col != 'class']
data = df[landmark_cols]

# Step 4: Normalize x, y, z coordinates (scale between 0 and 1 per row)
def normalize_row(row):
    coords = row.values.reshape(-1, 4)[:, :3]  # Only x, y, z (not visibility)
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    normalized = (coords - min_vals) / (max_vals - min_vals + 1e-6)
    return normalized.flatten()

normalized_data = data.apply(normalize_row, axis=1, result_type='expand')

# Step 5: Combine normalized data with labels
processed_df = pd.concat([labels, normalized_data], axis=1)
processed_df.columns = ['class'] + [f'lm{i}' for i in range(1, normalized_data.shape[1]+1)]

# Save for training later
processed_df.to_csv('/Users/shivasaimuthyam/Downloads/processed_landmarks.csv', index=False)
print("✅ Preprocessing complete! Data saved as 'processed_landmarks.csv'")

