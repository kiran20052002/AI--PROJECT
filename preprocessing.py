import os
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = r"C:\Users\avanc\Downloads\dataverse_files\HAM10000_images_part_1"
METADATA_PATH = r"C:\Users\avanc\Downloads\dataverse_files\HAM10000_metadata"
OUTPUT_DIR = r"C:\Users\avanc\Downloads\dataverse_files\processed_dataset"

# Skincare concerns and severities
CONCERNS = ['Acne', 'Aging', 'Hyperpigmentation', 'Dryness', 'Sensitive Skin']
SEVERITIES = ['mild', 'moderate', 'severe']
EXPECTED_CLASSES = [f"{concern}_{severity}" for concern in CONCERNS for severity in SEVERITIES]

# HAM10000 lesion types to skincare concerns (approximate mapping)
LESION_TO_CONCERN = {
    'akiec': 'Aging',
    'bcc': 'Sensitive Skin',
    'bkl': 'Hyperpigmentation',
    'df': 'Dryness',
    'mel': 'Hyperpigmentation',
    'nv': 'Aging',
    'vasc': 'Sensitive Skin'
}

def preprocess_dataset():
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    
    # Map lesion types to concerns and assign random severities
    metadata['concern'] = metadata['dx'].map(LESION_TO_CONCERN)
    metadata = metadata.dropna(subset=['concern'])  # Drop rows with unmapped concerns
    metadata['severity'] = np.random.choice(SEVERITIES, size=len(metadata))
    metadata['class'] = metadata['concern'] + '_' + metadata['severity']

    # Verify classes
    unique_classes = metadata['class'].unique()
    print(f"Unique classes found in metadata: {sorted(unique_classes)}")
    if set(unique_classes) != set(EXPECTED_CLASSES):
        print(f"Warning: Expected {EXPECTED_CLASSES}, but found {sorted(unique_classes)}")

    # Create output directory structure (flattened: concern_severity)
    for split in ['train', 'val']:
        for class_name in EXPECTED_CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    # Split into train and validation
    train_df, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['class'], random_state=42)

    # Copy images to new structure
    for df, split in [(train_df, 'train'), (val_df, 'val')]:
        for _, row in df.iterrows():
            image_id = row['image_id']
            class_name = row['class']
            src_path = os.path.join(DATASET_PATH, f"{image_id}.jpg")
            dst_path = os.path.join(OUTPUT_DIR, split, class_name, f"{image_id}.jpg")
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Missing image: {src_path}")

    # Verify directory structure
    train_dirs = os.listdir(os.path.join(OUTPUT_DIR, 'train'))
    print(f"Training directories created: {sorted(train_dirs)}")
    assert len(train_dirs) == 15, f"Expected 15 directories, but found {len(train_dirs)}: {train_dirs}"

if __name__ == "__main__":
    # Clean up previous processed data
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    preprocess_dataset()
    print("Dataset preprocessing complete.")