import zipfile
import os
import tensorflow as tf

zip_path = r"C:\Users\DELL\Desktop\project\Dataset.zip"
extract_to = r"C:\Users\DELL\Desktop\project\Dataset"

# Unzip if not already done
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Dataset extracted successfully.")
else:
    print("📁 Dataset already extracted.")

# Path to extracted dataset
dataset_path = extract_to

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(180, 180),
    batch_size=32
)

# Output class names
print("✅ Images loaded successfully!")
print("Classes found:", train_ds.class_names)
