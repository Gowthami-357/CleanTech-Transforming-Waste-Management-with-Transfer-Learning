import tensorflow as tf
import os

# 📁 Set your dataset folder path here
data_dir = r"C:\Users\DELL\Desktop\project\Dataset"  # Folder must contain subfolders for each class

# ✅ Load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# ✅ Print confirmation info to terminal
print("✅ Classes found:", train_ds.class_names)
print("✅ Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("✅ Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())

for images, labels in train_ds.take(1):
    print("✅ First batch loaded")
    print(" - Image batch shape:", images.shape)
    print(" - Label batch shape:", labels.shape)
    print(" - First 5 labels:", labels[:5].numpy())
    