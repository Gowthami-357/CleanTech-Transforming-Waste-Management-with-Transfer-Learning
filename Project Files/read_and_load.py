import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ==== Step 1: Paths ====
zip_path = r"C:\Users\DELL\Desktop\project\Dataset.zip"
extract_to = r"C:\Users\DELL\Desktop\project\Dataset"
model_save_path = "waste_classifier_model.h5"

# ==== Step 2: Extract the dataset ====
if not os.path.exists(extract_to):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Dataset extracted successfully.")
else:
    print("📁 Dataset already extracted.")

# ==== Step 3: Set up data generators ====
img_size = (224, 224)  # VGG16 default input size
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    extract_to,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    extract_to,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# ==== Step 4: Load VGG16 base model ====
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze convolutional base

# ==== Step 5: Build the model ====
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# ==== Step 6: Compile the model ====
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==== Step 7: Set up callbacks ====
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ==== Step 8: Train the model ====
print("📊 Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# ==== Step 9: Evaluate the model ====
print("✅ Training complete. Evaluating on validation set:")
model.evaluate(val_generator)

# ==== Step 10: Plot accuracy ====
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ==== Step 11: Save manually as healthy_vs_rotten.h5 ====
final_model_path = "healthy_vs_rotten.h5"
model.save(final_model_path)
print(f"💾 Final trained model saved as '{final_model_path}'")

