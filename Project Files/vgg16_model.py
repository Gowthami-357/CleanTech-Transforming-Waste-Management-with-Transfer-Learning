import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==== Step 1: Paths ====
zip_path = r"C:\Users\DELL\Desktop\project\Dataset.zip"
extract_to = r"C:\Users\DELL\Desktop\project\Dataset"
final_model_path = r"C:\Users\DELL\Desktop\project\healthy_vs_rotten.h5"

# ==== Step 2: Extract Dataset ====
if not os.path.exists(extract_to):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Dataset extracted successfully.")
else:
    print("📁 Dataset already extracted.")

# ==== Step 3: Data Generators ====
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

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

# ==== Step 4: Load VGG16 Base ====
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

# ==== Step 5: Build Model ====
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==== Step 6: Train Model ====
checkpoint = ModelCheckpoint(final_model_path, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("📊 Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# ==== Step 7: Evaluate ====
print("✅ Training complete. Evaluating on validation set:")
model.evaluate(val_generator)

# ==== Step 8: Plot Accuracy ====
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ==== Step 9: Save Final Model ====
model.save(final_model_path)
print(f"💾 Model saved as '{final_model_path}'")

# ==== Step 10: Predict Specific Images & Save to CSV ====

# Labels list (as fixed list)
labels = [0, 1, 2]

# Image paths for prediction
test_images = [
    r"C:\Users\DELL\Desktop\project\test\cardboard127.jpeg",
    r"C:\Users\DELL\Desktop\project\test\TEST_BIODEG_HFL_100.jpeg",
    r"C:\Users\DELL\Desktop\project\test\TRAIN.2_BIODEG_ORI_1036.jpg"
]

# Load model
model = tf.keras.models.load_model(final_model_path)
results = []

for img_path in test_images:
    if os.path.exists(img_path):
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            predicted_label = labels[predicted_class]

            results.append({"filename": os.path.basename(img_path), "prediction": predicted_label})

            plt.imshow(img)
            plt.title(f"Predicted Label: {predicted_label}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    else:
        print(f"⚠️ File not found: {img_path}")

# Save results to CSV
output_csv = r"C:\Users\DELL\Desktop\project\test\predictions.csv"
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"📝 Predictions saved to: {output_csv}")
