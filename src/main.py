import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# 1. DATA PREPARATION


# Générateur d'entraînement (avec augmentation)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1
    # horizontal_flip=False (important en médical)
)

# Générateur validation/test (sans augmentation)
val_test_gen = ImageDataGenerator(rescale=1./255)

# Chargement des données
train_data = train_gen.flow_from_directory(
    'chest_xray/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_data = val_test_gen.flow_from_directory(
    'chest_xray/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_data = val_test_gen.flow_from_directory(
    'chest_xray/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False   # IMPORTANT pour évaluation
)


# 2. MODEL (CNN)


model = Sequential([

    # Bloc 1
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Bloc 2
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Bloc 3
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Classifieur
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 3. TRAINING


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stop]
)


# 4. EVALUATION


# Prédictions
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Labels réels
y_test = test_data.classes

# Rapport de classification
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonie']))

# Matrice de confusion
print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 5. VISUALISATION


# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()
