import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from math import ceil
import numpy as np

# Directories
train_dir = 'C:/Users/Järjestelmänvalvoja/Downloads/archive/Covid19-dataset/train'
test_dir = 'C:/Users/Järjestelmänvalvoja/Downloads/archive/Covid19-dataset/test'

# Target size for resizing images
target_size = (150, 150)

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2  # 20% of training data for validation
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

# Calculate steps per epoch
train_steps = ceil(train_generator.samples / train_generator.batch_size)
validation_steps = ceil(validation_generator.samples / validation_generator.batch_size)
test_steps = ceil(test_generator.samples / test_generator.batch_size)

# Logging
print(f'Training samples: {train_generator.samples}')
print(f'Validation samples: {validation_generator.samples}')
print(f'Test samples: {test_generator.samples}')
print(f'Steps per epoch: {train_steps}')
print(f'Validation steps: {validation_steps}')
print(f'Test steps: {test_steps}')

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Using a pretrained model (VGG16) and fine-tuning it
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

base_model.trainable = False  # Freeze the base model layers initially

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
try:
    model.summary()
except Exception as e:
    print(f'Error during model summary: {e}')

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=10,
        class_weight=class_weights,
        callbacks=[reduce_lr]
    )
except Exception as e:
    print(f'Error during training: {e}')

# Unfreeze some layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:  # Freeze the first N layers, adjust as necessary
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=15,
        class_weight=class_weights,
        callbacks=[reduce_lr]
    )
except Exception as e:
    print(f'Error during training: {e}')

# Evaluate the model
try:
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    print(f'Test accuracy: {test_acc}')
except Exception as e:
    print(f'Error during evaluation: {e}')

# Save the model
model.save('covid19_classification_model.h5')
