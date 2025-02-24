import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201, Xception, InceptionV3 # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from glob import glob
import cv2

def prepare_dataset(data_dir, image_size=(299, 299)):
    """
    Prepare dataset by organizing images into proper structure.

    The dataset includes four unique HER2 expression classes as determined by pathologists:

    HER2 0: No HER2 protein expression observed.
    HER2 1+: Low level of HER2 protein expression.
    HER2 2+: Ambiguous or borderline HER2 expression, difficult to classify as positive or negative.
    HER2 3+: High level of HER2 protein expression.

    These classes are crucial for guiding treatment decisions:
    - HER2-positive cases (typically HER2 3+) may benefit from HER2-targeted therapies
      like Trastuzumab or Lapatinib.
    - Accurate classification ensures effective treatment planning.

    Args:
        data_dir (str): Path to the dataset directory containing the train folder.
        image_size (tuple): Target size for image resizing (default is 299x299).

    Returns:
        train_flow (ImageDataGenerator): Training data generator.
        val_flow (ImageDataGenerator): Validation data generator.
        num_classes (int): Number of unique classes in the dataset.
    """
    print(f"Preparing dataset from {data_dir}")

    # Get all image paths
    image_paths = glob(os.path.join(data_dir, 'train', '*.png'))
    if not image_paths:
        raise ValueError(f"No images found in {os.path.join(data_dir, 'train')}")

    print(f"Found {len(image_paths)} images in {os.path.join(data_dir, 'train')}")

    # Extract HER2 levels from filenames
    images = []
    labels = []

    for idx, path in enumerate(image_paths):
        try:
            print(f"Processing image {idx + 1}/{len(image_paths)}: {path}")

            # Extract HER2 level from filename (handling "1+", "3+", etc.)
            her2_level_str = path.split('_')[-1].split('.')[0]  # Get HER2 level part
            her2_level = int(her2_level_str.rstrip('+'))  # Remove '+' and convert to int

            # HER2 Levels:
            # 0  -> No HER2 expression.
            # 1+ -> Low HER2 expression.
            # 2+ -> Borderline expression, difficult to classify.
            # 3+ -> High HER2 expression.

            # Load and preprocess image
            img = load_img(path, target_size=image_size)
            img_array = img_to_array(img)
            img_array = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)  # Bilateral Filtering for noise reduction
            img_array = img_array / 255.0  # Normalize

            images.append(img_array)
            labels.append(her2_level)

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue

    if not images:
        raise ValueError("No valid images could be processed")

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Get unique classes
    unique_classes = np.unique(y)
    print(f"Found {len(unique_classes)} unique classes: {unique_classes}")

    # Convert labels to categorical
    y = tf.keras.utils.to_categorical(y)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create data generators
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    val_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    # Create generator flows
    train_flow = train_generator.flow(X_train, y_train, batch_size=32)
    val_flow = val_generator.flow(X_val, y_val, batch_size=32)

    return train_flow, val_flow, len(unique_classes)

def create_enhanced_model(base_model, input_shape=(299, 299, 3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Ensure base model is not trainable initially
    base_model.trainable = False
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def create_ensemble_model(num_classes):
    input_shape = (299, 299, 3)

    # Base models
    densenet_base = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    googlenet_base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Enhanced models
    densenet_model = create_enhanced_model(densenet_base, input_shape, num_classes)
    xception_model = create_enhanced_model(xception_base, input_shape, num_classes)
    googlenet_model = create_enhanced_model(googlenet_base, input_shape, num_classes)

    # Ensemble
    ensemble_input = Input(shape=input_shape)
    densenet_output = densenet_model(ensemble_input)
    xception_output = xception_model(ensemble_input)
    googlenet_output = googlenet_model(ensemble_input)

    ensemble_output = tf.keras.layers.Average()([densenet_output, xception_output, googlenet_output])

    return Model(inputs=ensemble_input, outputs=ensemble_output)

def plot_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_generator, class_names):
    # Predict the values from the validation dataset
    val_images, val_labels = next(iter(val_generator))
    predictions = np.argmax(model.predict(val_images), axis=1)
    true_labels = np.argmax(val_labels, axis=1)

    # Compute the confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names)

    # Compute and print the classification report and F1 score
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

def train_model(data_dir, epochs=50):
    # Prepare data
    train_generator, val_generator, num_classes = prepare_dataset(data_dir)

    # Create and compile model
    model = create_ensemble_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    )

    return model, history

# Main script
if _name_ == "_main_":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Define directories
    base_dir = 'BCI_dataset'
    he_dir = os.path.join(base_dir, 'HE')
    ihc_dir = os.path.join(base_dir,'IHC')

    # Train on H&E images
    print("Training model on H&E images...")
    try:
        he_model, he_history = train_model(he_dir)
        print("\nH&E Training completed")

        # Plot training history
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(he_history.history['accuracy'], label='Training Accuracy')
        plt.plot(he_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy (H&E)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(he_history.history['loss'], label='Training Loss')
        plt.plot(he_history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (H&E)')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error training H&E model: {str(e)}")

    # Train on IHC images
    print("\nTraining model on IHC images...")
    try:
        ihc_model, ihc_history = train_model(ihc_dir)
        print("\nIHC Training completed")

        # Plot training history
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(ihc_history.history['accuracy'], label='Training Accuracy')
        plt.plot(ihc_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy (IHC)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(ihc_history.history['loss'], label='Training Loss')
        plt.plot(ihc_history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (IHC)')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error training IHC model: {str(e)}")
