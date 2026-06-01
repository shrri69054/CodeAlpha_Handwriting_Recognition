# =============================================================
# TASK 3: Handwritten Character Recognition — CodeAlpha ML Internship
# =============================================================
# Objective  : Identify handwritten digits AND characters
# Datasets   : MNIST (digits 0-9) + EMNIST Letters (A-Z)
# Model      : Convolutional Neural Network (CNN)
# =============================================================
# Install requirements:
#   pip install tensorflow scikit-learn matplotlib seaborn
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)


# ------------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------------
def load_mnist():
    """Load MNIST digit dataset (28x28 grayscale, 10 classes)."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]
    print(f"  MNIST  — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, [str(i) for i in range(10)]


def load_emnist_letters():
    """
    Load EMNIST Letters via tensorflow_datasets.
    Falls back to a synthetic demo if the package is unavailable.
    """
    try:
        import tensorflow_datasets as tfds
        ds_train, ds_test = tfds.load(
            "emnist/letters", split=["train", "test"], as_supervised=True
        )
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.image.transpose(image)   # EMNIST needs transposing
            label = label - 1                    # Labels are 1-26 → 0-25
            return image, label

        train_data = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
        test_data  = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

        X_train = np.concatenate([x.numpy() for x, _ in train_data])
        y_train = np.concatenate([y.numpy() for _, y in train_data])
        X_test  = np.concatenate([x.numpy() for x, _ in test_data])
        y_test  = np.concatenate([y.numpy() for _, y in test_data])
        labels  = [chr(ord("A") + i) for i in range(26)]
        print(f"  EMNIST — Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test, labels

    except ImportError:
        print("  ⚠  tensorflow_datasets not found — generating synthetic letter data.")
        print("     Install with: pip install tensorflow-datasets")
        return generate_synthetic_letters()


def generate_synthetic_letters(n_train=13000, n_test=2600):
    """
    Synthetic 28×28 grayscale 'letter' images (random patches per class).
    Used as demo fallback only.
    """
    np.random.seed(42)
    n_classes = 26
    labels = [chr(ord("A") + i) for i in range(n_classes)]

    def make_images(n):
        X, y = [], []
        per_class = n // n_classes
        for cls in range(n_classes):
            for _ in range(per_class):
                img = np.random.rand(28, 28, 1).astype("float32") * 0.3
                # Add a class-specific pattern
                r, c = (cls // 5) * 5 + 2, (cls % 5) * 5 + 2
                img[r:r+6, c:c+6, 0] += 0.7
                img = np.clip(img, 0, 1)
                X.append(img)
                y.append(cls)
        return np.array(X), np.array(y)

    X_train, y_train = make_images(n_train)
    X_test,  y_test  = make_images(n_test)
    print(f"  Synthetic letters — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, labels


# ------------------------------------------------------------------
# 2. BUILD CNN
# ------------------------------------------------------------------
def build_cnn(input_shape, n_classes):
    """
    Two-block CNN suitable for both MNIST and EMNIST.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.25),

        # Classifier head
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ------------------------------------------------------------------
# 3. TRAIN WITH DATA AUGMENTATION
# ------------------------------------------------------------------
def train_model(model, X_train, y_train, X_test, y_test,
                epochs=30, batch_size=128, augment=True):
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=4,
                          factor=0.5, min_lr=1e-6, verbose=1),
    ]

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
        )
        datagen.fit(X_train)
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    return history


# ------------------------------------------------------------------
# 4. VISUALISATION
# ------------------------------------------------------------------
def plot_training_history(history, title="CNN Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(history.history["accuracy"],     label="Train", color="#2ecc71", lw=2)
    ax1.plot(history.history["val_accuracy"], label="Val",   color="#e74c3c", lw=2)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train", color="#3498db", lw=2)
    ax2.plot(history.history["val_loss"], label="Val",   color="#e67e22", lw=2)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    # Show only up to 20 classes for readability
    n = min(len(class_names), 20)
    mask = y_true < n
    cm_small = confusion_matrix(y_true[mask], y_pred[mask])
    sns.heatmap(cm_small, annot=(n <= 10), fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names[:n],
                yticklabels=class_names[:n])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=7)


def visualise_predictions(X_test, y_test, y_pred, class_names, n=20, title="Predictions"):
    """Show a grid of test images with predicted vs true labels."""
    indices = np.random.choice(len(X_test), n, replace=False)
    cols = 5
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for i, idx in enumerate(indices):
        ax = axes[i // cols][i % cols]
        ax.imshow(X_test[idx].squeeze(), cmap="gray")
        pred  = class_names[y_pred[idx]]
        truth = class_names[y_test[idx]]
        color = "green" if pred == truth else "red"
        ax.set_title(f"P:{pred}  T:{truth}", fontsize=8, color=color)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*64)
    print("  CodeAlpha ML — Task 3: Handwritten Character Recognition")
    print("="*64)

    # ══════════════════════════════════════════════════════════════
    # PART A: MNIST Digits
    # ══════════════════════════════════════════════════════════════
    print("\n  ── Part A: MNIST Digits (0–9) ──")
    X_train, y_train, X_test, y_test, class_names = load_mnist()

    cnn_mnist = build_cnn((28, 28, 1), n_classes=10)
    cnn_mnist.summary()

    print("\n  Training on MNIST …")
    hist_mnist = train_model(cnn_mnist, X_train, y_train,
                             X_test, y_test, epochs=30, augment=True)

    y_pred_mnist = np.argmax(cnn_mnist.predict(X_test), axis=1)
    acc_mnist    = accuracy_score(y_test, y_pred_mnist)
    print(f"\n  MNIST Test Accuracy : {acc_mnist:.2%}")
    print(classification_report(y_test, y_pred_mnist, target_names=class_names))

    # Plots
    fig = plot_training_history(hist_mnist, "MNIST CNN Training")
    fig.savefig("task3_mnist_training.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(y_test, y_pred_mnist, class_names, "Confusion Matrix — MNIST", ax)
    plt.tight_layout()
    plt.savefig("task3_mnist_confusion.png", dpi=150, bbox_inches="tight")

    fig = visualise_predictions(X_test, y_test, y_pred_mnist,
                                class_names, n=20, title="MNIST Predictions")
    fig.savefig("task3_mnist_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()

    cnn_mnist.save("task3_mnist_cnn.keras")
    print("  💾 MNIST model saved → task3_mnist_cnn.keras")

    # ══════════════════════════════════════════════════════════════
    # PART B: EMNIST Letters (A–Z)
    # ══════════════════════════════════════════════════════════════
    print("\n  ── Part B: EMNIST Letters (A–Z) ──")
    X_trL, y_trL, X_tsL, y_tsL, letter_names = load_emnist_letters()

    cnn_emnist = build_cnn((28, 28, 1), n_classes=26)

    print("\n  Training on EMNIST Letters …")
    hist_emnist = train_model(cnn_emnist, X_trL, y_trL,
                              X_tsL, y_tsL, epochs=30, augment=True)

    y_pred_emnist = np.argmax(cnn_emnist.predict(X_tsL), axis=1)
    acc_emnist    = accuracy_score(y_tsL, y_pred_emnist)
    print(f"\n  EMNIST Test Accuracy : {acc_emnist:.2%}")
    print(classification_report(y_tsL, y_pred_emnist, target_names=letter_names))

    fig = plot_training_history(hist_emnist, "EMNIST CNN Training")
    fig.savefig("task3_emnist_training.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_confusion_matrix(y_tsL, y_pred_emnist, letter_names,
                          "Confusion Matrix — EMNIST Letters", ax)
    plt.tight_layout()
    plt.savefig("task3_emnist_confusion.png", dpi=150, bbox_inches="tight")

    fig = visualise_predictions(X_tsL, y_tsL, y_pred_emnist,
                                letter_names, n=20,
                                title="EMNIST Letter Predictions")
    fig.savefig("task3_emnist_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()

    cnn_emnist.save("task3_emnist_cnn.keras")
    print("  💾 EMNIST model saved → task3_emnist_cnn.keras")

    print("\n  📊 All plots saved.")
    print("\n  ✅ Task 3 complete!\n")
