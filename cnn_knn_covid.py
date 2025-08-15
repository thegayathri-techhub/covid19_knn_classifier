<<<<<<< HEAD
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Constants
IMG_SIZE = 128
DATASET_PATH = 'dataset/'
CATEGORIES = ['COVID', 'NORMAL', 'PNEUMONIA', 'LUNG_OPACITY']
USE_MASKS = True  # Set to False if you don't want mask preprocessing
APPLY_AUGMENTATION = True

# Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

def load_images():
    data, labels = [], []
    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category, 'images')
        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)
                print(f"Loading {img_path}")  # Debug line
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load {img_path}")  # Debug line for failed loads
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error loading {file}: {e}")  # Error handling
                continue
    return np.array(data), np.array(labels)

    #data, labels = [], []
    #for category in CATEGORIES:
     #   img_folder = os.path.join(DATASET_PATH, category, 'images')
      #  mask_folder = os.path.join(DATASET_PATH, category, 'masks')

       # for file in os.listdir(img_folder):
        #    try:
         #       img_path = os.path.join(img_folder, file)
          #      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
           #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            #    if USE_MASKS:
             #       mask_path = os.path.join(mask_folder, file)
              #      if os.path.exists(mask_path):
               #         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                #        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                 #       img = cv2.bitwise_and(img, mask)

                #if APPLY_AUGMENTATION:
                 #   img = datagen.random_transform(img)

                #data.append(img)
                #labels.append(category)
            #except:
             #   continue
    #return np.array(data), np.array(labels)

def build_feature_extractor():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten()
    ])
    return model

def extract_features(model, X):
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    features = model.predict(X, verbose=0)
    return features

def plot_confusion_matrix(cm, classes, fold_num):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Fold {fold_num}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold_{fold_num}.png')
    plt.close()

def show_sample_predictions(X_test, y_test, y_pred, label_encoder, fold_num):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    indices = np.random.choice(len(X_test), size=9, replace=False)

    for ax, idx in zip(axes.flatten(), indices):
        img = X_test[idx]
        true_label = label_encoder.inverse_transform([y_test[idx]])[0]
        pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Sample Predictions - Fold {fold_num}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'sample_predictions_fold_{fold_num}.png')
    plt.close()

def create_pdf_report(fold_num):
    pdf_path = f"fold_{fold_num}_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 770, f"Fold {fold_num} Report")

    # Confusion Matrix
    c.setFont("Helvetica", 12)
    c.drawString(50, 730, "Confusion Matrix:")
    c.drawImage(ImageReader(f"confusion_matrix_fold_{fold_num}.png"), 50, 450, width=500, preserveAspectRatio=True)

    # Sample Predictions
    c.drawString(50, 420, "Sample Predictions:")
    c.drawImage(ImageReader(f"sample_predictions_fold_{fold_num}.png"), 50, 150, width=500, preserveAspectRatio=True)

    c.save()

def plot_fold_metrics(accs, precs, recs):
    folds = np.arange(1, len(accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(folds, accs, label='Accuracy', marker='o')
    plt.plot(folds, precs, label='Precision', marker='s')
    plt.plot(folds, recs, label='Recall', marker='^')
    plt.xticks(folds)
    plt.title('Fold-wise Performance Metrics')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fold_metrics.png')
    plt.close()

def hybrid_cnn_knn(X, y, n_neighbors=3, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    fold = 1
    all_reports = []

    accuracies, precisions, recalls = [], [], []

    for train_idx, test_idx in kf.split(X):
        print(f"\n📁 Fold {fold}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        feature_extractor = build_feature_extractor()
        feature_extractor.compile(optimizer='adam', loss='mse')  # Dummy compile

        train_features = extract_features(feature_extractor, X_train)
        test_features = extract_features(feature_extractor, X_test)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_features, y_train)
        y_pred = knn.predict(test_features)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        plot_confusion_matrix(cm, label_encoder.classes_, fold)
        show_sample_predictions(X_test, y_test, y_pred, label_encoder, fold)
        create_pdf_report(fold)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)
        all_reports.append(report)
        fold += 1

    plot_fold_metrics(accuracies, precisions, recalls)
    return all_reports

if __name__ == '__main__':
    print("🔍 Loading images...")
    X, y = load_images()
    print(f"✅ Loaded {len(X)} images.")
    reports = hybrid_cnn_knn(X, y, n_neighbors=3, k_folds=5)
=======
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Constants
IMG_SIZE = 128
DATASET_PATH = 'dataset/'
CATEGORIES = ['COVID', 'NORMAL', 'PNEUMONIA', 'LUNG_OPACITY']
USE_MASKS = True  # Set to False if you don't want mask preprocessing
APPLY_AUGMENTATION = True

# Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

def load_images():
    data, labels = [], []
    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category, 'images')
        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)
                print(f"Loading {img_path}")  # Debug line
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load {img_path}")  # Debug line for failed loads
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error loading {file}: {e}")  # Error handling
                continue
    return np.array(data), np.array(labels)

    #data, labels = [], []
    #for category in CATEGORIES:
     #   img_folder = os.path.join(DATASET_PATH, category, 'images')
      #  mask_folder = os.path.join(DATASET_PATH, category, 'masks')

       # for file in os.listdir(img_folder):
        #    try:
         #       img_path = os.path.join(img_folder, file)
          #      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
           #     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            #    if USE_MASKS:
             #       mask_path = os.path.join(mask_folder, file)
              #      if os.path.exists(mask_path):
               #         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                #        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                 #       img = cv2.bitwise_and(img, mask)

                #if APPLY_AUGMENTATION:
                 #   img = datagen.random_transform(img)

                #data.append(img)
                #labels.append(category)
            #except:
             #   continue
    #return np.array(data), np.array(labels)

def build_feature_extractor():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten()
    ])
    return model

def extract_features(model, X):
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    features = model.predict(X, verbose=0)
    return features

def plot_confusion_matrix(cm, classes, fold_num):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Fold {fold_num}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold_{fold_num}.png')
    plt.close()

def show_sample_predictions(X_test, y_test, y_pred, label_encoder, fold_num):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    indices = np.random.choice(len(X_test), size=9, replace=False)

    for ax, idx in zip(axes.flatten(), indices):
        img = X_test[idx]
        true_label = label_encoder.inverse_transform([y_test[idx]])[0]
        pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Sample Predictions - Fold {fold_num}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'sample_predictions_fold_{fold_num}.png')
    plt.close()

def create_pdf_report(fold_num):
    pdf_path = f"fold_{fold_num}_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 770, f"Fold {fold_num} Report")

    # Confusion Matrix
    c.setFont("Helvetica", 12)
    c.drawString(50, 730, "Confusion Matrix:")
    c.drawImage(ImageReader(f"confusion_matrix_fold_{fold_num}.png"), 50, 450, width=500, preserveAspectRatio=True)

    # Sample Predictions
    c.drawString(50, 420, "Sample Predictions:")
    c.drawImage(ImageReader(f"sample_predictions_fold_{fold_num}.png"), 50, 150, width=500, preserveAspectRatio=True)

    c.save()

def plot_fold_metrics(accs, precs, recs):
    folds = np.arange(1, len(accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(folds, accs, label='Accuracy', marker='o')
    plt.plot(folds, precs, label='Precision', marker='s')
    plt.plot(folds, recs, label='Recall', marker='^')
    plt.xticks(folds)
    plt.title('Fold-wise Performance Metrics')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fold_metrics.png')
    plt.close()

def hybrid_cnn_knn(X, y, n_neighbors=3, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    fold = 1
    all_reports = []

    accuracies, precisions, recalls = [], [], []

    for train_idx, test_idx in kf.split(X):
        print(f"\n📁 Fold {fold}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        feature_extractor = build_feature_extractor()
        feature_extractor.compile(optimizer='adam', loss='mse')  # Dummy compile

        train_features = extract_features(feature_extractor, X_train)
        test_features = extract_features(feature_extractor, X_test)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_features, y_train)
        y_pred = knn.predict(test_features)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        plot_confusion_matrix(cm, label_encoder.classes_, fold)
        show_sample_predictions(X_test, y_test, y_pred, label_encoder, fold)
        create_pdf_report(fold)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        print(report)
        all_reports.append(report)
        fold += 1

    plot_fold_metrics(accuracies, precisions, recalls)
    return all_reports

if __name__ == '__main__':
    print("🔍 Loading images...")
    X, y = load_images()
    print(f"✅ Loaded {len(X)} images.")
    reports = hybrid_cnn_knn(X, y, n_neighbors=3, k_folds=5)
>>>>>>> 067a9e7a842f661e5db300a9645f5f5730abcf7d
