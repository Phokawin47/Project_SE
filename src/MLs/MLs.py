import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ===============================
# CONFIG (แก้ path ตรงนี้จุดเดียว)
# ===============================
BASE_PATH = r"C:\Users\ASUS\Desktop\New folder\dataset_resplit_aug_2"

# ===============================
# 1. Load YOLO Dataset
# ===============================
def load_dataset(split_path):

    split_path = Path(split_path)
    images_path = split_path / "images"
    labels_path = split_path / "labels"

    X, y = [], []

    print(f"\nLoading data from {split_path}...")

    for img_file in images_path.glob("*.[jJ][pP][gG]"):

        label_file = labels_path / (img_file.stem + ".txt")

        if not label_file.exists():
            continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue

        # ดึงข้อมูลจาก YOLO format: class x_center y_center width height
        parts = lines[0].strip().split() # เพิ่ม .strip() ตัดช่องว่างหัวท้าย
        if len(parts) < 5: # เช็คว่ามีข้อมูลครบ 5 ตัวไหม (class, x, y, w, h)
            continue
        class_id = int(float(parts[0]))
        x_c, y_c, w, h = map(float, parts[1:])

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # แปลงพิกัดจาก Normalize (0-1) เป็นพิกเซลจริง
        img_h, img_w = img.shape[:2]
        x1 = int((x_c - w/2) * img_w)
        y1 = int((y_c - h/2) * img_h)
        x2 = int((x_c + w/2) * img_w)
        y2 = int((y_c + h/2) * img_h)

        # Crop เอาเฉพาะส่วนที่เป็นมือ (ป้องกันพิกัดติดลบหรือเกินขนาดรูป)
        hand_crop = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        
        if hand_crop.size == 0:
            continue

        # นำส่วนที่ตัดมาไป Resize และทำ HOG
        img_resized = resize(hand_crop, (128, 128))
        gray = rgb2gray(img_resized)
        
        # คราวนี้ HOG จะเห็นแค่ "นิ้ว" เน้นๆ แล้วครับ
        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

        X.append(feat)
        y.append(class_id)

    print(f"Loaded {len(X)} samples")

    return np.array(X), np.array(y)

# ===============================
# 2. Train + Save Best Model
# ===============================
def train_and_save(models, name, save_dir, X_train, y_train, X_valid, y_valid):

    best_acc = 0
    best_model = None

    print(f"\nTraining {name}...")

    for model in models:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_valid, model.predict(X_valid))
        print(f"  {model} -> Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_dir, "best_model.pkl"))

    print(f"Best {name} Val Accuracy: {best_acc:.4f}")

    return best_model

# ===============================
# 3. MAIN
# ===============================
if __name__ == "__main__":

    X_train, y_train = load_dataset(os.path.join(BASE_PATH, "train"))
    X_valid, y_valid = load_dataset(os.path.join(BASE_PATH, "valid"))
    X_test,  y_test  = load_dataset(os.path.join(BASE_PATH, "test"))

    if len(X_train) == 0:
        print("❌ No training data found. Check dataset path.")
        exit()

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # ===============================
    # Model Definitions
    # ===============================
    lr_models = [
        LogisticRegression(max_iter=1000),
        LogisticRegression(C=0.1, max_iter=1000),
        LogisticRegression(C=1.0, max_iter=2000)
    ]

    knn_models = [
        KNeighborsClassifier(n_neighbors=3, weights='distance'),
        KNeighborsClassifier(n_neighbors=5)
        
    ]

    rf_models = [
        RandomForestClassifier(n_estimators=100),
        RandomForestClassifier(max_depth=20, n_estimators=300)
    ]

    svm_models = [
        SVC(kernel='rbf', C=1),
        SVC(C=10, gamma='scale', kernel='rbf')
    ]

    # ===============================
    # Train
    # ===============================
    lr_model  = train_and_save(lr_models,  "Logistic Regression", "models/logistic_regression", X_train, y_train, X_valid, y_valid)
    knn_model = train_and_save(knn_models, "kNN",                 "models/knn",               X_train, y_train, X_valid, y_valid)
    rf_model  = train_and_save(rf_models,  "Random Forest",       "models/random_forest",     X_train, y_train, X_valid, y_valid)
    svm_model = train_and_save(svm_models, "SVM",                 "models/svm",               X_train, y_train, X_valid, y_valid)

    # ===============================
    # Evaluation
    # ===============================
    models = {
        "Logistic Regression": lr_model,
        "kNN": knn_model,
        "Random Forest": rf_model,
        "SVM": svm_model
    }

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average='macro', zero_division=0)
        }

    df_results = pd.DataFrame(results).T
    print("\nTest Results:")
    print(df_results)

    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/test_results.csv")

    # ===============================
    # Plot Accuracy
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.bar(df_results.index, df_results["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("results/accuracy_comparison.png")
    plt.show()

    print("\n✅ Finished Successfully")