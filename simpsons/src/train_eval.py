# src/train_eval.py
import os, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

# Resolve raiz do projeto preferindo a pasta do script; só cai para o CWD se o script não tiver data/simpsons
def resolve_base_dir():
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "data" / "simpsons").exists():
        return script_root
    if (cwd / "data" / "simpsons").exists():
        return cwd
    return script_root

BASE_DIR = resolve_base_dir()
DATA_DIR = BASE_DIR / "data" / "simpsons"
OUTPUTS_DIR = BASE_DIR / "outputs"
IMG_SIZE = (96, 96)   # 96x96 p/ agilizar
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys")

def load_dataset(root):
    """
    Lê imagens em:
        root/Train/<classe>/*.bmp
        root/Valid/<classe>/*.bmp
    e retorna (X_train, y_train), (X_valid, y_valid), class_names.
    """
    root = Path(root)
    data = {"Train": {"X": [], "y": []}, "Valid": {"X": [], "y": []}}

    train_dir = root / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Pasta de treino não encontrada: {train_dir}. Execute o script a partir da raiz do projeto 'simpsons' e verifique a estrutura data/simpsons/Train.")

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print("Classes detectadas:", class_names)

    for split in ["Train", "Valid"]:
        split_path = root / split
        print(f"Lendo imagens de: {split_path}")
        for label, cls in enumerate(class_names):
            class_dir = split_path / cls
            for imgp in glob.glob(str(class_dir / "*")):
                try:
                    img = Image.open(imgp).convert("RGB").resize(IMG_SIZE)
                    data[split]["X"].append(np.array(img))
                    data[split]["y"].append(label)
                except Exception:
                    pass

    X_train = np.array(data["Train"]["X"])
    y_train = np.array(data["Train"]["y"])
    X_valid = np.array(data["Valid"]["X"])
    y_valid = np.array(data["Valid"]["y"])
    return (X_train, y_train), (X_valid, y_valid), class_names


def extract_features(images, use_lbp=False):
    feats = []
    for img in images:
        # HOG (skimage espera escala [0..1] e canal único)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32)/255.0
        hog_vec = hog(gray_f, **HOG_PARAMS)

        # Histograma de cores (RGB, 8 bins por canal => 24)
        hist = []
        for c in range(3):
            h = cv2.calcHist([img],[c],None,[8],[0,256]).flatten()
            h = h / (h.sum() + 1e-8)
            hist.append(h)
        color_hist = np.concatenate(hist, axis=0)

        feat = np.concatenate([hog_vec, color_hist], axis=0)
        feats.append(feat)
    return np.vstack(feats)

def build_ensemble_estimators():
    """
    Cria 20 estimadores com hiperparâmetros variados para composição do ensemble (voting).
    """
    estimators = []

    # KNN (5 variações)
    estimators.append(("knn3", KNeighborsClassifier(n_neighbors=3, weights="uniform")))
    estimators.append(("knn5", KNeighborsClassifier(n_neighbors=5, weights="uniform")))
    estimators.append(("knn7", KNeighborsClassifier(n_neighbors=7, weights="distance")))
    estimators.append(("knn9", KNeighborsClassifier(n_neighbors=9, weights="distance")))
    estimators.append(("knn11", KNeighborsClassifier(n_neighbors=11, weights="distance")))

    # SVM RBF (5 variações, probabilidade ligada p/ voting soft)
    estimators.append(("svm_c1", SVC(kernel="rbf", C=1, gamma="scale", probability=True)))
    estimators.append(("svm_c10", SVC(kernel="rbf", C=10, gamma="scale", probability=True)))
    estimators.append(("svm_c100_g1e3", SVC(kernel="rbf", C=100, gamma=1e-3, probability=True)))
    estimators.append(("svm_c50_g1e3", SVC(kernel="rbf", C=50, gamma=1e-3, probability=True)))
    estimators.append(("svm_c20_g1e4", SVC(kernel="rbf", C=20, gamma=1e-4, probability=True)))

    # Random Forest (4 variações)
    estimators.append(("rf100", RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)))
    estimators.append(("rf200_d20", RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)))
    estimators.append(("rf300_d10", RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)))
    estimators.append(("rf400", RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)))

    # Decision Tree (3 variações)
    estimators.append(("dt_full", DecisionTreeClassifier(random_state=42)))
    estimators.append(("dt_d10", DecisionTreeClassifier(max_depth=10, random_state=42)))
    estimators.append(("dt_d20", DecisionTreeClassifier(max_depth=20, random_state=42)))

    # MLP (3 variações)
    estimators.append(("mlp100", MLPClassifier(hidden_layer_sizes=(100,), max_iter=400, random_state=42)))
    estimators.append(("mlp120_60", MLPClassifier(hidden_layer_sizes=(120, 60), max_iter=400, random_state=42)))
    estimators.append(("mlp80_40_20", MLPClassifier(hidden_layer_sizes=(80, 40, 20), max_iter=400, random_state=42)))

    return estimators


def evaluate_with_cv(X_train, y_train, X_valid, y_valid, class_names):
    results = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # k=10 exigido
    classifiers = {
        "knn": (KNeighborsClassifier(), {"clf__n_neighbors":[3,5,7,9], "clf__weights":["uniform","distance"]}),
        "svm_rbf": (SVC(probability=True), {"clf__C":[1,10,100], "clf__gamma":["scale", 1e-3, 1e-4], "clf__kernel":["rbf"]}),
        "decision_tree": (DecisionTreeClassifier(random_state=42), {"clf__max_depth":[None,10,20,30], "clf__min_samples_split":[2,4]}),
        "random_forest": (RandomForestClassifier(n_jobs=-1, random_state=42), {"clf__n_estimators":[100,200,300], "clf__max_depth":[None,15,25]}),
        "mlp": (MLPClassifier(max_iter=500, random_state=42), {"clf__hidden_layer_sizes":[(100,), (120,60), (80,40,20)], "clf__alpha":[1e-4, 1e-3]})
    }

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    for name, (est, grid) in classifiers.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
        gcv = GridSearchCV(pipe, grid, cv=skf, n_jobs=-1, scoring="f1_macro", refit=True, verbose=0)
        gcv.fit(X_train, y_train)

        # CV metrics (best model)
        cv_best_f1 = gcv.best_score_

        # Hold-out validation (Valid split)
        y_valid_pred = gcv.predict(X_valid)
        val_acc = accuracy_score(y_valid, y_valid_pred)
        val_f1 = f1_score(y_valid, y_valid_pred, average="macro")
        val_prec = precision_score(y_valid, y_valid_pred, average="macro", zero_division=0)
        val_rec = recall_score(y_valid, y_valid_pred, average="macro", zero_division=0)
        val_cm = confusion_matrix(y_valid, y_valid_pred)

        results.append({
            "model": name,
            "best_params": gcv.best_params_,
            "cv_best_f1_macro": cv_best_f1,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "val_precision_macro": val_prec,
            "val_recall_macro": val_rec,
            "val_cm": val_cm.tolist()
        })

        # salva matriz de confusão validação
        plt.figure(figsize=(6,5))
        plt.imshow(val_cm, interpolation="nearest")
        plt.title(f"Confusion Matrix (Valid) - {name}")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(OUTPUTS_DIR / f"confusion_{name}.png", bbox_inches="tight")
        plt.close()

        # relatório de validação
        report = classification_report(y_valid, y_valid_pred, target_names=class_names, digits=4)
        with open(OUTPUTS_DIR / f"report_{name}_valid.txt","w") as f:
            f.write(report)

    # Ensemble com 20 classificadores (voting soft)
    ensemble_estimators = build_ensemble_estimators()
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
    ensemble_pipe = Pipeline([("scaler", StandardScaler()), ("ensemble", ensemble)])

    # CV do ensemble
    ensemble_cv_pred = cross_val_predict(ensemble_pipe, X_train, y_train, cv=skf, n_jobs=-1)
    ensemble_cv_f1 = f1_score(y_train, ensemble_cv_pred, average="macro")

    # Treino completo + validação
    ensemble_pipe.fit(X_train, y_train)
    ensemble_val_pred = ensemble_pipe.predict(X_valid)
    ensemble_val_acc = accuracy_score(y_valid, ensemble_val_pred)
    ensemble_val_f1 = f1_score(y_valid, ensemble_val_pred, average="macro")
    ensemble_val_prec = precision_score(y_valid, ensemble_val_pred, average="macro", zero_division=0)
    ensemble_val_rec = recall_score(y_valid, ensemble_val_pred, average="macro", zero_division=0)
    ensemble_val_cm = confusion_matrix(y_valid, ensemble_val_pred)

    results.append({
        "model": "ensemble_voting_soft_20",
        "best_params": "predefined set (20 estimadores variados)",
        "cv_best_f1_macro": ensemble_cv_f1,
        "val_accuracy": ensemble_val_acc,
        "val_f1_macro": ensemble_val_f1,
        "val_precision_macro": ensemble_val_prec,
        "val_recall_macro": ensemble_val_rec,
        "val_cm": ensemble_val_cm.tolist()
    })

    plt.figure(figsize=(6,5))
    plt.imshow(ensemble_val_cm, interpolation="nearest")
    plt.title("Confusion Matrix (Valid) - ensemble_voting_soft_20")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(OUTPUTS_DIR / "confusion_ensemble.png", bbox_inches="tight")
    plt.close()

    report = classification_report(y_valid, ensemble_val_pred, target_names=class_names, digits=4)
    with open(OUTPUTS_DIR / "report_ensemble_valid.txt","w") as f:
        f.write(report)


    pd.DataFrame(results).drop(columns=["val_cm"]).to_csv(OUTPUTS_DIR / "metrics_summary.csv", index=False)
    with open(OUTPUTS_DIR / "results.json","w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    (X_train_img, y_train), (X_valid_img, y_valid), class_names = load_dataset(DATA_DIR)
    print(f"Train: {X_train_img.shape}, Valid: {X_valid_img.shape}, classes: {len(class_names)} -> {class_names}")

    X_train = extract_features(X_train_img)
    X_valid = extract_features(X_valid_img)
    print(f"Feature dim (train): {X_train.shape}, (valid): {X_valid.shape}")

    results = evaluate_with_cv(X_train, y_train, X_valid, y_valid, class_names)
    print(f"OK! Veja {OUTPUTS_DIR}/ para métricas, relatórios e matrizes de confusão (inclui ensemble com 20 classificadores).")
