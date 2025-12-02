# src/train_eval_improved.py
import os, glob, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from collections import Counter

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Metrics
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix, classification_report)

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Balancing
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

# Feature extraction
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configuration
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
OUTPUTS_DIR = BASE_DIR / "outputs_improved"
IMG_SIZE = (128, 128)  # Aumentado para melhor resolução

# Parâmetros otimizados baseados em pesquisa
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),  # Células menores = mais detalhes
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True
)

LBP_PARAMS = dict(
    P=8,  # Número de pontos
    R=1,  # Raio
    method='uniform'
)


def augment_image(img):
    """Aplica data augmentation para criar variações da imagem"""
    augmented = [img]  # Inclui original
    
    # Flip horizontal
    augmented.append(np.fliplr(img))
    
    # Rotações pequenas
    for angle in [-15, 15]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated)
    
    # Ajustes de brilho
    pil_img = Image.fromarray(img)
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(pil_img)
        bright = np.array(enhancer.enhance(factor))
        augmented.append(bright)
    
    # Pequenos shifts
    for dx, dy in [(5, 0), (-5, 0), (0, 5), (0, -5)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(shifted)
    
    return augmented


def load_dataset_with_augmentation(root, augment_minority=True, minority_threshold=20):
    """Carrega dataset com data augmentation para classes minoritárias"""
    root = Path(root)
    data = {"Train": {"X": [], "y": []}, "Valid": {"X": [], "y": []}}
    
    train_dir = root / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Pasta de treino não encontrada: {train_dir}")
    
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print("Classes detectadas:", class_names)
    
    # Primeira passagem: contar amostras
    class_counts = {cls: 0 for cls in class_names}
    for split in ["Train"]:
        split_path = root / split
        for cls in class_names:
            class_dir = split_path / cls
            class_counts[cls] = len(list(class_dir.glob("*")))
    
    print("\nDistribuição original (Train):")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} amostras")
    
    # Carregar com augmentation
    for split in ["Train", "Valid"]:
        split_path = root / split
        print(f"\nLendo imagens de: {split_path}")
        
        for label, cls in enumerate(class_names):
            class_dir = split_path / cls
            is_minority = (split == "Train" and augment_minority and 
                          class_counts[cls] < minority_threshold)
            
            for imgp in glob.glob(str(class_dir / "*")):
                try:
                    img = Image.open(imgp).convert("RGB").resize(IMG_SIZE)
                    img_array = np.array(img)
                    
                    if is_minority:
                        # Aplicar augmentation para classes minoritárias
                        augmented = augment_image(img_array)
                        for aug_img in augmented:
                            data[split]["X"].append(aug_img)
                            data[split]["y"].append(label)
                    else:
                        data[split]["X"].append(img_array)
                        data[split]["y"].append(label)
                except Exception as e:
                    print(f"Erro ao carregar {imgp}: {e}")
                    pass
    
    X_train = np.array(data["Train"]["X"])
    y_train = np.array(data["Train"]["y"])
    X_valid = np.array(data["Valid"]["X"])
    y_valid = np.array(data["Valid"]["y"])
    
    print(f"\nApós augmentation (Train):")
    train_counter = Counter(y_train)
    for label, cls in enumerate(class_names):
        print(f"  {cls}: {train_counter[label]} amostras")
    
    return (X_train, y_train), (X_valid, y_valid), class_names


def preprocess_image(img):
    """Pré-processamento avançado da imagem"""
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    return gray_clahe, gray


def extract_lbp_features(gray_img):
    """Extrai características LBP (Local Binary Patterns)"""
    lbp = local_binary_pattern(gray_img, LBP_PARAMS['P'], 
                               LBP_PARAMS['R'], LBP_PARAMS['method'])
    
    # Histograma LBP (59 bins para 'uniform')
    n_bins = LBP_PARAMS['P'] + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, 
                          range=(0, n_bins), density=True)
    
    return hist


def extract_color_moments(img):
    """Extrai momentos de cor (média, desvio padrão, skewness)"""
    moments = []
    for channel in range(3):
        pixels = img[:, :, channel].flatten()
        moments.append(np.mean(pixels))
        moments.append(np.std(pixels))
        moments.append(np.mean((pixels - np.mean(pixels))**3)**(1/3))
    return np.array(moments)


def extract_enhanced_features(images):
    """Extração de características aprimorada"""
    all_features = []
    
    for img in images:
        # Pré-processamento
        gray_clahe, gray_original = preprocess_image(img)
        
        # 1. HOG com CLAHE
        gray_f = gray_clahe.astype(np.float32) / 255.0
        hog_vec = hog(gray_f, **HOG_PARAMS)
        
        # 2. LBP
        lbp_vec = extract_lbp_features(gray_clahe)
        
        # 3. Histograma de cores RGB (16 bins para mais detalhe)
        color_hist = []
        for c in range(3):
            hist = cv2.calcHist([img], [c], None, [16], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            color_hist.append(hist)
        color_hist = np.concatenate(color_hist)
        
        # 4. Momentos de cor
        color_moments = extract_color_moments(img)
        
        # 5. Histograma HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_hist = []
        for c in range(3):
            hist = cv2.calcHist([hsv], [c], None, [8], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            hsv_hist.append(hist)
        hsv_hist = np.concatenate(hsv_hist)
        
        # Concatenar todas as características
        feature_vector = np.concatenate([
            hog_vec,           # ~8000 features (HOG com cells menores)
            lbp_vec,           # 10 features (LBP)
            color_hist,        # 48 features (RGB 16 bins)
            color_moments,     # 9 features (momentos RGB)
            hsv_hist          # 24 features (HSV)
        ])
        
        all_features.append(feature_vector)
    
    return np.vstack(all_features)


def build_enhanced_ensemble_estimators(class_weights=None):
    """Cria ensemble aprimorado com class weights"""
    estimators = []
    
    # KNN (6 variações com pesos diferentes)
    for k in [3, 5, 7, 9, 11, 13]:
        weight = "distance" if k > 5 else "uniform"
        estimators.append((f"knn{k}", KNeighborsClassifier(
            n_neighbors=k, weights=weight, n_jobs=-1)))
    
    # SVM RBF (6 variações com class_weight)
    for C, gamma in [(1, 'scale'), (10, 'scale'), (50, 1e-3), 
                     (100, 1e-3), (100, 1e-4), (20, 1e-4)]:
        estimators.append((f"svm_c{C}_g{gamma}", SVC(
            kernel="rbf", C=C, gamma=gamma, probability=True,
            class_weight='balanced', random_state=42)))
    
    # Random Forest (5 variações com class_weight)
    for n_est, max_d in [(150, None), (200, 25), (300, 20), 
                         (400, None), (250, 15)]:
        estimators.append((f"rf{n_est}_d{max_d}", RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d, 
            class_weight='balanced', n_jobs=-1, random_state=42)))
    
    # Gradient Boosting (3 variações) - NOVO
    for lr, n_est in [(0.1, 100), (0.05, 150), (0.1, 200)]:
        estimators.append((f"gb_lr{lr}_n{n_est}", GradientBoostingClassifier(
            learning_rate=lr, n_estimators=n_est, random_state=42)))
    
    # Decision Tree (2 variações com class_weight)
    for max_d in [15, 25]:
        estimators.append((f"dt_d{max_d}", DecisionTreeClassifier(
            max_depth=max_d, class_weight='balanced', random_state=42)))
    
    # MLP (3 variações)
    for layers in [(150,), (120, 60), (100, 50, 25)]:
        estimators.append((f"mlp{'_'.join(map(str, layers))}", MLPClassifier(
            hidden_layer_sizes=layers, max_iter=500, 
            early_stopping=True, random_state=42)))
    
    return estimators


def evaluate_with_smote_and_cv(X_train, y_train, X_valid, y_valid, class_names):
    """Avaliação com SMOTE e validação cruzada"""
    results = []
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Calcular class weights
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train), 
                                         y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\nClass weights calculados: {class_weight_dict}")
    
    # Aplicar SMOTE apenas no treino
    print("\nAplicando SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Após SMOTE:")
    smote_counter = Counter(y_train_smote)
    for label, cls in enumerate(class_names):
        print(f"  {cls}: {smote_counter[label]} amostras")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Classificadores com class_weight onde aplicável
    classifiers = {
        "knn": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9, 11], 
             "clf__weights": ["uniform", "distance"]}
        ),
        "svm_rbf": (
            SVC(probability=True, class_weight='balanced'),
            {"clf__C": [1, 10, 50, 100], 
             "clf__gamma": ["scale", 1e-3, 1e-4]}
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            {"clf__max_depth": [10, 15, 20, 25, 30]}
        ),
        "random_forest": (
            RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced'),
            {"clf__n_estimators": [150, 200, 300, 400], 
             "clf__max_depth": [None, 20, 25]}
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [100, 150, 200], 
             "clf__learning_rate": [0.05, 0.1, 0.15]}
        ),
        "mlp": (
            MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
            {"clf__hidden_layer_sizes": [(150,), (120, 60), (100, 50, 25)],
             "clf__alpha": [1e-4, 1e-3, 1e-2]}
        )
    }
    
    for name, (est, grid) in classifiers.items():
        print(f"\nTreinando {name}...")
        
        # Usar RobustScaler em vez de StandardScaler (melhor para outliers)
        pipe = Pipeline([("scaler", RobustScaler()), ("clf", est)])
        
        gcv = GridSearchCV(pipe, grid, cv=skf, n_jobs=-1, 
                          scoring="f1_macro", refit=True, verbose=1)
        gcv.fit(X_train_smote, y_train_smote)
        
        # Métricas no conjunto de validação (sem SMOTE)
        y_valid_pred = gcv.predict(X_valid)
        val_acc = accuracy_score(y_valid, y_valid_pred)
        val_f1 = f1_score(y_valid, y_valid_pred, average="macro")
        val_prec = precision_score(y_valid, y_valid_pred, average="macro", zero_division=0)
        val_rec = recall_score(y_valid, y_valid_pred, average="macro", zero_division=0)
        val_cm = confusion_matrix(y_valid, y_valid_pred)
        
        results.append({
            "model": name,
            "best_params": gcv.best_params_,
            "cv_best_f1_macro": gcv.best_score_,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "val_precision_macro": val_prec,
            "val_recall_macro": val_rec,
            "val_cm": val_cm.tolist()
        })
        
        # Salvar matriz de confusão
        plt.figure(figsize=(8, 6))
        plt.imshow(val_cm, interpolation="nearest", cmap='Blues')
        plt.title(f"Confusion Matrix (Valid) - {name}\nF1={val_f1:.3f}")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Adicionar valores na matriz
        thresh = val_cm.max() / 2
        for i, j in np.ndindex(val_cm.shape):
            plt.text(j, i, format(val_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if val_cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / f"confusion_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Relatório
        report = classification_report(y_valid, y_valid_pred, 
                                      target_names=class_names, digits=4)
        with open(OUTPUTS_DIR / f"report_{name}_valid.txt", "w") as f:
            f.write(report)
        
        print(f"  Validação - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Ensemble aprimorado
    print("\n" + "="*60)
    print("Treinando ENSEMBLE APRIMORADO (25 classificadores)...")
    print("="*60)
    
    ensemble_estimators = build_enhanced_ensemble_estimators(class_weight_dict)
    ensemble = VotingClassifier(estimators=ensemble_estimators, 
                               voting="soft", n_jobs=-1)
    ensemble_pipe = Pipeline([("scaler", RobustScaler()), ("ensemble", ensemble)])
    
    # Treinar ensemble
    ensemble_pipe.fit(X_train_smote, y_train_smote)
    
    # Avaliar no conjunto de validação
    ensemble_val_pred = ensemble_pipe.predict(X_valid)
    ensemble_val_acc = accuracy_score(y_valid, ensemble_val_pred)
    ensemble_val_f1 = f1_score(y_valid, ensemble_val_pred, average="macro")
    ensemble_val_prec = precision_score(y_valid, ensemble_val_pred, 
                                       average="macro", zero_division=0)
    ensemble_val_rec = recall_score(y_valid, ensemble_val_pred, 
                                    average="macro", zero_division=0)
    ensemble_val_cm = confusion_matrix(y_valid, ensemble_val_pred)
    
    results.append({
        "model": "ensemble_voting_soft_25_improved",
        "best_params": "25 estimadores com SMOTE + class_weight + augmentation",
        "cv_best_f1_macro": "N/A",
        "val_accuracy": ensemble_val_acc,
        "val_f1_macro": ensemble_val_f1,
        "val_precision_macro": ensemble_val_prec,
        "val_recall_macro": ensemble_val_rec,
        "val_cm": ensemble_val_cm.tolist()
    })
    
    # Visualizar matriz do ensemble
    plt.figure(figsize=(10, 8))
    plt.imshow(ensemble_val_cm, interpolation="nearest", cmap='RdYlGn')
    plt.title(f"ENSEMBLE - Confusion Matrix\nAcc={ensemble_val_acc:.3f}, F1={ensemble_val_f1:.3f}", 
             fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = ensemble_val_cm.max() / 2
    for i, j in np.ndindex(ensemble_val_cm.shape):
        plt.text(j, i, format(ensemble_val_cm[i, j], 'd'),
                ha="center", va="center", fontsize=12,
                color="white" if ensemble_val_cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_ensemble_improved.png", 
               dpi=150, bbox_inches="tight")
    plt.close()
    
    # Relatório do ensemble
    report = classification_report(y_valid, ensemble_val_pred, 
                                  target_names=class_names, digits=4)
    with open(OUTPUTS_DIR / "report_ensemble_improved.txt", "w") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE FINAL - Acc: {ensemble_val_acc:.4f}, F1: {ensemble_val_f1:.4f}")
    print(f"{'='*60}\n")
    
    # Salvar resultados
    df_results = pd.DataFrame(results).drop(columns=["val_cm"])
    df_results.to_csv(OUTPUTS_DIR / "metrics_summary_improved.csv", index=False)
    
    with open(OUTPUTS_DIR / "results_improved.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Criar tabela comparativa
    print("\n" + "="*80)
    print("RESUMO COMPARATIVO (Validação)")
    print("="*80)
    print(f"{'Modelo':<35} {'Acurácia':>10} {'F1-Macro':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<35} {r['val_accuracy']:>10.4f} "
              f"{r['val_f1_macro']:>10.4f} {r['val_precision_macro']:>10.4f} "
              f"{r['val_recall_macro']:>10.4f}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("SISTEMA DE CLASSIFICAÇÃO APRIMORADO - SIMPSONS")
    print("="*80)
    print("\nMelhorias implementadas:")
    print("  ✓ Data Augmentation para classes minoritárias")
    print("  ✓ SMOTE para balanceamento adicional")
    print("  ✓ Pré-processamento com CLAHE")
    print("  ✓ Descritores adicionais: LBP, Color Moments, HSV")
    print("  ✓ HOG otimizado (pixels_per_cell=4x4)")
    print("  ✓ Class weights nos classificadores")
    print("  ✓ RobustScaler (melhor para outliers)")
    print("  ✓ Gradient Boosting adicionado")
    print("  ✓ Ensemble com 25 classificadores")
    print("="*80 + "\n")
    
    # Carregar dataset com augmentation
    (X_train_img, y_train), (X_valid_img, y_valid), class_names = \
        load_dataset_with_augmentation(DATA_DIR, augment_minority=True)
    
    print(f"\nDimensões finais:")
    print(f"  Train: {X_train_img.shape}")
    print(f"  Valid: {X_valid_img.shape}")
    print(f"  Classes: {class_names}")
    
    # Extrair características aprimoradas
    print("\nExtraindo características aprimoradas...")
    X_train = extract_enhanced_features(X_train_img)
    X_valid = extract_enhanced_features(X_valid_img)
    
    print(f"\nDimensionalidade das características:")
    print(f"  Train: {X_train.shape}")
    print(f"  Valid: {X_valid.shape}")
    
    # Treinar e avaliar
    results = evaluate_with_smote_and_cv(X_train, y_train, X_valid, y_valid, class_names)
    
    print(f"\n{'='*80}")
    print(f"CONCLUÍDO! Resultados salvos em: {OUTPUTS_DIR}/")
    print(f"{'='*80}")
    print("\nArquivos gerados:")
    print(f"  • metrics_summary_improved.csv")
    print(f"  • results_improved.json")
    print(f"  • Matrizes de confusão (PNG)")
    print(f"  • Relatórios de classificação (TXT)")
    print("="*80)