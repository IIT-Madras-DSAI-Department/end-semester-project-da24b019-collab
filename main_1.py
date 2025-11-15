import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import cdist
from algorithms import KNearestNeighbors, BaggingClassifier, calculate_f1_score  



# ------------------------------------------------------------
# PCA (SVD Implementation)
# ------------------------------------------------------------
def apply_pca(X_train, X_val, n_components=100):
    print(f"\nApplying PCA with {n_components} components...")

    # Center data
    mean_vec = np.mean(X_train, axis=0)
    X_center = X_train - mean_vec

    # SVD decomposition
    U, S, Vt = np.linalg.svd(X_center, full_matrices=False)

    # Select top principal components
    components = Vt[:n_components]

    # Project data
    X_train_pca = X_center.dot(components.T)
    X_val_pca = (X_val - mean_vec).dot(components.T)

    print(f"PCA completed: {X_train_pca.shape}, {X_val_pca.shape}")

    return X_train_pca, X_val_pca



# ------------------------------------------------------------
# Load and preprocess MNIST data
# ------------------------------------------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    Y = df['label'].to_numpy()

    drop_cols = ['label']
    if 'even' in df.columns:
        drop_cols.append('even')
        print(f"Dropping column 'even' from {file_path}")

    X = df.drop(columns=drop_cols, axis=1).to_numpy()

    return X / 255.0, Y



# ------------------------------------------------------------
# Bias–Variance Evaluation WITH PCA
# ------------------------------------------------------------
def evaluate_bias_variance(
        X_train, Y_train, X_val, Y_val,
        BASE_K=5, N_ESTIMATORS=7, M_runs=3,
        PCA_COMPONENTS=100):

    print("\n===== Evaluating Bias–Variance Trade-off =====")

    # Apply PCA only once
    X_train_pca, X_val_pca = apply_pca(X_train, X_val, PCA_COMPONENTS)

    all_preds = []

    for i in range(M_runs):
        print(f"  -> Training repeat {i+1}/{M_runs}")

        base_model = KNearestNeighbors(k=BASE_K)
        bagger = BaggingClassifier(
            base_estimator=base_model,
            n_estimators=N_ESTIMATORS,
            sample_size_ratio=0.7,
            random_state=200 + i
        )

        bagger.fit(X_train_pca, Y_train)
        pred = bagger.predict(X_val_pca)
        all_preds.append(pred)

    all_preds = np.array(all_preds)   # (M_runs, N_samples)

    # Majority vote (average model)
    def vote(arr):
        return np.argmax(np.bincount(arr, minlength=10))

    mean_pred = np.apply_along_axis(vote, 0, all_preds)

    # ---- Bias ----
    bias = np.mean(mean_pred != Y_val)

    # ---- Variance ----
    variance_vals = []
    for j in range(len(Y_val)):
        disagreement = np.sum(all_preds[:, j] != mean_pred[j])
        variance_vals.append(disagreement / M_runs)

    variance = np.mean(variance_vals)

    # ---- Total Error ----
    total_error = np.mean(all_preds != Y_val)

    return bias, variance



# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
start_time = time.time()
print("Starting MNIST Data Load...")

try:
    X_train, Y_train = load_and_preprocess_data("MNIST_train.csv")
    X_val,   Y_val   = load_and_preprocess_data("MNIST_validation.csv")

    print("Data Loaded Successfully.")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")



    # ---------------- PCA ----------------
    PCA_COMPONENTS = 100
    X_train_pca, X_val_pca = apply_pca(X_train, X_val, PCA_COMPONENTS)



    # ---------------- TRAIN ENSEMBLE ----------------
    BASE_K = 5
    N_ESTIMATORS = 10
    base = KNearestNeighbors(k=BASE_K)

    ensemble = BaggingClassifier(
        base_estimator=base,
        n_estimators=N_ESTIMATORS,
        sample_size_ratio=0.8,
        random_state=42
    )

    print("\nTraining Final Bagging Classifier...")
    ensemble.fit(X_train_pca, Y_train)
    train_time = time.time() - start_time
    print(f"Training Complete in {train_time:.2f} seconds.")



    # ---------------- VALIDATION ----------------
    print("\nPredicting on Validation Set...")
    Y_pred = ensemble.predict(X_val_pca)

    macro_f1, cm = calculate_f1_score(Y_val, Y_pred)

    print("\n===== SYSTEM PERFORMANCE =====")
    print(f"KNN (k={BASE_K}) + Bagging ({N_ESTIMATORS})")
    print(f"PCA Components: {PCA_COMPONENTS}")
    print(f"Training Time: {train_time:.2f} sec")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)



    # ---------------- BIAS–VARIANCE ----------------
    bias, variance= evaluate_bias_variance(
        X_train, Y_train, X_val, Y_val,
        BASE_K=BASE_K,
        N_ESTIMATORS=N_ESTIMATORS,
        M_runs=8,
        PCA_COMPONENTS=PCA_COMPONENTS
    )

    print("\n===== BIAS–VARIANCE RESULTS =====")
    print(f"Bias Error:     {bias:.4f}")
    print(f"Variance Error: {variance:.4f}")




except Exception as e:
    print("Error during execution:", e)

   

