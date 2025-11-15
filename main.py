import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import cdist
from algorithms import KNearestNeighbors, BaggingClassifier, calculate_f1_score   # If stored separately


# ------------------------------------------------------------
# Load and preprocess MNIST data
# ------------------------------------------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Extract target label
    Y = df['label'].to_numpy()

    # Drop target + drop 'even' column if present
    drop_cols = ['label']
    if 'even' in df.columns:
        drop_cols.append('even')
        print(f"Dropping column 'even' from {file_path}")

    X = df.drop(columns=drop_cols, axis=1).to_numpy()

    # Normalize pixel values
    X = X / 255.0
    return X, Y



# ------------------------------------------------------------
# Bias–Variance Evaluation Function
# ------------------------------------------------------------
def evaluate_bias_variance(X_train, Y_train, X_val, Y_val,
                           BASE_K=5, N_ESTIMATORS=7, M_runs=3):

    print("\n===== Evaluating Bias–Variance Trade-off =====")
    all_preds = []

    for i in range(M_runs):
        print(f"  -> Training repeat {i+1}/{M_runs}")

        # New models each run (different randomness)
        base_model = KNearestNeighbors(k=BASE_K)
        bagger = BaggingClassifier(
            base_estimator=base_model,
            n_estimators=N_ESTIMATORS,
            sample_size_ratio=0.7,
            random_state=100 + i
        )

        bagger.fit(X_train, Y_train)
        pred = bagger.predict(X_val)
        all_preds.append(pred)

    all_preds = np.array(all_preds)    # shape = (M_runs, N_samples)

    # Majority vote across M models
    def vote(arr):
        return np.argmax(np.bincount(arr, minlength=10))

    mean_pred = np.apply_along_axis(vote, 0, all_preds)

    # --------- Bias ----------
    bias = np.mean(mean_pred != Y_val)

    # --------- Variance ----------
    variance_vals = []
    M = M_runs
    for j in range(len(Y_val)):
        disagreement = np.sum(all_preds[:, j] != mean_pred[j])
        variance_vals.append(disagreement / M)

    variance = np.mean(variance_vals)

    # --------- Total Error ----------
    total_error = np.mean(all_preds != Y_val)

    return bias, variance, total_error



# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
start_time = time.time()
print("Starting MNIST Data Load...")

try:
    # Load train/validation MNIST CSV
    X_train, Y_train = load_and_preprocess_data("MNIST_train.csv")
    X_val,   Y_val   = load_and_preprocess_data("MNIST_validation.csv")

    print("Data Loaded Successfully.")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")



    # --------------------------------------------------------
    # TRAIN ENSEMBLE (Bagging + KNN)
    # --------------------------------------------------------
    BASE_K = 5
    N_ESTIMATORS = 7

    base = KNearestNeighbors(k=BASE_K)
    ensemble = BaggingClassifier(
        base_estimator=base,
        n_estimators=N_ESTIMATORS,
        sample_size_ratio=0.8,
        random_state=42
    )

    print("\nTraining Final Bagging Classifier...")
    ensemble.fit(X_train, Y_train)
    train_time = time.time() - start_time
    print(f"Training Complete in {train_time:.2f} seconds.")



    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------
    print("\nPredicting on Validation Set...")
    Y_pred = ensemble.predict(X_val)

    # F1 Score
    macro_f1, cm = calculate_f1_score(Y_val, Y_pred)

    print("\n===== SYSTEM PERFORMANCE =====")
    print(f"Base Estimator: KNN (k={BASE_K})")
    print(f"Bagging Estimators: {N_ESTIMATORS}")
    print(f"Training Time: {train_time:.2f} sec")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)



    # --------------------------------------------------------
    # BIAS–VARIANCE TRADE-OFF
    # --------------------------------------------------------
    bias, variance, total_err = evaluate_bias_variance(
        X_train, Y_train, X_val, Y_val,
        BASE_K=BASE_K,
        N_ESTIMATORS=N_ESTIMATORS,
        M_runs=3      # Repeat model 8 times
    )

    print("\n===== BIAS–VARIANCE RESULTS =====")
    print(f"Bias Error:     {bias:.4f}")
    print(f"Variance Error: {variance:.4f}")
    print(f"Total Error:    {total_err:.4f}")

    # ---------- Optimality Check ----------
    print("\n===== TRAINING OPTIMALITY CHECK =====")
    if bias < 0.05 and variance < 0.03:
        print("Training is WELL OPTIMIZED ✔")
    elif bias < 0.10 and variance < 0.05:
        print("Training is reasonably good — minor tuning possible.")
    else:
        print("Training NOT optimal — model underfits or overfits.")


except Exception as e:
    print("Error during execution:", e)

