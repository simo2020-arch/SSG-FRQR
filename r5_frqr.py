# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:37:18 2025

@author: UCL81
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:58:39 2025
@author: UCL81
"""
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import defaultdict
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import seaborn as sns




def bootstrap_stability_selection(X, y, base_selector, n_bootstrap=5, stability_threshold=0.6):
    n_features = X.shape[1]
    selection_matrix = np.zeros((n_bootstrap, n_features))
    runtimes = []
    gamma_scores = []

    for i in range(n_bootstrap):
        X_sample, y_sample = resample(X, y)
        start_time = time.time()
        selected, gamma = base_selector(X_sample, y_sample)
        elapsed = time.time() - start_time

        print(f"[{i+1}/{n_bootstrap}] Features: {selected}, Time: {elapsed:.2f}s")
        selection_matrix[i, selected] = 1
        runtimes.append(elapsed)
        gamma_scores.append(gamma)

    stability_scores = selection_matrix.mean(axis=0)
    selected_features = np.where(stability_scores >= stability_threshold)[0]
    return selected_features, stability_scores, selection_matrix, runtimes, gamma_scores



def compute_kuncheva_index(selection_matrix, k):
    b = selection_matrix.shape[1]
    n = selection_matrix.shape[0]
    total = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            overlap = np.sum(selection_matrix[i] * selection_matrix[j])
            total += overlap
            pairs += 1
    expected_overlap = (k ** 2) / b
    kuncheva = (total / pairs - expected_overlap) / (k - expected_overlap) if k != expected_overlap else 0
    return kuncheva


def compute_average_jaccard(selection_matrix):
    n = selection_matrix.shape[0]
    total = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = selection_matrix[i]
            b = selection_matrix[j]
            intersection = np.sum(np.logical_and(a, b))
            union = np.sum(np.logical_or(a, b))
            total += intersection / union if union > 0 else 0
            pairs += 1
    return total / pairs

'''
def cluster_data(X, n_clusters=None):
    """
    Cluster the input data using MiniBatchKMeans with adaptive cluster size.

    Parameters:
    -----------
    X : ndarray
        The input data matrix (n_samples × n_features).
    n_clusters : int or None
        Number of clusters. If None, uses a heuristic: min(20, max(5, n // 10)).

    Returns:
    --------
    labels : ndarray
        Cluster labels for each sample.
    centroids : ndarray
        Coordinates of cluster centroids.
    """
    if n_clusters is None:
        n = X.shape[0]
        n_clusters = min(50, max(5, n // 10))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_
'''
def cluster_data(X, n_clusters=None):
    if n_clusters is None:
        n_clusters = min(50, max(5, X.shape[0] // 5))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init='auto',
        batch_size=2048,
        random_state=42
    )
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_

def approximate_similarity(X, clusters, labels):
    centroid_data = clusters[labels]
    dists = euclidean_distances(centroid_data, centroid_data)
    sim = np.exp(-dists ** 2)
    return sim


def compute_fdd(similarity_matrix, y):
    n = len(y)
    pos = np.zeros(n)

    for i in range(n):
        same_class_mask = (y == y[i]).astype(float)
        weighted_sim = similarity_matrix[i] * same_class_mask
        pos[i] = np.sum(weighted_sim) / np.sum(same_class_mask) if np.sum(same_class_mask) > 0 else 0

    return np.mean(pos)




def compute_interaction_gain(X, y, current_set, candidate, sim_func):
    if not current_set:
        gamma_cand = sim_func(X[:, [candidate]], y)
        return gamma_cand, gamma_cand
    gamma_prev = sim_func(X[:, current_set], y)
    gamma_new = sim_func(X[:, current_set + [candidate]], y)
    interaction = gamma_new - gamma_prev
    return gamma_new, interaction

def interaction_aware_frqr(X, y, max_iter=50, epsilon=1e-3):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    selected = []
    remaining = list(range(X.shape[1]))

    def sim_func(X_sub, y_sub):
        labels, centers = cluster_data(X_sub)
        sim = approximate_similarity(X_sub, centers, labels)
        return compute_fdd(sim, y_sub)

    best_gamma = -np.inf  # Initialize with worst possible score
    for iteration in range(max_iter):
        best_gain = -np.inf
        best_feature = None
        gamma_new_for_best = None

        for f in remaining:
            gamma_new, gain = compute_interaction_gain(X, y, selected, f, sim_func)
            #if gain > best_gain:
            if gain > best_gain and gain > 0:
                best_gain = gain
                best_feature = f
                gamma_new_for_best = gamma_new

        # Stop if no improvement
        if best_feature is None and gamma_new_for_best is None and best_gain < epsilon:
            print(f"Stopped at iteration {iteration} due to no further gain.")
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_gamma = gamma_new_for_best

        print(f"Best Feature in Iteration {iteration}: {best_feature}, Gain: {best_gain:.10f}, γ: {best_gamma:.10f}")

    return selected, best_gamma



def plot_runtimes(times, gamma_scores):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(times, 'g-o', label='Runtime (s)')
    ax2.plot(gamma_scores, 'b-s', label='FDD')

    ax1.set_xlabel('Bootstrap Iteration')
    ax1.set_ylabel('Runtime (s)', color='g')
    ax2.set_ylabel('FDD', color='b')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='b')
    #plt.title("Runtime and FDD across Bootstraps")
    fig.tight_layout()
    plt.show()
    
def compute_redundancy(X, selected_features):
    subset = X[:, selected_features]
    corr_matrix = np.corrcoef(subset, rowvar=False)
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    avg_redundancy = np.mean(np.abs(corr_matrix[upper_tri_indices]))
    return avg_redundancy



def evaluate_classifiers(X, y, selected_features):
    X_sel = X[:, selected_features]
    class_counts = np.bincount(y)
    min_class_count = np.min(class_counts)
    n_splits = max(5, min_class_count)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_results = defaultdict(list)
    svm_results = defaultdict(list)
    cart_results = defaultdict(list)

    for train_idx, test_idx in skf.split(X_sel, y):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # SVM
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)

        # CART
        cart = DecisionTreeClassifier(random_state=42)
        cart.fit(X_train, y_train)
        y_pred_cart = cart.predict(X_test)

        for metric, func in zip(['Accuracy', 'F1', 'Precision', 'Recall'],
                                [accuracy_score, f1_score, precision_score, recall_score]):
            if metric == 'Accuracy':
                rf_results[metric].append(func(y_test, y_pred_rf))
                svm_results[metric].append(func(y_test, y_pred_svm))
                cart_results[metric].append(func(y_test, y_pred_cart))
            else:
                rf_results[metric].append(func(y_test, y_pred_rf, average='macro'))
                svm_results[metric].append(func(y_test, y_pred_svm, average='macro'))
                cart_results[metric].append(func(y_test, y_pred_cart, average='macro'))

    # Return mean ± std for each classifier
    rf_summary = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in rf_results.items()}
    svm_summary = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in svm_results.items()}
    cart_summary = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in cart_results.items()}

    return rf_summary, svm_summary, cart_summary



def inject_label_noise(y, noise_level=0.1, seed=42):
    np.random.seed(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    unique_labels = np.unique(y)
    for idx in noisy_indices:
        current_label = y[idx]
        new_label = np.random.choice(unique_labels[unique_labels != current_label])
        y_noisy[idx] = new_label
    return y_noisy

def evaluate_cross_domain(X_s, y_s, X_t, y_t, selected_features):
    X_s_sel = X_s[:, selected_features]
    X_t_sel = X_t[:, selected_features]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_s_sel, y_s)
    y_pred_s = clf.predict(X_s_sel)
    y_pred_t = clf.predict(X_t_sel)

    acc_s = accuracy_score(y_s, y_pred_s)
    acc_t = accuracy_score(y_t, y_pred_t)
    generalization_gap = acc_s - acc_t
    gamma_combined = 0.7 * acc_s + 0.3 * acc_t
    return acc_s, acc_t, generalization_gap, gamma_combined

# ------------------------------ USER SETUP ------------------------------ #
source_path = "Leukemia-4 class dataset_source.csv"
target_path = "Leukemia-4 class dataset_target.csv"

df_source = pd.read_csv(source_path)
df_target = pd.read_csv(target_path)

X_s = df_source.iloc[:, :-1].values
y_s = df_source.iloc[:, -1].values
X_t = df_target.iloc[:, :-1].values
y_t = df_target.iloc[:, -1].values


# ---------------------- R5-FRQR + EVALUATION ------------------------ #
selected_features, stability_scores, selection_matrix, runtimes, gamma_scores = bootstrap_stability_selection(
    X_s, y_s, interaction_aware_frqr, n_bootstrap=5, stability_threshold=0.1
)

redundancy_score = compute_redundancy(X_s, selected_features)
print("Redundancy Score:", redundancy_score)
# Visualize
plot_runtimes(runtimes, gamma_scores)


kuncheva = compute_kuncheva_index(selection_matrix, len(selected_features))
jaccard = compute_average_jaccard(selection_matrix)

print("Kuncheva Index:", kuncheva)
print("Average Jaccard Index:", jaccard)


# Classifier performance
rf_metrics, svm_metrics, cart_metrics = evaluate_classifiers(X_s, y_s, selected_features)

print("Random Forest Results (mean ± std):", rf_metrics)
print("SVM Results (mean ± std):", svm_metrics)
print("CART Results (mean ± std):", cart_metrics)



# Robustness to noise
y_s_noisy = inject_label_noise(deepcopy(y_s), noise_level=0.1)
rf_clean, _, _ = evaluate_classifiers(X_s, y_s, selected_features)
rf_noisy, _, _ = evaluate_classifiers(X_s, y_s_noisy, selected_features)
#robustness_index = rf_noisy["Accuracy"] / rf_clean["Accuracy"]
print("Clean Accuracy:", rf_clean["Accuracy"])
print("Noisy Accuracy:", rf_noisy["Accuracy"])
#print("Robustness Index:", robustness_index)

# Cross-domain evaluation
acc_s, acc_t, domain_gap, gamma_blend = evaluate_cross_domain(X_s, y_s, X_t, y_t, selected_features)
print("Source Accuracy:", acc_s)
print("Target Accuracy:", acc_t)
print("∆Domain (Δ):", domain_gap)
print("Domain-adaptive Gamma:", gamma_blend)
print("selected features:", selected_features)
print("Runtimes:", runtimes)
print("stability scores:", stability_scores)

import csv

# Flatten the classifier results into a dict
all_results = {
    "Redundancy Score": redundancy_score,
    "Kuncheva Index": kuncheva,
    "Average Jaccard Index": jaccard,
    "Random Forest Accuracy (±)": rf_metrics["Accuracy"],
    "Random Forest F1 (±)": rf_metrics["F1"],
    "Random Forest Precision (±)": rf_metrics["Precision"],
    "Random Forest Recall (±)": rf_metrics["Recall"],
    "SVM Accuracy (±)": svm_metrics["Accuracy"],
    "SVM F1 (±)": svm_metrics["F1"],
    "SVM Precision (±)": svm_metrics["Precision"],
    "SVM Recall (±)": svm_metrics["Recall"],
    "CART Accuracy (±)": cart_metrics["Accuracy"],
    "CART F1 (±)": cart_metrics["F1"],
    "CART Precision (±)": cart_metrics["Precision"],
    "CART Recall (±)": cart_metrics["Recall"],
    "Clean Accuracy": rf_clean["Accuracy"],
    "Noisy Accuracy": rf_noisy["Accuracy"],
    #"Robustness Index": robustness_index,
    "Source Accuracy": acc_s,
    "Target Accuracy": acc_t,
    "∆Domain (Δ)": domain_gap,
    "Domain-adaptive Gamma": gamma_blend,
    "Num Selected Features": len(selected_features)
}

# Save metrics to CSV
results_df = pd.DataFrame([all_results])
results_df.to_csv("evaluation_summary.csv", index=False)

# Also save detailed arrays for reproducibility
np.savetxt("selected_features.csv", selected_features, delimiter=",", fmt="%d", header="Selected Feature Indices", comments="")
np.savetxt("runtimes.csv", runtimes, delimiter=",", header="Runtime per Bootstrap (s)", comments="")
np.savetxt("stability_scores.csv", stability_scores, delimiter=",", header="Stability Score per Feature", comments="")


