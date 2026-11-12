import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from skrebate import ReliefF

# ---------------- CONFIG ----------------
RANDOM_STATE = 42
N_SAMPLE = 20000        # stratified subsample size (increase later if time allows)X = sample.drop(columns=["Label", "Class"])

N_FEATURES = 10         # features kept after Relief
RELIEF_SAMPLE = 3000    # rows used to compute Relief weights (Relief is O(n^2))
SMOTE_K = 3             # same k as the base paper

# ---------------- 1. LOAD + SUBSAMPLE ----------------
df = pd.read_csv("cleaned_data/cleaned_CICIDS2017.csv")
print("Loaded:", df.shape)

sample, _ = train_test_split(df, train_size=N_SAMPLE, stratify=df["Class"],
                             random_state=RANDOM_STATE)
print("\nSampled class distribution:")
print(sample["Class"].value_counts())

X = sample.drop(columns=["Label", "Class", "Source_File"])   # drop Label to avoid leakage
y = sample["Class"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = list(le.classes_)
print("Classes:", class_names)

# ---------------- 2. TRAIN/TEST SPLIT (stratified 70/30) ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, stratify=y_enc, random_state=RANDOM_STATE)

# drop zero-variance features (decided on training data only)
keep = X_train.columns[X_train.nunique() > 1]
X_train, X_test = X_train[keep], X_test[keep]
print(f"\nFeatures after zero-variance removal: {len(keep)}")

# ---------------- 3. SCALING (fit on train only) ----------------
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=keep, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=keep, index=X_test.index)

# ---------------- 4. RELIEF FEATURE SELECTION (train only) ----------------
Xr, _, yr, _ = train_test_split(X_train_s, y_train, train_size=RELIEF_SAMPLE,
                                stratify=y_train, random_state=RANDOM_STATE)
print("\nRunning ReliefF...")
t0 = time.time()
relief = ReliefF(n_features_to_select=N_FEATURES, n_neighbors=10, n_jobs=-1)
relief.fit(Xr.values, yr)
print(f"Relief done in {time.time()-t0:.1f}s")

ranking = pd.Series(relief.feature_importances_, index=keep).sort_values(ascending=False)
ranking.to_csv("outputs/relief_feature_ranking.csv", header=["relief_weight"])
selected = list(ranking.index[:N_FEATURES])
print("\nTop features:")
print(ranking.head(N_FEATURES))

X_train_sel = X_train_s[selected]
X_test_sel = X_test_s[selected]

# ---------------- 5. SMOTE (train only) ----------------
print("\nTrain class counts before SMOTE:", np.bincount(y_train))
smote = SMOTE(k_neighbors=SMOTE_K, random_state=RANDOM_STATE)
X_bal, y_bal = smote.fit_resample(X_train_sel, y_train)
print("Train class counts after SMOTE: ", np.bincount(y_bal))
print("Test class counts (untouched):  ", np.bincount(y_test))

# ---------------- 6. MODELS (hyperparameters from the base paper) ----------------
models = {
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, metric="manhattan", weights="distance", n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_split=10,
                                            min_samples_leaf=5, class_weight="balanced",
                                            random_state=RANDOM_STATE),
    "SVM-RBF": SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced"),
    "SVM-Polynomial": SVC(kernel="poly", degree=2, C=10, gamma=0.01, coef0=1, class_weight="balanced"),
}

def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)   # SVC: one-vs-rest scores, avoids slow probability=True

y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
support = y_bin.sum(axis=0)

summary_rows, classwise_rows, cms = [], [], {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    t0 = time.time()
    model.fit(X_bal, y_bal)
    train_time = time.time() - t0

    y_pred = model.predict(X_test_sel)
    scores = get_scores(model, X_test_sel)

    aucs = [roc_auc_score(y_bin[:, i], scores[:, i]) for i in range(len(class_names))]
    auc_weighted = np.average(aucs, weights=support)
    auc_macro = np.mean(aucs)

    pw, rw, fw, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    summary_rows.append({
        "Model": name, "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_w": pw, "Recall_w": rw, "F1_w": fw,
        "Precision_macro": pm, "Recall_macro": rm, "F1_macro": fm,
        "AUC_weighted": auc_weighted, "AUC_macro": auc_macro,
        "Train_time_s": round(train_time, 1),
    })

    rep = classification_report(y_test, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    for c in class_names:
        classwise_rows.append({"Model": name, "Class": c, "Precision": rep[c]["precision"],
                               "Recall": rep[c]["recall"], "F1": rep[c]["f1-score"],
                               "Support": rep[c]["support"],
                               "AUC": aucs[class_names.index(c)]})
    cms[name] = confusion_matrix(y_test, y_pred)
    print(f"  done in {train_time:.1f}s | acc={summary_rows[-1]['Accuracy']:.4f} "
          f"| AUC_w={auc_weighted:.4f} | AUC_macro={auc_macro:.4f}")

# ---------------- 7. SAVE RESULTS ----------------
summary = pd.DataFrame(summary_rows).round(4)
classwise = pd.DataFrame(classwise_rows).round(4)
summary.to_csv("outputs/results_summary.csv", index=False)
classwise.to_csv("outputs/class_wise_results.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, cm) in zip(axes.ravel(), cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=200)

print("\n=== OVERALL RESULTS ===")
print(summary.to_string(index=False))
print("\n=== CLASS-WISE RESULTS ===")
print(classwise.to_string(index=False))
print("\nSaved: results_summary.csv, class_wise_results.csv, relief_feature_ranking.csv, confusion_matrices.png")
