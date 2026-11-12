import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from skrebate import ReliefF
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline



Path("outputs").mkdir(exist_ok=True)

# ---------------- CONFIG ----------------
SEEDS = [42, 7, 21, 99, 123]
N_SAMPLE = 5000
TUNE_SEED = 42

N_FEATURES = 10
RELIEF_SAMPLE = 3000
SMOTE_K = 3

df = pd.read_csv("cleaned_data/cleaned_CICIDS2017.csv")
print("Loaded:", df.shape)


def prepare(seed):
    """Subsample -> split -> scale -> Relief. Everything fitted on train only."""
    sample, _ = train_test_split(
        df,
        train_size=N_SAMPLE,
        stratify=df["Class"],
        random_state=seed
    )

    X = sample.drop(columns=[
    "Label",
    "Class",
    "Source_File",
    "Fwd Header Length.1"
])
    le = LabelEncoder()
    y = le.fit_transform(sample["Class"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    return X_tr, X_te, y_tr, y_te, list(X_tr.columns), list(le.classes_)


# ---------------- 1. SVM TUNING (training data of TUNE_SEED only; SMOTE inside CV) ----------------

print("\n=== SVM TUNING ===")

X_tr, X_te, y_tr, y_te, sel, class_names = prepare(TUNE_SEED)

grid = {
    "svm__C": [1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1]
}

best = {}
tuning_rows = []

for key, kw in [
    ("rbf", dict(kernel="rbf")),
    ("poly", dict(kernel="poly", degree=2, coef0=1))
]:

    t0 = time.time()

    pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("variance", VarianceThreshold()),
    ("relief", ReliefF(
        n_features_to_select=N_FEATURES,
        n_neighbors=10,
        n_jobs=1
    )),
    ("smote", SMOTE(
        k_neighbors=SMOTE_K,
        random_state=TUNE_SEED
    )),
    ("svm", SVC(**kw))
])

    gs = GridSearchCV(
        pipe,
        grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1
    )

    gs.fit(X_tr, y_tr)

    best[key] = {
        "C": gs.best_params_["svm__C"],
        "gamma": gs.best_params_["svm__gamma"]
    }

    for p, s in zip(
        gs.cv_results_["params"],
        gs.cv_results_["mean_test_score"]
    ):
        tuning_rows.append({
            "Kernel": key,
            "C": p["svm__C"],
            "gamma": p["svm__gamma"],
            "CV_F1_macro": round(s, 4)
        })

    print(
        f"{key}: best={best[key]} "
        f"(CV macro-F1={gs.best_score_:.4f}) "
        f"in {time.time()-t0:.0f}s"
    )

pd.DataFrame(tuning_rows).to_csv(
    "outputs/tuning_results.csv",
    index=False
)



# ---------------- 2. MODEL BUILDER ----------------

def build_models(strategy, seed):

    # Apply class weights ONLY for the ClassWeight strategy
    cw = "balanced" if strategy == "ClassWeight" else None

    m = {}

    # KNN does not support class_weight
    if strategy != "ClassWeight":
        m["KNN (k=7)"] = KNeighborsClassifier(
            n_neighbors=7,
            metric="manhattan",
            weights="distance",
            n_jobs=1
        )

    m["Decision Tree"] = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=cw,
        random_state=seed
    )

    # Original paper hyperparameters: evaluate under SMOTE
    if strategy == "SMOTE":
        m["SVM-RBF (paper)"] = SVC(
            kernel="rbf",
            C=100,
            gamma="scale"
        )

        m["SVM-Poly (paper)"] = SVC(
            kernel="poly",
            degree=2,
            C=10,
            gamma=0.01,
            coef0=1
        )

    # Tuned SVMs: evaluate under all applicable strategies
    m["SVM-RBF (tuned)"] = SVC(
        kernel="rbf",
        **best["rbf"],
        class_weight=cw
    )

    m["SVM-Poly (tuned)"] = SVC(
        kernel="poly",
        degree=2,
        coef0=1,
        **best["poly"],
        class_weight=cw
    )

    return m





# ---------------- 3. MULTI-SEED EXPERIMENTS ----------------

# ---------------- 3. MULTI-SEED EXPERIMENTS ----------------

rows = []
relief_counts = {}

for seed in SEEDS:
    print(f"\n=== SEED {seed} ===")

    X_tr, X_te, y_tr, y_te, _, class_names = prepare(seed)

    labels = list(range(len(class_names)))
    y_bin = label_binarize(y_te, classes=labels)
    support = y_bin.sum(axis=0)
    oa = class_names.index("Other_Attacks")

    print("Test counts:", dict(zip(class_names, support)))

    selected_features_recorded = False

    for strategy in ["SMOTE", "ClassWeight", "NoResampling"]:

        models = build_models(strategy, seed)

        for name, model in models.items():

            # Apply SMOTE only for the SMOTE strategy
            sampler = (
                SMOTE(k_neighbors=SMOTE_K, random_state=seed)
                if strategy == "SMOTE"
                else "passthrough"
            )

            # Complete preprocessing and model pipeline
            pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("variance", VarianceThreshold()),
    ("relief", ReliefF(
        n_features_to_select=N_FEATURES,
        n_neighbors=10,
        n_jobs=1
    )),
    ("smote", sampler),
    ("model", model)
])

            t0 = time.time()

            pipeline.fit(X_tr, y_tr)

            fit_time = time.time() - t0

            y_pred = pipeline.predict(X_te)

            if hasattr(pipeline, "predict_proba"):
                scores = pipeline.predict_proba(X_te)
            else:
                scores = pipeline.decision_function(X_te)

            aucs = [
                roc_auc_score(y_bin[:, i], scores[:, i])
                for i in labels
            ]

            # Class-wise metrics
            p, r, f1, _ = precision_recall_fscore_support(
                y_te,
                y_pred,
                labels=labels,
                zero_division=0
            )

            # Macro metrics
            _, _, f1_macro, _ = precision_recall_fscore_support(
                y_te,
                y_pred,
                average="macro",
                zero_division=0
            )

            # Weighted metrics
            p_weighted, r_weighted, f1_weighted, _ = (
                precision_recall_fscore_support(
                    y_te,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )

            rows.append({
                "Seed": seed,
                "Strategy": strategy,
                "Model": name,
                "Accuracy": accuracy_score(y_te, y_pred),
                "Precision_weighted": p_weighted,
                "Recall_weighted": r_weighted,
                "F1_weighted": f1_weighted,
                "F1_macro": f1_macro,
                "AUC_weighted": np.average(aucs, weights=support),
                "AUC_macro": np.mean(aucs),
                "OA_Precision": p[oa],
                "OA_Recall": r[oa],
                "OA_F1": f1[oa],
                "Fit_time_s": round(fit_time, 1)
            })

            # Record ReliefF-selected features once per seed
                        # Record ReliefF-selected features once per seed
            if not selected_features_recorded:

                variance = pipeline.named_steps["variance"]
                relief = pipeline.named_steps["relief"]

                remaining_features = X_tr.columns[
                    variance.get_support()
                ]

                ranking = pd.Series(
                    relief.feature_importances_,
                    index=remaining_features
                ).sort_values(ascending=False)

                for feature in ranking.index[:N_FEATURES]:
                    relief_counts[feature] = (
                        relief_counts.get(feature, 0) + 1
                    )

                selected_features_recorded = True

    print(
                f"  [{strategy:11s}] {name:17s} "
                f"acc={rows[-1]['Accuracy']:.4f} "
                f"AUCw={rows[-1]['AUC_weighted']:.4f} "
                f"OA_F1={f1[oa]:.3f} "
                f"({fit_time:.1f}s)"
            )

# ---------------- SAVE RESULTS ----------------

res = pd.DataFrame(rows)
res.to_csv("outputs/multiseed_runs.csv", index=False)

# ---------------- 4. SUMMARIES ----------------
res = pd.DataFrame(rows)
res.to_csv("outputs/multiseed_runs.csv", index=False)

metrics = [
    "Accuracy",
    "Precision_weighted",
    "Recall_weighted",
    "F1_weighted",
    "F1_macro",
    "AUC_weighted",
    "AUC_macro",
    "OA_Precision",
    "OA_Recall",
    "OA_F1"
]
g = res.groupby(["Strategy", "Model"])[metrics]
summary = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
summary.to_csv("outputs/multiseed_summary.csv")
print("\n=== MEAN ± STD OVER", len(SEEDS), "SEEDS (OA = Other_Attacks) ===")
print(summary.to_string())

core = ["KNN (k=7)", "Decision Tree", "SVM-RBF (tuned)", "SVM-Poly (tuned)"]
r = res[(res.Strategy == "SMOTE") & res.Model.isin(core)].copy()
r["rank_AUC_weighted"] = r.groupby("Seed")["AUC_weighted"].rank(ascending=False)
r["rank_AUC_macro"] = r.groupby("Seed")["AUC_macro"].rank(ascending=False)
print("\n=== MEAN RANK (1 = best), SMOTE, tuned SVMs ===")
print(r.groupby("Model")[["rank_AUC_weighted", "rank_AUC_macro"]].mean().round(2))
diff = sum(list(s.sort_values("AUC_weighted", ascending=False).Model) !=
           list(s.sort_values("AUC_macro", ascending=False).Model) for _, s in r.groupby("Seed"))
print(f"Seeds where weighted vs macro AUC ordering differs: {diff}/{len(SEEDS)}")

stab = pd.Series(relief_counts).sort_values(ascending=False)
stab.to_csv("outputs/relief_stability.csv", header=["times_selected"])
print(f"\n=== RELIEF FEATURE STABILITY (times in top {N_FEATURES}, out of {len(SEEDS)} seeds) ===")
print(stab.to_string())
print("\nSaved: multiseed_runs.csv, multiseed_summary.csv, tuning_results.csv, relief_stability.csv")