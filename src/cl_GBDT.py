import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


train_path = "/Users/hhhhu_7/z5524742/9417/group_pj/DataPreProcess/cleaned_features_train.csv"
val_path   = "/Users/hhhhu_7/z5524742/9417/group_pj/DataPreProcess/cleaned_features_val.csv"
test_path  = "/Users/hhhhu_7/z5524742/9417/group_pj/DataPreProcess/cleaned_features_test.csv"

horizons = [1, 6, 12, 24]
lag_list = [1, 2, 3, 6, 12, 24] 
lag_cols = [f"CO_lag{lag}" for lag in lag_list]

train = pd.read_csv(train_path)
val   = pd.read_csv(val_path)
test  = pd.read_csv(test_path)

for df in [train, val, test]:
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

def co_to_class(value: float) -> int:
    if value < 1.5:
        return 0
    elif value < 2.5:
        return 1
    else:
        return 2

for df in [train, val, test]:
    df["CO_class"] = df["CO(GT)"].apply(co_to_class)


for df in [train, val, test]:
    for lag in lag_list:
        df[f"CO_lag{lag}"] = df["CO(GT)"].shift(lag)

models = {
    "GBDT": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=300,
        random_state=42
    )
}

results = []

def prepare_xy(df: pd.DataFrame, horizon: int):
    df_h = df.copy()

    label_col = f"CO_t+{horizon}"
    df_h[label_col] = df_h["CO_class"].shift(-horizon)

    df_h = df_h.dropna(subset=[label_col] + lag_cols)

    drop_cols = ["DateTime", "CO(GT)", "CO_class", label_col]

    feat_cols = [c for c in df_h.columns if c not in drop_cols]

    X = df_h[feat_cols].values
    y = df_h[label_col].astype(int).values
    baseline_pred = df_h["CO_class"].astype(int).values

    return X, y, baseline_pred


for h in horizons:
    print(f"\n{'-' * 80}")
    print(f"Horizon = {h} hour")

    X_train, y_train, base_train = prepare_xy(train, h)
    X_val,   y_val,   base_val   = prepare_xy(val,   h)
    X_test,  y_test,  base_test  = prepare_xy(test,  h)

    for split_name, y_true, baseline_pred in [
        ("val",  y_val,  base_val),
        ("test", y_test, base_test)
    ]:
        acc = accuracy_score(y_true, baseline_pred)
        f1  = f1_score(y_true, baseline_pred, average="macro")
        print(f"[h={h}] Baseline on {split_name:4s} - acc={acc:.4f}, macroF1={f1:.4f}")

        results.append({
            "horizon": h,
            "split": split_name,
            "model": "Baseline",
            "accuracy": acc,
            "macro_f1": f1
        })

    for model_name, clf in models.items():
        print(f"Training: {model_name} (h={h})")
        clf.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("val",  X_val,  y_val),
            ("test", X_test, y_test)
        ]:
            y_pred = clf.predict(X_split)
            acc = accuracy_score(y_split, y_pred)
            f1  = f1_score(y_split, y_pred, average="macro")

            print(f"[h={h}] {model_name:10s} on {split_name:4s} - acc={acc:.4f}, F1={f1:.4f}")

            results.append({
                "horizon": h,
                "split": split_name,
                "model": model_name,
                "accuracy": acc,
                "macro_f1": f1
            })

            if split_name == "test":
                cm = confusion_matrix(y_split, y_pred)
                plt.figure(figsize=(4, 3))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["low", "mid", "high"],
                    yticklabels=["low", "mid", "high"]
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"CO - {model_name} (h={h}, test)")
                plt.tight_layout()
                plt.savefig(f"CM_CO_{model_name}_h{h}_test.png", dpi=150)
                plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv("classification_CO_lag_results.csv", index=False)
print("\nSaved results to classification_CO_lag_results.csv")

df_test = results_df[results_df["split"] == "test"].copy()

plot_models = ["Baseline", "GBDT", "MLP"]
colors = ["gray", "blue", "red"]

# ---- Accuracy curve ----
plt.figure(figsize=(8, 5))
for model_name, color in zip(plot_models, colors):
    sub = df_test[df_test["model"] == model_name].set_index("horizon").loc[horizons]
    plt.plot(horizons, sub["accuracy"], marker="o", label=model_name, color=color)

plt.title("CO Classification Accuracy vs Forecast Horizon")
plt.xlabel("Forecast Horizon (hours)")
plt.ylabel("Accuracy")
plt.xticks(horizons)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("CO_accuracy_curve.png", dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
for model_name, color in zip(plot_models, colors):
    sub = df_test[df_test["model"] == model_name].set_index("horizon").loc[horizons]
    plt.plot(horizons, sub["macro_f1"], marker="o", label=model_name, color=color)

plt.title("CO Classification Macro-F1 vs Forecast Horizon")
plt.xlabel("Forecast Horizon (hours)")
plt.ylabel("Macro F1-score")
plt.xticks(horizons)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("CO_f1_curve.png", dpi=200)
plt.show()