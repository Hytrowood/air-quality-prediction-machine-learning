# train_c6h6_exclude_s2_multi.py
import os
import re
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint


TRAIN_CSV = "./DataPreProcess/cleaned_features_train.csv"
VALID_CSV = "./DataPreProcess/cleaned_features_val.csv"
TEST_CSV  = "./DataPreProcess/cleaned_features_test.csv"

DATETIME_COL = "DateTime"
SENTINEL = -200
TARGET = "C6H6(GT)"                        
HORIZONS = [1, 6, 12, 24]                 


POLLUTANTS_GT = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]


SENSOR_COLS = ["PT08.S1(CO)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
EXCLUDED_SENSORS = ["PT08.S2(NMHC)"]       

METEO_COLS  = ["T", "RH", "AH"]
TIME_COLS   = ["Hour", "Weekday", "Month_sin", "Month_cos"]  
ALLOWED_HISTORY_REGEX = re.compile(r".*_(lag\d+|ma\d+)$", re.IGNORECASE)

OUT_DIR = "rf_c6h6_excl_s2_multi"
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(OUT_DIR, "models"); os.makedirs(MODELS_DIR, exist_ok=True)
PRED_DIR = os.path.join(OUT_DIR, "predictions"); os.makedirs(PRED_DIR, exist_ok=True)


def cyc_sin_cos(series: pd.Series, period: int, offset: int = 0):
    angle = 2 * math.pi * (series - offset) / period
    return np.sin(angle), np.cos(angle)

def ensure_time_features(df: pd.DataFrame, datetime_col: str = DATETIME_COL):

    if datetime_col not in df.columns:
        raise ValueError("lack of DateTime ")
    if not np.issubdtype(df[datetime_col].dtype, np.datetime64):
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    if "Hour" not in df.columns:
        df["Hour"] = df[datetime_col].dt.hour
    if "Weekday" not in df.columns:
        df["Weekday"] = df[datetime_col].dt.weekday
    if "Month" not in df.columns:
        df["Month"] = df[datetime_col].dt.month

    if ("Month_sin" not in df.columns) or ("Month_cos" not in df.columns):
        s, c = cyc_sin_cos(df["Month"].astype(int), period=12, offset=1)
        df["Month_sin"], df["Month_cos"] = s, c

    if "Month" in df.columns:
        df.drop(columns=["Month"], inplace=True)
    return df

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[c] == SENTINEL, c] = np.nan
    df = ensure_time_features(df, DATETIME_COL)
    return df

def select_safe_feature_columns(df: pd.DataFrame) -> list[str]:
    allowed_base = set(SENSOR_COLS + METEO_COLS + TIME_COLS)
    safe = []
    for col in df.columns:
        if col == DATETIME_COL:
            continue
        if col in POLLUTANTS_GT:
            continue
        if col in EXCLUDED_SENSORS:
            continue
        if col in allowed_base:
            safe.append(col); continue
        if ALLOWED_HISTORY_REGEX.match(col):
            safe.append(col); continue
        if re.search(r"(lead|t\+|future)", col, re.IGNORECASE):
            raise ValueError(f"Suspected future feature columns detected{col}")
    leaks = [c for c in safe if "(GT)" in c or c in EXCLUDED_SENSORS]
    if leaks:
        raise ValueError(f"Detected leaked/excluded columns enter the feature:{leaks}")
    return safe

def make_xy_for_horizon(df: pd.DataFrame, target_col: str, h: int):

    df = df.copy()
    y_col = f"{target_col}_lead{h}"
    if y_col not in df.columns:
        df[y_col] = df[target_col].shift(-h)

    df_feat = df.drop(columns=[y_col])

    feat_cols = select_safe_feature_columns(df_feat)
    X_all = df_feat[feat_cols].select_dtypes(include=[np.number]).copy()
    y_all = df[y_col].astype(float)

    mask = ~y_all.isna()
    X = X_all.loc[mask].reset_index(drop=True)
    y = y_all.loc[mask].reset_index(drop=True)

    dt = df.loc[mask, DATETIME_COL].reset_index(drop=True)
    idx = np.flatnonzero(mask.to_numpy())  
    return X, y, dt, idx

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

train_df = load_and_prepare(TRAIN_CSV)
valid_df = load_and_prepare(VALID_CSV)
test_df  = load_and_prepare(TEST_CSV)

rf_base = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    ))
])

param_dist = {
    "rf__n_estimators": randint(400, 900),
    "rf__max_depth": randint(8, 20),
    "rf__min_samples_split": randint(2, 10),
    "rf__min_samples_leaf": randint(1, 8),
    "rf__max_features": ["sqrt", "log2", 0.5],
    "rf__bootstrap": [True, False]
}

summary_rows = []

print("\n" + "="*88)
print(f"Training multi-horizon (no leakage, exclude S2): {TARGET} @ {HORIZONS}")
print("="*88)

for h in HORIZONS:
    print(f"\n--- Horizon = {h}h ---")
    X_train, y_train, _dt_train, idx_train = make_xy_for_horizon(train_df, TARGET, h)
    X_valid, y_valid, _dt_valid, idx_valid = make_xy_for_horizon(valid_df, TARGET, h)
    X_test,  y_test,  dt_test,  idx_test  = make_xy_for_horizon(test_df,  TARGET, h)

    baseline = rf_base
    baseline.fit(X_train, y_train)
    valid_pred_base = baseline.predict(X_valid)
    base_valid_rmse = rmse(y_valid, valid_pred_base)
    base_valid_r2 = r2_score(y_valid, valid_pred_base)
    print(f"[Baseline RF][h={h}] valid RMSE={base_valid_rmse:.4f}, R2={base_valid_r2:.4f}")

    X_tv = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
    y_tv = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)
    test_fold = np.concatenate([
        -1 * np.ones(len(X_train), dtype=int),
         0 * np.ones(len(X_valid), dtype=int)
    ])
    ps = PredefinedSplit(test_fold=test_fold)

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=ps, verbose=1, random_state=42, n_jobs=-1, refit=False
    )
    search.fit(X_tv, y_tv)
    best_params = search.best_params_
    best_cv_rmse = -search.best_score_
    print(f"[Search][h={h}] Best params: {best_params}")
    print(f"[Search][h={h}] Best valid RMSE (cv metric): {best_cv_rmse:.4f}")

    rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1, **{k.split("__")[1]: v for k, v in best_params.items()})
    tuned_pipe = Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("rf", rf_tuned)])
    tuned_pipe.fit(X_train, y_train)
    valid_pred_best = tuned_pipe.predict(X_valid)
    best_valid_rmse = rmse(y_valid, valid_pred_best)
    best_valid_r2 = r2_score(y_valid, valid_pred_best)
    print(f"[Best RF][h={h}] valid RMSE={best_valid_rmse:.4f}, R2={best_valid_r2:.4f}")

    rf_final = RandomForestRegressor(random_state=42, n_jobs=-1, **{k.split("__")[1]: v for k, v in best_params.items()})
    final_pipe = Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("rf", rf_final)])
    final_pipe.fit(X_tv, y_tv)
    test_pred = final_pipe.predict(X_test)
    test_rmse = rmse(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    print(f"[Final Test][h={h}] RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

    y_naive = test_df.loc[idx_test, TARGET].to_numpy()
    naive_rmse = rmse(y_test.to_numpy(), y_naive.astype(float))

    safe_name = TARGET.replace("(GT)", "").replace("(", "_").replace(")", "")
    pred_df = pd.DataFrame({
        DATETIME_COL: dt_test.values,
        f"{TARGET}_actual_h{h}": y_test.values,
        f"{TARGET}_pred_h{h}": test_pred
    })
    pred_path = os.path.join(PRED_DIR, f"pred_{safe_name}_exclS2_h{h}.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[h={h}] Saved predictions -> {pred_path}")

    model_path = os.path.join(MODELS_DIR, f"rf_{safe_name}_exclS2_h{h}.joblib")
    joblib.dump(final_pipe, model_path)
    print(f"[h={h}] Saved model -> {model_path}")

    rf_used: RandomForestRegressor = final_pipe.named_steps["rf"]
    fi = pd.Series(rf_used.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    fi_path = os.path.join(OUT_DIR, f"feature_importance_{safe_name}_exclS2_h{h}.csv")
    fi.to_csv(fi_path, header=["importance"])
    print(f"[h={h}] Saved feature importance -> {fi_path}")

    summary_rows.append({
        "target": TARGET,
        "excluded_features": ";".join(EXCLUDED_SENSORS),
        "horizon_h": h,
        "naive_RMSE_test": round(naive_rmse, 4),
        "baseline_valid_RMSE": round(base_valid_rmse, 4),
        "best_valid_RMSE": round(best_valid_rmse, 4),
        "best_valid_R2": round(best_valid_r2, 4),
        "test_RMSE": round(test_rmse, 4),
        "test_R2": round(test_r2, 4),
        "pred_csv": pred_path,
        "model_path": model_path,
        "feat_importance_csv": fi_path
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "summary_C6H6_exclS2_multi.csv")
summary_df.to_csv(summary_path, index=False)
print("\nSaved summary ->", summary_path)
print("Done.")
