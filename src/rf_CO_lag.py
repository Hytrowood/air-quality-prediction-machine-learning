import os, re, math, joblib
import numpy as np, pandas as pd
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
TARGET = "CO(GT)"  

LAGS_TARGET = [1, 2, 3, 6, 12]
LAGS_SENS   = [1, 3, 6]
SENSOR_LAG_SOURCES = ["PT08.S1(CO)", "PT08.S5(O3)", "PT08.S3(NOx)"] 

SENSOR_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
METEO_COLS  = ["T", "RH", "AH"]
TIME_COLS   = ["Hour", "Weekday", "Month", "Month_sin", "Month_cos"]
ALLOWED_HISTORY_REGEX = re.compile(r".*_(lag\d+|ma\d+)$")

OUT_DIR = "rf_CO_lag_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(OUT_DIR, "models"); os.makedirs(MODELS_DIR, exist_ok=True)
PRED_DIR = os.path.join(OUT_DIR, "predictions"); os.makedirs(PRED_DIR, exist_ok=True)

def cyc_sin_cos(series: pd.Series, period: int, offset: int = 0):
    angle = 2 * math.pi * (series - offset) / period
    return np.sin(angle), np.cos(angle)

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if DATETIME_COL in df.columns:
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    for c in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[c] == SENTINEL, c] = np.nan
    if ("Month_sin" not in df.columns) or ("Month_cos" not in df.columns):
        if "Month" not in df.columns:
            raise ValueError("lack of Month and not found Month_sin/Month_cos。")
        m = df["Month"].astype(int)
        s, c = cyc_sin_cos(m, 12, 1)
        df["Month_sin"], df["Month_cos"] = s, c

    if "Month" in df.columns:
        df = df.drop(columns=["Month"])
    if DATETIME_COL in df.columns:
        df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    return df

def add_lags_inplace(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"未找到目标列 {TARGET}")
    for k in LAGS_TARGET:
        df[f"CO_lag{k}"] = df[TARGET].shift(k)
    for src in SENSOR_LAG_SOURCES:
        if src in df.columns:
            for k in LAGS_SENS:
                df[f"{src}_lag{k}"] = df[src].shift(k)

def trim_head_by_max_lag(df: pd.DataFrame) -> pd.DataFrame:
    maxlag = max(LAGS_TARGET + LAGS_SENS) if (LAGS_TARGET or LAGS_SENS) else 0
    return df.iloc[maxlag:].reset_index(drop=True) if maxlag > 0 else df

def select_safe_feature_columns(df: pd.DataFrame, target_col: str):
    cols = []
    for col in df.columns:
        if col == DATETIME_COL: continue
        if col.endswith("(GT)") or "(GT)" in col:
            continue
        if col in SENSOR_COLS or col in METEO_COLS or col in ["Month_sin","Month_cos","Hour","Weekday"]:
            cols.append(col); continue
        if ALLOWED_HISTORY_REGEX.match(col):  
            cols.append(col); continue
        if "(GT)" not in col:
            cols.append(col)
    leak = [c for c in cols if "(GT)" in c]
    if leak:
        raise ValueError(f"Detected leakage feature columns:{leak}")
    return cols

def make_xy(df: pd.DataFrame, target_col: str):
    safe_cols = select_safe_feature_columns(df, target_col)
    X = df[safe_cols].select_dtypes(include=[np.number]).copy()
    y = df[target_col].astype(float)
    return X, y

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

train_df = load_and_prepare(TRAIN_CSV)
valid_df = load_and_prepare(VALID_CSV)
test_df  = load_and_prepare(TEST_CSV)

for _df in (train_df, valid_df, test_df):
    add_lags_inplace(_df)
train_df = trim_head_by_max_lag(train_df)
valid_df = trim_head_by_max_lag(valid_df)
test_df  = trim_head_by_max_lag(test_df)

rf_base = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", n_jobs=-1, random_state=42
    ))
])

param_dist = {
    "rf__n_estimators": randint(400, 1100),
    "rf__max_depth": randint(8, 28),
    "rf__min_samples_split": randint(2, 10),
    "rf__min_samples_leaf": randint(1, 8),
    "rf__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    "rf__bootstrap": [True, False]
}

print("\n" + "="*88)
print(f"Training target (no leakage, with lags): {TARGET}")
print("="*88)

X_train, y_train = make_xy(train_df, TARGET)
X_valid, y_valid = make_xy(valid_df, TARGET)
X_test,  y_test  = make_xy(test_df,  TARGET)

rf_base.fit(X_train, y_train)
valid_pred_base = rf_base.predict(X_valid)
base_valid_rmse = rmse(y_valid, valid_pred_base)
base_valid_r2 = r2_score(y_valid, valid_pred_base)
print(f"[Baseline RF][{TARGET}] valid RMSE={base_valid_rmse:.4f}, R2={base_valid_r2:.4f}")

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
    n_iter=35,
    scoring="neg_root_mean_squared_error",
    cv=ps, verbose=1, random_state=42, n_jobs=-1, refit=False
)
search.fit(X_tv, y_tv)
best_params = search.best_params_
best_cv_rmse = -search.best_score_
print(f"[Search][{TARGET}] Best params: {best_params}")
print(f"[Search][{TARGET}] Best valid RMSE (cv metric): {best_cv_rmse:.4f}")

rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1, **{k.split("__")[1]: v for k, v in best_params.items()})
tuned_pipe = Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("rf", rf_tuned)])
tuned_pipe.fit(X_train, y_train)
valid_pred_best = tuned_pipe.predict(X_valid)
best_valid_rmse = rmse(y_valid, valid_pred_best)
best_valid_r2 = r2_score(y_valid, valid_pred_best)
print(f"[Best RF][{TARGET}] valid RMSE={best_valid_rmse:.4f}, R2={best_valid_r2:.4f}")

rf_final = RandomForestRegressor(random_state=42, n_jobs=-1, **{k.split("__")[1]: v for k, v in best_params.items()})
final_pipe = Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("rf", rf_final)])
final_pipe.fit(X_tv, y_tv)
test_pred = final_pipe.predict(X_test)
test_rmse = rmse(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)
print(f"[Final Test][{TARGET}] RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

safe_name = TARGET.replace("(GT)", "").replace("(", "_").replace(")", "")
pred_df = pd.DataFrame({
    DATETIME_COL: test_df[DATETIME_COL].values if DATETIME_COL in test_df.columns else np.arange(len(y_test)),
    f"{TARGET}_actual": y_test.values,
    f"{TARGET}_pred": test_pred
})
pred_path = os.path.join(PRED_DIR, f"pred_{safe_name}.csv")
pred_df.to_csv(pred_path, index=False)
print(f"[{TARGET}] Saved predictions -> {pred_path}")

model_path = os.path.join(MODELS_DIR, f"rf_{safe_name}.joblib")
joblib.dump(final_pipe, model_path)
print(f"[{TARGET}] Saved model -> {model_path}")

rf_used: RandomForestRegressor = final_pipe.named_steps["rf"]
feature_names = X_test.columns
importances = pd.Series(rf_used.feature_importances_, index=feature_names).sort_values(ascending=False)
fi_path = os.path.join(OUT_DIR, f"feature_importance_{safe_name}.csv")
importances.to_csv(fi_path, header=["importance"])
print(f"[{TARGET}] Saved feature importance -> {fi_path}")

y_true_test_series = test_df[TARGET].reset_index(drop=True)
y_naive = y_true_test_series.shift(1).bfill().to_numpy()
naive_rmse = rmse(y_true_test_series.to_numpy().astype(float), y_naive.astype(float))

summary_df = pd.DataFrame([{
    "target": TARGET,
    "naive_RMSE_test": round(naive_rmse, 4),
    "baseline_valid_RMSE": round(base_valid_rmse, 4),
    "best_valid_RMSE": round(best_valid_rmse, 4),
    "best_valid_R2": round(best_valid_r2, 4),
    "test_RMSE": round(test_rmse, 4),
    "test_R2": round(test_r2, 4),
    "pred_csv": pred_path,
    "model_path": model_path,
    "feat_importance_csv": fi_path
}])
summary_path = os.path.join(OUT_DIR, "summary_metrics.csv")
summary_df.to_csv(summary_path, index=False)
print("\nSaved summary metrics ->", summary_path)
print("Done.")
