
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ---------- paths ----------
DATA_DIR = Path(".")  # 改成你的目录也行；默认当前目录
train_csv = DATA_DIR / "cleaned_features_train.csv"
val_csv   = DATA_DIR / "cleaned_features_val.csv"
test_csv  = DATA_DIR / "cleaned_features_test.csv"

assert train_csv.exists() and val_csv.exists() and test_csv.exists(), "CSV files not found."

# ---------- load ----------
train = pd.read_csv(train_csv)
val   = pd.read_csv(val_csv)
test  = pd.read_csv(test_csv)

# 可选：如仍有 NaN，先丢弃（通常不需要）
train = train.dropna()
val   = val.dropna()
test  = test.dropna()

# ---------- targets ----------
candidate_targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
targets = [c for c in candidate_targets if c in train.columns and c in val.columns and c in test.columns]
if not targets:
    raise RuntimeError(f"No expected target columns found. Columns={train.columns.tolist()}")

def feature_cols(df, tgts):
    # 使用除目标列外的所有列作为特征（包括 Hour/Weekday/Month 等）
    return [c for c in df.columns if c not in tgts]

X_train = train[feature_cols(train, targets)]
X_val   = val[feature_cols(val, targets)]
X_test  = test[feature_cols(test, targets)]

# 小网格（速度快，效果够用）
C_grid       = [1.0, 5.0, 10.0, 50.0]
epsilon_grid = [0.05, 0.1, 0.2]
gamma_grid   = ['scale', 0.1, 0.01]

results = []
plot_dir = Path("./svr_plots")
plot_dir.mkdir(exist_ok=True)

for target in targets:
    y_train = train[target].values
    y_val   = val[target].values
    y_test  = test[target].values

    pipe = Pipeline([('svr', SVR(kernel='rbf'))])

    # ----- tune on validation -----
    best_rmse = None
    best_params = None
    for C in C_grid:
        for eps in epsilon_grid:
            for gam in gamma_grid:
                pipe.set_params(svr__C=C, svr__epsilon=eps, svr__gamma=gam)
                pipe.fit(X_train, y_train)
                pred_val = pipe.predict(X_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
                if (best_rmse is None) or (rmse_val < best_rmse):
                    best_rmse = rmse_val
                    best_params = dict(C=C, epsilon=eps, gamma=gam)

    # ----- retrain on train+val with best params -----
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    final = Pipeline([('svr', SVR(kernel='rbf',
                                  C=best_params['C'],
                                  epsilon=best_params['epsilon'],
                                  gamma=best_params['gamma']))])
    final.fit(X_tv, y_tv)

    # ----- evaluate on test -----
    pred_test = final.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    results.append({
        'target': target,
        'best_C': best_params['C'],
        'best_epsilon': best_params['epsilon'],
        'best_gamma': best_params['gamma'],
        'val_RMSE_best': float(best_rmse),
        'test_RMSE': float(rmse_test)
    })

    # ----- plot: truth vs prediction on test -----
    fig = plt.figure(figsize=(10, 4))
    idx = np.arange(len(y_test))
    plt.plot(idx, y_test, label='Truth')
    plt.plot(idx, pred_test, label='SVR Pred')
    plt.title(f"SVR Test Predictions vs Truth - {target}")
    plt.xlabel("Test Time Index")
    plt.ylabel(target)
    plt.legend()
    fig.tight_layout()
    out_png = plot_dir / f"svr_{target.replace('(GT)','').replace('/','_')}_test_plot.png"
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {out_png}")

# ----- save results table -----
results_df = pd.DataFrame(results)
results_df.to_csv("svr_results.csv", index=False)
print("\nDetected targets:", targets)
print("Saved results table: svr_results.csv")
print("Saved plots in: ./svr_plots")