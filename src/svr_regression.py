
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def build_X(df: pd.DataFrame, targets):

    X = df.drop(columns=[c for c in targets if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    return X


def dropna_pairwise(X: np.ndarray, y: np.ndarray):

    mask = np.isfinite(X).all(axis=1)
    if mask.sum() < len(mask):
        mask &= np.isfinite(y)
    return X[mask], y[mask]


def main():
  
    DATA_DIR = Path(".")
    train = pd.read_csv(DATA_DIR / "cleaned_features_train.csv")
    val = pd.read_csv(DATA_DIR / "cleaned_features_val.csv")
    test = pd.read_csv(DATA_DIR / "cleaned_features_test.csv")

    print("Train shape:", train.shape)
    print("Val shape:", val.shape)
    print("Test shape:", test.shape)

 
    drop_candidates = [
        "DateTime", "datetime", "date", "time", "Date", "Time",
        "Unnamed: 0", "index"
    ]

    for df in (train, val, test):
        for col in drop_candidates:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

 
    targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    targets = [t for t in targets if t in train.columns]
    print("Detected targets:", targets)

 
    X_train = build_X(train, targets)
    X_val = build_X(val, targets)
    X_test = build_X(test, targets)


    common_cols = sorted(
        set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
    )
    X_train = X_train[common_cols].copy()
    X_val = X_val[common_cols].copy()
    X_test = X_test[common_cols].copy()

    print(f"Feature count: {len(common_cols)}")
    print("Example features:", common_cols[:10])

 
    y_train_all = {t: train[t].values for t in targets}
    y_val_all = {t: val[t].values for t in targets}
    y_test_all = {t: test[t].values for t in targets}

    C_grid = [1.0, 5.0, 10.0, 50.0]
    epsilon_grid = [0.05, 0.1, 0.2]
    gamma_grid = ['scale', 0.1, 0.01]

    results = []
    plot_dir = DATA_DIR / "svr_plots"
    plot_dir.mkdir(exist_ok=True)

 
    for target in targets:
        print("\n===============================")
        print("Training SVR for target:", target)
        print("===============================")

        y_tr = y_train_all[target]
        y_va = y_val_all[target]
        y_te = y_test_all[target]


        Xtr, ytr = dropna_pairwise(X_train.values, y_tr)
        Xva, yva = dropna_pairwise(X_val.values, y_va)
        Xte, yte = dropna_pairwise(X_test.values, y_te)

 
        pipe = Pipeline([('svr', SVR(kernel='rbf'))])
        best_rmse = None
        best_params = None


        for C in C_grid:
            for eps in epsilon_grid:
                for gam in gamma_grid:
                    pipe.set_params(svr__C=C, svr__epsilon=eps, svr__gamma=gam)
                    pipe.fit(Xtr, ytr)
                    pred_val = pipe.predict(Xva)
                    rmse_val = float(
                        np.sqrt(mean_squared_error(yva, pred_val))
                    )
                    if (best_rmse is None) or (rmse_val < best_rmse):
                        best_rmse = rmse_val
                        best_params = dict(C=C, epsilon=eps, gamma=gam)

        print("Best params on val:", best_params)
        print("Best val RMSE:", best_rmse)

        X_tv = np.vstack([Xtr, Xva])
        y_tv = np.concatenate([ytr, yva])

     
        final = Pipeline([
            ('svr', SVR(kernel='rbf',
                        C=best_params["C"],
                        epsilon=best_params["epsilon"],
                        gamma=best_params["gamma"]))
        ])
        final.fit(X_tv, y_tv)

       
        pred_test = final.predict(Xte)
        rmse_test = float(np.sqrt(mean_squared_error(yte, pred_test)))
        print("Test RMSE:", rmse_test)

        results.append({
            "target": target,
            "best_C": best_params["C"],
            "best_epsilon": best_params["epsilon"],
            "best_gamma": best_params["gamma"],
            "val_RMSE_best": best_rmse,
            "test_RMSE": rmse_test
        })

       
        plt.figure(figsize=(10, 4))
        idx = np.arange(len(yte))
        plt.plot(idx, yte, label="Truth")
        plt.plot(idx, pred_test, label="SVR Pred")
        plt.title(f"SVR Test Predictions vs Truth - {target}")
        plt.xlabel("Test Sample Index")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()

        out_png = plot_dir / f"svr_{target.replace('(GT)', '')}_test_plot.png"
        plt.savefig(out_png, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {out_png}")


    res_df = pd.DataFrame(results).sort_values("test_RMSE")
    print("\nSVR summary:")
    print(res_df)

    out_csv = DATA_DIR / "svr_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()