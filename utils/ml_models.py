# utils/ml_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def random_forest_volatility(df_feat: pd.DataFrame):
    """
    Train RF to predict next-step volatility_12.
    Assumes df_feat has columns from prepare_features().
    """
    if df_feat is None or df_feat.empty:
        # Return a dummy model + empty preds to keep the page alive
        return RandomForestRegressor(n_estimators=10, random_state=42), pd.Series(dtype=float, name="rf_vol_pred")

    # y = next-step target
    y = df_feat["volatility_12"].shift(-1)
    X = df_feat[["ret","vol","hl_spread","ma_12","ma_48","ma_ratio","volatility_12"]].copy()

    # Clean finite
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).astype(float)

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]

    if len(X) < 100:  # not enough samples, return safe outputs
        return RandomForestRegressor(n_estimators=10, random_state=42), pd.Series(dtype=float, name="rf_vol_pred")

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    preds = pd.Series(model.predict(X), index=X.index, name="rf_vol_pred")
    return model, preds
