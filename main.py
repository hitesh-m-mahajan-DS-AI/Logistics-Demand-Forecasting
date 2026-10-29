"""Leakage-aware one-step forecasting of public UCI logistics observations."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent
SOURCE_NAME = "Daily_Demand_Forecasting_Orders.csv"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/409/daily%2Bdemand%2Bforecasting%2Borders.zip"
SOURCE_SHA256 = "adf8dda021beae76b13b1cbb5b3c075d064f03b3353a711c1ddc357c6f89e924"
FEATURES = ["lag_1", "lag_2", "lag_3"]
TRAIN_END = 50


def load_source(path=None):
    """Use the bundled 5 KB CSV; download the original only when it is absent."""
    path = Path(path) if path is not None else ROOT / "data" / SOURCE_NAME
    if not path.exists():
        with urlopen(SOURCE_URL, timeout=30) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                data = archive.read(SOURCE_NAME)
        if sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded source differs from the audited version; review before use.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source checksum mismatch. Use the unmodified UCI CSV.")
    # Periods are thousands separators: 539.577 means 539577, not 539.577.
    raw = pd.read_csv(path, sep=";", thousands=".")
    if raw.shape != (60, 13) or raw.isna().any().any():
        raise ValueError("Expected 60 complete observations with 13 source columns.")
    if not all(pd.api.types.is_integer_dtype(raw[column]) for column in raw.columns):
        raise ValueError("Source values must parse as integer counts/calendar codes.")
    data = raw.iloc[:, [0, 1, 12]].copy()
    data.columns = ["week_of_month", "day_of_week_code", "total_orders"]
    data.insert(0, "observation", range(1, len(data) + 1))
    if (data["total_orders"] <= 0).any() or not data["day_of_week_code"].between(2, 6).all():
        raise ValueError("Unexpected target values or weekday codes.")
    if not data["week_of_month"].between(1, 5).all():
        raise ValueError("Unexpected week-of-month code.")
    return raw, data


def audit_source(raw, data):
    """Record evidence about units, target leakage and limited date information."""
    expected_next = data["day_of_week_code"].shift(1).map({2: 3, 3: 4, 4: 5, 5: 6, 6: 2})
    breaks = data.loc[expected_next.notna() & (data["day_of_week_code"] != expected_next)]
    return {
        "source_url": SOURCE_URL, "csv_sha256": SOURCE_SHA256, "rows": len(data),
        "license": "CC BY 4.0", "doi": "https://doi.org/10.24432/C5BC8T",
        "periods_are_thousands_separators": True,
        "target_equals_type_A_plus_B_plus_C": bool((raw.iloc[:, 4:7].sum(axis=1) == data["total_orders"]).all()),
        "target_equals_nonurgent_plus_urgent_plus_fiscal": bool((raw.iloc[:, [2, 3, 7]].sum(axis=1) == data["total_orders"]).all()),
        "weekday_cycle_breaks_at_observations": breaks["observation"].tolist(),
        "week_field_decreases_at_observations": data.loc[data["week_of_month"].diff() < 0, "observation"].tolist(),
        "ordering_note": "UCI labels this time-series data. File order is preserved, but there are no dates "
                         "and two weekday-cycle breaks; consecutive business-day chronology is not verified. "
                         "All horizons are next observation under the source-order assumption.",
    }


def make_features(data):
    """Shift BEFORE using the rolling window: no current-target value is a feature."""
    frame = data[["observation", "total_orders"]].copy()
    for lag in [1, 2, 3]:
        frame[f"lag_{lag}"] = frame["total_orders"].shift(lag)
    frame["trailing_mean_3"] = frame["total_orders"].shift(1).rolling(3).mean()
    return frame.dropna().reset_index(drop=True)


def evaluate(frame, train_end=TRAIN_END):
    """Fit once through observation 50; score rolling one-step inputs for 51-60."""
    train = frame[frame["observation"] <= train_end]
    test = frame[frame["observation"] > train_end]
    if train.empty or test.empty:
        raise ValueError("Training and holdout sets must both contain observations.")
    model = LinearRegression()
    model.fit(train[FEATURES], train["total_orders"])
    predictions = test[["observation", "total_orders"]].rename(columns={"total_orders": "actual"}).copy()
    predictions["last_observation"] = test["lag_1"].to_numpy()
    predictions["trailing_mean_3"] = test["trailing_mean_3"].to_numpy()
    # Forecasts are counts; apply the same fixed zero floor to regression outputs.
    predictions["linear_regression"] = np.maximum(model.predict(test[FEATURES]), 0)
    scores = []
    for name in ["last_observation", "trailing_mean_3", "linear_regression"]:
        scores.append({"method": name,
                       "mae_orders": float(mean_absolute_error(predictions["actual"], predictions[name])),
                       "rmse_orders": float(root_mean_squared_error(predictions["actual"], predictions[name]))})
    scores = pd.DataFrame(scores).sort_values("mae_orders").reset_index(drop=True)
    return model, predictions.reset_index(drop=True), scores


def next_observation_forecasts(data, frame):
    """After evaluation, refit on all available labels for an unscored illustration."""
    model = LinearRegression().fit(frame[FEATURES], frame["total_orders"])
    recent = data["total_orders"].iloc[-3:]
    row = pd.DataFrame([[recent.iloc[-1], recent.iloc[-2], recent.iloc[-3]]], columns=FEATURES)
    return pd.DataFrame([
        {"observation": len(data) + 1, "method": "last_observation", "predicted_orders": float(recent.iloc[-1])},
        {"observation": len(data) + 1, "method": "trailing_mean_3", "predicted_orders": float(recent.mean())},
        {"observation": len(data) + 1, "method": "linear_regression", "predicted_orders": float(max(model.predict(row)[0], 0))},
    ])


def save_charts(data, predictions, scores, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    colors = {"last_observation": "#778899", "trailing_mean_3": "#D97732", "linear_regression": "#007F78"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    axes[0].plot(data["observation"], data["total_orders"], color="#142D3D", label="Actual orders")
    axes[0].axvline(TRAIN_END + 0.5, color="#D97732", linestyle="--", label="Holdout begins")
    axes[0].set(title="History: 60 observations in original source order", ylabel="Orders")
    axes[0].legend(loc="upper right")
    axes[1].plot(predictions["observation"], predictions["actual"], color="#142D3D", marker="o", label="Actual")
    for method, color in colors.items():
        axes[1].plot(predictions["observation"], predictions[method], color=color, marker=".", label=method)
    axes[1].set(title="Final 10 observations: one-step forecasts, not a 10-step forecast",
                xlabel="Source observation (no calendar date provided)", ylabel="Orders")
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "forecast_comparison.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.barh(scores["method"], scores["mae_orders"], color=[colors[name] for name in scores["method"]])
    axis.invert_yaxis()
    axis.set(title="Holdout MAE: lower is better (10 observations)", xlabel="Mean absolute error (orders)")
    fig.savefig(output / "mae_comparison.png", dpi=160)
    plt.close(fig)


def main():
    start = time.perf_counter()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    raw, data = load_source()
    audit = audit_source(raw, data)
    frame = make_features(data)
    model, predictions, scores = evaluate(frame)
    forecasts = next_observation_forecasts(data, frame)
    data.to_csv(ROOT / "data" / "cleaned_demand.csv", index=False)
    frame.to_csv(output / "lagged_features.csv", index=False)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    scores.to_csv(output / "model_comparison.csv", index=False)
    forecasts.to_csv(output / "next_observation_forecasts.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "coefficient": model.coef_}).to_csv(output / "coefficients.csv", index=False)
    (output / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    save_charts(data, predictions, scores, output)
    summary = {
        "source_observations": len(data), "first_training_target_observation": 4,
        "last_training_target_observation": TRAIN_END, "training_examples": int((frame["observation"] <= TRAIN_END).sum()),
        "holdout_examples": len(predictions), "holdout_observations": predictions["observation"].tolist(),
        "features": FEATURES, "scores": scores.to_dict(orient="records"),
        "holdout_lowest_mae_method": scores.iloc[0]["method"],
        "holdout_model_intercept": float(model.intercept_),
        "weekday_cycle_breaks": audit["weekday_cycle_breaks_at_observations"],
        "evaluation": "Fixed training fit; sequential one-step inputs use only already observed prior totals. "
                      "Source order assumed, exact dates unverified. No hyperparameter tuning; "
                      "lowest-error comparison is descriptive, not deployment selection.",
        "run_seconds": round(time.perf_counter() - start, 3),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(scores.to_string(index=False))
    print(f"\nTraining examples: {summary['training_examples']}; holdout: {len(predictions)}")
    print("Lowest observed holdout MAE:", summary["holdout_lowest_mae_method"])
    print("Source weekday-cycle breaks:", audit["weekday_cycle_breaks_at_observations"])
    print("Next-observation forecasts (unscored illustration):\n", forecasts.to_string(index=False))


if __name__ == "__main__":
    main()
