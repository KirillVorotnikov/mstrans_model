"""Autonomous selective forecasting pipeline.

Required user data files:
  - labels_day_train.csv
  - labels_day_test.csv
  - test_submission.csv

The script builds, in one run:
  calendar LightGBM -> seasonal ARIMA 30% -> route 5 correction ->
  Chronos-2 20% -> selective per-route CatBoost.

Missing Python packages are installed automatically. Chronos-2 weights are
downloaded from Hugging Face on the first run and then reused from the cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys
import warnings


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))


def ensure_dependencies() -> None:
    requirements = []
    checks = {
        "numpy": "numpy>=1.26,<3",
        "pandas": "pandas>=2.2,<4",
        "lightgbm": "lightgbm>=4,<5",
        "statsmodels": "statsmodels>=0.14,<1",
        "catboost": "catboost==1.2.10",
        "chronos": "chronos-forecasting==2.3.2",
        "pyarrow": "pyarrow>=15",
    }
    for module, package in checks.items():
        if importlib.util.find_spec(module) is None:
            requirements.append(package)
    if requirements:
        print("Installing:", ", ".join(requirements), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])


ensure_dependencies()

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from catboost import CatBoostRegressor
from chronos import Chronos2Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
ACTIVE_ROUTES = [route for route in ROUTES if route != 5]
SELECTED_CATBOOST_ROUTES = [11, 12, 25, 28]
CATBOOST_WEIGHTS = {11: 0.35, 12: 0.10, 25: 0.35, 28: 0.45}
KEYS = ["route", "date", "hour"]
CHRONOS_WEIGHT = 0.20
CHRONOS_QUANTILE = "0.7"
LIGHTGBM_N_ESTIMATORS = 536  # fixed from the completed Sep-Oct validation


CATBOOST_PARAMS = {
    11: dict(
        loss_function="Quantile:alpha=0.7", iterations=1500,
        learning_rate=0.09739630194717583, depth=4,
        l2_leaf_reg=0.005817066538204014,
        random_strength=0.011762426499887868,
        bagging_temperature=0.3954299769492494, border_count=239,
    ),
    12: dict(
        loss_function="MAE", iterations=1500,
        learning_rate=0.13944792632118025, depth=4,
        l2_leaf_reg=0.13444305532609163,
        random_strength=0.7017156693033213,
        bagging_temperature=0.8848219633796027, border_count=217,
    ),
    25: dict(
        loss_function="Quantile:alpha=0.7", iterations=1500,
        learning_rate=0.03428475736257873, depth=5,
        l2_leaf_reg=0.3119140149980465,
        random_strength=1.1005657369494315e-08,
        bagging_temperature=0.431920380647568, border_count=122,
    ),
    28: dict(
        loss_function="RMSE", iterations=1500,
        learning_rate=0.04835199705442417, depth=5,
        l2_leaf_reg=0.0034603811977855785,
        random_strength=8.767862485337334e-07,
        bagging_temperature=0.5720601179359943, border_count=153,
    ),
}
for params in CATBOOST_PARAMS.values():
    params.update(
        random_seed=SEED,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def find_data_file(filename: str, alternatives: list[str]) -> Path:
    candidates = [ROOT / path for path in alternatives]
    candidates += [WORKSPACE / path for path in alternatives]
    candidates += [Path.cwd() / path for path in alternatives]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        matches = sorted(kaggle.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Cannot find {filename}. Put it in data/ or next to final.py."
    )


TRAIN_PATH = find_data_file(
    "labels_day_train.csv",
    ["data/labels_day_train.csv", "data/labels/labels_day_train.csv", "labels_day_train.csv"],
)
HISTORY_TEST_PATH = find_data_file(
    "labels_day_test.csv",
    ["data/labels_day_test.csv", "data/labels/labels_day_test.csv", "labels_day_test.csv"],
)
def russian_calendar():
    dates = pd.date_range("2025-01-01", "2026-12-31", freq="D")
    weekends = set(dates[dates.dayofweek >= 5])
    extra_off = set(pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-05-01", "2025-05-02", "2025-05-08", "2025-05-09",
        "2025-06-12", "2025-06-13", "2025-11-03", "2025-11-04",
        "2025-12-31",
    ]))
    extra_off |= set(pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        "2026-01-09", "2026-02-23", "2026-03-09", "2026-05-01",
        "2026-05-11", "2026-06-12", "2026-11-04", "2026-12-31",
    ]))
    off_days = (weekends | extra_off) - {pd.Timestamp("2025-11-01")}
    preholidays = set(pd.to_datetime([
        "2025-03-07", "2025-04-30", "2025-06-11", "2025-11-01",
        "2026-04-30", "2026-05-08", "2026-06-11", "2026-11-03",
    ]))
    return off_days, preholidays


OFF_DAYS, PREHOLIDAYS = russian_calendar()
LEGACY_HOLIDAYS = set(pd.to_datetime([
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
    "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
    "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-02",
    "2025-05-08", "2025-05-09", "2025-06-12", "2025-06-13",
    "2025-11-03", "2025-11-04", "2025-12-31",
])) | {date for date in OFF_DAYS if date.year == 2026}


def seasonal_features(frame: pd.DataFrame, exact_calendar: bool) -> pd.DataFrame:
    frame = frame.copy()
    dt = pd.to_datetime(frame["date"])
    frame["hour_val"] = frame["hour"]
    frame["dayofweek"] = dt.dt.dayofweek
    frame["day"] = dt.dt.day
    frame["month"] = dt.dt.month
    frame["dayofyear"] = dt.dt.dayofyear
    frame["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    frame["quarter"] = dt.dt.quarter
    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype(int)
    frame["is_month_start"] = dt.dt.is_month_start.astype(int)
    frame["is_month_end"] = dt.dt.is_month_end.astype(int)
    if exact_calendar:
        frame["is_off_day"] = dt.isin(OFF_DAYS).astype(int)
        frame["is_holiday"] = (
            (frame["dayofweek"] < 5) & frame["is_off_day"].eq(1)
        ).astype(int)
        frame["is_preholiday"] = dt.isin(PREHOLIDAYS).astype(int)
        frame["is_working_weekend"] = (
            frame["is_weekend"].eq(1) & frame["is_off_day"].eq(0)
        ).astype(int)
        frame["is_extra_off_day"] = (
            (frame["dayofweek"] < 5) & frame["is_off_day"].eq(1)
        ).astype(int)
    else:
        frame["is_holiday"] = dt.isin(LEGACY_HOLIDAYS).astype(int)
        frame["is_off_day"] = (
            frame["is_weekend"].eq(1) | frame["is_holiday"].eq(1)
        ).astype(int)
        frame["is_preholiday"] = (
            dt + pd.Timedelta(days=1)
        ).isin(LEGACY_HOLIDAYS).astype(int)
    for name, period, value in [
        ("hour", 24.0, frame["hour_val"]),
        ("dow", 7.0, frame["dayofweek"]),
        ("doy", 365.25, frame["dayofyear"]),
        ("month", 12.0, frame["month"]),
    ]:
        frame[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    hour_in_week = frame["dayofweek"] * 24 + frame["hour_val"]
    for k in [1, 2, 3]:
        frame[f"fourier_week_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_week / 168.0)
        frame[f"fourier_week_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_week / 168.0)
    hour_in_year = (frame["dayofyear"] - 1) * 24 + frame["hour_val"]
    for k in [1, 2]:
        frame[f"fourier_year_sin_{k}"] = np.sin(2 * np.pi * k * hour_in_year / 8766.0)
        frame[f"fourier_year_cos_{k}"] = np.cos(2 * np.pi * k * hour_in_year / 8766.0)
    return frame


def add_base_profiles(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    source = result.loc[train_mask]
    specs = [
        (["route", "hour", "dayofweek"], "rhd", ["mean", "median", "std"]),
        (["route", "hour", "is_off_day"], "rho", ["mean", "median"]),
        (["route", "hour"], "rh", ["mean", "median"]),
    ]
    for keys, tag, stats in specs:
        profile = source.groupby(keys)["boardings"].agg(stats).reset_index()
        profile = profile.rename(
            columns={stat: f"profile_{tag}_{stat}" for stat in stats}
        )
        result = result.merge(profile, on=keys, how="left", validate="many_to_one")
    return result


def calendar_matrix(dates: pd.DatetimeIndex) -> np.ndarray:
    dow = dates.dayofweek.to_numpy()
    dayofyear = dates.dayofyear.to_numpy()
    return np.concatenate([
        np.eye(7, dtype=np.float32)[dow],
        np.asarray(dates.isin(OFF_DAYS), dtype=np.float32)[:, None],
        np.asarray(dates.isin(PREHOLIDAYS), dtype=np.float32)[:, None],
        np.column_stack([
            np.sin(2 * np.pi * dayofyear / 365.25),
            np.cos(2 * np.pi * dayofyear / 365.25),
        ]).astype(np.float32),
    ], axis=1)


def arima_forecast(
    history_grid: pd.DataFrame,
    train_end: pd.Timestamp,
    forecast_dates: pd.DatetimeIndex,
) -> np.ndarray:
    train = history_grid.loc[history_grid["date"] <= train_end, KEYS + ["boardings"]].copy()
    daily = train.groupby(["route", "date"], as_index=False)["boardings"].sum()
    train_dates = pd.date_range(train["date"].min(), train_end, freq="D")
    train_exog = calendar_matrix(train_dates)[:, 7:]
    future_exog = calendar_matrix(forecast_dates)[:, 7:]
    daily_forecasts = np.zeros((len(ROUTES), len(forecast_dates)), dtype=np.float32)
    for route_index, route in enumerate(ROUTES):
        series = (
            daily.loc[daily["route"] == route]
            .set_index("date")["boardings"]
            .reindex(train_dates, fill_value=0.0)
            .astype(float)
        )
        if series.sum() == 0:
            continue
        model = SARIMAX(
            np.log1p(series.to_numpy()), exog=train_exog,
            order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), trend="n",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=100)
        forecast = np.expm1(fitted.forecast(len(forecast_dates), exog=future_exog))
        daily_forecasts[route_index] = np.maximum(0, forecast)
        print(f"ARIMA route {route} done", flush=True)
    daily_totals = train.groupby(["route", "date"])["boardings"].transform("sum")
    train["hour_share"] = np.where(
        daily_totals > 0, train["boardings"] / daily_totals, 0.0
    )
    train["dayofweek"] = train["date"].dt.dayofweek
    train["is_off_day"] = train["date"].isin(OFF_DAYS).astype(int)
    profile = train.groupby(
        ["route", "hour", "dayofweek", "is_off_day"]
    )["hour_share"].mean()
    hourly = np.zeros((len(ROUTES), len(forecast_dates), 24), dtype=np.float32)
    for route_index, route in enumerate(ROUTES):
        fallback = train.loc[train["route"] == route].groupby("hour")["hour_share"].mean()
        for day_index, date in enumerate(forecast_dates):
            off = int(date in OFF_DAYS)
            shares = np.array([
                profile.get((route, hour, date.dayofweek, off), np.nan)
                for hour in range(24)
            ])
            if not np.isfinite(shares).all() or shares.sum() <= 0:
                shares = np.array([fallback.get(hour, 0.0) for hour in range(24)])
            shares /= max(shares.sum(), 1e-9)
            hourly[route_index, day_index] = daily_forecasts[route_index, day_index] * shares
    return hourly.reshape(-1)


def route5_external_profile(
    history_grid: pd.DataFrame,
    forecast: pd.DataFrame,
    history_end: pd.Timestamp,
) -> np.ndarray:
    active = history_grid.loc[
        (history_grid["route"] != 5) & (history_grid["date"] <= history_end)
    ].copy()
    active["dow"] = active["date"].dt.dayofweek
    active["off"] = active["date"].isin(OFF_DAYS).astype(int)
    daily_total = active.groupby(["route", "date"])["boardings"].transform("sum")
    active["hour_share"] = np.where(
        daily_total > 0, active["boardings"] / daily_total, 0.0
    )
    share_profile = active.groupby(["off", "hour"])["hour_share"].mean()
    daily = active.groupby(["route", "date"], as_index=False)["boardings"].sum()
    route_mean = daily.groupby("route")["boardings"].transform("mean")
    daily["relative"] = daily["boardings"] / route_mean.clip(lower=1)
    daily["dow"] = daily["date"].dt.dayofweek
    daily["off"] = daily["date"].isin(OFF_DAYS).astype(int)
    daily["month"] = daily["date"].dt.month
    day_profile = daily.groupby(["off", "dow"])["relative"].median()
    month_profile = daily.groupby("month")["relative"].median()
    dates = pd.date_range("2025-12-16", "2025-12-31", freq="D")
    day_weights = np.array([
        day_profile.get((int(date in OFF_DAYS), date.dayofweek), 1.0) for date in dates
    ])
    first_week_days = dates <= pd.Timestamp("2025-12-22")
    day_weights[0] *= 0.35
    day_totals = day_weights * (40_000 / day_weights[first_week_days].sum())
    values = np.zeros(len(forecast), dtype=float)
    route5 = forecast["route"].eq(5)
    for date, total in zip(dates, day_totals):
        mask = route5 & forecast["date"].eq(date)
        rows = forecast.loc[mask]
        shares = np.array([
            share_profile.get((int(date in OFF_DAYS), hour), 0.0)
            for hour in rows["hour"]
        ])
        hours = rows["hour"].to_numpy()
        shares[(hours < 5) | (hours > 22)] = 0
        shares[hours == 5] *= 0.5
        shares[hours == 22] *= 0.15
        if date == pd.Timestamp("2025-12-16"):
            shares[hours < 16] = 0
        shares /= max(shares.sum(), 1e-9)
        values[np.flatnonzero(mask)] = total * shares
    # Reproduce the successful 85k variant: first round the original profile,
    # then shrink only 23-31 December.
    values = np.rint(np.maximum(values, 0))
    first_week = route5 & forecast["date"].between("2025-12-16", "2025-12-22")
    later = (
        route5
        & forecast["date"].between("2025-12-23", "2025-12-31")
    )
    values[later] *= (85_000 - values[first_week].sum()) / values[later].sum()

    # Route 5 has no usable training history. For 2026 use a conservative
    # scenario of 5.2k trips/day (below the reported first-week average),
    # modulated by profiles of the other routes. This is an explicit assumption
    # and must be replaced once route-5 observations exist.
    dates_2026 = pd.date_range("2026-01-01", "2026-12-31", freq="D")
    raw_factors = np.array([
        day_profile.get((int(date in OFF_DAYS), date.dayofweek), 1.0)
        * month_profile.get(date.month, 1.0)
        for date in dates_2026
    ])
    raw_factors /= raw_factors.mean()
    for date, total in zip(dates_2026, 5_200 * raw_factors):
        mask = route5 & forecast["date"].eq(date)
        if not mask.any():
            continue
        rows = forecast.loc[mask]
        hours = rows["hour"].to_numpy()
        shares = np.array([
            share_profile.get((int(date in OFF_DAYS), hour), 0.0)
            for hour in hours
        ])
        shares[(hours < 5) | (hours > 22)] = 0
        shares[hours == 5] *= 0.5
        shares[hours == 22] *= 0.15
        shares /= max(shares.sum(), 1e-9)
        values[np.flatnonzero(mask)] = total * shares
    return values


def chronos_calendar_frame(routes, dates):
    frame = pd.MultiIndex.from_product(
        [routes, pd.DatetimeIndex(dates)], names=["item_id", "timestamp"]
    ).to_frame(index=False)
    dt = frame["timestamp"]
    dow, doy = dt.dt.dayofweek, dt.dt.dayofyear
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["is_off_day"] = dt.isin(OFF_DAYS).astype(float)
    frame["is_preholiday"] = dt.isin(PREHOLIDAYS).astype(float)
    frame["is_month_start"] = dt.dt.is_month_start.astype(float)
    frame["is_month_end"] = dt.dt.is_month_end.astype(float)
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", type=int, choices=[2025, 2026], default=2025,
        help="2025 reproduces the competition submission; 2026 creates a full-year forecast.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main(year: int, requested_output: Path | None = None) -> None:
    print("Train:", TRAIN_PATH)
    print("History test:", HISTORY_TEST_PATH)
    raw = pd.concat([
        pd.read_csv(TRAIN_PATH, sep=";"),
        pd.read_csv(HISTORY_TEST_PATH, sep=";"),
    ], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    history_end = raw["date"].max()
    if year == 2025:
        template_path = find_data_file(
            "test_submission.csv", ["data/test_submission.csv", "test_submission.csv"]
        )
        target_template = pd.read_csv(template_path, sep=";")[KEYS]
        target_template["date"] = pd.to_datetime(target_template["date"])
        print("Template:", template_path)
    else:
        target_dates = pd.date_range("2026-01-01", "2026-12-31", freq="D")
        target_template = pd.MultiIndex.from_product(
            [ROUTES, target_dates, range(24)], names=KEYS
        ).to_frame(index=False)
        print("2026 template generated:", len(target_template), "rows")
    target_dates = pd.DatetimeIndex(sorted(target_template["date"].unique()))
    model_forecast_dates = pd.date_range(
        history_end + pd.Timedelta(days=1), target_dates[-1], freq="D"
    )
    if target_dates[0] < model_forecast_dates[0]:
        raise ValueError("Forecast target overlaps the observed history")
    print(
        "Model horizon:", model_forecast_dates[0].date(), "..",
        model_forecast_dates[-1].date(), f"({len(model_forecast_dates)} days)",
    )
    all_dates = pd.date_range(raw["date"].min(), model_forecast_dates[-1], freq="D")
    grid = pd.MultiIndex.from_product(
        [ROUTES, all_dates, range(24)], names=KEYS
    ).to_frame(index=False)
    grid = grid.merge(raw[KEYS + ["boardings"]], on=KEYS, how="left", validate="one_to_one")
    grid["boardings"] = grid["boardings"].fillna(0.0)

    # 1. Calendar LightGBM with the already selected final hyperparameters.
    lgb_grid = seasonal_features(grid, exact_calendar=True)
    lgb_params = dict(
        objective="regression_l1", metric="mae", boosting_type="gbdt",
        n_estimators=LIGHTGBM_N_ESTIMATORS,
        learning_rate=0.03, num_leaves=63,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    final_train_mask = lgb_grid["date"] <= history_end
    final_featured = add_base_profiles(lgb_grid, final_train_mask)
    feature_columns = [
        column for column in final_featured.columns
        if column not in {"route", "date", "boardings"}
    ]
    final_train = final_featured.loc[final_train_mask]
    future = final_featured.loc[
        final_featured["date"].isin(model_forecast_dates)
    ].copy()
    calendar_model = lgb.LGBMRegressor(**lgb_params)
    calendar_model.fit(final_train[feature_columns], final_train["boardings"])
    calendar_prediction = np.maximum(
        0, calendar_model.predict(future[feature_columns])
    )
    print("LightGBM done; fixed n_estimators=", LIGHTGBM_N_ESTIMATORS, flush=True)

    # 2. ARIMA and the exact 30% blend construction used by the winning file.
    arima_prediction = arima_forecast(grid, history_end, model_forecast_dates)
    blend20_rounded = np.rint(np.maximum(
        0, 0.80 * calendar_prediction + 0.20 * arima_prediction
    ))
    arima_rounded = np.rint(np.maximum(arima_prediction, 0))
    reconstructed_calendar = (blend20_rounded - 0.20 * arima_rounded) / 0.80
    baseline_prediction = np.rint(np.maximum(
        0, 0.70 * reconstructed_calendar + 0.30 * arima_rounded
    )).astype(int)
    baseline = future[KEYS].copy().reset_index(drop=True)
    baseline["prediction"] = baseline_prediction
    route5_values = route5_external_profile(grid, baseline, history_end)
    route5 = baseline["route"].eq(5).to_numpy()
    baseline.loc[route5, "prediction"] = np.rint(route5_values[route5]).astype(int)
    print("Baseline route 5:", baseline.loc[route5, "prediction"].sum(), flush=True)

    # 3. Chronos-2 daily forecast and hourly redistribution.
    history_dates = pd.date_range(raw["date"].min(), history_end, freq="D")
    daily_observed = raw.groupby(["route", "date"], as_index=False)["boardings"].sum()
    daily = pd.MultiIndex.from_product(
        [ACTIVE_ROUTES, history_dates], names=["route", "date"]
    ).to_frame(index=False)
    daily = daily.merge(daily_observed, on=["route", "date"], how="left")
    daily["boardings"] = daily["boardings"].fillna(0.0)
    context = chronos_calendar_frame(ACTIVE_ROUTES, history_dates)
    context = context.merge(
        daily, left_on=["item_id", "timestamp"], right_on=["route", "date"],
        validate="one_to_one",
    ).drop(columns=["route", "date"]).rename(columns={"boardings": "target"})
    future_covariates = chronos_calendar_frame(ACTIVE_ROUTES, model_forecast_dates)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print("Chronos device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map=device, dtype=dtype,
    )
    chronos_daily = pipeline.predict_df(
        context, future_df=future_covariates,
        prediction_length=len(model_forecast_dates), quantile_levels=[0.7],
        batch_size=100, context_length=512, cross_learning=True, freq="D",
    ).rename(columns={"item_id": "route", "timestamp": "date"})
    chronos_daily[CHRONOS_QUANTILE] = chronos_daily[CHRONOS_QUANTILE].clip(lower=0)
    active = baseline.loc[baseline["route"] != 5].copy().rename(
        columns={"prediction": "baseline"}
    )
    daily_baseline = active.groupby(["route", "date"])["baseline"].transform("sum")
    active["hour_share"] = np.divide(
        active["baseline"], daily_baseline,
        out=np.full(len(active), 1 / 24, dtype=float),
        where=daily_baseline.to_numpy() > 0,
    )
    active = active.merge(
        chronos_daily[["route", "date", CHRONOS_QUANTILE]],
        on=["route", "date"], validate="many_to_one",
    )
    active["chronos_hourly"] = active["hour_share"] * active[CHRONOS_QUANTILE]
    active["chronos_blend"] = (
        (1 - CHRONOS_WEIGHT) * active["baseline"]
        + CHRONOS_WEIGHT * active["chronos_hourly"]
    )

    # 4. Fixed selective CatBoost models (only routes with validated gain).
    cat_grid = pd.MultiIndex.from_product(
        [SELECTED_CATBOOST_ROUTES, all_dates, range(24)], names=KEYS
    ).to_frame(index=False)
    cat_grid = cat_grid.merge(
        raw[KEYS + ["boardings"]], on=KEYS, how="left", validate="one_to_one"
    )
    cat_grid["boardings"] = cat_grid["boardings"].fillna(0.0)
    cat_grid = seasonal_features(cat_grid, exact_calendar=False)
    cat_train_mask = cat_grid["date"] <= history_end
    cat_grid = add_base_profiles(cat_grid, cat_train_mask)
    cat_features = [
        column for column in cat_grid.columns
        if column not in {"route", "date", "boardings"}
    ]
    cat_parts = []
    for route in SELECTED_CATBOOST_ROUTES:
        train_route = cat_grid.loc[cat_train_mask & cat_grid["route"].eq(route)]
        future_route = cat_grid.loc[
            cat_grid["date"].isin(model_forecast_dates) & cat_grid["route"].eq(route)
        ]
        model = CatBoostRegressor(**CATBOOST_PARAMS[route])
        model.fit(train_route[cat_features], train_route["boardings"])
        part = future_route[KEYS].copy()
        part["catboost_prediction"] = np.maximum(
            0, model.predict(future_route[cat_features])
        )
        cat_parts.append(part)
        print(f"CatBoost route {route} done", flush=True)
    cat_predictions = pd.concat(cat_parts, ignore_index=True)
    active = active.merge(cat_predictions, on=KEYS, how="left", validate="one_to_one")
    active["catboost_weight"] = active["route"].map(CATBOOST_WEIGHTS).fillna(0.0)
    active["prediction"] = (
        (1 - active["catboost_weight"]) * active["chronos_blend"]
        + active["catboost_weight"] * active["catboost_prediction"].fillna(0.0)
    )

    # 5. Restore route 5, template order, and save.
    output = baseline.copy()
    output["prediction"] = output["prediction"].astype(float)
    lookup = active.set_index(KEYS)["prediction"]
    active_mask = output["route"] != 5
    output.loc[active_mask, "prediction"] = pd.MultiIndex.from_frame(
        output.loc[active_mask, KEYS]
    ).map(lookup)
    output["prediction"] = np.rint(output["prediction"].clip(lower=0)).astype(int)
    output = target_template[KEYS].merge(
        output, on=KEYS, how="left", validate="one_to_one"
    )
    assert output["prediction"].notna().all()
    assert len(output) == len(target_template)
    assert not output.duplicated(KEYS).any()
    output["prediction"] = output["prediction"].astype(int)
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    if requested_output is not None:
        output_path = requested_output
    elif Path("/kaggle/working").exists():
        filename = "submission_selective.csv" if year == 2025 else "submission_selective_2026.csv"
        output_path = Path("/kaggle/working") / filename
    else:
        filename = "submission_selective.csv" if year == 2025 else "submission_selective_2026.csv"
        output_path = ROOT / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep=";", index=False)
    print("Saved:", output_path)
    print("Rows:", len(output), "total:", output["prediction"].sum())
    print("Route 5:", output.loc[output["route"] == 5, "prediction"].sum())


if __name__ == "__main__":
    args = parse_args()
    main(args.year, args.output)
