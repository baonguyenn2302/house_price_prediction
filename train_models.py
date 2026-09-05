from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / 'data' / 'houseprice.csv'
OUTPUTS_DIR = ROOT / 'outputs'
MODELS_DIR = ROOT / 'models'

OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

NUMERIC_COLUMNS = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
CATEGORICAL_COLUMNS = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state', 'Location']


def extract_location(address: str | float | int | None) -> str:
    if address is None or (isinstance(address, str) and not address.strip()):
        return 'Other'
    if not isinstance(address, str):
        return 'Other'
    parts = [part.strip() for part in address.split(',') if part.strip()]
    if not parts:
        return 'Other'
    return parts[-1] if parts[-1] else 'Other'


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                'numeric',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                NUMERIC_COLUMNS,
            ),
            (
                'categorical',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=250, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
    }


def save_prediction_plots(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title(f'{model_name} - Actual vs Predicted')

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Predicted Price')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name} - Residual Plot')

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f'{model_name.lower().replace(" ", "_")}_prediction.png', dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price']).reset_index(drop=True)
    df = df[df['Price'] > 0].reset_index(drop=True)
    df['Location'] = df['Address'].map(extract_location)

    X = df.drop(columns=['Address', 'Price'])
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    summary_rows = []
    best_model_name = None
    best_model = None
    best_rmse = float('inf')

    for model_name, estimator in build_models().items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', estimator),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        summary_rows.append({
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_model_name = model_name

    summary = pd.DataFrame(summary_rows).sort_values('RMSE')
    summary.to_csv(OUTPUTS_DIR / 'house_price_model_summary.csv', index=False)

    if best_model is not None:
        best_model.fit(X, y)
        joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
        y_pred_best = best_model.predict(X_test)
        save_prediction_plots(best_model_name, y_test.reset_index(drop=True), y_pred_best)

    print('\nHouse price model summary:')
    print(summary.round(4).to_string(index=False))
    print(f'\nBest model: {best_model_name}')


if __name__ == '__main__':
    main()
