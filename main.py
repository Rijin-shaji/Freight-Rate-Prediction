import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "E:/project spotter/train-test.csv"
VAL_PATH = "E:/project spotter/validation.csv"
DEC_PATH = "E:/project spotter/december-chart-inputs.csv"
VAL_TEMPLATE_PATH = "E:/project spotter/validation-predictions-template.csv"
VAL_OUTPUT_PATH = "E:/project spotter/validation_predictions.csv"
DEC_OUTPUT_PATH = "E:/project spotter/december_chart_inputs.csv"

df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.isna().sum())

weight_median = df['weight'].median()
market_index_median = df['market_index'].median()
df['weight'] = df['weight'].fillna(weight_median)
df['market_index'] = df['market_index'].fillna(market_index_median)

df['month'] = df['date'].dt.month
df['dow'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['dow'] >= 5).astype(int)

US_HOLIDAYS = pd.to_datetime(['2025-12-25', '2025-12-24', '2025-12-31', '2025-11-27'])

def days_to_nearest_holiday(date):
    return min(abs((date - h).days) for h in US_HOLIDAYS)

df['days_to_holiday'] = df['date'].apply(days_to_nearest_holiday)

df['lane'] = df['pickup'] + '__' + df['delivery']

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[['equipment_enc', 'lane_enc']] = enc.fit_transform(df[['equipment', 'lane']])

feature_cols = [
    'distance', 'weight', 'equipment_enc', 'lane_enc',
    'market_index', 'quote_signal',
    'month', 'dow', 'is_weekend', 'days_to_holiday'
]

X = df[feature_cols]
y = np.log1p(df['posted_rate'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(X_train, y_train)

pred = np.expm1(model.predict(X_test))
actual = np.expm1(y_test)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
r2 = r2_score(actual, pred)

print("\n=== Model A (full-feature) hold-out results ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R²  :", r2)

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nModel A feature importance")
print(importances)

final_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
final_model.fit(X, y)

val = pd.read_csv(VAL_PATH, parse_dates=['date'])

print(val.shape)
print(val.columns.tolist())
print(val.isna().sum())

val['weight'] = val['weight'].fillna(weight_median)
val['market_index'] = val['market_index'].fillna(market_index_median)

val['month'] = val['date'].dt.month
val['dow'] = val['date'].dt.dayofweek
val['is_weekend'] = (val['dow'] >= 5).astype(int)
val['days_to_holiday'] = val['date'].apply(days_to_nearest_holiday)

val['lane'] = val['pickup'] + '__' + val['delivery']
val[['equipment_enc', 'lane_enc']] = enc.transform(val[['equipment', 'lane']])

X_val = val[feature_cols]
val_pred = np.expm1(final_model.predict(X_val))

output = val[['load_id']].copy()
output['predicted_rate'] = np.round(val_pred, 2)
output.to_csv(VAL_OUTPUT_PATH, index=False)

print(output.head())
print(output.shape)
print(output['predicted_rate'].describe())

template = pd.read_csv(VAL_TEMPLATE_PATH)
print("\nload_id order matches template:", (template['load_id'] == output['load_id']).all())

cal_feature_cols = [
    'distance', 'weight', 'equipment_enc', 'lane_enc',
    'month', 'dow', 'is_weekend', 'days_to_holiday'
]

X_cal = df[cal_feature_cols]
y_cal = y

cal_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
cal_model.fit(X_cal, y_cal)

cal_importances = pd.Series(
    cal_model.feature_importances_, index=cal_feature_cols
).sort_values(ascending=False)
print("\n=== Model B (calendar-only) feature importance ===")
print(cal_importances)

dec = pd.read_csv(DEC_PATH, parse_dates=['date'])

print(dec.shape)
print(dec.columns.tolist())
print(dec)

dec['month'] = dec['date'].dt.month
dec['dow'] = dec['date'].dt.dayofweek
dec['is_weekend'] = (dec['dow'] >= 5).astype(int)
dec['days_to_holiday'] = dec['date'].apply(days_to_nearest_holiday)

dec['lane'] = dec['pickup'] + '__' + dec['delivery']
dec[['equipment_enc', 'lane_enc']] = enc.transform(dec[['equipment', 'lane']])

X_dec = dec[cal_feature_cols]
dec_pred = np.expm1(cal_model.predict(X_dec))

dec_out = dec[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date']].copy()
dec_out['date'] = dec_out['date'].dt.strftime('%Y-%m-%d')
dec_out['predicted_rate'] = np.round(dec_pred, 2)

dec_out.to_csv(DEC_OUTPUT_PATH, index=False)

print(dec_out)