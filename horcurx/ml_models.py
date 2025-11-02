import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# --- 1. COPIED FROM YOUR JUPYTER NOTEBOOK ---
# We must define the class here so joblib can re-create the object
class HybridDemandForecaster:
    def __init__(self, n_estimators=200, max_depth=25, random_state=42):
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.sarimax_models = {}
        self.feature_importances_ = None

    def _fit_sarimax_per_brand(self, X_train, y_train, brand_col="brand_encoded", year_col="Year"):
        print("\n📈 Training SARIMAX Models for Each Brand...")
        self.sarimax_models = {}
        failed = 0

        for brand, group in X_train.groupby(brand_col):
            try:
                df = group.copy()
                df["target"] = y_train.loc[group.index]

                df[year_col] = pd.to_datetime(df[year_col].astype(int), format="%Y", errors="coerce")
                df = df.dropna(subset=[year_col, "target"])
                df = df.sort_values(year_col)

                yearly_series = df.groupby(year_col)["target"].mean()
                if yearly_series.shape[0] < 3:
                    continue

                full_years = pd.date_range(start=yearly_series.index.min(),
                                           end=yearly_series.index.max(),
                                           freq="YS")
                yearly_series = yearly_series.reindex(full_years).interpolate(method="linear")

                model = SARIMAX(yearly_series, order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 2),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                fitted = model.fit(disp=False)
                self.sarimax_models[brand] = fitted

            except Exception as e:
                failed += 1
                print(f"   ⚠️ Could not fit SARIMAX for brand {brand}: {e}")

        print(f"✅ Time-series models trained for {len(self.sarimax_models)} brands ({failed} failed).")

    def train(self, X_train, y_train, verbose=True):
        if verbose:
            print("=" * 60)
            print("🎯 Training Hybrid Demand Forecasting Model")
            print("   Components: SARIMAX (Time Series) + Random Forest (Features)")
            print("=" * 60)
            print(f"\n📊 Training samples: {len(X_train)}")
            print(f"📊 Features used: {len(X_train.columns)}")
            print(f"📊 Target range: {y_train.min():.2f} - {y_train.max():.2f}")

        if "brand_encoded" in X_train.columns and "Year" in X_train.columns:
            self._fit_sarimax_per_brand(X_train, y_train)
        else:
            print("⚠️ Skipping SARIMAX (no brand/year info available).")

        if verbose:
            print("\n🌲 Training Random Forest Component...")
        self.rf_model.fit(X_train, y_train)

        preds = self.rf_model.predict(X_train)
        mae = mean_absolute_error(y_train, preds)
        rmse = np.sqrt(mean_squared_error(y_train, preds))
        r2 = r2_score(y_train, preds)

        self.feature_importances_ = pd.Series(
            self.rf_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        if verbose:
            print("\n📈 Training Metrics (Hybrid Model):")
            print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
            print("\n🔍 Top 5 Important Features:")
            print(self.feature_importances_.head())

        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def predict(self, X_test):
        rf_preds = self.rf_model.predict(X_test)
        hybrid_preds = rf_preds.copy()

        if "brand_encoded" in X_test.columns and len(self.sarimax_models) > 0:
            # Note: The notebook output shows SARIMAX trained 0 models,
            # so this block likely won't run, which is fine.
            # The RF model is the primary predictor.
            for i, row in X_test.iterrows():
                brand = row["brand_encoded"]
                if brand in self.sarimax_models:
                    try:
                        sarimax_pred = self.sarimax_models[brand].forecast(steps=1)[0]
                        hybrid_preds[i] = 0.7 * rf_preds[i] + 0.3 * sarimax_pred
                    except Exception:
                        pass

        return hybrid_preds

    def evaluate(self, X_test, y_test, verbose=True):
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        if verbose:
            print("\n============================================================")
            print("📊 Model Evaluation Results (Test Data)")
            print("============================================================")
            print(f"MAE :  {mae:.2f}")
            print(f"RMSE:  {rmse:.2f}")
            print(f"R²   :  {r2:.4f}")
            print("============================================================")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def save(self, filepath="hybrid_demand_model.pkl"):
        data = {
            "rf_model": self.rf_model,
            "sarimax_models": self.sarimax_models,
            "feature_importances": self.feature_importances_,
        }
        joblib.dump(data, filepath)
        print(f"💾 Model saved successfully to {filepath}\n")

    @staticmethod
    def load(filepath="hybrid_demand_model.pkl"):
        try:
            data = joblib.load(filepath)
            model = HybridDemandForecaster()
            model.rf_model = data["rf_model"]
            model.sarimax_models = data["sarimax_models"]
            model.feature_importances_ = data["feature_importances"]
            print(f"✅ Hybrid Demand Model loaded successfully from {filepath}")
            return model
        except FileNotFoundError:
            print(f"⚠️ MODEL FILE NOT FOUND at {filepath}. API will use mock data.")
            return None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. API will use mock data.")
            return None

# --- 2. THE "CONTRACT" ---
# This feature list MUST match the notebook EXACTLY
# From cell 120: ['brand_encoded', 'car_age', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'price_per_km', 'Year', 'Selling_Price', 'price_x_age', 'price_x_mileage', 'age_x_mileage']
DEMAND_MODEL_FEATURES = [
    "brand_encoded", "car_age", "Kilometers_Driven", "Fuel_Type",
    "Transmission", "price_per_km", "Year", "Selling_Price",
    "price_x_age", "price_x_mileage", "age_x_mileage"
]

# --- 3. LOAD THE MODEL ON STARTUP ---
# This code runs ONCE when main.py imports this file.
demand_model = HybridDemandForecaster.load("final_demand_model.pkl")

# --- 4. PREDICTION FUNCTION ---
# This is the new function our API will call.
def get_demand_score_for_car(features_dict: dict) -> float:
    """
    Takes a dictionary of features, prepares them,
    and returns a demand score (0-1).
    """
    if demand_model is None:
        # Fallback if the .pkl file wasn't found
        return 0.65 # Return a neutral mock score

    try:
        # Create a single-row DataFrame from the dict
        # This automatically handles matching and ordering features
        df = pd.DataFrame([features_dict], columns=DEMAND_MODEL_FEATURES)
        
        # Fill any features we didn't have with 0
        df = df.fillna(0) 
        
        # Get the prediction (it's a numpy array)
        prediction = demand_model.predict(df)
        
        # Return the first (and only) score
        return float(prediction[0])
    except Exception as e:
        print(f"⚠️ Demand model prediction failed: {e}")
        print(f"Features provided: {features_dict}")
        return 0.5 # Return a neutral score on failure
