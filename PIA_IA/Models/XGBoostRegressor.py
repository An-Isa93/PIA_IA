from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class XGBR:
    def __init__(self, dataset, X_train, X_test, y_train, y_test):
        self.df = dataset.copy()
        if 'date' in self.df.columns:
            self.df.drop(columns=['date'], inplace=True)
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        self.xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.y_pred = None
        self.metrics = None

    def train_model(self):
        self.xgb.fit(self.X_train, self.y_train)
        self.y_pred = self.xgb.predict(self.X_test)
        self.metrics = self._get_metrics()

    def predict_and_get_metrics(self, X):
        y_pred = self.xgb.predict(X)
        mae, mse, rmse, r2 = self.metrics
        lower, upper = self._get_bounds(y_pred, rmse)

        return y_pred, lower, upper, self.metrics

    def _get_bounds(self, y, rmse):
        z = 1.28  # para 80%
        lower = y - z * rmse
        upper = y + z * rmse

        return lower, upper

    def _get_metrics(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        
        return [mae, mse, rmse, r2]
    
    def summary(self):
        mae, mse, rmse, r2 = self.metrics
        print("XGBoost Regressor Summary:")
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")