from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ProphetTimeSeries:
    def __init__(self, dataset, X_train=None, X_test=None, y_train=None, y_test=None):
        self.df_rough = dataset.copy()
        self.df_prophet = None
        self.regressors = None

        self.preprocess()
        
        self._prophet = Prophet(
            daily_seasonality=True,   # capturar patrones diarios
            weekly_seasonality=True,  # capturar patrones semanales
            yearly_seasonality=True,   # capturar patrones anuales
            changepoint_prior_scale=1.5,
        ) 
        
        self.adjust_model()

        self.metrics = None

    def preprocess(self):
        # Seleccionar columnas a usar como regresores (todas menos 'date' y 'Energy_Consumption')
        self.regressors = [c for c in self.df_rough.columns if c not in ['date','Energy_Consumption']]
        
        # Rellenar NaN con la media
        self.df_rough[self.regressors] = self.df_rough[self.regressors].fillna(self.df_rough[self.regressors].mean())
        self.df_rough['Energy_Consumption'].fillna(self.df_rough['Energy_Consumption'].mean(), inplace=True)

        # Preparar DataFrame para Prophet
        self.df_prophet = self.df_rough[['date','Energy_Consumption'] + self.regressors].rename(columns={'date':'ds', 'Energy_Consumption':'y'})

    def adjust_model(self):
        # Ajustar modelo
        for reg in self.regressors:
            self._prophet.add_regressor(reg)

        self._prophet.fit(self.df_prophet)

    def train_model(self):
        # 'Training' ... just for metrics purposes
        future = self._prophet.make_future_dataframe(periods=30)

        for reg in self.regressors:
            last_value = self.df_prophet[reg].iloc[-1]
            future[reg] = pd.concat([self.df_prophet[reg], pd.Series([last_value]*30)]).reset_index(drop=True)
        
        # Get metrics given this 30 predictions
        forecast = self._prophet.predict(future)

        self.metrics = self._get_metrics(forecast)

    def predict_and_get_metrics(self, X):
        # X = df {ds: date, regressor1, regressor2...}
        y_pred = self._prophet.predict(X)
        mae, mse, rmse, r2, mape = self.metrics
        lower, upper = self._get_bounds(y_pred)

        return y_pred, lower, upper, self.metrics

    def _get_bounds(self, y):
        lower = y['yhat_lower'].values[0]
        upper = y['yhat_upper'].values[0]

        return lower, upper

    def _get_metrics(self, forecast):
        # Valores reales
        y_true = self.df_prophet['y'].values

        # Valores predichos por Prophet (solo las fechas históricas)
        y_pred = forecast['yhat'].values[:len(y_true)]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
        
        r2 = r2_score(y_true, y_pred)

        return [mae, mse, rmse, r2, mape]
    
    def summary(self):
        mae, mse, rmse, r2, mape = self.metrics
        print("Prophet Summary:")
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"MAPE: {mape:.4f}")