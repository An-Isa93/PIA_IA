from django.shortcuts import render
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
import matplotlib.pyplot as plt
from Models.LinearRegression import RegresionLineal
from Models.Prophet import ProphetTimeSeries
from Models.RandomForestRegressor import RandomForestRegression
from Models.XGBoostRegressor import XGBR


def home(request):
    context = {}
    if request.method == "POST":
        # Leer dataset
        df = pd.read_csv("Datasets/energy_consumption_merged.csv")
        # Convertir fecha correctamente
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        df.fillna(df.mean(), inplace=True)

        # Modelo seleccionado en el formulario
        model_choice = request.POST.get("models")
        print(model_choice)

        # Para Prophet se usa la fecha, para los demás se elimina
        if model_choice in ["1", "2", "3"]:
            df = df.drop(columns=['date'])
            
        # Separar variables
        X = df.drop(columns=['Energy_Consumption'])
        y = df['Energy_Consumption']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "1":
            model = RegresionLineal(df, X_train, X_test, y_train, y_test)
        elif model_choice == "2":
            model = RandomForestRegression(df, X_train, X_test, y_train, y_test)
        elif model_choice == "3":
            model = XGBR(df, X_train, X_test, y_train, y_test)
        else:
            model = ProphetTimeSeries(df, X_train, X_test, y_train, y_test)
        print("Columnas usadas para entrenar:", X.columns.tolist())
        if model:
            model.train_model()
            model.summary()
            context["result"] = "Modelo entrenado correctamente"
            if model_choice in ["1", "2", "3"]:
                # Definir nombres de variables según el dataset
                features = [
                    "Air_Temperature", "Humidity", "SolarRadiation",
                    "Cool_Temperature", "Heat_Temperature",
                    "Interior_Temperature", "Exterior_Temperature",
                    "Total_Occupants", "Wifi_Users"
                ]
                input_data = []
                for f in features:
                    val = request.POST.get(f)
                    if val:
                        input_data.append(float(val))
                    

                # Solo predecir si el usuario llenó todos los campos
                if len(input_data) == len(features):
                    y_pred, lower, upper, metrics = model.predict_and_get_metrics(np.array([input_data]))
                   
                    context["prediction"] = (
                                            f"Predicción: {round(y_pred[0], 2)} kW<br>"
                                            f"<span style='white-space: nowrap;'>Rango de confianza: {round(lower[0], 2)} – {round(upper[0], 2)}kW"
                                            )

                else:
                    context["prediction"] = "⚠️ Faltan valores para predecir."

            # Si es modelo Prophet (serie de tiempo)
            elif model_choice == "4":
                date_input = request.POST.get("date")
                if date_input:
                    try:
                        # Convertir fecha correctamente
                        date_input = pd.to_datetime(date_input, format='%Y-%m-%d', errors='coerce')

                        # Extraer todas las variables del formulario
                        features = [
                            "Air_Temperature", "Humidity", "SolarRadiation",
                            "Cool_Temperature", "Heat_Temperature",
                            "Interior_Temperature", "Exterior_Temperature",
                            "Total_Occupants", "Wifi_Users"
                        ]

                        # Crear diccionario con valores numéricos
                        input_data = {}
                        for f in features:
                            val = request.POST.get(f)
                            input_data[f] = float(val) if val else 0.0

                        # Crear DataFrame futuro con todas las variables
                        future = pd.DataFrame({
                            'ds': [date_input],
                            'Air_Temperature': [input_data["Air_Temperature"]],
                            'Humidity': [input_data["Humidity"]],
                            'SolarRadiation': [input_data["SolarRadiation"]],
                            'Cool_Temperature': [input_data["Cool_Temperature"]],
                            'Heat_Temperature': [input_data["Heat_Temperature"]],
                            'Interior_Temperature': [input_data["Interior_Temperature"]],
                            'Exterior_Temperature': [input_data["Exterior_Temperature"]],
                            'Total_Occupants': [input_data["Total_Occupants"]],
                            'Wifi_Users': [input_data["Wifi_Users"]]
                        })

                        # Predecir con Prophet
                        y_pred, lower, upper, metrics = model.predict_and_get_metrics(future)

                        # Extraer el valor numérico de la predicción (depende de cómo Prophet devuelva y_pred)
                        y_value = y_pred["yhat"].iloc[0]  # primer valor de la predicción

                        context["prediction"] = (
                            f"Predicción para {date_input.date()}: {round(y_value, 2)} kW <br>"
                            f"<span style='white-space: nowrap;'>Rango de confianza: {round(lower, 2)} - {round(upper, 2)} kW"
                        )
                    except Exception as e:
                        context["prediction"] = f"Error al predecir con Prophet: {e}"
                else:
                    context["prediction"] = "⚠️ No se proporcionó una fecha para Prophet."
    return render(request,"home.html", context)

def graphs(request):
    df = pd.read_csv("Datasets/energy_consumption_merged.csv")
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df.fillna(df.mean(), inplace=True)
    df = df.drop(columns=['date'])
    # Variables
    X = df.drop(columns=['Energy_Consumption'])
    y = df['Energy_Consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    graficas = {}

    # === 1️⃣ Linear Regression ===
    lr = RegresionLineal(df, X_train, X_test, y_train, y_test)
    lr.train_model()
    y_pred_lr = lr.predict_and_get_metrics(X_test)


    '''plt.figure(figsize=(6, 5))
    plt.style.use('seaborn-v0_8')
    plt.scatter(y_test, y_pred_lr[0], alpha=0.6)
    plt.xlabel("Real Energy Consumption")
    plt.ylabel("Predicted")
    plt.title("Linear Regression")
    plt.grid(True)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graficas['Linear Regression'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()
    context = {"graficas": graficas}'''
    results = pd.DataFrame({
    'Real': y_test,
    'Predicted': y_pred_lr[0]
    })

    # Crear gráfica interactiva
    fig = px.scatter(
        results,
        x='Real',
        y='Predicted',
        title='Linear Regression - Real vs Predicted',
        labels={'Real': 'Real Energy Consumption', 'Predicted': 'Predicted'},
        template='plotly_white'
    )
    fig.update_layout(
    width=600,   # ancho en píxeles
    height=500,  # alto en píxeles
    margin=dict(l=20, r=20, t=40, b=20),  # márgenes más compactos
    modebar=dict(orientation='v')
    )
    # Convertir la figura a HTML
    graficas['Linear Regression'] = fig.to_html(full_html=False)
    context = {"graficas": graficas}
    return render(request, "graphs.html", context)
