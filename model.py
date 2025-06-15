import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats


df = pd.read_csv("flights.csv")

# Предварительная обработка
df = df.dropna(subset=['ARRIVAL_DELAY'])
df['AIRLINE'] = df['AIRLINE'].astype(str)
df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str)
df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].astype(str)

# Выбор признаков
features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 
            'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 
            'TAXI_OUT', 'DISTANCE', 'SCHEDULED_TIME']
target = 'ARRIVAL_DELAY'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Попытка загрузить сохраненную модель
try:
    model = joblib.load('flight_delay_model.pkl')
    print("Загружена сохраненная модель")
except:
    print("Обучение новой модели...")
    # Создание и обучение модели
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 
                                   'DEPARTURE_DELAY', 'TAXI_OUT', 'DISTANCE', 'SCHEDULED_TIME']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    joblib.dump(model, 'flight_delay_model.pkl')
    print("Модель обучена и сохранена")

# Оценка модели
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")