import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Загрузка и подготовка данных ---
try:
    df = pd.read_csv("flights.csv", low_memory=False)
    df.columns = df.columns.str.strip()

    # Объединяем YEAR, MONTH, DAY или DAY_OF_MONTH в дату
    if {"YEAR", "MONTH", "DAY"}.issubset(df.columns):
        df["FL_DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
    elif {"YEAR", "MONTH", "DAY_OF_MONTH"}.issubset(df.columns):
        df["FL_DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY_OF_MONTH"]])
    else:
        raise ValueError("Не найдены подходящие колонки: YEAR, MONTH, DAY или DAY_OF_MONTH")
except Exception as e:
    messagebox.showerror("Ошибка загрузки", str(e))
    exit()

# --- Обучение/загрузка модели ---
df_model = df.dropna(subset=['ARRIVAL_DELAY'])
df_model['AIRLINE'] = df_model['AIRLINE'].astype(str)
df_model['ORIGIN_AIRPORT'] = df_model['ORIGIN_AIRPORT'].astype(str)
df_model['DESTINATION_AIRPORT'] = df_model['DESTINATION_AIRPORT'].astype(str)

features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT',
            'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY',
            'TAXI_OUT', 'DISTANCE', 'SCHEDULED_TIME']
target = 'ARRIVAL_DELAY'

X = df_model[features]
y = df_model[target]

try:
    model = joblib.load('flight_delay_model.pkl')
except:
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE',
                                'DEPARTURE_DELAY', 'TAXI_OUT', 'DISTANCE', 'SCHEDULED_TIME']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X, y)
    joblib.dump(model, 'flight_delay_model.pkl')

# --- Настройка интерфейса ---
root = tk.Tk()
root.title("Анализ задержек рейсов")

plot_types = ["lineplot", "barplot", "boxplot", "histplot"]
agg_funcs = {"Среднее": "mean", "Максимум": "max", "Минимум": "min"}

frame = ttk.Frame(root, padding=10)
frame.pack(fill=tk.BOTH, expand=True)

def label(row, text):
    ttk.Label(frame, text=text).grid(row=row, column=0, sticky=tk.W)

def entry(row, default=""):
    e = ttk.Entry(frame)
    e.insert(0, default)
    e.grid(row=row, column=1, sticky=tk.EW)
    return e

label(0, "Начальная дата (YYYY-MM-DD):")
start_entry = entry(0, "2015-01-01")

label(1, "Конечная дата (YYYY-MM-DD):")
end_entry = entry(1, "2015-12-31")

label(2, "Тип графика:")
plot_combo = ttk.Combobox(frame, values=plot_types)
plot_combo.set("lineplot")
plot_combo.grid(row=2, column=1, sticky=tk.EW)

label(3, "Столбец X:")
x_combo = ttk.Combobox(frame, values=list(df.columns))
x_combo.set("FL_DATE")
x_combo.grid(row=3, column=1, sticky=tk.EW)

label(4, "Столбец Y:")
y_combo = ttk.Combobox(frame, values=list(df.columns))
y_combo.set("DEPARTURE_DELAY")
y_combo.grid(row=4, column=1, sticky=tk.EW)

label(5, "Агрегация:")
agg_combo = ttk.Combobox(frame, values=list(agg_funcs.keys()))
agg_combo.set("Среднее")
agg_combo.grid(row=5, column=1, sticky=tk.EW)

# Контейнер для графиков с увеличенной рабочей областью
graph_container = tk.Canvas(root, height=800, width=1200)
graph_container.pack(fill=tk.BOTH, expand=True)

graphs_frame = ttk.Frame(graph_container)
graph_container.create_window((0, 0), window=graphs_frame, anchor="nw")

# Счётчик графиков
graph_counter = {"count": 0}  # обёртка для изменения внутри функции

def draw_plot():
    try:
        start = pd.to_datetime(start_entry.get())
        end = pd.to_datetime(end_entry.get())
        if start > end:
            raise ValueError("Неверный диапазон дат")

        data = df[(df["FL_DATE"] >= start) & (df["FL_DATE"] <= end)]
        if data.empty:
            messagebox.showinfo("Нет данных", "Нет данных в этом диапазоне")
            return

        x = x_combo.get()
        y = y_combo.get()
        agg = agg_funcs[agg_combo.get()]
        plot = plot_combo.get()

        if x == "FL_DATE":
            data = data.groupby("FL_DATE")[y].agg(agg).reset_index()

        fig, ax = plt.subplots(figsize=(5, 3))
        if plot in ["lineplot", "barplot", "scatter"]:
            getattr(sns, plot)(data=data, x=x, y=y, ax=ax)
        else:
            getattr(sns, plot)(data=data, x=y, ax=ax)
        ax.set_title(f"{plot} по {x} и {y}")

        canvas = FigureCanvasTkAgg(fig, master=graphs_frame)
        canvas_widget = canvas.get_tk_widget()

        # Расчёт позиции (по 2 графика в строку)
        row = graph_counter["count"] // 2
        col = graph_counter["count"] % 2
        canvas_widget.grid(row=row, column=col, padx=10, pady=10)

        canvas.draw()
        graph_counter["count"] += 1

    except Exception as e:
        messagebox.showerror("Ошибка построения", str(e))

ttk.Button(frame, text="Построить график", command=draw_plot).grid(row=6, column=0, columnspan=2, pady=10)

# --- Раздел прогнозирования задержки ---
ttk.Label(frame, text="Прогноз задержки").grid(row=7, column=0, columnspan=2)

entries = {}
inputs = {
    "MONTH": 1, "DAY": 1, "DAY_OF_WEEK": 1, "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "ATL", "DESTINATION_AIRPORT": "LAX",
    "SCHEDULED_DEPARTURE": 600, "DEPARTURE_DELAY": 0,
    "TAXI_OUT": 10, "DISTANCE": 2000, "SCHEDULED_TIME": 300
}

row = 8
for key, default in inputs.items():
    ttk.Label(frame, text=key).grid(row=row, column=0)
    entry = ttk.Entry(frame)
    entry.insert(0, str(default))
    entry.grid(row=row, column=1)
    entries[key] = entry
    row += 1

result_label = ttk.Label(frame, text="")
result_label.grid(row=row, column=0, columnspan=2, pady=5)

def predict_delay():
    try:
        input_data = {k: entries[k].get() for k in entries}
        for k in ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE',
                  'DEPARTURE_DELAY', 'TAXI_OUT', 'DISTANCE', 'SCHEDULED_TIME']:
            input_data[k] = int(input_data[k])

        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        result_label.config(text=f"Прогнозируемая задержка прибытия: {prediction:.1f} мин.")
    except Exception as e:
        result_label.config(text="Ошибка прогноза")
        print(e)

ttk.Button(frame, text="Предсказать задержку", command=predict_delay).grid(row=row + 1, column=0, columnspan=2)

root.mainloop()
