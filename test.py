import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# === Загрузка и предобработка ===
df = pd.read_csv("flights.csv", low_memory=False)
df.columns = df.columns.str.strip()

if {"YEAR", "MONTH", "DAY_OF_MONTH"}.issubset(df.columns):
    df["FL_DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY_OF_MONTH"]])
else:
    df["FL_DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])

# === Модель по твоему коду ===
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    model = joblib.load('flight_delay_model.pkl')
except:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE',
                                    'DEPARTURE_DELAY', 'TAXI_OUT', 'DISTANCE', 'SCHEDULED_TIME']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
        ])
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])
    model.fit(X_train, y_train)
    joblib.dump(model, 'flight_delay_model.pkl')

# === Интерфейс ===
root = tk.Tk()
root.title("Flight Dashboard")
root.geometry("1600x900")

# Верхняя панель делится на левую (графики) и правую (управление)
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas_frame = tk.Frame(left_frame)
canvas_frame.pack(fill=tk.BOTH, expand=True)

graph_widgets = []  # Хранение всех нарисованных графиков

right_frame = tk.Frame(main_frame, padx=20)
right_frame.pack(side=tk.RIGHT, fill=tk.Y)

# ==== Левая панель: графики ====
plot_types = ["lineplot", "barplot", "boxplot", "histplot"]
agg_funcs = {"Среднее": "mean", "Максимум": "max", "Минимум": "min", "Счет": "count"}

top_controls = tk.Frame(right_frame)
top_controls.pack(pady=10)

ttk.Label(top_controls, text="Начало:").grid(row=0, column=0)
start_entry = ttk.Entry(top_controls)
start_entry.insert(0, "2015-01-01")
start_entry.grid(row=0, column=1)

ttk.Label(top_controls, text="Конец:").grid(row=1, column=0)
end_entry = ttk.Entry(top_controls)
end_entry.insert(0, "2015-12-31")
end_entry.grid(row=1, column=1)

ttk.Label(top_controls, text="Тип графика:").grid(row=2, column=0)
plot_combo = ttk.Combobox(top_controls, values=plot_types)
plot_combo.set("lineplot")
plot_combo.grid(row=2, column=1)

ttk.Label(top_controls, text="X ось:").grid(row=3, column=0)
x_combo = ttk.Combobox(top_controls, values=list(df.columns))
x_combo.set("FL_DATE")
x_combo.grid(row=3, column=1)

ttk.Label(top_controls, text="Y ось:").grid(row=4, column=0)
y_combo = ttk.Combobox(top_controls, values=list(df.columns))
y_combo.set("DEPARTURE_DELAY")
y_combo.grid(row=4, column=1)

ttk.Label(top_controls, text="Агрегация:").grid(row=5, column=0)
agg_combo = ttk.Combobox(top_controls, values=list(agg_funcs.keys()))
agg_combo.set("Среднее")
agg_combo.grid(row=5, column=1)

canvas_frame = tk.Frame(left_frame)
canvas_frame.pack(fill=tk.BOTH, expand=True)

# Хранилище для графиков
graph_widgets = []

def clear_graphs():
    for widget in graph_widgets:
        widget.destroy()
    graph_widgets.clear()

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

        # создаём новый график и размещаем в сетке
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        if plot in ["lineplot", "barplot", "scatter"]:
            getattr(sns, plot)(data=data, x=x, y=y, ax=ax)
        else:
            getattr(sns, plot)(data=data, x=y, ax=ax)
        ax.set_title(f"{plot} по {x} и {y}")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        widget = canvas.get_tk_widget()

        # размещаем по 2 в строку
        row = len(graph_widgets) // 2
        col = len(graph_widgets) % 2
        widget.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        canvas.draw()
        graph_widgets.append(widget)

    except Exception as e:
        messagebox.showerror("Ошибка построения", str(e))


ttk.Button(right_frame, text="Построить график", command=draw_plot).pack(pady=5)
ttk.Button(right_frame, text="Очистить все графики", command=clear_graphs).pack(pady=5)


# --- Прогноз задержки (с вводом вручную, справа) ---
ttk.Label(right_frame, text="--- Прогноз задержки ---", font=("Arial", 11, "bold")).pack(pady=(30, 5))

# Контейнер с полями ввода
predict_frame = ttk.Frame(right_frame)
predict_frame.pack()

entries = {}

default_values = {
    'MONTH': 6,
    'DAY': 15,
    'DAY_OF_WEEK': 6,
    'AIRLINE': 'AA',
    'ORIGIN_AIRPORT': 'ATL',
    'DESTINATION_AIRPORT': 'LAX',
    'SCHEDULED_DEPARTURE': 900,
    'DEPARTURE_DELAY': 5,
    'TAXI_OUT': 12,
    'DISTANCE': 2200,
    'SCHEDULED_TIME': 300
}

for i, col in enumerate(features):
    ttk.Label(predict_frame, text=col).grid(row=i, column=0, sticky=tk.W)
    e = ttk.Entry(predict_frame)
    e.insert(0, str(default_values[col]))
    e.grid(row=i, column=1)
    entries[col] = e

result_label = ttk.Label(right_frame, text="", font=("Arial", 10, "bold"))
result_label.pack(pady=10)

def predict_manual():
    try:
        row = []
        for col in features:
            val = entries[col].get()
            val = int(val) if col not in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'] else str(val)
            row.append(val)
        df_input = pd.DataFrame([row], columns=features)
        prediction = model.predict(df_input)[0]
        result_label.config(text=f"Прогноз: {prediction:.1f} минут")
    except Exception as e:
        result_label.config(text="Ошибка прогноза")
        print("Ошибка:", e)

ttk.Button(right_frame, text="Предсказать", command=predict_manual).pack(pady=5)

root.mainloop()
