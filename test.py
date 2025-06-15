import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# --- Настройка интерфейса ---
root = tk.Tk()
root.title("Анализ задержек рейсов")

plot_types = ["lineplot", "barplot", "scatter", "boxplot", "histplot"]
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
start_entry = entry(0, "2020-01-01")

label(1, "Конечная дата (YYYY-MM-DD):")
end_entry = entry(1, "2020-12-31")

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
y_combo.set("DEP_DELAY")
y_combo.grid(row=4, column=1, sticky=tk.EW)

label(5, "Агрегация:")
agg_combo = ttk.Combobox(frame, values=list(agg_funcs.keys()))
agg_combo.set("Среднее")
agg_combo.grid(row=5, column=1, sticky=tk.EW)

# --- График ---
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# --- Построение графика ---
def draw_plot():
    try:
        start = pd.to_datetime(start_entry.get())
        end = pd.to_datetime(end_entry.get())
        if start > end:
            raise ValueError("Начальная дата позже конечной")

        data = df[(df["FL_DATE"] >= start) & (df["FL_DATE"] <= end)]
        if data.empty:
            messagebox.showinfo("Нет данных", "В выбранном периоде нет данных")
            return

        x = x_combo.get()
        y = y_combo.get()
        plot = plot_combo.get()
        agg = agg_funcs[agg_combo.get()]

        # Если ось X — дата, агрегируем
        if x == "FL_DATE":
            data = data.groupby("FL_DATE")[y].agg(agg).reset_index()

        ax.clear()
        if plot in ["lineplot", "barplot", "scatter"]:
            getattr(sns, plot)(data=data, x=x, y=y, ax=ax)
        else:  # histplot, boxplot — только по Y
            getattr(sns, plot)(data=data, x=y, ax=ax)

        ax.set_title(f"{plot} по {x} и {y} ({agg_combo.get().lower()})")
        canvas.draw()
    except Exception as err:
        messagebox.showerror("Ошибка построения", str(err))

ttk.Button(frame, text="Построить график", command=draw_plot).grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
