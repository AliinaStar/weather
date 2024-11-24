import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored
from tabulate import tabulate
from scipy.signal import savgol_filter

class WeatherDataFetcher:
    def __init__(self, required_columns: list[str] = None, date_column: str = "date", date_format: str = "%Y%m%d"):
        """
        Ініціалізує клас DataFetcher.

        :param required_columns: Список необхідних колонок (може бути None, якщо валідація не потрібна).
        :param date_column: Назва колонки з датою (якщо є).
        :param date_format: Формат дати для перевірки (якщо є колонка з датою).
        """
        self.required_columns = required_columns or []
        self.date_column = date_column
        self.date_format = date_format
    def _fetch_weather_data(self, source: str) -> pd.DataFrame:
        """
               Завантажує дані з файлу та виконує перевірки.

               :param source: Шлях до CSV-файлу.
               :return: DataFrame з даними.
               """
        try:
            data = pd.read_csv(source)


            if self.required_columns:
                missing_columns = [col for col in self.required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Відсутні необхідні колонки: {', '.join(missing_columns)}")


            if self.date_column in data.columns:
                data[self.date_column] = pd.to_datetime(data[self.date_column], format=self.date_format,
                                                        errors='coerce')
                if data[self.date_column].isna().any():
                    raise ValueError(f"Некоректний формат дати в колонці '{self.date_column}'.")
            return data
        except Exception as e:
            print(f"Помилка: {e}")
        return pd.DataFrame()

class WeatherDataPlotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._prepare_data()

    def _prepare_data(self):
        try:
            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d', errors='coerce')
            self.data['decade'] = (self.data['date'].dt.year // 10) * 10
        except Exception as e:
            print(f"Помилка під час підготовки даних: {e}")

    def plot_temperature_with_scales(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")


            grouped = self.data.groupby('decade').agg({
                'min_temp': 'mean',
                'mean_temp': 'mean',
                'max_temp': 'mean'
            }).reset_index()


            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(grouped['decade']))
            bar_width = 0.3


            colors = {
                'min_temp': 'blue',
                'mean_temp': 'yellow',
                'max_temp': 'red'
            }

            for i, decade in enumerate(x):
                for temp_type, color in colors.items():
                    temp_value = grouped[temp_type][i]
                    ax.bar(
                        decade, temp_value, width=bar_width, color=color, alpha=0.7, edgecolor='black'
                    )


                    scale_y = np.linspace(0, temp_value, 5)  # 5 позначок
                    for y in scale_y:
                        ax.plot(
                            [decade - bar_width / 4, decade + bar_width / 4],  # маленькі горизонтальні штрихи
                            [y, y],
                            color='black',
                            linewidth=0.5
                        )


                    ax.scatter(
                        [decade], [0], color=color, s=200, edgecolor='black', zorder=3
                    )


            ax.set_xticks(x)
            ax.set_xticklabels(grouped['decade'])
            ax.set_xlabel("Десятиліття")
            ax.set_ylabel("Температура (°C)")
            ax.set_title("Температури по десятиліттях у вигляді термометрів зі шкалами")
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")

    def plot_snow_depth_by_decade(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")



            if 'snow_depth' not in self.data.columns or self.data['snow_depth'].dropna().empty:
                raise ValueError("Немає даних про висоту снігу.")


            grouped_data = self.data.dropna(subset=['snow_depth']).groupby('decade')


            decades = list(grouped_data.groups.keys())
            num_decades = len(decades)
            cols = 3
            rows = (num_decades + cols - 1) // cols


            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
            axes = axes.flatten()


            for idx, (decade, group) in enumerate(grouped_data):
                sns.kdeplot(
                    group['snow_depth'],
                    fill=True,
                    ax=axes[idx],
                    alpha=0.5,
                    color='skyblue'
                )
                axes[idx].set_title(f"{decade}s", fontsize=14)
                axes[idx].set_xlabel("Висота снігу (см)", fontsize=12)
                axes[idx].set_ylabel("Щільність", fontsize=12)
                axes[idx].grid(axis='y', linestyle='--', alpha=0.7)


            for idx in range(num_decades, len(axes)):
                fig.delaxes(axes[idx])


            fig.suptitle("Висота снігу по десятиліттях: Сугроби (KDE)", fontsize=16)
            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")

    def plot_radiation(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")

            grouped = self.data.groupby('decade')
            decades = list(grouped.groups.keys())
            num_decades = len(decades)
            cols = 3
            rows = (num_decades + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten()

            for idx, (decade, group) in enumerate(grouped):
                axes[idx].scatter(
                    x=group['global_radiation'],
                    y=group['mean_temp'],
                    alpha=0.5,
                    color='lightgreen'
                )

                sns.regplot(
                    data=group,
                    x='global_radiation',
                    y='mean_temp',
                    ax=axes[idx], 
                    color='green',
                    scatter=False
                )

                axes[idx].set_title(f"{decade}s")
                axes[idx].set_xlabel("Глобальна радіація")
                axes[idx].set_ylabel("Середня температура")
                axes[idx].grid(True, linestyle='--', alpha=0.7)

            for idx in range(num_decades, len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")

    def plot_precipitation(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")

            grouped_data = self.data.groupby('decade')['precipitation'].mean().reset_index()

            fig, ax = plt.subplots(figsize=(12, 6))
            x = grouped_data['decade']
            y = grouped_data['precipitation']
            max_value = max(y)
            for i, (x_val, y_val) in enumerate(zip(x, y)):
                ax.bar(
                    x=x_val,
                    height=max_value * 1.3,
                    width=1.5,
                    color='none',
                    edgecolor='Black',
                    linewidth=1.0,
                    zorder=4
                )

                ax.bar(
                    x=x_val,
                    height=y_val,
                    width=1.5,
                    color='lightblue',
                    edgecolor='none',
                    zorder=3
                )
            ax.set_title("Середня кількість опадів по десятиліттях")
            ax.set_xlabel("Десятиліття")
            ax.set_ylabel("Середня кількість опадів")
            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")
     
     
    def cloud_cover(self):
        """
        Візуалізує покриття хмарами для кожного десятиліття у вигляді хмарок з точок.
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if 'cloud_cover' not in self.data.columns or self.data['cloud_cover'].dropna().empty:
                raise ValueError("Немає даних про покриття хмарами.")

            grouped_data = self.data.groupby('decade')['cloud_cover'].apply(list)
            
            decades = grouped_data.index
            num_decades = len(decades)
            cols = 3  
            rows = (num_decades + cols - 1) // cols  

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
            axes = axes.flatten() 

            for idx, (decade, cloud_values) in enumerate(grouped_data.items()):

                np.random.seed(42 + idx) 
                x = np.random.normal(loc=50, scale=15, size=len(cloud_values))
                y = np.random.normal(loc=50, scale=10, size=len(cloud_values))
                sizes = np.array(cloud_values) * 5


                axes[idx].scatter(x, y, s=sizes, alpha=0.5, color='lightblue', edgecolor='blue' )
                axes[idx].set_title(f"{decade}s", fontsize=14)
                axes[idx].set_xlim(0, 100)
                axes[idx].set_ylim(0, 100)
                axes[idx].set_xlabel("Ширина розташування(придумати шось нормальне)", fontsize=12)
                axes[idx].set_ylabel("Висота розташування(придумати шось нормальне)", fontsize=12)
                axes[idx].grid(True, linestyle='--', alpha=0.7)


            for idx in range(num_decades, len(axes)):
                fig.delaxes(axes[idx])

            fig.suptitle("Покриття хмарами для кожного десятиліття", fontsize=16, y=1.02)
            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")

    def sunshine(self):
        """
        Візуалізує кількість сонця для кожного десятиліття у формі сонця.
        Один промінь = один рік, довжина і колір залежать від кількості сонця.
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
    
            if 'decade' not in self.data.columns or 'sunshine' not in self.data.columns or 'date' not in self.data.columns:
                raise ValueError("Дані повинні містити колонки 'decade', 'sunshine' і 'date'.")
    
            self.data['year'] = self.data['date'].dt.year
            grouped_data = self.data.groupby(['decade', 'year'])['sunshine'].sum().reset_index()
    
            decades = grouped_data['decade'].unique()
    
            cols = 3 
            rows = (len(decades) + cols - 1) // cols 
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
        
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for idx, decade in enumerate(decades):
                decade_data = grouped_data[grouped_data['decade'] == decade]
                years = decade_data['year']
                sunshine_values = decade_data['sunshine']
    
                norm = plt.Normalize(vmin=sunshine_values.min(), vmax=sunshine_values.max())
                cmap = plt.cm.YlOrRd  
                colors = cmap(norm(sunshine_values))
    
                ax = axes[idx]
                ax.set_aspect('equal')
                num_years = len(years)
                angles = np.linspace(0, 2 * np.pi, num_years, endpoint=False)  # Кути для променів
    
                for angle, sun_value, year, color in zip(angles, sunshine_values, years, colors):
                    x = np.cos(angle) * sun_value  # X-координата кінця променя
                    y = np.sin(angle) * sun_value  # Y-координата кінця променя
    
                    ax.plot([0, x], [0, y], color=color, lw=2, alpha=0.8)
    
                    ax.text(x * 1.1, y * 1.1, str(year), fontsize=8, ha='center', va='center')
    
                ax.scatter(0, 0, s=500, color='yellow', zorder=4)
    
                ax.set_xlim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
                ax.set_ylim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
                ax.set_title(f"{decade}s", fontsize=12)
                ax.axis('off') 
    
            for idx in range(len(decades), len(axes)):
                fig.delaxes(axes[idx])

            
            fig.suptitle("Сонячна активність за десятиліттями", fontsize=16, y=1.02)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.05, label="Сонячні години")
    
            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")

    def sunshine_year(self, year):
        """
        Візуалізує кількість сонця для конкретного року у формі сонця.
        Один промінь = один день, довжина і колір залежать від кількості сонця.
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")

            if 'sunshine' not in self.data.columns or 'date' not in self.data.columns:
                raise ValueError("Дані повинні містити колонки 'sunshine' і 'date'.")

            self.data['year'] = self.data['date'].dt.year
            year_data = self.data[self.data['year'] == year]

            if year_data.empty:
                raise ValueError(f"Немає даних для року {year}.")

            year_data = year_data.sort_values(by='date')
            days = year_data['date'].dt.day_of_year
            sunshine_values = year_data['sunshine']

            norm = plt.Normalize(vmin=sunshine_values.min(), vmax=sunshine_values.max())
            cmap = plt.cm.YlOrRd  # Жовто-червоний градієнт
            colors = cmap(norm(sunshine_values))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')

            num_days = len(days)
            angles = np.linspace(0, 2 * np.pi, num_days, endpoint=False)  # Кути для променів

            for angle, sun_value, day_of_year, color in zip(angles, sunshine_values, days, colors):
                x = np.cos(angle) * sun_value  # X-координата кінця променя
                y = np.sin(angle) * sun_value  # Y-координата кінця променя


                ax.plot([0, x], [0, y], color=color, lw=0.7, alpha=0.8)

                if day_of_year % 30 == 0:  
                    ax.text(
                        x * 1.1, y * 1.1,
                        str(day_of_year), fontsize=6, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", edgecolor="gray", alpha=0.8)
                    )

            ax.scatter(0, 0, s=500, color='gold', zorder=3)

            ax.set_xlim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
            ax.set_ylim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
            ax.set_title(f"Сонячна активність за {year} рік", fontsize=12, fontweight='regular')
            ax.axis('off')  # Прибираємо осі

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.05, label="Сонячні години")

            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")

    def weather_report(self):
        """
        Створює комплексний текстовий звіт про погоду
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
    
            total_records = len(self.data)
            date_range = (self.data['date'].min(), self.data['date'].max())
            print(colored("\nЗагальна інформація:", "green"))
            print(f"- Кількість записів: {total_records}")
            print(f"- Період даних: {date_range[0].date()} — {date_range[1].date()}")
            
            avg_sunshine = self.data['sunshine'].mean()
            max_sunshine = self.data['sunshine'].max()
            sunniest_day = self.data.loc[self.data['sunshine'].idxmax(), 'date']
    
            print(colored(f"\nСередній сонячний світловий день: {avg_sunshine:.2f} годин/день"))
            print(f"Найсонячніший день: {sunniest_day.date()} ({max_sunshine:.2f} годин)\n")
    
            mean_temp = self.data['mean_temp'].mean()
            max_temp = self.data['max_temp'].max()
            min_temp = self.data['min_temp'].min()
            hottest_day = self.data.loc[self.data['max_temp'].idxmax(), 'date']
            coldest_day = self.data.loc[self.data['min_temp'].idxmin(), 'date']
    
            temp_data = [
                ["Середня температура (°C)", f"{mean_temp:.2f}"],
                ["Максимальна температура (°C)", f"{max_temp:.2f} (Дата: {hottest_day.date()})"],
                ["Мінімальна температура (°C)", f"{min_temp:.2f} (Дата: {coldest_day.date()})"]
            ]
            print(colored("Температура:", "blue"))
            print(tabulate(temp_data, headers=["Метрика", "Значення"], tablefmt="fancy_grid"))
            print()
    
            total_precipitation = self.data['precipitation'].sum()
            avg_precipitation = self.data['precipitation'].mean()
            max_precipitation_day = self.data.loc[self.data['precipitation'].idxmax(), 'date']

            precipitation_data = [
                ["Загальна кількість опадів (мм)", f"{total_precipitation:.2f}"],
                ["Середньодобова кількість опадів (мм)", f"{avg_precipitation:.2f}"],
                ["Найбільше опадів (мм)", f"{max_precipitation_day.date()}"]
            ]
            print(colored("Опади:", "blue"))
            print(tabulate(precipitation_data, headers=["Метрика", "Значення"], tablefmt="fancy_grid"))
            print()
            
        except Exception as e:
            print(colored(f"Помилка при створенні звіту: {e}", "red"))

    def cloud_pressure_plot(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")

            grouped_data = self.data.groupby(['decade', 'date'])['pressure'].mean().reset_index()
            decades = grouped_data['decade'].unique()
            num_decades = len(decades)

            cols = 3
            rows = (num_decades + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
            axes = axes.flatten()

            for i, decade in enumerate(decades):
                decade_data = grouped_data[grouped_data['decade'] == decade]

                ax = axes[i]
                y_pressure = decade_data['pressure'].to_numpy()
                dates = decade_data['date']

                ax.plot(
                    dates,
                    y_pressure,
                    color='lightblue',
                    linewidth=2,
                )

                if len(y_pressure) > 5:
                    p = np.polyfit(range(len(dates)), y_pressure, deg=5)
                    y_poly = np.polyval(p, range(len(dates)))
                    ax.plot(
                        dates,
                        y_poly,
                        color='blue',
                    )

                ax.set_title(f"Середній тиск: {decade}s", fontsize=14)
                ax.set_xlabel("Дата")
                ax.set_ylabel("Середній тиск")
                ax.grid(True, linestyle='--', alpha=0.5)

            fig.suptitle("Середній тиск по датах для кожного десятиліття", fontsize=16)

            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")



fetcher = WeatherDataFetcher()
df = fetcher._fetch_weather_data(r"E:\projects\Python\PP\src\london_weather.csv")

if not df.empty:
    plotter = WeatherDataPlotter(df)
    print("Дані успішно завантажені!")
    plotter.plot_temperature_with_scales()
    plotter.plot_snow_depth_by_decade()
    plotter.plot_radiation()
    plotter.plot_precipitation()
    plotter.cloud_cover()
    plotter.cloud_pressure_plot()
    plotter.sunshine()
    plotter.sunshine_year()
    plotter.weather_report()
