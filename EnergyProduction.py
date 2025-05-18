import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.filterwarnings('ignore')

class EnergyProductionSimulator:
    def __init__(self, excel_file_path):
        self.excel_file_path = excel_file_path
        self.df = self.load_and_prepare_data()
        self.models, self.poly = self.train_polynomial_regression_models()
        self.prediction_years = 20
        self.create_interactive_ui()

    def load_and_prepare_data(self):
        df = pd.read_excel(self.excel_file_path, header=10)

        year_column = [col for col in df.columns if 'year' in col.lower() or 'total' in col.lower()][0]
        fossil_columns = [col for col in df.columns if 'fossil' in col.lower() and 'production' in col.lower()]
        renewable_columns = [col for col in df.columns if 'renewable' in col.lower() and 'production' in col.lower()]

        rename_dict = {
            year_column: 'Annual Total',
            fossil_columns[0] if fossil_columns else None: 'Total Fossil Fuels Production',
            renewable_columns[0] if renewable_columns else None: 'Total Renewable Energy Production'
        }
        rename_dict = {k: v for k, v in rename_dict.items() if k is not None and v is not None}
        df = df.rename(columns=rename_dict)

        imputer = SimpleImputer(strategy='mean')
        for col in rename_dict.values():
            df[col] = imputer.fit_transform(df[[col]])

        return df[list(rename_dict.values())]

    def train_polynomial_regression_models(self, degree=2):
        X = self.df[['Annual Total']].values

        # Log-transform target values for exponential-like modeling
        y_fossil = np.log(self.df[['Total Fossil Fuels Production']].values + 1)
        y_renewable = np.log(self.df[['Total Renewable Energy Production']].values + 1)
        
        total_prod = (self.df['Total Fossil Fuels Production'] + self.df['Total Renewable Energy Production'].values)
        y_total = np.log(total_prod + 1)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train_fossil, y_test_fossil = train_test_split(X_poly, y_fossil, test_size=0.2, random_state=42)
        _, _, y_train_renewable, y_test_renewable = train_test_split(X_poly, y_renewable, test_size=0.2, random_state=42)
        _, _, y_train_total, y_test_total = train_test_split(X_poly, y_total, test_size=0.2, random_state=42)

        # Train linear models on log-scale data
        fossil_model = LinearRegression().fit(X_train, y_train_fossil)
        renewable_model = LinearRegression().fit(X_train, y_train_renewable)
        demand_model = LinearRegression().fit(X_train, y_train_total)

        # Store demand model for later inverse-transform use
        self.demand_model = demand_model

        # Wrap models to return inverse-transformed predictions
        def fossil_predict(X_input): return np.exp(fossil_model.predict(X_input)) - 1
        def renewable_predict(X_input): return np.exp(renewable_model.predict(X_input)) - 1
        def demand_predict(X_input): return np.exp(demand_model.predict(X_input)) - 1

        # Store as callable functions
        self.fossil_predict = fossil_predict
        self.renewable_predict = renewable_predict
        self.demand_predict = demand_predict

        return (fossil_predict, renewable_predict), poly

    def predict_adjusted_future_energy_production(self, fossil_decrease_rate, renewable_increase_rate):
        future_years = np.arange(self.df['Annual Total'].max() + 1, self.df['Annual Total'].max() + self.prediction_years + 1).reshape(-1, 1)
        future_years_poly = self.poly.transform(future_years)

        base_fossil = self.fossil_predict(future_years_poly).flatten()
        base_renewable = self.renewable_predict(future_years_poly).flatten()

        last_year = self.df['Annual Total'].max()
        last_fossil = self.df.loc[self.df['Annual Total'] == last_year, 'Total Fossil Fuels Production'].values[0]
        last_renewable = self.df.loc[self.df['Annual Total'] == last_year, 'Total Renewable Energy Production'].values[0]

        adjusted_fossil, adjusted_renewable = [last_fossil], [last_renewable]
        for _ in range(1, len(future_years)):
            adjusted_fossil.append(adjusted_fossil[-1] * (1 - fossil_decrease_rate))
            adjusted_renewable.append(adjusted_renewable[-1] * (1 + renewable_increase_rate))

        blend_factor = np.linspace(1, 0, len(future_years))
        adjusted_fossil_poly = base_fossil * blend_factor + np.array(adjusted_fossil) * (1 - blend_factor)
        adjusted_renewable_poly = base_renewable * blend_factor + np.array(adjusted_renewable) * (1 - blend_factor)

        # Predict energy demand using demand model
        predicted_demand = self.demand_predict(future_years_poly).flatten()
        total_supply = adjusted_fossil_poly + adjusted_renewable_poly
        success_score = np.mean(total_supply >= predicted_demand) * 100

        return future_years, adjusted_fossil_poly, adjusted_renewable_poly, predicted_demand, success_score

    def update_plot(self, *args):
        fossil_rate = self.fossil_slider.get() / 100.0
        renewable_rate = self.renewable_slider.get() / 100.0

        self.fossil_value_label.config(text=f"{fossil_rate:.2f} ({fossil_rate * 100:.0f}%)")
        self.renewable_value_label.config(text=f"{renewable_rate:.2f} ({renewable_rate * 100:.0f}%)")

        future_years, future_fossil, future_renewable, predicted_demand, score = self.predict_adjusted_future_energy_production(fossil_rate, renewable_rate)

        self.ax.clear()
        self.ax.fill_between(self.df['Annual Total'], 0, self.df['Total Fossil Fuels Production'], color='blue', alpha=0.3, label='Fossil Fuels (Historical)')
        self.ax.fill_between(self.df['Annual Total'], self.df['Total Fossil Fuels Production'], self.df['Total Fossil Fuels Production'] + self.df['Total Renewable Energy Production'], color='green', alpha=0.3, label='Renewable Energy (Historical)')

        self.ax.fill_between(future_years.flatten(), 0, future_fossil.flatten(), color='blue', alpha=0.5, linestyle='--', label='Fossil Fuels (Predicted)')
        self.ax.fill_between(future_years.flatten(), future_fossil.flatten(), future_fossil.flatten() + future_renewable.flatten(), color='green', alpha=0.5, linestyle='--', label='Renewable Energy (Predicted)')

        self.ax.plot(self.df['Annual Total'], self.df['Total Fossil Fuels Production'] + self.df['Total Renewable Energy Production'], color='purple', linewidth=2, label='Total Energy (Historical)')
        self.ax.plot(future_years.flatten(), future_fossil.flatten() + future_renewable.flatten(), color='red', linestyle='--', linewidth=2, label='Total Energy (Predicted)')

        # Updated: Plot demand using trained model
        self.ax.plot(future_years.flatten(), predicted_demand, color='black', linestyle=':', linewidth=2, label='Projected Energy Demand')

        crossover_index = next((i for i, (f, r) in enumerate(zip(future_fossil, future_renewable)) if r > f), None)
        if crossover_index is not None:
            crossover_year = int(future_years[crossover_index])
            self.ax.axvline(x=crossover_year, color='black', linestyle='--', alpha=0.7)
            self.ax.text(crossover_year + 0.5, self.ax.get_ylim()[1] * 0.9, f'Crossover: {crossover_year}', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            self.crossover_label.config(text=f"Renewable > Fossil in: {crossover_year}")
        else:
            self.crossover_label.config(text="No crossover detected in prediction period")

        self.ax.set_title("Energy Production: Historical and Predicted", fontsize=16)
        self.ax.set_xlabel("Year", fontsize=12)
        self.ax.set_ylabel("Production (Quadrillion Btu)", fontsize=12)
        self.ax.legend()
        self.ax.grid(alpha=0.7)
        self.canvas.draw()

        self.score_label.config(text=f"Energy Sufficiency Score: {score:.2f}%")
        self.last_score = score
        
    def check_sufficiency(self):
        fossil_rate = self.fossil_slider.get() / 100.0
        renewable_rate = self.renewable_slider.get() / 100.0

        # Get updated forecast and new score
        _, _, _, predicted_demand, success_score = self.predict_adjusted_future_energy_production(fossil_rate, renewable_rate)

        if success_score < 70:
            messagebox.showwarning(
                "Low Supply Score",
                "Warning: The adjusted energy supply may fall short of independently projected demand.\n\n"
                "Try adjusting the sliders to create a more balanced and sustainable energy mix."
            )


    def create_interactive_ui(self):
        self.root = tk.Tk()
        self.root.title("Energy Production Simulator - Educational Edition")
        self.root.geometry("1200x900")

        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Fossil Fuels Decrease Rate:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.fossil_slider = ttk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, length=300, command=self.update_plot)
        self.fossil_slider.grid(column=1, row=0, padx=5, pady=5)
        self.fossil_slider.set(6)
        self.fossil_value_label = ttk.Label(control_frame, text="0.06 (6%)")
        self.fossil_value_label.grid(column=2, row=0, padx=5, pady=5)

        ttk.Label(control_frame, text="Renewable Energy Increase Rate:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.renewable_slider = ttk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, length=300, command=self.update_plot)
        self.renewable_slider.grid(column=1, row=1, padx=5, pady=5)
        self.renewable_slider.set(2)
        self.renewable_value_label = ttk.Label(control_frame, text="0.02 (2%)")
        self.renewable_value_label.grid(column=2, row=1, padx=5, pady=5)

        self.crossover_label = ttk.Label(control_frame, text="Crossover prediction: calculating...", font=("Arial", 10, "bold"))
        self.crossover_label.grid(column=0, row=2, columnspan=3, padx=5, pady=10)

        self.score_label = ttk.Label(control_frame, text="Energy Sufficiency Score: Calculating...", font=("Arial", 10, "bold"))
        self.score_label.grid(column=0, row=3, columnspan=3, padx=5, pady=5)

        self.evaluate_button = ttk.Button(control_frame, text="Check Energy Sufficiency", command=self.check_sufficiency)
        self.evaluate_button.grid(column=0, row=4, columnspan=3, pady=10)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot()
        self.root.mainloop()

def main():
    simulator = EnergyProductionSimulator('EnergyDataSS.xlsx')

if __name__ == "__main__":
    main()
