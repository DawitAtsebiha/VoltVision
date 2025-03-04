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
from tkinter import ttk
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
        
        # Find the relevant columns dynamically
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
    
    def train_polynomial_regression_models(self, degree=5):
        X = self.df[['Annual Total']].values
        y_fossil = self.df[['Total Fossil Fuels Production']].values
        y_renewable = self.df[['Total Renewable Energy Production']].values
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        X_train, X_test, y_train_fossil, y_test_fossil = train_test_split(X_poly, y_fossil, test_size=0.2, random_state=42)
        _, _, y_train_renewable, y_test_renewable = train_test_split(X_poly, y_renewable, test_size=0.2, random_state=42)
        
        fossil_model = LinearRegression()
        fossil_model.fit(X_train, y_train_fossil)
        
        renewable_model = LinearRegression()
        renewable_model.fit(X_train, y_train_renewable)
        
        y_pred_fossil = fossil_model.predict(X_test)
        rmse_fossil = np.sqrt(mean_squared_error(y_test_fossil, y_pred_fossil))
        r2_fossil = r2_score(y_test_fossil, y_pred_fossil)
        
        y_pred_renewable = renewable_model.predict(X_test)
        rmse_renewable = np.sqrt(mean_squared_error(y_test_renewable, y_pred_renewable))
        r2_renewable = r2_score(y_test_renewable, y_pred_renewable)
        
        print(f"Fossil Fuels Model - RMSE: {rmse_fossil:.2f}, R²: {r2_fossil:.2f}")
        print(f"Renewable Energy Model - RMSE: {rmse_renewable:.2f}, R²: {r2_renewable:.2f}")
        
        return (fossil_model, renewable_model), poly
    
    def predict_adjusted_future_energy_production(self, fossil_decrease_rate, renewable_increase_rate):
        future_years = np.arange(self.df['Annual Total'].max() + 1, self.df['Annual Total'].max() + self.prediction_years + 1).reshape(-1, 1)
        future_years_poly = self.poly.transform(future_years)
        
        model_fossil_poly, model_renewable_poly = self.models
        base_fossil = model_fossil_poly.predict(future_years_poly).flatten()
        base_renewable = model_renewable_poly.predict(future_years_poly).flatten()
        
        # Get last known values from data
        last_year = self.df['Annual Total'].max()
        last_fossil = self.df.loc[self.df['Annual Total'] == last_year, 'Total Fossil Fuels Production'].values[0]
        last_renewable = self.df.loc[self.df['Annual Total'] == last_year, 'Total Renewable Energy Production'].values[0]
        
        # Apply rates year by year 
        adjusted_fossil = [last_fossil]
        adjusted_renewable = [last_renewable]
        
        for i in range(1, len(future_years)):
            new_fossil = adjusted_fossil[-1] * (1 - fossil_decrease_rate)
            new_renewable = adjusted_renewable[-1] * (1 + renewable_increase_rate)
            adjusted_fossil.append(new_fossil)
            adjusted_renewable.append(new_renewable)
        
        # Add model factor to smooth transition from historical to predicted
        # This blends the polynomial model with the simple growth model
        blend_factor = np.linspace(1, 0, len(future_years))  # Transition from model to pure rates
        
        adjusted_fossil_poly = base_fossil * blend_factor + np.array(adjusted_fossil) * (1 - blend_factor)
        adjusted_renewable_poly = base_renewable * blend_factor + np.array(adjusted_renewable) * (1 - blend_factor)
        
        return future_years, adjusted_fossil_poly, adjusted_renewable_poly
    
    def update_plot(self, *args):
        fossil_decrease_rate = self.fossil_slider.get() / 100.0
        renewable_increase_rate = self.renewable_slider.get() / 100.0
        
        # Update the labels to show the current values
        self.fossil_value_label.config(text=f"{fossil_decrease_rate:.2f} ({fossil_decrease_rate*100:.0f}%)")
        self.renewable_value_label.config(text=f"{renewable_increase_rate:.2f} ({renewable_increase_rate*100:.0f}%)")
        
        future_years, future_fossil, future_renewable = self.predict_adjusted_future_energy_production(
            fossil_decrease_rate, renewable_increase_rate)
        
        # Clear the existing plot
        self.ax.clear()
        
        # Recreate the plot with new data
        self.ax.fill_between(self.df['Annual Total'], 0, self.df['Total Fossil Fuels Production'], 
                            color='blue', alpha=0.3, label='Fossil Fuels (Historical)')
        self.ax.fill_between(self.df['Annual Total'], self.df['Total Fossil Fuels Production'], 
                            self.df['Total Fossil Fuels Production'] + self.df['Total Renewable Energy Production'], 
                            color='green', alpha=0.3, label='Renewable Energy (Historical)')
        
        self.ax.fill_between(future_years.flatten(), 0, future_fossil.flatten(), 
                            color='blue', alpha=0.5, linestyle='--', label='Fossil Fuels (Predicted)')
        self.ax.fill_between(future_years.flatten(), future_fossil.flatten(), 
                            future_fossil.flatten() + future_renewable.flatten(), 
                            color='green', alpha=0.5, linestyle='--', label='Renewable Energy (Predicted)')
        
        self.ax.plot(self.df['Annual Total'], self.df['Total Fossil Fuels Production'] + self.df['Total Renewable Energy Production'], 
                    color='purple', linewidth=2, label='Total Energy (Historical)')
        self.ax.plot(future_years.flatten(), future_fossil.flatten() + future_renewable.flatten(), 
                    color='red', linestyle='--', linewidth=2, label='Total Energy (Predicted)')
        
        # Calculate the crossover point (if any)
        crossover_index = None
        for i in range(len(future_fossil)):
            if future_renewable[i] > future_fossil[i]:
                crossover_index = i
                break
        
        if crossover_index is not None:
            crossover_year = future_years.flatten()[crossover_index]
            self.ax.axvline(x=crossover_year, color='black', linestyle='--', alpha=0.7)
            self.ax.text(crossover_year + 0.5, self.ax.get_ylim()[1] * 0.9, 
                         f'Crossover: {int(crossover_year)}', 
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            self.crossover_label.config(text=f"Renewable > Fossil in: {int(crossover_year)}")
        else:
            self.crossover_label.config(text="No crossover detected in prediction period")
        
        # Add titles and labels
        self.ax.set_title("Energy Production: Historical and Predicted", fontsize=16)
        self.ax.set_xlabel("Year", fontsize=12)
        self.ax.set_ylabel("Production (Quadrillion Btu)", fontsize=12)
        self.ax.legend()
        self.ax.grid(alpha=0.7)
        
        # Redraw the canvas
        self.canvas.draw()
    
    def create_interactive_ui(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Energy Production Simulator")
        self.root.geometry("1200x800")
        
        # Create a frame for the controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create sliders for adjusting rates
        ttk.Label(control_frame, text="Fossil Fuels Decrease Rate:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.fossil_slider = ttk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, length=300, command=self.update_plot)
        self.fossil_slider.grid(column=1, row=0, padx=5, pady=5)
        self.fossil_slider.set(6)  # Default 6% decrease
        self.fossil_value_label = ttk.Label(control_frame, text="0.06 (6%)")
        self.fossil_value_label.grid(column=2, row=0, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Renewable Energy Increase Rate:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.renewable_slider = ttk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, length=300, command=self.update_plot)
        self.renewable_slider.grid(column=1, row=1, padx=5, pady=5)
        self.renewable_slider.set(2)  # Default 2% increase
        self.renewable_value_label = ttk.Label(control_frame, text="0.02 (2%)")
        self.renewable_value_label.grid(column=2, row=1, padx=5, pady=5)
        
        # Add crossover information
        self.crossover_label = ttk.Label(control_frame, text="Crossover prediction: calculating...", font=("Arial", 10, "bold"))
        self.crossover_label.grid(column=0, row=2, columnspan=3, padx=5, pady=10)
        
        # Create a frame for the matplotlib figure
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Create the figure and axis
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create the canvas and add it to the frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Generate the initial plot
        self.update_plot()
        
        # Start the mainloop
        self.root.mainloop()

def main():
    excel_file_path = 'EnergyDataSS.xlsx'
    simulator = EnergyProductionSimulator(excel_file_path)

if __name__ == "__main__":
    main()