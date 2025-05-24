from common_functions.shared_imports import *

from features.engineer_features import build_features, select_features
from features.adjustment_on_features import detect_multicollinearity, split_data
from model.model_training import train_models
from model.model_evaluation import cross_validate_models, evaluate_models
from data.load_and_clean import load_and_clean_data
from common_functions.log import logger

# Enable interactive mode for multiple simultaneous plots
#plt.ion()

class GenerationPredictor:
    """Ontario hourly generation-mix prediction & analysis system with robust overfitting prevention."""

    engineer_features = build_features
    load_and_clean_data = load_and_clean_data

    def __init__(self, test_size: float = 0.2, val_size: float = 0.15, random_state: int = 42):
        if not 0 < test_size < 1 or not 0 < val_size < 1:
            raise ValueError("test_size and val_size must be in (0,1)")
        if test_size + val_size >= 0.8:  # Ensure sufficient training data
            raise ValueError("test_size + val_size must be < 0.8 to ensure adequate training data")
            
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # Updated installed capacities (MW)
        self.capacity = {
            'Nuclear': 12900,
            'Hydro': 8900,
            'Wind': 5500,
            'Solar': 1500,
            'Biofuel': 400,
            'Other': 200,
            'Gas': 10500
        }

        # Initialize all attributes
        self.fuel_types: List[str] = []
        self.generation_data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []

        self.models: Dict[str, MultiOutputRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_selector: Optional[SelectKBest] = None
        self.predictions: Dict[str, pd.DataFrame] = {}
        self.val_predictions: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.cv_scores: Dict[str, Dict[str, float]] = {}

    def detect_multicollinearity(self, X, threshold=0.95):
        return detect_multicollinearity(self, X, threshold)

    def select_features(self, X, y, max_features=20):
        return select_features(self, X, y, max_features)

    def split_data(self):
        split_data(self)

    def train_models(self):
        train_models(self)

    def cross_validate_models(self):
        cross_validate_models(self)

    def evaluate_models(self):
        evaluate_models(self)

    def plot_results(self, show_days: int = 30):
        """Enhanced plotting with overfitting detection and multiple simultaneous windows."""
        if not self.metrics:
            raise RuntimeError("Evaluate models before plotting")
        
        # Model comparison plot
        fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig1.suptitle("Model Performance Analysis", fontsize=16)
        
        models = list(self.metrics.keys())
        val_r2 = [self.metrics[m]['validation']['R2'] for m in models]
        test_r2 = [self.metrics[m]['test']['R2'] for m in models]
        cv_r2 = [self.cv_scores[m]['mean_r2'] for m in models if m in self.cv_scores]
        
        x_pos = np.arange(len(models))
        
        # Validation vs Test R²
        axes[0,0].bar(x_pos - 0.2, val_r2, 0.4, label='Validation', alpha=0.8)
        axes[0,0].bar(x_pos + 0.2, test_r2, 0.4, label='Test', alpha=0.8)
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('R²')
        axes[0,0].set_title('Validation vs Test R²')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Cross-validation scores
        if cv_r2:
            cv_std = [self.cv_scores[m]['std_r2'] for m in models if m in self.cv_scores]
            axes[0,1].bar(range(len(cv_r2)), cv_r2, yerr=cv_std, capsize=5, alpha=0.8)
            axes[0,1].set_xlabel('Models')
            axes[0,1].set_ylabel('Cross-Validation R²')
            axes[0,1].set_title('Cross-Validation Performance')
            axes[0,1].set_xticks(range(len(cv_r2)))
            axes[0,1].set_xticklabels([m for m in models if m in self.cv_scores], rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # Overfitting detection
        r2_diff = np.array(val_r2) - np.array(test_r2)
        colors = ['red' if diff > 0.1 else 'orange' if diff > 0.05 else 'green' for diff in r2_diff]
        axes[1,0].bar(x_pos, r2_diff, color=colors, alpha=0.7)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Mild overfitting')
        axes[1,0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Strong overfitting')
        axes[1,0].set_xlabel('Models')
        axes[1,0].set_ylabel('Val R² - Test R²')
        axes[1,0].set_title('Overfitting Detection')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(models, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Model stability (CV std)
        if cv_r2:
            axes[1,1].bar(range(len(cv_std)), cv_std, alpha=0.8, color='purple')
            axes[1,1].set_xlabel('Models')
            axes[1,1].set_ylabel('CV Standard Deviation')
            axes[1,1].set_title('Model Stability (Lower is Better)')
            axes[1,1].set_xticks(range(len(cv_std)))
            axes[1,1].set_xticklabels([m for m in models if m in self.cv_scores], rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Select best model based on balanced performance
        valid_models = [m for m in models if self.metrics[m]['test']['R2'] > 0 and self.metrics[m]['test']['R2'] < 0.99]
        
        if valid_models:
            best_model = min(valid_models, 
                           key=lambda x: abs(self.metrics[x]['validation']['R2'] - self.metrics[x]['test']['R2']))
        else:
            best_model = max(models, key=lambda x: self.metrics[x]['test']['R2'])
        
        logger.info(f"Selected model: {best_model}")
        
        # Time series plot for best model (in separate window)
        self._plot_time_series(best_model, show_days)
        
        # Feature importance plot (in separate window)
        self._plot_feature_importance(best_model)
        
        # Fuel-specific performance plot (in separate window)
        self._plot_fuel_performance()

    def _plot_time_series(self, model_name: str, show_days: int):
        """Plot time series results for a specific model as three separate figures."""
        # Define data window
        end_date = self.y_test.index[-1]
        start_date = end_date - pd.Timedelta(days=show_days)
        mask = (self.y_test.index >= start_date)
        y_actual = self.y_test.loc[mask]
        y_pred = self.predictions[model_name].loc[mask]

        # 1) Total Generation
        fig_total = plt.figure(figsize=(10, 5))
        fig_total.suptitle(f'Total Generation - {model_name}', fontsize=14)
        plt.plot(y_actual.index, y_actual.sum(axis=1), label='Actual Total', linewidth=2)
        plt.plot(y_pred.index,   y_pred.sum(axis=1),   label='Predicted Total', linewidth=2, linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Generation (MW)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2) Generation by Fuel Type
        fig_fuel = plt.figure(figsize=(10, 5))
        fig_fuel.suptitle(f'Generation by Fuel Type - {model_name}', fontsize=14)
        for fuel in self.fuel_types:
            plt.plot(y_actual.index, y_actual[fuel],      label=f'{fuel} Actual', linewidth=1)
            plt.plot(y_pred.index,   y_pred[fuel],        linestyle='--', label=f'{fuel} Predicted', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Generation (MW)')
        plt.legend(loc='upper right', ncol=2)
        plt.grid(True)
        plt.show()

        # 3) Prediction Error by Fuel Type
        fig_error = plt.figure(figsize=(10, 5))
        fig_error.suptitle(f'Prediction Error by Fuel Type - {model_name}', fontsize=14)
        errors = y_actual - y_pred
        for fuel in self.fuel_types:
            plt.plot(errors.index, errors[fuel], label=f'{fuel} Error', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Error (MW)')
        plt.legend(loc='upper right', ncol=2)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the GenerationPredictor end-to-end pipeline"
    )
    parser.add_argument(
        "csv_path",
        help="Path to the CSV file containing the generation data"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to reserve for the test set"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Proportion of data to reserve for the validation set"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for random operations to ensure reproducibility"
    )
    parser.add_argument(
        "--show_days",
        type=int,
        default=30,
        help="Number of days to display in the final time-series plot"
    )
    args = parser.parse_args()

    predictor = GenerationPredictor(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    predictor.load_and_clean_data(args.csv_path, start_date='2015-01-01')
    predictor.split_data()
    predictor.train_models()
    predictor.cross_validate_models()
    predictor.evaluate_models()
    predictor.plot_results(show_days=args.show_days)
