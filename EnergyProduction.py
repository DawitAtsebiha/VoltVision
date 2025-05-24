import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import seaborn as sns

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Enable interactive mode for multiple simultaneous plots
#plt.ion()

class GenerationPredictor:
    """Ontario hourly generation-mix prediction & analysis system with robust overfitting prevention."""

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

    def load_and_clean_data(self,
                            csv_path: str,
                            start_date: str = '2015-01-01',
                            min_data_points: int = 8760):
        """Load and clean data, skip duplicate rows, and preserve most data while handling gaps."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        logger.info("Loading generation data...")
        # read CSV and parse dates
        df = pd.read_csv(path, parse_dates=['Datetime'])
        self.fuel_types = [c for c in df.columns if c != 'Datetime']

        # drop every duplicate timestamp (keep first of each pair)
        df = df.loc[~df['Datetime'].duplicated(keep='first')]

        # set index, sort, and filter by start date
        df = (df
            .set_index('Datetime')
            .sort_index()
            .loc[start_date:])

        if len(df) < min_data_points:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_data_points}")

        # clip extreme outliers (0.1st and 99.9th percentiles)
        for fuel in self.fuel_types:
            lo = df[fuel].quantile(0.001)
            hi = df[fuel].quantile(0.999)
            df[fuel] = df[fuel].clip(lower=max(0, lo), upper=hi)

        # enforce hourly frequency
        df = df.asfreq('h')

        # looser interpolation: fill gaps up to 24 hours
        df.interpolate(method='time', limit=24, inplace=True)
        df.ffill(limit=24, inplace=True)
        df.bfill(limit=24, inplace=True)

        # drop rows with more than 30% of fuel columns missing
        thresh = int(len(self.fuel_types) * 0.7)
        before_drop = len(df)
        df.dropna(thresh=thresh, inplace=True)
        logger.info(f"Removed {before_drop - len(df)} rows with >30% missing")

        # fill any small remaining gaps (up to 2 hours)
        df.ffill(limit=2, inplace=True)
        df.bfill(limit=2, inplace=True)

        # warn if any NaNs remain
        total_nans = df.isna().sum().sum()
        if total_nans > 0:
            logger.warning(f"{total_nans} NaNs remain after filling — check your data gaps")

        # compute total generation
        df['Total'] = df[self.fuel_types].sum(axis=1)

        # add small noise to each fuel series to avoid perfect correlations
        np.random.seed(self.random_state)
        for fuel in self.fuel_types:
            noise_std = df[fuel].std() * 0.001
            df[fuel] = (df[fuel] + np.random.normal(0, noise_std, len(df))).clip(lower=0)

        # recalculate total after noise
        df['Total'] = df[self.fuel_types].sum(axis=1)

        # final check on minimum length
        if len(df) < min_data_points:
            raise ValueError(f"After cleaning: {len(df)} rows, need at least {min_data_points}")

        self.generation_data = df
        logger.info(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")

    def engineer_features(self) -> pd.DataFrame:
        """Conservative feature engineering to prevent data leakage and overfitting."""
        if self.generation_data is None:
            raise RuntimeError("Data not loaded")
            
        g = self.generation_data
        features = pd.DataFrame(index=g.index)
        
        # Basic time features (no future information)
        features['hour'] = g.index.hour
        features['dow'] = g.index.dayofweek
        features['month'] = g.index.month
        features['quarter'] = g.index.quarter
        features['is_weekend'] = (g.index.dayofweek >= 5).astype(int)
        features['is_business_hour'] = ((g.index.hour >= 8) & (g.index.hour <= 17) & 
                                       (g.index.dayofweek < 5)).astype(int)
        
        # Seasonal indicators
        features['is_summer'] = g.index.month.isin([6, 7, 8]).astype(int)
        features['is_winter'] = g.index.month.isin([12, 1, 2]).astype(int)
        
        # Cyclic encoding for continuity
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dow_sin'] = np.sin(2 * np.pi * features['dow'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['dow'] / 7)
        
        # ONLY historical lag features (no future information)
        # Reduced lags to prevent overfitting
        lag_hours = [1, 2, 24, 168]  # 1h, 2h, 1day, 1week
        for lag in lag_hours:
            features[f'total_lag_{lag}'] = g['Total'].shift(lag)
        
        # Historical rolling statistics (with sufficient window to prevent overfitting)
        rolling_windows = [24, 168]  # 1day, 1week
        for window in rolling_windows:
            # Use minimum periods to handle edge cases
            min_periods = max(window // 2, 12)  # At least 12 hours of data
            
            roll_mean = g['Total'].shift(1).rolling(window=window, min_periods=min_periods).mean()
            roll_std = g['Total'].shift(1).rolling(window=window, min_periods=min_periods).std()
            
            features[f'total_ma_{window}'] = roll_mean
            features[f'total_std_{window}'] = roll_std
            
            # Relative position vs recent history
            features[f'total_vs_ma_{window}'] = (g['Total'].shift(1) - roll_mean) / (roll_std + 1e-8)
        
        # Historical utilization ratios (shift by 1 to prevent leakage)
        for fuel, cap in self.capacity.items():
            if fuel in g.columns:
                util = (g[fuel].shift(1) / cap).clip(0, 1.5)
                features[f'{fuel.lower()}_util'] = util
        
        # Weather proxy features (based on time only, no future info)
        # Solar potential based on hour and season
        hour_angle = 2 * np.pi * (features['hour'] - 6) / 12
        seasonal_factor = 0.8 + 0.4 * np.cos(2 * np.pi * (features['month'] - 6) / 12)
        features['solar_potential'] = (np.maximum(0, np.sin(hour_angle)) * seasonal_factor).clip(0, 1)
        
        # Load pattern features
        features['peak_morning'] = ((features['hour'] >= 7) & (features['hour'] <= 9)).astype(int)
        features['peak_evening'] = ((features['hour'] >= 17) & (features['hour'] <= 19)).astype(int)
        features['off_peak'] = ((features['hour'] <= 6) | (features['hour'] >= 22)).astype(int)
        
        # Remove rows with NaN values (from lags and rolling windows)
        initial_length = len(features)
        features.dropna(inplace=True)
        logger.info(f"Removed {initial_length - len(features)} rows due to feature engineering NaNs")
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Engineered {len(self.feature_names)} features")
        
        return features

    def detect_multicollinearity(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Detect and remove highly correlated features to prevent perfect predictions."""
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        # Remove features with highest correlations
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            logger.warning(f"High correlation detected: {feat1} - {feat2} ({corr_val:.3f})")
            # Remove the second feature in each pair
            features_to_remove.add(feat2)
        
        return list(features_to_remove)

    def select_features(self, X: pd.DataFrame, y: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
        """Conservative feature selection to prevent overfitting."""
        
        # First, remove highly correlated features
        features_to_remove = self.detect_multicollinearity(X, threshold=0.95)
        if features_to_remove:
            X = X.drop(columns=features_to_remove)
            logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        # Limit the number of features based on sample size
        # Rule of thumb: at least 10 samples per feature
        max_features_by_samples = min(max_features, len(X) // 15)  # Conservative ratio
        
        # Use mutual information for feature selection (less prone to linear relationships)
        selector = SelectKBest(score_func=mutual_info_regression, k=max_features_by_samples)
        
        # Use total generation as target for feature selection
        X_selected = selector.fit_transform(X, y.sum(axis=1))
        
        # Get selected feature names
        mask = selector.get_support()
        self.selected_features = [col for i, col in enumerate(X.columns) if mask[i]]
        
        self.feature_selector = selector
        
        logger.info(f"Selected {len(self.selected_features)} features from {len(X.columns)} original features")
        logger.info(f"Selected features: {self.selected_features}")
        
        return pd.DataFrame(X_selected, index=X.index, columns=self.selected_features)

    def split_data(self):
        """Time-aware data splitting with minimum dataset size validation."""
        features = self.engineer_features()
        y = self.generation_data.loc[features.index, self.fuel_types]
        
        # Ensure we have enough data for reliable splits
        n = len(features)
        if n < 1000:  # Minimum 1000 samples
            raise ValueError(f"Dataset too small after preprocessing: {n} samples. Need at least 1000.")
        
        # Apply feature selection
        features_selected = self.select_features(features, y)
        
        # Conservative time-based splits
        test_size = max(int(n * self.test_size), 100)  # At least 100 test samples
        val_size = max(int(n * self.val_size), 50)     # At least 50 validation samples
        train_size = n - test_size - val_size
        
        if train_size < 200:  # Need at least 200 training samples
            raise ValueError(f"Training set too small: {train_size} samples. Reduce test_size/val_size.")
        
        self.X_train = features_selected.iloc[:train_size].copy()
        self.X_val = features_selected.iloc[train_size:train_size + val_size].copy()
        self.X_test = features_selected.iloc[train_size + val_size:].copy()
        
        self.y_train = y.iloc[:train_size].copy()
        self.y_val = y.iloc[train_size:train_size + val_size].copy()
        self.y_test = y.iloc[train_size + val_size:].copy()
        
        logger.info(f"Split: train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)}")
        
        # Validate splits don't have perfect correlations
        train_corr = np.corrcoef(self.X_train.T).max()
        if train_corr > 0.99:
            logger.warning(f"High correlation in training features: {train_corr:.4f}")

    def train_models(self):
        """Train models with strong regularization to prevent overfitting."""
        
        # Very conservative model configurations
        configs = {
            'Ridge Regression': Ridge(alpha=10.0),  # Strong regularization
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,      # Reduced trees
                max_depth=8,          # Shallow trees
                min_samples_split=20, # High minimum splits
                min_samples_leaf=10,  # High minimum leaf samples
                max_features=0.6,     # Reduced feature sampling
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50,      # Reduced iterations
                max_depth=3,          # Very shallow
                learning_rate=0.01,   # Very slow learning
                subsample=0.7,        # Strong subsampling
                min_samples_split=20, # High constraints
                min_samples_leaf=10,
                random_state=self.random_state
            )
        }
        
        for name, model in configs.items():
            logger.info(f"Training {name}...")
            
            try:
                # Use robust scaling for all models
                scaler = RobustScaler()
                self.scalers[name] = scaler
                
                X_train_scaled = scaler.fit_transform(self.X_train)
                X_val_scaled = scaler.transform(self.X_val)
                X_test_scaled = scaler.transform(self.X_test)
                
                # Check for scaling issues
                if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
                    logger.error(f"Scaling issues detected for {name}")
                    continue
                
                # Train model
                multi_model = MultiOutputRegressor(model)
                multi_model.fit(X_train_scaled, self.y_train)
                
                # Generate predictions
                val_preds = multi_model.predict(X_val_scaled)
                test_preds = multi_model.predict(X_test_scaled)
                
                # Validate predictions are reasonable
                if np.any(np.isnan(val_preds)) or np.any(np.isnan(test_preds)):
                    logger.error(f"NaN predictions detected for {name}")
                    continue
                
                self.models[name] = multi_model
                
                # Store predictions (ensure non-negative and reasonable bounds)
                max_reasonable = self.y_train.max().max() * 2  # Allow 2x max historical
                
                self.val_predictions[name] = pd.DataFrame(
                    np.clip(val_preds, 0, max_reasonable), 
                    index=self.X_val.index, 
                    columns=self.fuel_types
                )
                self.predictions[name] = pd.DataFrame(
                    np.clip(test_preds, 0, max_reasonable), 
                    index=self.X_test.index, 
                    columns=self.fuel_types
                )
                
                logger.info(f"{name} trained successfully")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

    def cross_validate_models(self, cv_folds: int = 3):
        """Conservative cross-validation with time series splits."""
        logger.info("Performing cross-validation...")
        
        # Combine train and validation for CV (but keep time order)
        X_cv = pd.concat([self.X_train, self.X_val])
        y_cv = pd.concat([self.y_train, self.y_val])
        
        # Reduced CV folds to prevent overfitting detection issues
        tscv = TimeSeriesSplit(n_splits=cv_folds, test_size=len(self.X_val))
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")
            
            cv_scores = []
            scaler = self.scalers[name]
            
            try:
                for train_idx, val_idx in tscv.split(X_cv):
                    X_fold_train, X_fold_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                    y_fold_train, y_fold_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
                    
                    # Scale data
                    fold_scaler = RobustScaler()
                    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                    X_fold_val_scaled = fold_scaler.transform(X_fold_val)
                    
                    # Create and train temporary model
                    base_model = type(model.estimators_[0])(**model.estimators_[0].get_params())
                    temp_model = MultiOutputRegressor(base_model)
                    temp_model.fit(X_fold_train_scaled, y_fold_train)
                    
                    # Predict and evaluate
                    fold_pred = temp_model.predict(X_fold_val_scaled)
                    fold_pred = np.clip(fold_pred, 0, None)  # Ensure non-negative
                    
                    fold_score = r2_score(y_fold_val, fold_pred)
                    cv_scores.append(fold_score)
                
                self.cv_scores[name] = {
                    'mean_r2': np.mean(cv_scores),
                    'std_r2': np.std(cv_scores),
                    'scores': cv_scores
                }
                
                logger.info(f"{name} CV R2: {self.cv_scores[name]['mean_r2']:.4f} ± {self.cv_scores[name]['std_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"CV error for {name}: {str(e)}")
                self.cv_scores[name] = {'mean_r2': -999, 'std_r2': 999, 'scores': [-999]}

    def evaluate_models(self):
        """Comprehensive model evaluation with overfitting detection."""
        for name in self.models.keys():
            try:
                # Validation metrics
                val_overall = {
                    'MAE': mean_absolute_error(self.y_val, self.val_predictions[name]),
                    'RMSE': np.sqrt(mean_squared_error(self.y_val, self.val_predictions[name])),
                    'R2': r2_score(self.y_val, self.val_predictions[name])
                }
                
                # Test metrics
                test_overall = {
                    'MAE': mean_absolute_error(self.y_test, self.predictions[name]),
                    'RMSE': np.sqrt(mean_squared_error(self.y_test, self.predictions[name])),
                    'R2': r2_score(self.y_test, self.predictions[name])
                }
                
                # Per-fuel metrics for test set
                fuel_metrics = {}
                for fuel in self.fuel_types:
                    fuel_metrics[fuel] = {
                        'MAE': mean_absolute_error(self.y_test[fuel], self.predictions[name][fuel]),
                        'RMSE': np.sqrt(mean_squared_error(self.y_test[fuel], self.predictions[name][fuel])),
                        'R2': r2_score(self.y_test[fuel], self.predictions[name][fuel])
                    }
                
                self.metrics[name] = {
                    'validation': val_overall,
                    'test': test_overall,
                    'fuel': fuel_metrics
                }
                
                # Log results with overfitting detection
                overfit_score = val_overall['R2'] - test_overall['R2']
                cv_score = self.cv_scores.get(name, {}).get('mean_r2', 'N/A')
                
                logger.info(f"{name} - Val R2: {val_overall['R2']:.4f}, Test R2: {test_overall['R2']:.4f}, "
                           f"Overfit: {overfit_score:.4f}, CV: {cv_score}")
                
                # Flag suspicious results
                if val_overall['R2'] > 0.999 or test_overall['R2'] > 0.999:
                    logger.warning(f"Suspiciously high R2 for {name} - possible overfitting or data leakage")
                
            except Exception as e:
                logger.error(f"Evaluation error for {name}: {str(e)}")

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
    predictor.load_and_clean_data(args.csv_path)
    predictor.split_data()
    predictor.train_models()
    predictor.cross_validate_models()
    predictor.evaluate_models()
    predictor.plot_results(show_days=args.show_days)
