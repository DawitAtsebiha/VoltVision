from common_functions.shared_imports import *
from common_functions.log import logger

def build_features(self) -> pd.DataFrame:
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