from common_functions.shared_imports import *

from common_functions.log import logger
from features.engineer_features import build_features, select_features

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


def split_data(self):
    """Time-aware data splitting with minimum dataset size validation."""
    build_features = self.engineer_features()

    y = self.generation_data.loc[build_features.index, self.fuel_types]
    
    # Ensure we have enough data for reliable splits
    n = len(build_features)
    if n < 1000:  # Minimum 1000 samples
        raise ValueError(f"Dataset too small after preprocessing: {n} samples. Need at least 1000.")
    
    # Apply feature selection
    features_selected = self.select_features(build_features, y)
    
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
