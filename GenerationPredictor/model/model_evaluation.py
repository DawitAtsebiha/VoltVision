from common_functions.shared_imports import *

from common_functions.log import logger

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
