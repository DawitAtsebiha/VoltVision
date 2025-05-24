from common_functions.shared_imports import *
from common_functions.log import logger

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