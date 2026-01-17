"""
Train machine learning model for reference matching
OPTIMIZED VERSION with proper parameters and tuning
"""

from catboost import CatBoostClassifier, CatBoostRanker, Pool
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import json
import pickle

class ModelTrainer:
    def __init__(self, model_type='ranker', use_gpu=False):
        """
        Args:
            model_type: 'classifier' or 'ranker'
            use_gpu: Whether to use GPU acceleration
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.model = None
        self.best_params = None
        self.training_history = {}
        
        # Initialize model with optimized parameters
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with optimal parameters"""
        
        if self.model_type == 'classifier':
            self.model = CatBoostClassifier(
                # Core parameters
                iterations=500,
                learning_rate=0.03,
                depth=6,
                
                # Loss function
                loss_function='Logloss',
                eval_metric='AUC',
                
                # Class imbalance handling - CRITICAL!
                auto_class_weights='Balanced',
                
                # Regularization
                l2_leaf_reg=3.0,
                random_strength=1.0,
                bagging_temperature=1.0,
                
                # Overfitting prevention
                early_stopping_rounds=50,
                od_type='Iter',
                
                # Performance
                task_type='GPU' if self.use_gpu else 'CPU',
                devices='0' if self.use_gpu else None,
                thread_count=-1,
                
                # Reproducibility
                random_seed=42,
                verbose=50,
                
                # Other
                use_best_model=True,
                bootstrap_type='Bayesian',
            )
        
        elif self.model_type == 'ranker':
            self.model = CatBoostRanker(
                # Core parameters - OPTIMIZED for ranking
                iterations=800,           # More iterations for better convergence
                learning_rate=0.05,       # Slightly higher LR works well with ranking
                depth=8,                  # Deeper trees for complex ranking patterns
                
                # Loss function - YetiRank is best for NDCG optimization
                loss_function='YetiRank',
                
                # Evaluation metrics - use NDCG without colon format for compatibility
                eval_metric='NDCG',
                custom_metric=['MAP', 'PrecisionAt:top=5', 'RecallAt:top=5'],
                
                # Regularization - tuned for ranking
                l2_leaf_reg=5.0,          # Slightly higher regularization
                random_strength=0.5,      # Reduced randomness
                bagging_temperature=0.8,  # Moderate bagging
                
                # Overfitting prevention
                early_stopping_rounds=100,  # More patience for ranking
                od_type='Iter',
                
                # Tree structure
                min_data_in_leaf=5,       # Prevent overfitting on small groups
                grow_policy='SymmetricTree',
                
                # Performance
                task_type='GPU' if self.use_gpu else 'CPU',
                devices='0' if self.use_gpu else None,
                thread_count=-1,
                
                # Reproducibility
                random_seed=42,
                verbose=100,              # Less frequent logging
                
                # Other
                use_best_model=True,
                bootstrap_type='Bayesian',
            )
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def prepare_data_classifier(self, pairs, features_df, labels):
        """
        Prepare data for classifier
        
        Args:
            pairs: List of pair dicts
            features_df: DataFrame with features
            labels: List of binary labels (0/1)
        
        Returns:
            X, y, cat_features
        """
        X = features_df.copy()
        y = labels
        
        # IMPORTANT: Only treat actual string/object columns as categorical
        # Do NOT treat binary numeric (0/1) features as categorical!
        # This was causing CatBoost error: "bad object for id: 0.0"
        cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object':
                cat_features.append(col)
        
        # Convert categorical features to string (required by CatBoost)
        for col in cat_features:
            X[col] = X[col].fillna('missing')
            X[col] = X[col].astype(str)
        
        # Ensure no NaN in numeric columns
        for col in X.columns:
            if col not in cat_features:
                # Fill NaN with 0 for numeric columns
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X, y, cat_features
    
    def prepare_data_ranker(self, pairs, features_df, labels):
        """
        Prepare data for ranker
        
        For ranking, we need group IDs: each BibTeX entry = one group
        
        Args:
            pairs: List of pair dicts
            features_df: DataFrame with features
            labels: List of binary labels
        
        Returns:
            Pool object for CatBoost ranker
        """
        X = features_df.copy()
        y = labels
        
        # Create group IDs (one per BibTeX entry)
        group_ids = []
        current_group = 0
        current_bibtex = None
        
        for pair in pairs:
            bibtex_key = pair['bibtex_key']
            if bibtex_key != current_bibtex:
                current_group += 1
                current_bibtex = bibtex_key
            group_ids.append(current_group)
        
        # IMPORTANT: Only treat actual string/object columns as categorical
        # Do NOT treat binary numeric (0/1) features as categorical!
        # This was causing CatBoost error: "bad object for id: 0.0"
        cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object':
                cat_features.append(col)
        
        # Convert categorical features to string (required by CatBoost)
        for col in cat_features:
            X[col] = X[col].fillna('missing')
            X[col] = X[col].astype(str)
        
        # Ensure no NaN in numeric columns
        for col in X.columns:
            if col not in cat_features:
                # Fill NaN with 0 for numeric columns
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return Pool(
            data=X,
            label=y,
            group_id=group_ids,
            cat_features=cat_features if cat_features else None  # None if no cat features
        )
    
    def train(self, train_pairs, train_features, train_labels,
              val_pairs, val_features, val_labels):
        """
        Train the model
        
        Args:
            train_pairs: Training pair dicts
            train_features: Training features DataFrame
            train_labels: Training labels
            val_pairs: Validation pair dicts
            val_features: Validation features DataFrame
            val_labels: Validation labels
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} model")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_labels)}")
        print(f"Validation samples: {len(val_labels)}")
        
        # Check class balance
        pos_count = sum(train_labels)
        neg_count = len(train_labels) - pos_count
        print(f"Class balance: {pos_count} positive / {neg_count} negative")
        
        if pos_count > 0:
            print(f"Imbalance ratio: 1:{neg_count/pos_count:.1f}")
        else:
            print("WARNING: No positive samples in training data!")
            return  # Cannot train without positive samples
        
        if self.model_type == 'classifier':
            # Prepare data
            X_train, y_train, cat_features = self.prepare_data_classifier(
                train_pairs, train_features, train_labels
            )
            X_val, y_val, _ = self.prepare_data_classifier(
                val_pairs, val_features, val_labels
            )
            
            # Train
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                verbose=50,
                plot=False
            )
        
        elif self.model_type == 'ranker':
            # Prepare data as Pool
            train_pool = self.prepare_data_ranker(
                train_pairs, train_features, train_labels
            )
            val_pool = self.prepare_data_ranker(
                val_pairs, val_features, val_labels
            )
            
            # Train
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=50,
                plot=False
            )
        
        # Store training history
        self.training_history = self.model.get_evals_result()
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best iteration: {self.model.get_best_iteration()}")
        print(f"{'='*60}\n")
    
    def tune_hyperparameters(self, train_pairs, train_features, train_labels,
                           val_pairs, val_features, val_labels):
        """
        Hyperparameter tuning using grid search
        
        KEY PARAMETERS TO TUNE:
        ----------------------
        1. learning_rate: [0.01, 0.03, 0.05]
        2. depth: [4, 6, 8]
        3. l2_leaf_reg: [1, 3, 5]
        
        Returns:
            best_params: Dict of best parameters
        """
        print(f"\n{'='*60}")
        print(f"Hyperparameter Tuning")
        print(f"{'='*60}\n")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.03, 0.05],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
        }
        
        best_score = -np.inf
        best_params = None
        
        from itertools import product
        
        # Grid search
        total_combinations = len(param_grid['learning_rate']) * \
                           len(param_grid['depth']) * \
                           len(param_grid['l2_leaf_reg'])
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        combination_num = 0
        for lr in param_grid['learning_rate']:
            for depth in param_grid['depth']:
                for l2 in param_grid['l2_leaf_reg']:
                    combination_num += 1
                    print(f"\n[{combination_num}/{total_combinations}] "
                          f"Testing: lr={lr}, depth={depth}, l2={l2}")
                    
                    # Create model with these params
                    if self.model_type == 'classifier':
                        model = CatBoostClassifier(
                            iterations=300,
                            learning_rate=lr,
                            depth=depth,
                            l2_leaf_reg=l2,
                            loss_function='Logloss',
                            eval_metric='AUC',
                            auto_class_weights='Balanced',
                            early_stopping_rounds=30,
                            random_seed=42,
                            verbose=False
                        )
                        
                        X_train, y_train, cat_features = self.prepare_data_classifier(
                            train_pairs, train_features, train_labels
                        )
                        X_val, y_val, _ = self.prepare_data_classifier(
                            val_pairs, val_features, val_labels
                        )
                        
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_val, y_val),
                            cat_features=cat_features,
                            verbose=False
                        )
                        
                        # Get best score
                        score = model.get_best_score()['validation']['AUC']
                    
                    else:  # ranker
                        model = CatBoostRanker(
                            iterations=300,
                            learning_rate=lr,
                            depth=depth,
                            l2_leaf_reg=l2,
                            loss_function='YetiRank',
                            eval_metric='NDCG',  # Use standard NDCG for compatibility
                            early_stopping_rounds=30,
                            random_seed=42,
                            verbose=False
                        )
                        
                        train_pool = self.prepare_data_ranker(
                            train_pairs, train_features, train_labels
                        )
                        val_pool = self.prepare_data_ranker(
                            val_pairs, val_features, val_labels
                        )
                        
                        model.fit(
                            train_pool,
                            eval_set=val_pool,
                            verbose=False
                        )
                        
                        # Try to get NDCG score with fallback
                        val_scores = model.get_best_score().get('validation', {})
                        score = val_scores.get('NDCG') or val_scores.get('NDCG:top=5') or \
                                next((v for k, v in val_scores.items() if 'NDCG' in k), 0.0)
                    
                    print(f"  Score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'learning_rate': lr,
                            'depth': depth,
                            'l2_leaf_reg': l2
                        }
                        print(f"  ✓ New best score!")
        
        print(f"\n{'='*60}")
        print(f"Best parameters found:")
        print(f"  learning_rate: {best_params['learning_rate']}")
        print(f"  depth: {best_params['depth']}")
        print(f"  l2_leaf_reg: {best_params['l2_leaf_reg']}")
        print(f"  Best score: {best_score:.4f}")
        print(f"{'='*60}\n")
        
        self.best_params = best_params
        
        # Re-initialize model with best params
        self.model.set_params(**best_params)
        
        return best_params
    
    def predict(self, X):
        """
        Make predictions
        
        For classifier: Returns probabilities of class 1
        For ranker: Returns ranking scores
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array of scores/probabilities
        """
        if self.model_type == 'classifier':
            # Return probability of positive class
            return self.model.predict_proba(X)[:, 1]
        else:
            # Return ranking scores
            return self.model.predict(X)
    
    def predict_top_k(self, bibtex_key, pairs, features_df, k=5, 
                     post_process=True):
        """
        Predict top-k candidates for one BibTeX entry
        
        Args:
            bibtex_key: BibTeX entry key
            pairs: All pairs for this BibTeX entry
            features_df: Features for all pairs
            k: Number of top candidates
            post_process: Whether to apply post-processing
        
        Returns:
            List of top-k candidate IDs sorted by score
        """
        # Get predictions
        scores = self.predict(features_df)
        
        # Create (candidate_id, score, candidate_data) tuples
        # IMPORTANT: Store only necessary data to avoid RecursionError
        candidate_scores = []
        for i, pair in enumerate(pairs):
            if pair['bibtex_key'] == bibtex_key:
                # Extract only needed data, avoid storing entire pair
                candidate_scores.append((
                    pair['candidate_id'],
                    scores[i],
                    pair.get('candidate_data', {})  # Only store candidate_data
                ))
        
        # Post-processing (if enabled)
        if post_process:
            # Get bibtex_data safely
            bibtex_data = pairs[0].get('bibtex_data', {}) if pairs else {}
            candidate_scores = self._post_process(
                bibtex_key, 
                candidate_scores,
                bibtex_data
            )
        
        # Sort by score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k candidate IDs
        return [cand_id for cand_id, score, _ in candidate_scores[:k]]
    
    def _post_process(self, bibtex_key, candidate_scores, bibtex_data):
        """
        Post-process predictions to improve ranking
        
        STRATEGIES:
        1. Exact title match → boost to top
        2. Year filter → penalize large differences
        3. First author match → boost score
        4. ArXiv ID extraction → exact match
        
        Args:
            bibtex_key: BibTeX entry key
            candidate_scores: List of (candidate_id, score, candidate_data) tuples
            bibtex_data: BibTeX entry data
        
        Returns:
            Modified candidate_scores
        """
        processed = []
        
        for cand_id, score, candidate_data in candidate_scores:
            new_score = score
            
            # Strategy 1: Exact title match
            if self._exact_title_match(bibtex_data, candidate_data):
                new_score = 1.0  # Force to top
            
            # Strategy 2: Year filter
            year_diff = self._year_difference(bibtex_data, candidate_data)
            if year_diff > 3:
                new_score *= 0.1  # Heavy penalty
            
            # Strategy 3: First author match
            if self._first_author_match(bibtex_data, candidate_data):
                new_score *= 1.2  # Boost by 20%
            
            # Strategy 4: ArXiv ID in BibTeX
            if self._has_arxiv_id(bibtex_data, cand_id):
                new_score = 1.0  # Force to top
            
            processed.append((cand_id, new_score, candidate_data))
        
        return processed
    
    def _exact_title_match(self, bibtex_data, candidate_data):
        """Check if titles match exactly (case-insensitive)"""
        title1 = bibtex_data.get('title', '').lower().strip()
        title2 = candidate_data.get('paper_title', '').lower().strip()
        return title1 == title2 and len(title1) > 0
    
    def _year_difference(self, bibtex_data, candidate_data):
        """Calculate year difference"""
        import re
        
        # Extract year from BibTeX
        year1 = bibtex_data.get('year', '')
        # Use type() instead of isinstance() to avoid recursion issues
        if type(year1).__name__ == 'str':
            match = re.search(r'\d{4}', year1)
            year1 = int(match.group()) if match else 0
        
        # Extract year from candidate
        year2_str = candidate_data.get('submission_date', '')
        if year2_str:
            try:
                year2 = int(str(year2_str)[:4])
            except (ValueError, TypeError):
                year2 = 0
        else:
            year2 = 0
        
        if year1 == 0 or year2 == 0:
            return 0
        
        return abs(year1 - year2)
    
    def _first_author_match(self, bibtex_data, candidate_data):
        """Check if first authors match"""
        authors1 = bibtex_data.get('authors', [])
        authors2 = candidate_data.get('authors', [])
        
        if not authors1 or not authors2:
            return False
        
        first1 = authors1[0].lower().strip()
        first2 = authors2[0].lower().strip()
        
        # Check if last names match
        last1 = first1.split()[-1]
        last2 = first2.split()[-1]
        
        return last1 == last2
    
    def _has_arxiv_id(self, bibtex_data, candidate_arxiv_id):
        """Check if BibTeX entry contains the arXiv ID"""
        import re
        
        # Search in all BibTeX fields
        bibtex_str = str(bibtex_data).lower()
        
        # Normalize arXiv ID (remove hyphens, dots)
        normalized_id = candidate_arxiv_id.replace('-', '').replace('.', '')
        
        # Check if ID appears anywhere
        if normalized_id.lower() in bibtex_str.replace('-', '').replace('.', ''):
            return True
        
        # Also check pattern like "arXiv:1606.03490"
        pattern = r'arxiv[:\s]*' + re.escape(candidate_arxiv_id.replace('-', '.'))
        if re.search(pattern, bibtex_str, re.IGNORECASE):
            return True
        
        return False
    
    def get_feature_importance(self, train_pool=None):
        """
        Get feature importance
        
        Args:
            train_pool: Optional training Pool for LossFunctionChange calculation.
                        If not provided, uses PredictionValuesChange (faster, no data needed).
        
        Returns:
            DataFrame with feature names and importance scores
        """
        feature_names = self.model.feature_names_
        
        # Use PredictionValuesChange by default (doesn't require training data)
        # LossFunctionChange is more accurate but requires the training dataset
        try:
            if train_pool is not None:
                importance = self.model.get_feature_importance(
                    data=train_pool, 
                    type='LossFunctionChange'
                )
            else:
                # PredictionValuesChange doesn't require training data
                importance = self.model.get_feature_importance(
                    type='PredictionValuesChange'
                )
        except Exception as e:
            print(f"Warning: Could not get feature importance with primary method: {e}")
            try:
                # Fallback to basic feature importance
                importance = self.model.get_feature_importance(type='FeatureImportance')
            except Exception:
                # Last resort: return zeros
                print("Warning: Using zero importance as fallback")
                importance = [0.0] * len(feature_names)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, path):
        """Save trained model"""
        # Check if model has been trained
        if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted():
            print("WARNING: Model not trained, skipping save.")
            return
            
        self.model.save_model(path)
        print(f"Model saved to {path}")
        
        # Also save metadata
        try:
            feature_importance = self.get_feature_importance().to_dict()
        except Exception as e:
            print(f"WARNING: Could not get feature importance: {e}")
            feature_importance = {}
            
        metadata = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'feature_importance': feature_importance
        }
        
        metadata_path = path.replace('.cbm', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, path):
        """Load trained model"""
        if self.model_type == 'classifier':
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRanker()
        
        self.model.load_model(path)
        print(f"Model loaded from {path}")
        
        # Load metadata if exists
        metadata_path = path.replace('.cbm', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_params = metadata.get('best_params')
                self.training_history = metadata.get('training_history')
            print(f"Metadata loaded from {metadata_path}")
        except FileNotFoundError:
            print("No metadata file found")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot (if None, display only)
        """
        import matplotlib.pyplot as plt
        
        if not self.training_history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Training vs validation metric
        metric_plotted = False
        
        if 'learn' in self.training_history:
            available_metrics = list(self.training_history['learn'].keys())
            print(f"Available metrics in training history: {available_metrics}")
            
            # Define metric priority based on model type
            if self.model_type == 'classifier':
                metric_priority = ['Logloss', 'AUC', 'Accuracy']
            else:
                # For ranker, try multiple NDCG variations that CatBoost might use
                metric_priority = [
                    'NDCG',              # Standard NDCG
                    'NDCG:top=5',        # NDCG with top parameter
                    'NDCG:type=Base',    # Base type NDCG
                    'NDCG:type=Exp',     # Exponential type NDCG
                    'PairLogitPairwise', # Alternative ranking metric
                    'YetiRank',          # Loss function as metric
                ]
            
            # Find the first available metric from priority list
            metric = None
            for m in metric_priority:
                if m in available_metrics:
                    metric = m
                    break
            
            # If no priority metric found, try to find any NDCG variant
            if metric is None:
                for m in available_metrics:
                    if 'NDCG' in m.upper():
                        metric = m
                        break
            
            # Last resort: use the first available metric
            if metric is None and available_metrics:
                metric = available_metrics[0]
            
            if metric:
                try:
                    train_metric = self.training_history['learn'][metric]
                    val_metric = self.training_history.get('validation', {}).get(metric, [])
                    
                    axes[0].plot(train_metric, label='Train', color='blue')
                    if val_metric:
                        axes[0].plot(val_metric, label='Validation', color='orange')
                    axes[0].set_xlabel('Iteration')
                    axes[0].set_ylabel(metric)
                    axes[0].set_title(f'Training History: {metric}')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    metric_plotted = True
                except Exception as e:
                    print(f"Warning: Could not plot metric {metric}: {e}")
        
        if not metric_plotted:
            axes[0].text(0.5, 0.5, 'No training metrics available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Training History (No Data)')
        
        # Plot 2: Feature importance (top 20)
        try:
            feat_importance = self.get_feature_importance().head(20)
            
            if len(feat_importance) > 0:
                axes[1].barh(range(len(feat_importance)), feat_importance['importance'])
                axes[1].set_yticks(range(len(feat_importance)))
                axes[1].set_yticklabels(feat_importance['feature'])
                axes[1].set_xlabel('Importance')
                axes[1].set_title('Top 20 Feature Importance')
                axes[1].grid(True, alpha=0.3, axis='x')
            else:
                axes[1].text(0.5, 0.5, 'No feature importance available',
                            ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Feature Importance (No Data)')
        except Exception as e:
            print(f"Warning: Could not get feature importance: {e}")
            axes[1].text(0.5, 0.5, f'Error: {str(e)[:50]}',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Feature Importance (Error)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()