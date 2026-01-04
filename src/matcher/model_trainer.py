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
                od_wait=50,
                
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
                # Core parameters
                iterations=500,
                learning_rate=0.03,
                depth=6,
                
                # Loss function - OPTIMIZED FOR RANKING
                loss_function='YetiRank',
                
                # Evaluation metrics
                eval_metric='NDCG:top=5',
                custom_metric=['MRR:top=5', 'PrecisionAt:top=5', 'RecallAt:top=5'],
                
                # Regularization
                l2_leaf_reg=3.0,
                random_strength=1.0,
                bagging_temperature=1.0,
                
                # Overfitting prevention
                early_stopping_rounds=50,
                od_type='Iter',
                od_wait=50,
                
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
        X = features_df
        y = labels
        
        # Identify categorical features
        cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'bool':
                cat_features.append(col)
            # Also check if numeric column has few unique values (likely categorical)
            elif X[col].nunique() < 10 and col not in ['year_diff', 'citation_count']:
                cat_features.append(col)
        
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
        X = features_df
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
        
        # Identify categorical features
        cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'bool':
                cat_features.append(col)
            elif X[col].nunique() < 10 and col not in ['year_diff', 'citation_count']:
                cat_features.append(col)
        
        return Pool(
            data=X,
            label=y,
            group_id=group_ids,
            cat_features=cat_features
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
        print(f"Imbalance ratio: 1:{neg_count/pos_count:.1f}")
        
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
                            eval_metric='NDCG:top=5',
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
                        
                        score = model.get_best_score()['validation']['NDCG:top=5']
                    
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
        
        # Create (candidate_id, score) tuples
        candidate_scores = []
        for i, pair in enumerate(pairs):
            if pair['bibtex_key'] == bibtex_key:
                candidate_scores.append((
                    pair['candidate_id'],
                    scores[i],
                    pair  # Store pair for post-processing
                ))
        
        # Post-processing (if enabled)
        if post_process:
            candidate_scores = self._post_process(
                bibtex_key, 
                candidate_scores,
                pairs[0]['bibtex_data']  # Get BibTeX data
            )
        
        # Sort by score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k candidate IDs
        return [cand_id for cand_id, score, pair in candidate_scores[:k]]
    
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
            candidate_scores: List of (candidate_id, score, pair) tuples
            bibtex_data: BibTeX entry data
        
        Returns:
            Modified candidate_scores
        """
        processed = []
        
        for cand_id, score, pair in candidate_scores:
            new_score = score
            candidate_data = pair['candidate_data']
            
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
            
            processed.append((cand_id, new_score, pair))
        
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
        if isinstance(year1, str):
            match = re.search(r'\d{4}', year1)
            year1 = int(match.group()) if match else 0
        
        # Extract year from candidate
        year2_str = candidate_data.get('submission_date', '')
        if year2_str:
            year2 = int(year2_str[:4])
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
    
    def get_feature_importance(self):
        """
        Get feature importance
        
        Returns:
            DataFrame with feature names and importance scores
        """
        feature_names = self.model.feature_names_
        importance = self.model.get_feature_importance()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, path):
        """Save trained model"""
        self.model.save_model(path)
        print(f"Model saved to {path}")
        
        # Also save metadata
        metadata = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'feature_importance': self.get_feature_importance().to_dict()
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
        
        # Plot 1: Training vs validation loss
        if self.model_type == 'classifier':
            metric = 'Logloss'
        else:
            metric = 'NDCG:top=5'
        
        if 'learn' in self.training_history:
            train_metric = self.training_history['learn'][metric]
            val_metric = self.training_history['validation'][metric]
            
            axes[0].plot(train_metric, label='Train')
            axes[0].plot(val_metric, label='Validation')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel(metric)
            axes[0].set_title(f'Training History: {metric}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Feature importance (top 20)
        feat_importance = self.get_feature_importance().head(20)
        
        axes[1].barh(range(len(feat_importance)), feat_importance['importance'])
        axes[1].set_yticks(range(len(feat_importance)))
        axes[1].set_yticklabels(feat_importance['feature'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 20 Feature Importance')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()