"""
Main script to run ML pipeline
"""

import argparse
from matcher import *
from utils import *
import config
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR)
    parser.add_argument('--model-type', default='ranker', choices=['classifier', 'ranker'])
    args = parser.parse_args()
    
    # Step 1: Prepare data
    print("Step 1: Creating m×n pairs...")
    data_prep = DataPreparation()
    all_pairs = data_prep.create_all_pairs(args.data_dir)
    print(f"   Total pairs created: {len(all_pairs)}")
    
    # Step 2: Label data
    print("Step 2: Labeling data...")
    labeler = Labeler()
    
    # Load manual labels (assume pre-labeled)
    manual_pubs = load_manual_labels()  # Your pre-labeled data
    auto_pubs = labeler.label_automatically(all_pairs)
    
    labels = {**manual_pubs, **auto_pubs}
    print(f"   Total labeled pairs: {len(labels)}")
    
    # Step 3: Split data
    print("Step 3: Splitting data...")
    train_pairs, val_pairs, test_pairs = data_prep.split_data(
        all_pairs, 
        manual_pub_ids=['2504-13946', '2504-13947', ...],
        auto_pub_ids=['2504-13950', '2504-13951', ...]
    )
    
    # Step 4: Extract features
    print("Step 4: Extracting features...")
    feature_extractor = FeatureExtractor()
    hierarchy_extractor = HierarchyFeatureExtractor(args.data_dir)
    
    # Extract for all sets
    X_train = extract_all_features(train_pairs, feature_extractor, hierarchy_extractor)
    X_val = extract_all_features(val_pairs, feature_extractor, hierarchy_extractor)
    X_test = extract_all_features(test_pairs, feature_extractor, hierarchy_extractor)
    
    y_train = [labels.get((p['bibtex_key'], p['candidate_id']), 0) for p in train_pairs]
    y_val = [labels.get((p['bibtex_key'], p['candidate_id']), 0) for p in val_pairs]
    y_test = [labels.get((p['bibtex_key'], p['candidate_id']), 0) for p in test_pairs]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Step 5: Train model
    print("Step 5: Training model...")
    trainer = ModelTrainer(model_type=args.model_type)
    
    cat_features = ['first_author_match', 'year_match', 'cited_in_intro']
    trainer.train(X_train, y_train, X_val, y_val, cat_features=cat_features)
    
    # Save model
    trainer.save_model(f'{args.output_dir}/model.cbm')
    print("   Model saved!")
    
    # Step 6: Predict on test set
    print("Step 6: Generating predictions...")
    test_predictions = generate_predictions(test_pairs, trainer, top_k=5)
    
    # Step 7: Evaluate
    print("Step 7: Evaluating...")
    evaluator = Evaluator()
    
    test_ground_truth = labeler.get_ground_truth(test_pairs)
    mrr = evaluator.calculate_mrr(test_predictions, test_ground_truth)
    
    print(f"\n{'='*50}")
    print(f"   Test MRR: {mrr:.4f}")
    print(f"{'='*50}\n")
    
    # Step 8: Save predictions
    print("Step 8: Saving predictions...")
    for pub_id in get_test_publication_ids(test_pairs):
        pub_preds = filter_predictions_by_pub(test_predictions, pub_id)
        pub_gt = filter_ground_truth_by_pub(test_ground_truth, pub_id)
        
        evaluator.save_predictions(
            publication_id=pub_id,
            predictions=pub_preds,
            ground_truth=pub_gt,
            partition='test',
            output_path=f'{args.data_dir}/{pub_id}/pred.json'
        )
    
    print("✅ ML Pipeline complete!")

def extract_all_features(pairs, feature_extractor, hierarchy_extractor):
    """Helper to extract all features"""
    all_features = []
    
    for pair in pairs:
        # Traditional features
        features = feature_extractor.extract_features(pair)
        
        # Hierarchy features
        hier_features = hierarchy_extractor.extract_features(pair['bibtex_key'])
        
        # Combine
        features.update(hier_features)
        all_features.append(features)
    
    return pd.DataFrame(all_features)

def generate_predictions(test_pairs, trainer, top_k=5):
    """Generate top-k predictions for each BibTeX entry"""
    predictions = {}
    
    # Group pairs by BibTeX entry
    grouped = group_pairs_by_bibtex(test_pairs)
    
    for bibtex_key, candidates in grouped.items():
        # Extract features for all candidates
        candidate_features = []
        candidate_ids = []
        
        for pair in candidates:
            features = extract_features_for_pair(pair)
            candidate_features.append(features)
            candidate_ids.append(pair['candidate_id'])
        
        # Predict scores
        scores = trainer.predict(candidate_features)
        
        # Sort and get top-k
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_k_indices = sorted_indices[:top_k]
        
        predictions[bibtex_key] = [candidate_ids[i] for i in top_k_indices]
    
    return predictions

if __name__ == '__main__':
    main()