"""
Main script to run ML pipeline
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from matcher import *
from utils import *
import config

def load_manual_labels(labels_path):
    """
    Load manual labels from JSON file
    
    Format:
    {
        "publication_id": {
            "bibtex_key": "arxiv_id",
            ...
        },
        ...
    }
    """
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Manual labels file not found at {labels_path}")
        return {}

def extract_all_features(pairs, feature_extractor, hierarchy_extractors):
    """
    Extract all features for a list of pairs
    
    Args:
        pairs: List of pair dicts
        feature_extractor: FeatureExtractor instance
        hierarchy_extractors: Dict of {pub_id: HierarchyFeatureExtractor}
    
    Returns:
        DataFrame with all features
    """
    all_features = []
    
    print("Extracting features...")
    for pair in tqdm(pairs, desc="Features"):
        # Traditional features
        features = feature_extractor.extract_features(pair)
        
        # Hierarchy features (if available)
        pub_id = pair['publication_id']
        if pub_id in hierarchy_extractors:
            hier_features = hierarchy_extractors[pub_id].extract_features(
                pair['bibtex_key']
            )
            features.update(hier_features)
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)

def get_labels_for_pairs(pairs, manual_labels, labeler):
    """
    Get labels for all pairs
    
    Args:
        pairs: List of pairs
        manual_labels: Dict of manual labels
        labeler: Labeler instance
    
    Returns:
        List of labels (0 or 1)
    """
    labels = []
    
    for pair in pairs:
        pub_id = pair['publication_id']
        bibtex_key = pair['bibtex_key']
        candidate_id = pair['candidate_id']
        
        # Check manual labels first
        if pub_id in manual_labels:
            if bibtex_key in manual_labels[pub_id]:
                true_id = manual_labels[pub_id][bibtex_key]
                label = 1 if candidate_id == true_id else 0
                labels.append(label)
                continue
        
        # Otherwise use automatic labeling
        label = labeler.automatic_label(pair)
        if label is None:
            label = 0  # Default to no match
        labels.append(label)
    
    return labels

def generate_predictions(test_pairs, trainer, feature_extractor, 
                        hierarchy_extractors, top_k=5):
    """
    Generate top-k predictions for each BibTeX entry in test set
    
    Args:
        test_pairs: Test pairs
        trainer: Trained ModelTrainer
        feature_extractor: FeatureExtractor instance
        hierarchy_extractors: Dict of hierarchy extractors
        top_k: Number of top candidates
    
    Returns:
        Dict: {bibtex_key: [top_k_candidate_ids]}
    """
    predictions = {}
    
    # Group pairs by BibTeX entry
    grouped = {}
    for pair in test_pairs:
        key = (pair['publication_id'], pair['bibtex_key'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pair)
    
    print("\nGenerating predictions...")
    for (pub_id, bibtex_key), pairs in tqdm(grouped.items(), desc="Predictions"):
        # Extract features for all candidates
        features = extract_all_features(pairs, feature_extractor, hierarchy_extractors)
        
        # Predict top-k
        top_k_ids = trainer.predict_top_k(
            bibtex_key=bibtex_key,
            pairs=pairs,
            features_df=features,
            k=top_k,
            post_process=config.ENABLE_POST_PROCESSING
        )
        
        predictions[f"{pub_id}:{bibtex_key}"] = top_k_ids
    
    return predictions

def get_ground_truth(pairs, manual_labels):
    """
    Extract ground truth from pairs and manual labels
    
    Returns:
        Dict: {bibtex_key: correct_arxiv_id}
    """
    ground_truth = {}
    
    for pair in pairs:
        pub_id = pair['publication_id']
        bibtex_key = pair['bibtex_key']
        
        key = f"{pub_id}:{bibtex_key}"
        
        if key not in ground_truth:
            if pub_id in manual_labels and bibtex_key in manual_labels[pub_id]:
                ground_truth[key] = manual_labels[pub_id][bibtex_key]
    
    return ground_truth

def save_predictions_by_publication(predictions, ground_truth, partition, data_dir):
    """
    Save predictions in pred.json format for each publication
    
    Args:
        predictions: Dict of predictions
        ground_truth: Dict of ground truth
        partition: 'train', 'valid', or 'test'
        data_dir: Root data directory
    """
    # Group by publication
    pub_predictions = {}
    pub_ground_truth = {}
    
    for key, pred_list in predictions.items():
        pub_id, bibtex_key = key.split(':', 1)
        
        if pub_id not in pub_predictions:
            pub_predictions[pub_id] = {}
            pub_ground_truth[pub_id] = {}
        
        pub_predictions[pub_id][bibtex_key] = pred_list
        if key in ground_truth:
            pub_ground_truth[pub_id][bibtex_key] = ground_truth[key]
    
    # Save for each publication
    for pub_id in pub_predictions.keys():
        pred_json = {
            "partition": partition,
            "groundtruth": pub_ground_truth[pub_id],
            "prediction": pub_predictions[pub_id]
        }
        
        output_path = Path(data_dir) / pub_id / 'pred.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(pred_json, f, indent=2)
        
        print(f"  Saved predictions for {pub_id}")

def main():
    parser = argparse.ArgumentParser(description='Reference Matching ML Pipeline')
    parser.add_argument('--data-dir', required=True, help='Data directory')
    parser.add_argument('--output-dir', default=config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--model-type', default=config.MODEL_TYPE, 
                       choices=['classifier', 'ranker'], help='Model type')
    parser.add_argument('--manual-labels', default='./manual_labels.json',
                       help='Path to manual labels JSON')
    parser.add_argument('--tune-hyperparams', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for training')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"REFERENCE MATCHING ML PIPELINE")
    print(f"{'='*70}\n")
    
    # Step 1: Load manual labels
    print("Step 1: Loading manual labels...")
    manual_labels = load_manual_labels(args.manual_labels)
    print(f"   Loaded labels for {len(manual_labels)} publications")
    
    # Step 2: Prepare data - create m×n pairs
    print("\nStep 2: Creating m×n pairs...")
    data_prep = DataPreparation()
    all_pairs = data_prep.create_all_pairs(args.data_dir)
    print(f"   Total pairs created: {len(all_pairs)}")
    
    # Statistics
    publications = set(p['publication_id'] for p in all_pairs)
    print(f"   Publications: {len(publications)}")
    
    # Step 3: Label data
    print("\nStep 3: Labeling data...")
    labeler = Labeler()
    
    labels = get_labels_for_pairs(all_pairs, manual_labels, labeler)
    
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"   Positive (matches): {pos_count}")
    print(f"   Negative (non-matches): {neg_count}")
    print(f"   Imbalance ratio: 1:{neg_count/pos_count:.1f}")
    
    # Step 4: Split data
    print("\nStep 4: Splitting data...")
    
    # Define which publications for each set
    all_pub_ids = list(publications)
    manual_pub_ids = list(manual_labels.keys())
    auto_pub_ids = [pid for pid in all_pub_ids if pid not in manual_pub_ids]
    
    # Simple split (you should customize this)
    test_pubs = (
        manual_pub_ids[:config.TEST_MANUAL_PUBS] + 
        auto_pub_ids[:config.TEST_AUTO_PUBS]
    )
    val_pubs = (
        manual_pub_ids[config.TEST_MANUAL_PUBS:config.TEST_MANUAL_PUBS+config.VAL_MANUAL_PUBS] +
        auto_pub_ids[config.TEST_AUTO_PUBS:config.TEST_AUTO_PUBS+config.VAL_AUTO_PUBS]
    )
    train_pubs = [pid for pid in all_pub_ids if pid not in test_pubs and pid not in val_pubs]
    
    # Split pairs
    train_pairs = [p for p in all_pairs if p['publication_id'] in train_pubs]
    val_pairs = [p for p in all_pairs if p['publication_id'] in val_pubs]
    test_pairs = [p for p in all_pairs if p['publication_id'] in test_pubs]
    
    train_labels = [labels[i] for i, p in enumerate(all_pairs) if p['publication_id'] in train_pubs]
    val_labels = [labels[i] for i, p in enumerate(all_pairs) if p['publication_id'] in val_pubs]
    test_labels = [labels[i] for i, p in enumerate(all_pairs) if p['publication_id'] in test_pubs]
    
    print(f"   Train: {len(train_pairs)} pairs from {len(train_pubs)} publications")
    print(f"   Val:   {len(val_pairs)} pairs from {len(val_pubs)} publications")
    print(f"   Test:  {len(test_pairs)} pairs from {len(test_pubs)} publications")
    
    # Step 5: Extract features
    print("\nStep 5: Extracting features...")
    feature_extractor = FeatureExtractor()
    
    # Load hierarchy extractors
    print("   Loading hierarchy data...")
    hierarchy_extractors = {}
    for pub_id in publications:
        hierarchy_path = Path(args.data_dir) / pub_id / 'hierarchy.json'
        if hierarchy_path.exists():
            hierarchy_extractors[pub_id] = HierarchyFeatureExtractor(str(hierarchy_path))
        else:
            print(f"   Warning: No hierarchy.json for {pub_id}")
    
    print("   Extracting training features...")
    X_train = extract_all_features(train_pairs, feature_extractor, hierarchy_extractors)
    
    print("   Extracting validation features...")
    X_val = extract_all_features(val_pairs, feature_extractor, hierarchy_extractors)
    
    print("   Extracting test features...")
    X_test = extract_all_features(test_pairs, feature_extractor, hierarchy_extractors)
    
    print(f"   Feature dimensions: {X_train.shape[1]} features")
    print(f"   Feature names: {list(X_train.columns[:10])}...")
    
    # Step 6: Train model
    print("\nStep 6: Training model...")
    trainer = ModelTrainer(model_type=args.model_type, use_gpu=args.use_gpu)
    
    # Hyperparameter tuning (optional)
    if args.tune_hyperparams or config.ENABLE_HYPERPARAMETER_TUNING:
        print("\n   Running hyperparameter tuning...")
        best_params = trainer.tune_hyperparameters(
            train_pairs, X_train, train_labels,
            val_pairs, X_val, val_labels
        )
        print(f"   Best parameters: {best_params}")
    
    # Train with best parameters
    trainer.train(
        train_pairs, X_train, train_labels,
        val_pairs, X_val, val_labels
    )
    
    # Save model
    model_path = Path(config.MODEL_DIR) / f'model_{args.model_type}.cbm'
    trainer.save_model(str(model_path))
    
    # Plot training history
    plot_path = Path(config.OUTPUT_DIR) / f'training_history_{args.model_type}.png'
    trainer.plot_training_history(save_path=str(plot_path))
    
    # Step 7: Feature importance analysis
    print("\nStep 7: Feature importance analysis...")
    feat_importance = trainer.get_feature_importance()
    print("\nTop 10 most important features:")
    print(feat_importance.head(10).to_string(index=False))
    
    # Save feature importance
    feat_importance.to_csv(
        Path(config.OUTPUT_DIR) / f'feature_importance_{args.model_type}.csv',
        index=False
    )
    
    # Step 8: Generate predictions
    print("\nStep 8: Generating predictions for test set...")
    test_predictions = generate_predictions(
        test_pairs, trainer, feature_extractor, 
        hierarchy_extractors, top_k=config.TOP_K
    )
    
    # Step 9: Evaluate
    print("\nStep 9: Evaluating on test set...")
    evaluator = Evaluator()
    
    test_ground_truth = get_ground_truth(test_pairs, manual_labels)
    
    # Calculate MRR
    mrr = evaluator.calculate_mrr(test_predictions, test_ground_truth)
    
    # Calculate additional metrics
    metrics = evaluator.calculate_metrics(test_predictions, test_ground_truth)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  MRR:    {mrr:.4f}")
    print(f"  Hit@1:  {metrics['hit@1']:.2%}")
    print(f"  Hit@3:  {metrics['hit@3']:.2%}")
    print(f"  Hit@5:  {metrics['hit@5']:.2%}")
    print(f"  Avg Rank: {metrics['avg_rank']:.2f}")
    print(f"{'='*70}\n")
    
    # Step 10: Save predictions
    print("Step 10: Saving predictions...")
    
    # Save test predictions
    save_predictions_by_publication(
        test_predictions, test_ground_truth, 'test', args.data_dir
    )
    
    # Also generate predictions for val and train (optional)
    print("\nGenerating predictions for validation set...")
    val_predictions = generate_predictions(
        val_pairs, trainer, feature_extractor,
        hierarchy_extractors, top_k=config.TOP_K
    )
    val_ground_truth = get_ground_truth(val_pairs, manual_labels)
    save_predictions_by_publication(
        val_predictions, val_ground_truth, 'valid', args.data_dir
    )
    
    print("\nGenerating predictions for training set...")
    train_predictions = generate_predictions(
        train_pairs, trainer, feature_extractor,
        hierarchy_extractors, top_k=config.TOP_K
    )
    train_ground_truth = get_ground_truth(train_pairs, manual_labels)
    save_predictions_by_publication(
        train_predictions, train_ground_truth, 'train', args.data_dir
    )
    
    # Step 11: Save evaluation report
    print("\nStep 11: Saving evaluation report...")
    report = {
        'model_type': args.model_type,
        'use_gpu': args.use_gpu,
        'hyperparameter_tuning': args.tune_hyperparams,
        'data_statistics': {
            'total_pairs': len(all_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'test_pairs': len(test_pairs),
            'positive_samples': pos_count,
            'negative_samples': neg_count,
            'imbalance_ratio': f"1:{neg_count/pos_count:.1f}",
        },
        'feature_statistics': {
            'num_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
        },
        'test_metrics': {
            'mrr': float(mrr),
            'hit@1': float(metrics['hit@1']),
            'hit@3': float(metrics['hit@3']),
            'hit@5': float(metrics['hit@5']),
            'avg_rank': float(metrics['avg_rank']),
        },
        'validation_metrics': {
            'mrr': float(evaluator.calculate_mrr(val_predictions, val_ground_truth)),
        },
        'top_features': feat_importance.head(20).to_dict('records'),
    }
    
    report_path = Path(config.OUTPUT_DIR) / f'evaluation_report_{args.model_type}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Report saved to {report_path}")
    
    print("\n✅ ML Pipeline completed successfully!\n")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()