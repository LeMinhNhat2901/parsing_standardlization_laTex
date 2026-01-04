"""
Evaluate model performance using MRR and other metrics
COMPLETE VERSION
"""

import json
from pathlib import Path
import numpy as np

class Evaluator:
    def __init__(self):
        self.predictions = {}
        self.ground_truth = {}
    
    def calculate_mrr(self, predictions, ground_truth):
        """
        Calculate Mean Reciprocal Rank
        
        MRR = (1/|Q|) * Σ(1/rank_i)
        
        where rank_i is the position of the correct match in the predicted list
        
        Args:
            predictions: Dict {bibtex_key: [top5_candidates]}
            ground_truth: Dict {bibtex_key: correct_arxiv_id}
        
        Returns:
            MRR score (float between 0 and 1)
        """
        reciprocal_ranks = []
        
        for bibtex_key in ground_truth.keys():
            if bibtex_key not in predictions:
                # No prediction for this key
                reciprocal_ranks.append(0.0)
                continue
            
            pred_list = predictions[bibtex_key]
            true_id = ground_truth[bibtex_key]
            
            # Find rank of correct answer
            if true_id in pred_list:
                rank = pred_list.index(true_id) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            else:
                # Correct answer not in top-k
                reciprocal_ranks.append(0.0)
        
        if len(reciprocal_ranks) == 0:
            return 0.0
        
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return mrr
    
    def calculate_hit_at_k(self, predictions, ground_truth, k):
        """
        Calculate Hit@k metric
        
        Hit@k = proportion of queries where correct answer is in top-k
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
            k: Rank cutoff
        
        Returns:
            Hit@k score (float between 0 and 1)
        """
        hits = 0
        total = 0
        
        for bibtex_key, true_id in ground_truth.items():
            if bibtex_key not in predictions:
                total += 1
                continue
            
            pred_list = predictions[bibtex_key][:k]  # Only consider top-k
            
            if true_id in pred_list:
                hits += 1
            
            total += 1
        
        if total == 0:
            return 0.0
        
        return hits / total
    
    def calculate_average_rank(self, predictions, ground_truth):
        """
        Calculate average rank of correct answer
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
        
        Returns:
            Average rank (float)
        """
        ranks = []
        
        for bibtex_key, true_id in ground_truth.items():
            if bibtex_key not in predictions:
                # Assign worst rank (beyond list)
                ranks.append(len(predictions.get(bibtex_key, [])) + 1)
                continue
            
            pred_list = predictions[bibtex_key]
            
            if true_id in pred_list:
                rank = pred_list.index(true_id) + 1
                ranks.append(rank)
            else:
                # Not in list - assign rank beyond list
                ranks.append(len(pred_list) + 1)
        
        if len(ranks) == 0:
            return 0.0
        
        return sum(ranks) / len(ranks)
    
    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate all evaluation metrics
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
        
        Returns:
            Dict with all metrics
        """
        metrics = {
            'mrr': self.calculate_mrr(predictions, ground_truth),
            'hit@1': self.calculate_hit_at_k(predictions, ground_truth, 1),
            'hit@3': self.calculate_hit_at_k(predictions, ground_truth, 3),
            'hit@5': self.calculate_hit_at_k(predictions, ground_truth, 5),
            'avg_rank': self.calculate_average_rank(predictions, ground_truth),
        }
        
        return metrics
    
    def generate_report(self, predictions, ground_truth, output_path=None):
        """
        Generate detailed evaluation report
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
            output_path: Path to save report (optional)
        
        Returns:
            Dict with detailed metrics
        """
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, ground_truth)
        
        # Per-query analysis
        per_query_results = []
        
        for bibtex_key, true_id in ground_truth.items():
            pred_list = predictions.get(bibtex_key, [])
            
            if true_id in pred_list:
                rank = pred_list.index(true_id) + 1
                found = True
            else:
                rank = None
                found = False
            
            per_query_results.append({
                'bibtex_key': bibtex_key,
                'true_id': true_id,
                'predictions': pred_list,
                'rank': rank,
                'found': found,
                'reciprocal_rank': 1.0/rank if rank else 0.0
            })
        
        # Sort by rank (errors first)
        per_query_results.sort(key=lambda x: (x['rank'] is None, x['rank']))
        
        report = {
            'summary': metrics,
            'num_queries': len(ground_truth),
            'num_found': sum(1 for r in per_query_results if r['found']),
            'num_not_found': sum(1 for r in per_query_results if not r['found']),
            'per_query_results': per_query_results,
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Detailed report saved to {output_path}")
        
        return report
    
    def print_error_analysis(self, predictions, ground_truth, bibtex_data=None):
        """
        Print detailed error analysis for debugging
        
        Args:
            predictions: Dict of predictions
            ground_truth: Dict of ground truth
            bibtex_data: Optional dict with BibTeX entry details
        """
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70 + "\n")
        
        errors = []
        
        for bibtex_key, true_id in ground_truth.items():
            pred_list = predictions.get(bibtex_key, [])
            
            if true_id not in pred_list:
                errors.append({
                    'bibtex_key': bibtex_key,
                    'true_id': true_id,
                    'predictions': pred_list
                })
        
        if len(errors) == 0:
            print("No errors! All predictions are correct.")
            return
        
        print(f"Total errors: {len(errors)} / {len(ground_truth)}")
        print(f"Error rate: {len(errors)/len(ground_truth):.2%}\n")
        
        print("Sample errors (first 5):\n")
        for i, error in enumerate(errors[:5], 1):
            print(f"{i}. BibTeX key: {error['bibtex_key']}")
            print(f"   True ID: {error['true_id']}")
            print(f"   Predictions: {error['predictions']}")
            
            if bibtex_data and error['bibtex_key'] in bibtex_data:
                entry = bibtex_data[error['bibtex_key']]
                print(f"   BibTeX title: {entry.get('title', 'N/A')[:60]}...")
            
            print()
    
    def save_predictions(self, publication_id, predictions, ground_truth, 
                        partition, output_path):
        """
        Save predictions in pred.json format
        
        Format:
        {
            "partition": "test",
            "groundtruth": {
                "bibtex_entry_name_1": "arxiv_id_1",
                "bibtex_entry_name_2": "arxiv_id_2"
            },
            "prediction": {
                "bibtex_entry_name_1": ["cand_1", "cand_2", "cand_3", "cand_4", "cand_5"],
                "bibtex_entry_name_2": ["cand_1", "cand_2", "cand_3", "cand_4", "cand_5"]
            }
        }
        
        Args:
            publication_id: Publication ID
            predictions: Dict of predictions for this publication
            ground_truth: Dict of ground truth for this publication
            partition: 'train', 'valid', or 'test'
            output_path: Path to save pred.json
        """
        # Filter to only this publication's entries
        pub_predictions = {}
        pub_ground_truth = {}
        
        for key, pred_list in predictions.items():
            if key.startswith(publication_id + ':'):
                bibtex_key = key.split(':', 1)[1]
                pub_predictions[bibtex_key] = pred_list
        
        for key, true_id in ground_truth.items():
            if key.startswith(publication_id + ':'):
                bibtex_key = key.split(':', 1)[1]
                pub_ground_truth[bibtex_key] = true_id
        
        pred_json = {
            "partition": partition,
            "groundtruth": pub_ground_truth,
            "prediction": pub_predictions
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(pred_json, f, indent=2)
        
        print(f"Saved predictions for {publication_id} to {output_path}")