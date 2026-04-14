"""
LASSO Feature Selection Script

This script applies LASSO (L1 regularization) to select the most important features
from acoustic, linguistic, and/or paralinguistic features based on configuration.
Selected feature indices are saved and can be loaded during model training.
"""

import os
import sys
import yaml
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import BasicLogger


class LassoFeatureSelector:
    """
    Applies LASSO feature selection to extract the most relevant features
    from acoustic, paralinguistic, and linguistic feature sets.
    """
    
    def __init__(self, config: Dict[str, Any], logger: BasicLogger):
        """
        Initialize the LASSO feature selector.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            logger (BasicLogger): Logger instance
        """
        self.config = config
        self.logger = logger
        self.scaler = StandardScaler()
        self.lasso_models = {}
        self.selected_features_indices = {}
        self.feature_names = {}
        
    def load_features_and_labels(self) -> Tuple[Dict[str, torch.Tensor], np.ndarray, List[str]]:
        """
        Load all features and labels from the dataset.
        
        Returns:
            Tuple containing:
                - features_dict: Dictionary with feature types as keys and tensors as values
                - labels: Array of labels
                - audio_ids: List of audio file identifiers
        """
        from binary_classification_optuna import (
            extract_features,
            get_data_to_split
        )
        
        extract_conf = self.config['model_extract_train_test']
        
        # Extract features if needed
        if not extract_conf['load_extracted_features']:
            self.logger.info("Extracting features...")
            labels_dict = extract_features(config=self.config)
        else:
            self.logger.info("Loading pre-extracted features...")
            # Load labels
            labels_dict = self._load_labels()
            
        # Get data to split
        features_id, _ = get_data_to_split(labels=labels_dict)
        
        return labels_dict, features_id
    
    def _load_labels(self) -> Dict:
        """
        Load labels from the condition file using the same approach as common_classification.
        
        Returns:
            Dictionary mapping audio files to labels
        """
        from src.common_classification import create_dataset_from_files, process_labels
        
        extract_conf = self.config['model_extract_train_test']
        
        # Create dataset from files (same way as in extract_features)
        dataset = create_dataset_from_files(model_conf=extract_conf)
        
        # Process labels (same way as in extract_features)
        labels_dict = process_labels(dataset=dataset)
        
        return labels_dict
    
    def extract_all_features(self, labels_dict: Dict, which_feature: str = 'aggregated') -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Extract all features (acoustic, paralinguistic, linguistic) for the dataset.
        
        Args:
            labels_dict (Dict): Dictionary mapping audio files to labels
            which_feature (str): Feature type ('aggregated', 'raw', 'frame')
            
        Returns:
            Tuple containing:
                - features: Combined feature tensor
                - labels: Label tensor
                - feature_type_ranges: List describing which columns belong to which feature type
        """
        from binary_classification_optuna import (
            get_features_and_labels
        )
        
        extract_conf = self.config['model_extract_train_test']
        feature_type = self.config['audioprocessor_data']['feature_type']
        
        keys_labels_list = list(labels_dict.keys())
        
        # Temporarily enable all feature types
        original_acoustic = extract_conf.get('use_acoustic_feat', False)
        original_paralin = extract_conf.get('use_paralinguistic_feat', False)
        original_linguistic = extract_conf.get('use_linguistic_feat', False)
        
        # Collect features for each type
        all_features = []
        all_labels = None
        feature_type_ranges = []
        current_idx = 0
        
        # Acoustic features
        if original_acoustic:
            self.logger.info("Extracting acoustic features...")
            extract_conf['use_acoustic_feat'] = True
            extract_conf['use_paralinguistic_feat'] = False
            extract_conf['use_linguistic_feat'] = False
            
            features, labels, _ = get_features_and_labels(
                labels_dict=labels_dict,
                keys_labels_list=keys_labels_list,
                which_feature=which_feature,
                feature_type=feature_type,
                model='RandomForest',
                model_conf=extract_conf,
                init_model=False
            )
            
            all_features.append(features)
            all_labels = labels
            feature_type_ranges.append(('acoustic', current_idx, current_idx + features.shape[1]))
            current_idx += features.shape[1]
            self.logger.info(f"Acoustic features shape: {features.shape}")
        
        # Paralinguistic features
        if original_paralin:
            self.logger.info("Extracting paralinguistic features...")
            extract_conf['use_acoustic_feat'] = False
            extract_conf['use_paralinguistic_feat'] = True
            extract_conf['use_linguistic_feat'] = False
            
            features, labels, _ = get_features_and_labels(
                labels_dict=labels_dict,
                keys_labels_list=keys_labels_list,
                which_feature=which_feature,
                feature_type=feature_type,
                model='RandomForest',
                model_conf=extract_conf,
                init_model=False
            )
            
            all_features.append(features)
            if all_labels is None:
                all_labels = labels
            feature_type_ranges.append(('paralinguistic', current_idx, current_idx + features.shape[1]))
            current_idx += features.shape[1]
            self.logger.info(f"Paralinguistic features shape: {features.shape}")
        
        # Linguistic features
        if original_linguistic:
            self.logger.info("Extracting linguistic features...")
            extract_conf['use_acoustic_feat'] = False
            extract_conf['use_paralinguistic_feat'] = False
            extract_conf['use_linguistic_feat'] = True
            
            features, labels, _ = get_features_and_labels(
                labels_dict=labels_dict,
                keys_labels_list=keys_labels_list,
                which_feature=which_feature,
                feature_type=feature_type,
                model='RandomForest',
                model_conf=extract_conf,
                init_model=False
            )
            
            all_features.append(features)
            if all_labels is None:
                all_labels = labels
            feature_type_ranges.append(('linguistic', current_idx, current_idx + features.shape[1]))
            current_idx += features.shape[1]
            self.logger.info(f"Linguistic features shape: {features.shape}")
        
        # Restore original settings
        extract_conf['use_acoustic_feat'] = original_acoustic
        extract_conf['use_paralinguistic_feat'] = original_paralin
        extract_conf['use_linguistic_feat'] = original_linguistic
        
        # Concatenate all features
        if len(all_features) > 0:
            combined_features = torch.cat(all_features, dim=1)
        else:
            raise ValueError("No features were extracted. Check configuration.")
        
        self.logger.info(f"Combined features shape: {combined_features.shape}")
        
        return combined_features, all_labels, feature_type_ranges
    
    def apply_lasso_selection(self, features: torch.Tensor, labels: torch.Tensor,
                             feature_type_ranges: List[Tuple[str, int, int]],
                             alphas: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Apply LASSO feature selection separately for each feature type.
        
        Args:
            features (torch.Tensor): Feature tensor
            labels (torch.Tensor): Label tensor
            feature_type_ranges (List[Tuple[str, int, int]]): Feature type ranges
            alphas (List[float]): List of alpha values for LassoCV
            
        Returns:
            Dictionary mapping feature types to selected feature indices
        """
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        # Convert to numpy
        X = features.numpy()
        y = labels.numpy()
        
        # Normalize features
        self.logger.info("Normalizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        selected_indices = {}
        
        for feat_type, start_idx, end_idx in feature_type_ranges:
            self.logger.info(f"\nApplying LASSO to {feat_type} features...")
            self.logger.info(f"Feature range: [{start_idx}:{end_idx}]")
            
            # Extract features for this type
            X_type = X_scaled[:, start_idx:end_idx]
            
            # Apply LassoCV with cross-validation
            lasso = LassoCV(
                alphas=alphas,
                cv=5,
                max_iter=10000,
                random_state=42,
                n_jobs=-1
            )
            
            lasso.fit(X_type, y)
            
            # Get selected features (non-zero coefficients)
            coefficients = np.abs(lasso.coef_)
            selected_mask = coefficients > 1e-5
            selected_feat_indices = np.where(selected_mask)[0]
            
            # Convert to global indices
            global_indices = selected_feat_indices + start_idx
            
            self.logger.info(f"Best alpha: {lasso.alpha_}")
            self.logger.info(f"Selected {len(selected_feat_indices)} out of {X_type.shape[1]} features")
            self.logger.info(f"Selection rate: {100 * len(selected_feat_indices) / X_type.shape[1]:.2f}%")
            
            # Store results
            selected_indices[feat_type] = global_indices
            self.lasso_models[feat_type] = lasso
        
        return selected_indices
    
    def save_selected_features(self, selected_indices: Dict[str, np.ndarray], 
                               output_path: str) -> None:
        """
        Save selected feature indices to a file.
        
        Args:
            selected_indices (Dict[str, np.ndarray]): Selected feature indices per type
            output_path (str): Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data to save
        data = {
            'selected_indices': selected_indices,
            'lasso_models': self.lasso_models,
            'scaler': self.scaler,
            'config': {
                'use_acoustic_feat': self.config['model_extract_train_test'].get('use_acoustic_feat', False),
                'use_paralinguistic_feat': self.config['model_extract_train_test'].get('use_paralinguistic_feat', False),
                'use_linguistic_feat': self.config['model_extract_train_test'].get('use_linguistic_feat', False),
                'feature_type': self.config['audioprocessor_data']['feature_type']
            }
        }
        
        # Save to pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Selected features saved to: {output_path}")
        
        # Also save a human-readable summary
        summary_path = output_path.replace('.pkl', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("LASSO Feature Selection Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total_original = 0
            total_selected = 0
            
            for feat_type, indices in selected_indices.items():
                f.write(f"{feat_type.upper()} Features:\n")
                f.write(f"  Selected: {len(indices)} features\n")
                f.write(f"  Indices: {indices.tolist()}\n\n")
                
                total_selected += len(indices)
            
            f.write(f"\nTotal selected features: {total_selected}\n")
        
        self.logger.info(f"Summary saved to: {summary_path}")
    
    @staticmethod
    def load_selected_features(filepath: str) -> Dict:
        """
        Load selected feature indices from a file.
        
        Args:
            filepath (str): Path to the saved feature selection file
            
        Returns:
            Dictionary containing selected indices and metadata
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data


def get_feature_combination_suffix(config: Dict[str, Any]) -> str:
    """
    Generate a descriptive suffix based on which feature types are enabled.
    Includes specific acoustic feature names (e.g., rasta, gfcc) when acoustic features are used.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    
    Returns:
        str: Suffix describing the feature combination (e.g., 'acoustic_rasta_gfcc_paralinguistic')
    """
    extract_conf = config['model_extract_train_test']
    audio_conf = config['audioprocessor_data']
    
    enabled_features = []
    
    # Add acoustic features with specific feature type names
    if extract_conf.get('use_acoustic_feat', False):
        # Get acoustic feature types and create short names
        feature_types = audio_conf.get('feature_type', [])
        if feature_types:
            # Extract short names from feature types
            # e.g., 'compare_2016_rasta' -> 'rasta', 'spafe_gfcc' -> 'gfcc'
            short_names = []
            for ft in feature_types:
                if 'compare_2016_' in ft:
                    short_names.append(ft.replace('compare_2016_', ''))
                elif 'spafe_' in ft:
                    short_names.append(ft.replace('spafe_', ''))
                else:
                    short_names.append(ft)
            enabled_features.append('acoustic_' + '_'.join(short_names))
        else:
            enabled_features.append('acoustic')
    
    if extract_conf.get('use_paralinguistic_feat', False):
        enabled_features.append('paralinguistic')
    if extract_conf.get('use_linguistic_feat', False):
        enabled_features.append('linguistic')
    
    if not enabled_features:
        return 'no_features'
    
    return '_'.join(enabled_features)


def run_lasso_feature_selection(config_path: str) -> None:
    """
    Main function to run LASSO feature selection.
    
    Args:
        config_path (str): Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    extract_conf = config['model_extract_train_test']
    
    # Setup logger
    log_file = extract_conf.get('log_file', 'lasso_selection.log').replace('.log', '_lasso.log')
    logger = BasicLogger(log_file).get_logger()
    
    logger.info("=" * 70)
    logger.info("Starting LASSO Feature Selection")
    logger.info("=" * 70)
    
    # Initialize selector
    selector = LassoFeatureSelector(config, logger)
    
    # Load features and labels
    logger.info("\nLoading dataset...")
    labels_dict, features_id = selector.load_features_and_labels()
    logger.info(f"Loaded {len(labels_dict)} samples")
    
    # Extract all features
    for which_feature in extract_conf.get('which_features', ['aggregated']):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing feature type: {which_feature}")
        logger.info(f"{'=' * 70}")
        
        features, labels, feature_type_ranges = selector.extract_all_features(
            labels_dict, 
            which_feature=which_feature
        )
        
        # Apply LASSO selection
        logger.info("\nApplying LASSO feature selection...")
        selected_indices = selector.apply_lasso_selection(
            features, 
            labels, 
            feature_type_ranges
        )
        
        # Save selected features with unique filename
        output_dir = extract_conf.get('path_extracted_features', './features/')
        feature_combo_suffix = get_feature_combination_suffix(config)
        output_file = os.path.join(
            output_dir, 
            f'lasso_selected_features_{which_feature}_{feature_combo_suffix}.pkl'
        )
        logger.info(f"Saving to: {os.path.basename(output_file)}")
        selector.save_selected_features(selected_indices, output_file)
    
    logger.info("\n" + "=" * 70)
    logger.info("LASSO Feature Selection Completed")
    logger.info("=" * 70)


if __name__ == '__main__':
    
    run_lasso_feature_selection('./binary_classification_conf.yaml')
