import os
import re
import h5py
import yaml
import torch
import random
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from src.logger import BasicLogger
from typing import Dict, Tuple, Any, List
from src.model.model_object_multiclass import ModelBuilder, DEFAULT_CONFIG
from src.common_classification import (configure_model_cores,
                                       initialize_metrics_dict_multiclass,
                                       extract_features,
                                       get_data_to_split,
                                       create_train_test_split,
                                       determine_model_name,
                                       determine_metrics_multiclass,
                                       save_metrics_to_csv_multiclass)


def load_and_test_model(config: dict,
                        labels_dict: dict, 
                        feature_type: list,
                        which_feature: str,
                        logger: BasicLogger, 
                        metrics_dict: Dict[str, float]) -> tuple[Dict[str, float], 
                                                                      pd.DataFrame]:   
    '''
    Load and test the model with the given features and labels.

    Args:
        config (dict):                   Configuration dictionary.
        labels_dict (dict):              Dictionary containing the labels.
        feature_type (list):             List of feature types to use.
        which_feature (str):             Which feature to use.
        logger (BasicLogger):            Logger for logging information.
        metrics_dict (Dict[str, float]): Dictionary to store evaluation metrics.

    Returns:
        tuple[Dict[str, float], pd.DataFrame]:
            - metrics_dict (Dict[str, float]): Updated metrics dictionary.
            - detailed_pd (pd.DataFrame):      DataFrame containing detailed predictions and correctness checks.
    '''
    
    # Evaluate all IDs
    features_id, _ = get_data_to_split(labels = labels_dict)
    features_id = features_id   
    
    # Initialize model
    model_builder = ModelBuilder(name = config['model'],
                                 save_name = config['model_name'],
                                 path_to_model = config['model_path'],
                                 app_logger = logger)  
    model_builder.build_model()
    
    # Load the model
    model_builder.load_model_from_a_serialized_object()
    
    # Test the model
    metrics_dict, detailed_pd, _ = test_model(model_builder = model_builder, 
                                              test_index = np.arange(features_id.shape[0]), 
                                              features_id =  features_id, 
                                              labels_dict = labels_dict,
                                              which_feature = which_feature,
                                              feature_type = feature_type,
                                              model = config['model'],
                                              model_conf = config,
                                              metrics_dict = metrics_dict)
    
    return metrics_dict, detailed_pd


def evaluate_model(config: Dict[str, Any]) -> None:
    '''
    Evaluate the model based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        None
    '''    
    
    # Check if the model exists
    if not os.path.isfile(config['model_evaluate']['model_path'] + '/' + config['model_evaluate']['model_name']):
        print('Model not found')
        return
    
    # Initialize metrics and logger
    eval_conf = config['model_evaluate']
    if eval_conf['num_cores'] == 'None': eval_conf['num_cores'] = None
    metrics_dict = initialize_metrics_dict_multiclass()
    logger = BasicLogger(eval_conf['log_file']).get_logger()
    
    for which_feature in eval_conf['which_features']:

        # Extract features and prepare data
        feature_types, labels_dict = extract_features(audioprocessor_data = config['audioprocessor_data'],
                                                      model_conf = eval_conf)
        
        # Load and test the model
        metrics, detailed_pd = load_and_test_model(config = eval_conf,
                                                   labels_dict = labels_dict,
                                                   feature_type = feature_types[0],
                                                   which_feature = which_feature,
                                                   logger = logger, 
                                                   metrics_dict = metrics_dict)
        
        # Save metrics to a CSV file
        _ = save_metrics_to_csv_multiclass(metrics_dict = metrics, 
                                           model = eval_conf['model'],
                                           model_name = eval_conf['model_name'],
                                           which_feature = which_feature,
                                           feature_type = feature_types[0],
                                           save_metrics_path = eval_conf['save_metrics_path'], 
                                           detailed_pd = detailed_pd, 
                                           detailed_metrics = eval_conf['detailed_metrics'])


def test_model(model_builder: ModelBuilder, 
               test_index: np.ndarray, 
               features_id: np.ndarray,
               labels_dict: dict,
               which_feature: str,
               feature_type: list,
               model: str, 
               model_conf: dict, 
               metrics_dict: Dict[str, float]) -> Tuple[Dict[str, float], 
                                                        pd.DataFrame,
                                                        Dict[str, List[float]]]: 

    '''
    Test the model with the given features and labels.

    Args:
        model_builder (ModelBuilder):    Trained model builder object.
        test_index (np.ndarray):         Test features index.
        features_id (np.ndarray):        Whole dataset features ID.
        labels_dict (dict):              Dictionary containing the labels.
        which_feature (str):             Which feature to use.
        feature_type (list):             List of feature types to use.
        model (str):                     Model type.
        model_conf: (dict):              Dictionary with the model configuration.
        metrics_dict (Dict[str, float]): Dictionary to store evaluation metrics.

    Returns:
        tuple[Dict[str, float], pd.DataFrame]:
            - metrics_dict (Dict[str, float]):    Updated metrics dictionary.
            - detailed_pd (pd.DataFrame):         DataFrame containing detailed predictions and correctness checks.
            - threshold_results (Dict[str, List[float]]): Dictionary containing metrics for all tested thresholds.
    '''    
    
    # Check if the model is in train mode
    if_mode_train = 'save_model_f1_threshold' in model_conf.keys()
    
    # Get labels and audio path as keys
    (labels_dict, 
     keys_labels_list,
     canonical_audio_order) = get_labels_dict(labels_dict = labels_dict, 
                                              id_index = test_index, 
                                              features_id = features_id,
                                              shuffle = False)
    
    # Get training features and labels
    test_features, _, audio_labels = get_features_and_labels(labels_dict = labels_dict,
                                                             keys_labels_list = keys_labels_list,
                                                             which_feature = which_feature,
                                                             feature_type = feature_type,
                                                             model = model,
                                                             model_conf = model_conf,
                                                             canonical_audio_order = canonical_audio_order,
                                                             init_model = False)
    
    # Apply LASSO feature selection if model was trained with it
    if model_conf.get('use_lasso_selection', False):
        model_conf_with_ft = {**model_conf, 'feature_type': feature_type}
        lasso_data = load_lasso_selected_features(model_conf_with_ft, which_feature)
        if lasso_data is not None:
            test_features = apply_lasso_feature_selection(test_features, lasso_data, model_conf)
       
    # Normalize features if needed
    test_features = normalize_features(features = test_features,
                                       model_mean_std_dict = model_builder.mean_std_model_dict)
    
    # Test the model and get metrics
    (test_predictions, 
     test_probabilities) = model_builder.evaluate_model(test_features)
    metrics_dict, detailed_pd = determine_metrics_multiclass(metrics_dict = metrics_dict, 
                                                               test_predictions = test_predictions,
                                                               test_probabilities = test_probabilities,
                                                               test_labels = audio_labels,
                                                               detailed_metrics = model_conf['detailed_metrics'])    
       
    # Save model
    if if_mode_train:
        if_f1_majority = metrics_dict['f1_score_majority_voting'][-1] >= model_conf['save_model_f1_threshold']
        if_f1_prob = metrics_dict['f1_score_average_prob'][-1] >= model_conf['save_model_f1_threshold']
        if if_f1_majority or if_f1_prob:
            model_builder.save_as_a_serialized_object(path_to_save = model_conf['model_folder'])
    
    return metrics_dict, detailed_pd


def normalize_features(features: torch.Tensor, 
                       model_mean_std_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    '''
    Normalize the feature tensor using the mean and standard deviation.

    Parameters:
        features (torch.Tensor):                        The feature tensor to be normalized.
        model_mean_std_dict (dict[str, torch.Tensor]): A dictionary containing the mean and standard deviation for each feature type.

    Returns:
        features (torch.Tensor):   The normalized feature tensor.
    '''
    if model_mean_std_dict:
        if features.shape[1] != model_mean_std_dict['mean'].shape[0]:
            features = pad_features(feature = features,
                                     max_size = [features.shape[0], 
                                                 model_mean_std_dict['mean'].shape[0]])
        features = (features - model_mean_std_dict['mean']) / model_mean_std_dict['std']
        features = replace_nan_with_zero(feature = features)
    return features
    
    
def load_lasso_selected_features(config: dict[str, Any], 
                                 which_feature: str) -> dict:
    '''
    Load LASSO selected feature indices from file.
    
    Args:
        config (dict[str, Any]): Configuration dictionary.
        which_feature (str):     Which feature type (e.g., 'aggregated').
    
    Returns:
        dict: Dictionary containing selected feature indices and metadata,
              or None if file doesn't exist or LASSO is not enabled.
    '''
    if not config.get('use_lasso_selection', False):
        return None
    
    # Build feature combination suffix to match the saved filename
    enabled_features = []
    
    # Add acoustic features with specific feature type names
    if config.get('use_acoustic_feat', False):
        # Get the feature types from config
        feature_types = config.get('feature_type', [])
        
        if feature_types:
            # Extract short names from feature types
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
    
    if config.get('use_paralinguistic_feat', False):
        enabled_features.append('paralinguistic')
    if config.get('use_linguistic_feat', False):
        enabled_features.append('linguistic')
    
    feature_combo_suffix = '_'.join(enabled_features) if enabled_features else 'no_features'
    
    lasso_file = os.path.join(
        config['path_extracted_features'],
        f'lasso_selected_features_{which_feature}_{feature_combo_suffix}.pkl'
    )
    
    if not os.path.exists(lasso_file):
        # Try legacy filename format (without feature combination suffix) for backwards compatibility
        legacy_lasso_file = os.path.join(
            config['path_extracted_features'],
            f'lasso_selected_features_{which_feature}.pkl'
        )
        if os.path.exists(legacy_lasso_file):
            print(f"Warning: Using legacy LASSO file format. Consider regenerating with the new naming convention.")
            lasso_file = legacy_lasso_file
        else:
            return None
    
    with open(lasso_file, 'rb') as f:
        lasso_data = pickle.load(f)
    
    return lasso_data


def apply_lasso_feature_selection(features: torch.Tensor,
                                  lasso_data: dict,
                                  config: dict[str, Any]) -> torch.Tensor:
    '''
    Apply LASSO feature selection to the feature tensor.
    
    Args:
        features (torch.Tensor):  Feature tensor of shape (n_samples, n_features).
        lasso_data (dict):        Dictionary containing selected feature indices.
        config (dict[str, Any]):  Configuration dictionary.
    
    Returns:
        torch.Tensor: Feature tensor with only selected features.
    '''
    if lasso_data is None:
        return features
    
    # Get selected indices based on which feature types are enabled
    selected_indices = []
    
    if config.get('use_acoustic_feat', False) and 'acoustic' in lasso_data['selected_indices']:
        selected_indices.extend(lasso_data['selected_indices']['acoustic'].tolist())
    
    if config.get('use_paralinguistic_feat', False) and 'paralinguistic' in lasso_data['selected_indices']:
        selected_indices.extend(lasso_data['selected_indices']['paralinguistic'].tolist())
    
    if config.get('use_linguistic_feat', False) and 'linguistic' in lasso_data['selected_indices']:
        selected_indices.extend(lasso_data['selected_indices']['linguistic'].tolist())
    
    if len(selected_indices) == 0:
        return features
    
    # Sort indices to maintain order
    selected_indices = sorted(set(selected_indices))
    
    # Select only the chosen features
    selected_features = features[:, selected_indices]
    
    return selected_features


def get_mean_std_dict(features: torch.Tensor,
                      config: dict[str, Any]) -> dict[str, torch.Tensor]:
    '''
    Determine the mean and standard deviation for the dataset.

    Args:
        features (torch.Tensor): A tensor containing the features for which the mean and std are to be calculated.
        config (Dict[str, Any]): Configuration dictionary.
        

    Returns:
        Dict[str, torch.Tensor]: A dictionary with the mean and standard deviation for the dataset.
    '''
    model_mean_std_dict = {}
    if config['normalize']:
        model_mean_std_dict['mean'] = features.mean(dim=0) 
        model_mean_std_dict['std'] = features.std(dim=0)   
       
    return model_mean_std_dict
    
    
def get_linguistic_features_hdf5(audio_file: str,
                                 hdf5_path: str) -> torch.Tensor:
    '''
    Retrieves the linguistic features for a given audio file from an HDF5 file.

    Args:
        audio_file (str):   Path to the audio file whose features are to be retrieved.
        hdf5_path (str):    Path to the HDF5 file containing the linguistic features.

    Returns:
        torch.Tensor:       A 1D tensor of linguistic feature values for the given audio file.

    Raises:
        FileNotFoundError:  If the HDF5 file does not exist.
        KeyError:           If the 'linguistic_features' key is not found in the HDF5 file.
        IndexError:         If the filename is not found in the HDF5 store.
    '''
    target_filename = os.path.basename(audio_file)
    try:
        transcript_series = pd.read_hdf(hdf5_path,
                                        key='linguistic_features',
                                        where=f'filename == "{target_filename}"').iloc[0]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error accessing HDF5 store {hdf5_path} or key: {e}")
        raise 
    except IndexError:
        print(f"Error: Filename '{target_filename}' not found in linguistic_features HDF5 store.")
        raise

    if 'filename' in transcript_series.index:
        feature_values = transcript_series.drop('filename').to_numpy()
    else:
        feature_values = transcript_series.to_numpy()

    feature_values = pd.to_numeric(feature_values, errors='coerce')
    feature = torch.tensor(feature_values, dtype=torch.float32)
    return feature

  
def get_paralinguistic_features_hdf5(audio_file: str,
                                     hdf5_path: str) -> torch.Tensor:
    '''
    Retrieves the paralinguistic features for a given audio file from an HDF5 file.

    Args:
        audio_file (str):   Path to the audio file whose features are to be retrieved.
        hdf5_path (str):    Path to the HDF5 file containing the paralinguistic features.

    Returns:
        torch.Tensor:       A 1D tensor of paralinguistic feature values for the given audio file.

    Raises:
        FileNotFoundError:  If the HDF5 file does not exist.
        KeyError:           If the 'paralinguistic' key is not found in the HDF5 file.
        IndexError:         If the filename is not found in the HDF5 store.
    '''
    target_filename = os.path.basename(audio_file)
    try:
        transcript_series = pd.read_hdf(hdf5_path,
                                        key='paralinguistic_features',
                                        where=f'filename == "{target_filename}"').iloc[0]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error accessing HDF5 store {hdf5_path} or key: {e}")
        raise
    except IndexError:
        print(f"Error: Filename '{target_filename}' not found in paralinguistic_features HDF5 store.")
        raise

    if 'filename' in transcript_series.index:
        feature_values = transcript_series.drop('filename').to_numpy()
    else:
        feature_values = transcript_series.to_numpy()

    feature_values = pd.to_numeric(feature_values, errors='coerce')
    feature = torch.tensor(feature_values, dtype=torch.float32)
    return feature


def get_paraling_ling_features(audio_file: str,
                               config: dict[str, Any],
                               max_size: list) -> torch.Tensor:
    '''
    Returns the 'linguistic' and 'paralinguistic' features for the given audio file.

    Args:
        audio_file (str):        The audio file name.
        config (Dict[str, Any]): Configuration dictionary.
        max_size (list):         A list containing the maximum number of frames and features.

    Returns:
        feature (torch.Tensor):  The linguistic and paralinguistic features for the given audio file.
    '''
    
    # Get the paralinguistic features
    if config['use_paralinguistic_feat']:
        paralinguistic_feature = get_paralinguistic_features_hdf5(audio_file = audio_file,
                                                                  hdf5_path = config['path_extracted_features'] + 'paralinguistic_features.h5')
    else:
        paralinguistic_feature = torch.tensor([])
        
    # Get the linguistic features
    if config['use_linguistic_feat']:
        linguistic_feature = get_linguistic_features_hdf5(audio_file = audio_file,
                                                          hdf5_path = config['path_extracted_features'] + 'linguistic_features.h5')
    else:
        linguistic_feature = torch.tensor([])
                    
    # Combine the linguistic and paralinguistic features
    feature = torch.cat((paralinguistic_feature, linguistic_feature), dim = 0)
    feature = replace_nan_with_zero(feature = feature)
    if max_size:
        feature = pad_features(feature = feature, 
                               max_size = [1, max_size])
    return feature
    

def process_paraling_ling_features(config: dict[str, Any],
                                   which_feature: str,
                                   keys_labels_list: list,
                                   labels_dict: dict,
                                   features: list,
                                   labels: list,
                                   audio_labels: list,
                                   features_dict: dict,
                                   if_frame_acoustic: bool,
                                   canonical_audio_order: list = None) -> tuple[list, 
                                                                                list, 
                                                                                list, 
                                                                                dict]:
    '''
    Process paralinguistic and linguistic features for a list of audio files.

    Args:
        config (dict[str, Any]):        Configuration dictionary.
        which_feature (str):            Which feature to use.
        keys_labels_list (list):        List of keys in the labels dictionary.
        labels_dict (dict):             Dictionary containing the labels.
        features (list):                List of feature tensors.
        labels (list):                  List of label tensors.   
        audio_labels (list):            List of audio label tuples.
        features_dict (dict):           Dictionary mapping audio labels to feature tensors.
        if_frame_acoustic (bool):       Whether to use frame-level acoustic features.
        canonical_audio_order (list):   List of canonical audio identifiers for padding missing audios.
        
    Returns:
        tuple[list, list, list, dict]: A tuple containing features, labels, audio_labels, and features_dict.
            - features (list):         List of feature tensors.
            - labels (list):           List of label tensors.   
            - audio_labels (list):     List of audio label tuples.
            - features_dict (dict):    Dictionary mapping audio labels to feature tensors.
    '''
    max_size_aux = features[0].shape[0] if config['use_acoustic_feat'] and 'aggregated' not in which_feature else None
    
    # Group audios by participant if using aggregated features with canonical order
    if which_feature == 'aggregated' and canonical_audio_order is not None:
        (features, 
        labels, 
        audio_labels, 
        features_dict) = process_paraling_ling_features_with_padding(keys_labels_list = keys_labels_list,
                                                                     labels_dict = labels_dict,
                                                                     config = config,
                                                                     max_size_aux = max_size_aux,
                                                                     if_frame_acoustic = if_frame_acoustic,
                                                                     features = features,
                                                                     labels = labels,
                                                                     audio_labels = audio_labels,
                                                                     features_dict = features_dict,
                                                                     canonical_audio_order = canonical_audio_order)
    else:
        for audio_file in keys_labels_list:
            feature = get_paraling_ling_features(audio_file = audio_file, 
                                                 config = config, 
                                                 max_size = max_size_aux)
            label = labels_dict[audio_file]
            audio_label = get_audio_labels(audio_file = audio_file,
                                           label = label,
                                           if_frame_acoustic = if_frame_acoustic)
        
            # Save the feature and label
            if which_feature == 'aggregated':
                if audio_label not in features_dict:
                    features_dict[audio_label] = feature
                    labels.append(label)
                    audio_labels.append(audio_label)
                else:
                    features_dict[audio_label] = torch.concat((features_dict[audio_label], feature), dim=0)
            else:
                features.append(feature)
                labels.append(label)
                audio_labels.append(audio_label)
    return features, labels, audio_labels, features_dict

def process_paraling_ling_features_with_padding(keys_labels_list: list,
                                                labels_dict: dict,
                                                config: dict[str, Any],
                                                max_size_aux: int,
                                                if_frame_acoustic: bool,
                                                features: list,
                                                labels: list,
                                                audio_labels: list,
                                                features_dict: dict,
                                                canonical_audio_order: list) -> tuple[list, list, list, dict]:
    '''
    Process paralinguistic and linguistic features with padding for missing audios.

    Args:
        keys_labels_list (list):        List of keys in the labels dictionary.
        labels_dict (dict):             Dictionary containing the labels.
        config (dict[str, Any]):        Configuration dictionary.
        max_size_aux (int):             The maximum size of the dataset.
        if_frame_acoustic (bool):       Whether to use frame-level acoustic features.
        features (list):                List of feature tensors.
        labels (list):                  List of label tensors.   
        audio_labels (list):            List of audio label tuples.
        features_dict (dict):           Dictionary mapping audio labels to feature tensors.
        canonical_audio_order (list):   List of canonical audio identifiers for padding missing audios.
        
    Returns:
        tuple[list, list, list, dict]: A tuple containing features, labels, audio_labels, and features_dict.
            - features (list):         List of feature tensors.
            - labels (list):           List of label tensors.   
            - audio_labels (list):     List of audio label tuples.
            - features_dict (dict):    Dictionary mapping audio labels to feature tensors.
    '''
    # Group audios by participant
    participant_audios = {}
    for audio_file in keys_labels_list:
        file_key, _ = extract_file_key_and_base_file_path(audio_file)
        if file_key not in participant_audios:
            participant_audios[file_key] = []
        participant_audios[file_key].append(audio_file)
    
    # Get feature size from first audio
    first_audio = keys_labels_list[0]
    first_feature = get_paraling_ling_features(audio_file = first_audio, 
                                               config = config, 
                                               max_size = max_size_aux)
    feature_size = first_feature.shape[0]
    
    # Process each participant
    for participant_id in participant_audios:
        audios_for_participant = participant_audios[participant_id]
        
        # Map existing audios by their audio number
        audio_map = {}
        for audio_file in audios_for_participant:
            audio_num = audio_file[-6:-4]
            audio_map[audio_num] = audio_file
        
        # Build feature tensor with padding for missing audios
        participant_features = []
        for audio_num in canonical_audio_order:
            if audio_num in audio_map:
                audio_file = audio_map[audio_num]
                feature = get_paraling_ling_features(audio_file = audio_file, 
                                                     config = config, 
                                                     max_size = max_size_aux)
            else:
                # Pad with zeros for missing audio
                feature = torch.zeros(feature_size, dtype=torch.float32)
            participant_features.append(feature)
        
        # Concatenate all features for this participant
        combined_feature = torch.concat(participant_features, dim=0)
        
        # Get label and audio_label from first audio of participant
        first_audio_file = audios_for_participant[0]
        label = labels_dict[first_audio_file]
        audio_label = get_audio_labels(audio_file = first_audio_file,
                                       label = label,
                                       if_frame_acoustic = if_frame_acoustic)
        
        # Save to features_dict
        if audio_label not in features_dict:
            features_dict[audio_label] = combined_feature
            labels.append(label)
            audio_labels.append(audio_label)
        else:
            features_dict[audio_label] = torch.concat((features_dict[audio_label], combined_feature), dim=0)
    
    return features, labels, audio_labels, features_dict


def get_ssl_features_hdf5(audio_file: str,
                          hdf5_path: str) -> torch.Tensor:
    '''
    Retrieves the paralinguistic features for a given audio file from an HDF5 file.

    Args:
        audio_file (str):   Path to the audio file whose features are to be retrieved.
        hdf5_path (str):    Path to the HDF5 file containing the paralinguistic features.

    Returns:
        torch.Tensor:       A 1D tensor of paralinguistic feature values for the given audio file.

    Raises:
        FileNotFoundError:  If the HDF5 file does not exist.
        KeyError:           If the 'paralinguistic' key is not found in the HDF5 file.
        IndexError:         If the filename is not found in the HDF5 store.
    '''
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        feature = hdf5_file[audio_file]['raw'][()]
        feature = check_if_tensor(feature)

    return feature


def get_ssl_features(audio_file: str,
                     config: dict[str, Any],
                     max_size: list) -> torch.Tensor:
    '''
    Returns the 'linguistic' and 'paralinguistic' features for the given audio file.

    Args:
        audio_file (str):        The audio file name.
        config (Dict[str, Any]): Configuration dictionary.
        max_size (list):         A list containing the maximum number of frames and features.

    Returns:
        feature (torch.Tensor):  The linguistic and paralinguistic features for the given audio file.
    '''
    
    # Get the hubert features
    if config['use_ssl_hubert']:
        ssl_hubert_feature = get_ssl_features_hdf5(audio_file = audio_file,
                                                   hdf5_path = config['path_extracted_features'] + 'ssl_hubert_pre_classifier.h5')
    else:
        ssl_hubert_feature = torch.tensor([])
        
    # Get the wavlm features
    if config['use_ssl_wavlm']:
        ssl_wavlm_feature = get_ssl_features_hdf5(audio_file = audio_file,
                                                  hdf5_path = config['path_extracted_features'] + 'ssl_wavlm_pre_classifier.h5')
    else:
        ssl_wavlm_feature = torch.tensor([])
    
    # Get the wav2vec features
    if config['use_ssl_wav2vec']:
        ssl_wav2vec_feature = get_ssl_features_hdf5(audio_file = audio_file,
                                                  hdf5_path = config['path_extracted_features'] + 'ssl_xlsr_300m_utterance.h5')
    else:
        ssl_wav2vec_feature = torch.tensor([])
                        
    # Combine the linguistic and paralinguistic features
    feature = torch.cat((ssl_hubert_feature, ssl_wavlm_feature, ssl_wav2vec_feature), dim = 0)
    feature = replace_nan_with_zero(feature = feature)
    return feature



def process_ssl_features(config: dict[str, Any],
                         which_feature: str,
                         keys_labels_list: list,
                         labels_dict: dict,
                         features: list,
                         labels: list,
                         audio_labels: list,
                         features_dict: dict,
                         if_frame_acoustic: bool,
                         canonical_audio_order: list = None) -> tuple[list, list, list, dict]:
    '''
    Process SSL features for a list of audio files.

    Args:
        config (dict[str, Any]):    Configuration dictionary.
        which_feature (str):        Which feature to use.
        keys_labels_list (list):    List of keys in the labels dictionary.
        labels_dict (dict):         Dictionary containing the labels.
        features (list):            List of feature tensors.
        labels (list):              List of label tensors.   
        audio_labels (list):        List of audio label tuples.
        features_dict (dict):       Dictionary mapping audio labels to feature tensors.
        if_frame_acoustic (bool):   Whether to use frame-level acoustic features.
        
    Returns:
        tuple[list, list, list, dict]: A tuple containing features, labels, audio_labels, and features_dict.
            - features (list):         List of feature tensors.
            - labels (list):           List of label tensors.   
            - audio_labels (list):     List of audio label tuples.
            - features_dict (dict):    Dictionary mapping audio labels to feature tensors.
    '''
    
    
    
    if_ssl = config['use_ssl_hubert'] or config['use_ssl_wavlm'] or config['use_ssl_wav2vec']
    max_size_aux = features[0].shape[0] if if_ssl and 'aggregated' not in which_feature else None

    if which_feature == 'aggregated' and canonical_audio_order is not None:
        (features, 
        labels, 
        audio_labels, 
        features_dict) = process_ssl_features_with_padding(
                                                            keys_labels_list = keys_labels_list,
                                                            labels_dict = labels_dict,
                                                            config = config,
                                                            max_size_aux = features[0].shape[0] if if_ssl and 'aggregated' not in which_feature else None,
                                                            if_frame_acoustic = if_frame_acoustic,
                                                            features = features,
                                                            labels = labels,
                                                            audio_labels = audio_labels,
                                                            features_dict = features_dict,
                                                            canonical_audio_order = canonical_audio_order)
    else:
        for audio_file in keys_labels_list:
            feature = get_ssl_features(audio_file = audio_file, 
                                    config = config, 
                                    max_size = max_size_aux)
            label = labels_dict[audio_file]
            audio_label = get_audio_labels(audio_file = audio_file,
                                        label = label,
                                        if_frame_acoustic = if_frame_acoustic)
        
            # Save the feature and label
            if which_feature == 'aggregated':
                if audio_label not in features_dict:
                    features_dict[audio_label] = feature
                    labels.append(label)
                    audio_labels.append(audio_label)
                else:
                    features_dict[audio_label] = torch.concat((features_dict[audio_label], feature), dim=0)
            else:
                features.append(feature)
                labels.append(label)
                audio_labels.append(audio_label)
    return features, labels, audio_labels, features_dict

def process_ssl_features_with_padding(keys_labels_list: list,
                                                labels_dict: dict,
                                                config: dict[str, Any],
                                                max_size_aux: int,
                                                if_frame_acoustic: bool,
                                                features: list,
                                                labels: list,
                                                audio_labels: list,
                                                features_dict: dict,
                                                canonical_audio_order: list) -> tuple[list, list, list, dict]:
    '''
    Process SSL features with padding for missing audios.

    Args:
        keys_labels_list (list):        List of keys in the labels dictionary.
        labels_dict (dict):             Dictionary containing the labels.
        config (dict[str, Any]):        Configuration dictionary.
        max_size_aux (int):             The maximum size of the dataset.
        if_frame_acoustic (bool):       Whether to use frame-level acoustic features.
        features (list):                List of feature tensors.
        labels (list):                  List of label tensors.   
        audio_labels (list):            List of audio label tuples.
        features_dict (dict):           Dictionary mapping audio labels to feature tensors.
        canonical_audio_order (list):   List of canonical audio identifiers for padding missing audios.
        
    Returns:
        tuple[list, list, list, dict]: A tuple containing features, labels, audio_labels, and features_dict.
            - features (list):         List of feature tensors.
            - labels (list):           List of label tensors.   
            - audio_labels (list):     List of audio label tuples.
            - features_dict (dict):    Dictionary mapping audio labels to feature tensors.
    '''
    # Group audios by participant
    participant_audios = {}
    for audio_file in keys_labels_list:
        file_key, _ = extract_file_key_and_base_file_path(audio_file)
        if file_key not in participant_audios:
            participant_audios[file_key] = []
        participant_audios[file_key].append(audio_file)
    
    # Get feature size from first audio
    first_audio = keys_labels_list[0]
    first_feature = get_ssl_features(audio_file = first_audio, 
                                               config = config, 
                                               max_size = max_size_aux)
    
    feature_size = first_feature.shape[0]

    # Process each participant
    for participant_id in participant_audios:
        audios_for_participant = participant_audios[participant_id]
        
        # Map existing audios by their audio number
        audio_map = {}
        for audio_file in audios_for_participant:
            audio_num = audio_file[-6:-4]
            audio_map[audio_num] = audio_file
        
        # Build feature tensor with padding for missing audios
        participant_features = []
        for audio_num in canonical_audio_order:
            if audio_num in audio_map:
                audio_file = audio_map[audio_num]
                feature = get_ssl_features(audio_file = audio_file, 
                                                     config = config, 
                                                     max_size = max_size_aux)
                
            else:
                # Pad with zeros for missing audio
                feature = torch.zeros(feature_size, dtype=torch.float32)
            participant_features.append(feature)
        
        # Concatenate all features for this participant
        combined_feature = torch.concat(participant_features, dim=0)
        
        # Get label and audio_label from first audio of participant
        first_audio_file = audios_for_participant[0]
        label = labels_dict[first_audio_file]
        audio_label = get_audio_labels(audio_file = first_audio_file,
                                       label = label,
                                       if_frame_acoustic = if_frame_acoustic)
        
        # Save to features_dict
        if audio_label not in features_dict:
            features_dict[audio_label] = combined_feature
            labels.append(label)
            audio_labels.append(audio_label)
        else:
            features_dict[audio_label] = torch.concat((features_dict[audio_label], combined_feature), dim=0)
    
    return features, labels, audio_labels, features_dict

        
def extract_file_key_and_base_file_path(audio_file: str) -> Tuple[str,
                                                                  str]:
    '''
    Extract the file key from an audio file path using a regex pattern and return the file key and base file path.

    Args:
        audio_file (str):           Path to the audio file.

    Returns:
        Tuple[str, str]: A tuple containing:
            - file_key (str):       The extracted file key.
            - base_file_path (str): The base file path.
    '''
    base_file_path = os.path.basename(audio_file)
    file_key_regex = r'^[^\-/_.,\s]+'
    match = re.match(file_key_regex, base_file_path)
    file_key = match.group(0)
    return file_key, base_file_path


def get_audio_labels(audio_file: str,
                     label: float,
                     if_frame_acoustic: bool) -> tuple:
    '''
    Generate a tuple of labels for a given audio file.

    Args:
        audio_file (str):         The path to the audio file.
        label (float):            The label associated with the audio file.
        if_frame_acoustic (bool): Flag indicating if the feature type is frame and acoustic.

    Returns:
        tuple: A tuple of labels including the provided label and additional metadata based on the dataset and feature type.
    '''
    file_key, base_file_path = extract_file_key_and_base_file_path(audio_file)
    if if_frame_acoustic:
        target_labels = (label, file_key, base_file_path)
    else:
        target_labels = (label, file_key)
    return target_labels
    
    
def replace_nan_with_zero(feature: torch.Tensor) -> torch.Tensor:
    '''
    Replace NaN and infinity values in the tensor with zeros.

    Parameters:
        feature (torch.Tensor): The input tensor.

    Returns:
        feature (torch.Tensor): The tensor with NaN and infinity values replaced by zeros.
    '''
    if torch.isnan(feature).any() or torch.isinf(feature).any():
        feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    return feature


def pad_features(feature: torch.Tensor, 
                 max_size: list = None,
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''
    Pads feature vectors with zeros to make them all the same shape.

    Args:
        feature (torch.Tensor):  Feature tensors.
        dtype (torch.dtype):     Data type for the padded features.
        max_size (list):         Maximum size for the feature vectors.

    Returns:
        torch.Tensor:            Tensor of padded feature vectors.
    '''
    pad_width = []
        
    # Calculate the padding size for each dimension
    for cur_dim, max_dim in zip(feature.shape[::-1], max_size[::-1]):
        pad_width.extend([0, max_dim - cur_dim])
    
    # Apply padding
    padded_feature = torch.nn.functional.pad(feature, pad_width, mode='constant', value=0).to(dtype)
    
    return padded_feature


def check_if_tensor(feature: np.array) -> torch.Tensor:
    """
    Ensure the input feature is a PyTorch tensor.

    Args:
        feature (np.array):     The input feature which may or may not be a tensor.

    Returns:
        feature (torch.Tensor): The input feature converted to a tensor if it was not already a tensor.
    """
    if not isinstance(feature, torch.Tensor): 
        feature = torch.tensor(feature)
    return feature


def retrieve_and_process_features(hdf5_file_path: str, 
                                  audio_file: str,
                                  which_feature: str,
                                  max_size: list,
                                  max_size_aux: int) -> torch.Tensor:
    '''
    Retrieve and process acoustic features from an HDF5 file based on the configuration.

    Parameters:
        hdf5_file_path (str): The path to the HDF5 file.
        audio_file (str):     The path to the audio file.
        which_feature (str):  Which feature to use.
        max_size (list):      A list containing the maximum number of frames and features.
        max_size_aux (int):   The maximum size of the dataset.
        
    Returns:
        feature (torch.Tensor): The feature tensor.
    '''
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        if which_feature == 'frame':
            feature = hdf5_file[audio_file]['raw'][()]
            feature = check_if_tensor(feature)
            
        # Retrieve and concatenate the aggregated and raw features
        elif which_feature == 'aggre_raw' or which_feature == 'aggre_frame':
            feature_aggregated = hdf5_file[audio_file]['aggregated'][()]
            feature_aggregated = check_if_tensor(feature_aggregated)
            feature_raw = hdf5_file[audio_file]['raw'][()]
            feature_raw = check_if_tensor(feature_raw)
            feature = torch.cat((feature_aggregated, feature_raw), dim=0)
            
        # Retrieve the specified feature
        else:
            feature = hdf5_file[audio_file][which_feature][()]
            feature = check_if_tensor(feature)
            
        # Pad features to ensure they have the same shape
        feature = replace_nan_with_zero(feature = feature)
        if 'frame' in which_feature:
            feature = pad_features(feature = feature,
                                   max_size = [max_size_aux, max_size[1]])
        elif 'aggregated' not in which_feature:
            feature = pad_features(feature = feature,
                                   max_size = max_size)
            
    # Reshape features to 1D in order to have #samples x #features
    if which_feature != 'frame':
        feature = feature.flatten()
        
    return feature


def process_acoustic_features(hdf5_file_path_list: list,
                              keys_labels_list: list,
                              labels_dict: dict,
                              which_feature: str,
                              max_size: list,
                              max_size_aux: int,
                              if_frame_acoustic: bool,
                              features: list,
                              labels: list,
                              audio_labels: list,
                              features_dict: dict,
                              canonical_audio_order: list = None) -> tuple[list, list, list, dict]:
    '''
    Process acoustic features from HDF5 files and collect features, labels, and audio_labels.

    Args:
        hdf5_file_path_list (list): List of HDF5 file paths.
        keys_labels_list (list):    List of keys in the labels dictionary.
        labels_dict (dict):         Dictionary containing the labels.
        which_feature (str):        Which feature to use.
        max_size (list):            A list containing the maximum number of frames and features.
        max_size_aux (int):         The maximum size of the dataset.
        if_frame_acoustic (bool):   Whether to use frame-level acoustic features.
        features (list):            List of feature tensors.
        labels (list):              List of label tensors.   
        audio_labels (list):        List of audio label tuples.
        features_dict (dict):       Dictionary mapping audio labels to feature tensors.
        canonical_audio_order (list): List of canonical audio identifiers for padding missing audios.
        
    Returns:
        tuple[list, list, list, dict]: A tuple containing features, labels, audio_labels, and features_dict.
            - features (list):         List of feature tensors.
            - labels (list):           List of label tensors.   
            - audio_labels (list):     List of audio label tuples.
            - features_dict (dict):    Dictionary mapping audio labels to feature tensors.
    '''
    for hdf5_file_path in hdf5_file_path_list:
        # Use padding version for aggregated features when canonical_audio_order is provided
        if which_feature == 'aggregated' and canonical_audio_order is not None:
            (features, 
             labels, 
             audio_labels, 
             features_dict) = process_acoustic_features_with_padding(hdf5_file_path = hdf5_file_path,
                                                                     keys_labels_list = keys_labels_list,
                                                                     labels_dict = labels_dict,
                                                                     which_feature = which_feature,
                                                                     max_size = max_size,
                                                                     max_size_aux = max_size_aux,
                                                                     if_frame_acoustic = if_frame_acoustic,
                                                                     features = features,
                                                                     labels = labels,
                                                                     audio_labels = audio_labels,
                                                                     features_dict = features_dict,
                                                                     canonical_audio_order = canonical_audio_order)
        else:
            for audio_file in keys_labels_list:
                feature = retrieve_and_process_features(hdf5_file_path = hdf5_file_path,
                                                        audio_file = audio_file,
                                                        which_feature = which_feature,
                                                        max_size = max_size,
                                                        max_size_aux = max_size_aux)
                label = labels_dict[audio_file]
                audio_label = get_audio_labels(audio_file = audio_file,
                                               label = label,
                                               if_frame_acoustic = if_frame_acoustic)

                # Save the feature and label
                if which_feature == 'aggregated':
                    if audio_label not in features_dict:
                        features_dict[audio_label] = feature
                        labels.append(label)
                        audio_labels.append(audio_label)
                    else:
                        features_dict[audio_label] = torch.concat((features_dict[audio_label], feature), dim=0)
                else:
                    features.append(feature)
                    labels.append(label)
                    audio_labels.append(audio_label)
    return features, labels, audio_labels, features_dict


def process_acoustic_features_with_padding(hdf5_file_path: str,
                                           keys_labels_list: list,
                                           labels_dict: dict,
                                           which_feature: str,
                                           max_size: list,
                                           max_size_aux: int,
                                           if_frame_acoustic: bool,
                                           features: list,
                                           labels: list,
                                           audio_labels: list,
                                           features_dict: dict,
                                           canonical_audio_order: list) -> tuple[list, 
                                                                                 list, 
                                                                                 list, 
                                                                                 dict]:
    '''
    Process acoustic features with padding for missing audios based on canonical audio order.

    Args:
        hdf5_file_path (str):           HDF5 file path.
        keys_labels_list (list):        List of keys in the labels dictionary.
        labels_dict (dict):             Dictionary containing the labels.
        which_feature (str):            Which feature to use.
        max_size (list):                A list containing the maximum number of frames and features.
        max_size_aux (int):             The maximum size of the dataset.
        if_frame_acoustic (bool):       Whether to use frame-level acoustic features.
        features (list):                List of feature tensors.
        labels (list):                  List of label tensors.   
        audio_labels (list):            List of audio label tuples.
        features_dict (dict):           Dictionary mapping audio labels to feature tensors.
        canonical_audio_order (list):   List of canonical audio identifiers for padding missing audios.
        
    Returns:
        tuple[list, list, list, dict]: A tuple containing features, labels, audio_labels, and features_dict.
            - features (list):         List of feature tensors.
            - labels (list):           List of label tensors.
            - audio_labels (list):     List of audio label tuples.
            - features_dict (dict):    Dictionary mapping audio labels to feature tensors.
    '''
    # Group audios by participant
    participant_audios = {}
    for audio_file in keys_labels_list:
        file_key, _ = extract_file_key_and_base_file_path(audio_file)
        if file_key not in participant_audios:
            participant_audios[file_key] = []
        participant_audios[file_key].append(audio_file)
    
    # Get feature size from first audio
    first_audio = keys_labels_list[0]
    first_feature = retrieve_and_process_features(hdf5_file_path = hdf5_file_path,
                                                  audio_file = first_audio,
                                                  which_feature = which_feature,
                                                  max_size = max_size,
                                                  max_size_aux = max_size_aux)
    feature_size = first_feature.shape[0]
    
    # Process each participant
    for participant_id in participant_audios:
        audios_for_participant = participant_audios[participant_id]
        
        # Map existing audios by their audio number
        audio_map = {}
        for audio_file in audios_for_participant:
            audio_num = audio_file[-6:-4]
            audio_map[audio_num] = audio_file
        
        # Build feature tensor with padding for missing audios
        participant_features = []
        for audio_num in canonical_audio_order:
            if audio_num in audio_map:
                audio_file = audio_map[audio_num]
                feature = retrieve_and_process_features(hdf5_file_path = hdf5_file_path,
                                                       audio_file = audio_file,
                                                       which_feature = which_feature,
                                                       max_size = max_size,
                                                       max_size_aux = max_size_aux)
            else:
                # Pad with zeros for missing audio
                feature = torch.zeros(feature_size, dtype=torch.float32)
            participant_features.append(feature)
        
        # Concatenate all features for this participant
        combined_feature = torch.concat(participant_features, dim=0)
        
        # Get label and audio_label from first audio of participant
        first_audio_file = audios_for_participant[0]
        label = labels_dict[first_audio_file]
        audio_label = get_audio_labels(audio_file = first_audio_file,
                                       label = label,
                                       if_frame_acoustic = if_frame_acoustic)
        
        # Save to features_dict
        if audio_label not in features_dict:
            features_dict[audio_label] = combined_feature
            labels.append(label)
            audio_labels.append(audio_label)
        else:
            features_dict[audio_label] = torch.concat((features_dict[audio_label], combined_feature), dim=0)
    
    return features, labels, audio_labels, features_dict


def determine_feature_and_labels(config: dict[str, Any],
                                 labels_dict: dict,
                                 keys_labels_list: list,
                                 hdf5_file_path_list: list,
                                 which_feature: str,
                                 max_size: list,
                                 max_size_aux: int,
                                 canonical_audio_order: list = None) -> tuple[torch.Tensor, 
                                                             torch.Tensor,
                                                             np.ndarray]:
    '''
    Get the features and labels from the provided dictionaries.
    
    Args:
        config (Dict[str, Any]):    Configuration dictionary.
        labels_dict (dict):         Dictionary containing the labels.
        keys_labels_list (list):    List of keys in the labels dictionary.
        hdf5_file_path_list (list): List of HDF5 file paths.
        which_feature (str):        Which feature to use.
        feature_type (list):        List of feature types to use.
        max_size (list):            A list containing the maximum number of frames and features.
        max_size_aux (int):         The maximum size of the dataset.
        canonical_audio_order (list): List of canonical audio identifiers for padding missing audios.
        
    Returns:
        tuple[torch.tensor, torch.Tensor, torch.Tensor]: 
            - features_tensor (torch.Tensor): Tensor of features.
            - labels_tensor (torch.Tensor):   Tensor of labels.
            - audio_labels_np (np.ndarray):   Array containing a tuple of labels including the provided label and additional metadata based on the dataset and feature type.
    '''
    if_frame_acoustic = 'frame' in which_feature and config['use_acoustic_feat']
    features_dict = {}
    features, labels, audio_labels = [], [], []
    if config['use_acoustic_feat']:
        features, labels, audio_labels, features_dict = process_acoustic_features(hdf5_file_path_list = hdf5_file_path_list,
                                                                                  keys_labels_list = keys_labels_list,
                                                                                  labels_dict = labels_dict,
                                                                                  which_feature = which_feature,
                                                                                  max_size = max_size,
                                                                                  max_size_aux = max_size_aux,
                                                                                  if_frame_acoustic = if_frame_acoustic,
                                                                                  features = features,
                                                                                  labels = labels,
                                                                                  audio_labels = audio_labels,
                                                                                  features_dict = features_dict,
                                                                                  canonical_audio_order = canonical_audio_order)
    if config['use_paralinguistic_feat'] or config['use_linguistic_feat']:
        features, labels, audio_labels, features_dict = process_paraling_ling_features(config = config,
                                                                                      which_feature = which_feature,
                                                                                      keys_labels_list = keys_labels_list,
                                                                                      labels_dict = labels_dict,
                                                                                      features = features,
                                                                                      labels = labels,
                                                                                      audio_labels = audio_labels,
                                                                                      features_dict = features_dict,
                                                                                      if_frame_acoustic = if_frame_acoustic,
                                                                                      canonical_audio_order = canonical_audio_order)
    if config['use_ssl_hubert'] or config['use_ssl_wavlm'] or config['use_ssl_wav2vec']:
        features, labels, audio_labels, features_dict = process_ssl_features(config = config,
                                                                             which_feature = which_feature,
                                                                             keys_labels_list = keys_labels_list,
                                                                             labels_dict = labels_dict,
                                                                             features = features,
                                                                             labels = labels,
                                                                             audio_labels = audio_labels,
                                                                             features_dict = features_dict,
                                                                             if_frame_acoustic = if_frame_acoustic,
                                                                             canonical_audio_order = canonical_audio_order)
    if features_dict:
        for key in features_dict:
            features.append(features_dict[key])            
            
    # Convert lists to tensors and return them
    features_tensor = torch.stack(features)
    labels_tensor = torch.tensor(labels)
    audio_labels_np = np.array(audio_labels)
    return features_tensor, labels_tensor, audio_labels_np


def determine_batch_weights(labels_dict: dict,
                            model: str) -> None:
    '''
    Establish the batch weights based on the ground truth of the dataset.
    
    Args:
        labels_dict (dict):       Dictionary containing the labels.
        model (str):              Model type.
    '''
    batch_weight_dict: {0: 0.5, 1: 0.5}     # type: ignore
    label_counts = Counter(labels_dict.values())
    total_labels = len(labels_dict) 
    batch_weight_dict = {label: count/total_labels for label, count in label_counts.items()}
    DEFAULT_CONFIG['batch_weight_dict'] = batch_weight_dict
    if model == 'NaiveBayes':
        DEFAULT_CONFIG['NaiveBayes']['priors'] = [batch_weight_dict[prior] for prior in range(len(batch_weight_dict))]
    elif model == 'IsolationForest':
        contamination = batch_weight_dict[1] if batch_weight_dict[1] <= 0.5 else 0.5
        DEFAULT_CONFIG['IsolationForest']['contamination'] = contamination
                             
            
def get_aggregated_size(keys_labels_list: list,
                        hdf5_file_path_list: list) -> int:
    '''
    Get the aggregated size for each HDF5 file.

    This function iterates over the list of HDF5 file paths and retrieves the size of the 'aggregated' dataset
    for a specific audio file. The size is determined by the shape of the 'aggregated' dataset in each HDF5 file.
    
    Args:
        keys_labels_list (list):    List of keys in the labels dictionary.
        hdf5_file_path_list (list): List of HDF5 file paths.
        
    Returns:
        max_aggregated_size (int): The maximum size of the 'aggregated' features in all HDF5 files.
    '''
    aggregated_size = []
    audio_file = keys_labels_list[0]
    for hdf5_path in hdf5_file_path_list:
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            aggregated_size.append(hdf5_file[audio_file]['aggregated'].shape[0])
    max_aggregated_size = max(aggregated_size)
    return max_aggregated_size


def set_max_size_based_on_feature(which_feature: str,
                                  keys_labels_list: list,
                                  hdf5_file_path_list: list,
                                  max_size: list,
                                  max_size_aux: int) -> tuple[list, int]:
    '''
    Sets the maximum size for the feature vectors based on the feature type and model input size.
    
    Args:
        which_feature (str):        Which feature to use.
        keys_labels_list (list):    List of keys in the labels dictionary.
        hdf5_file_path_list (list): List of HDF5 file paths.
        max_size (list):            A list containing the maximum number of frames and features.
        max_size_aux (int):         The maximum size of the dataset.
        
    Returns:
        tuple: A tuple containing:
            - max_size (list):      A list containing the maximum number of frames and features.
            - max_size_aux (int):   The maximum size of the dataset.
    '''    
    max_aggregated_size = get_aggregated_size(keys_labels_list = keys_labels_list,
                                              hdf5_file_path_list = hdf5_file_path_list)
    
    if which_feature == 'aggregated':
        max_size[0] = max_aggregated_size
    elif which_feature == 'aggre_frame':
        max_size_aux = max_size[0]
        max_size[0] = max_aggregated_size + 1
    elif which_feature == 'aggre_raw':
        max_size[0] = max_aggregated_size + max_size[0]
        
    return max_size, max_size_aux


def get_max_size(config: dict[str, Any],
                 max_size: list,
                 which_feature: str,
                 keys_labels_list: list,
                 hdf5_file_path_list: list,
                 num_para_ling_feat: int) -> tuple[list, int]:
    '''
    This function determines the maximum size for the feature vectors based on the feature type.
    
    Args:
        config (Dict[str, Any]):        Configuration dictionary.
        max_size (list):                A list containing the maximum number of frames and features.
        which_feature (str):            Which feature to use.
        keys_labels_list (list):        List of keys in the labels dictionary.
        hdf5_file_path_list (list):     List of HDF5 file paths.
        model (str):                    Model type.
        num_para_ling_feat (int):       Number of paralinguistic and linguistic features.
        
    Returns:
        tuple: A tuple containing:
            - max_size (list):      A list containing the maximum number of frames and features.
            - max_size_aux (int):   The maximum size of the dataset.
    '''
    max_size_aux = 0
    if which_feature == 'frame' and config['use_acoustic_feat']:
        max_size_aux = max_size[0]
        max_size[0] = 1
    else:
        if config['use_acoustic_feat']:
            max_size, max_size_aux = set_max_size_based_on_feature(which_feature = which_feature,
                                                                   keys_labels_list = keys_labels_list,
                                                                   hdf5_file_path_list = hdf5_file_path_list,
                                                                   max_size = max_size,
                                                                   max_size_aux = max_size_aux)
        elif config['use_paralinguistic_feat'] or config['use_linguistic_feat']:
            max_size = [1, num_para_ling_feat]
            
    return  max_size, max_size_aux


def get_num_para_ling_feat(config: dict[str, Any],
                           num_paralinguistic_feats: int = 39, 
                           num_linguistic_feats: int = 63) -> int:
    '''
    Get the number of paralinguistic and linguistic features.
    
    Args:
        config (Dict[str, Any]):        Configuration dictionary.
        num_paralinguistic_feats (int): Number of paralinguistic features.
        num_linguistic_feats (int):     Number of linguistic features.

    Returns:
        num_para_ling_feat (int):       The number of paralinguistic and linguistic features.
    '''
    num_para_ling_feat = 0
    if config['use_paralinguistic_feat']:
        num_para_ling_feat += num_paralinguistic_feats
    if config['use_linguistic_feat']:
        num_para_ling_feat += num_linguistic_feats
    return num_para_ling_feat
    
    
def determine_max_size(config: dict[str, Any],
                       keys_labels_list: list,
                       hdf5_file_path_list: list,
                       which_feature: str) -> tuple[list, 
                                                    int]:
    '''
    This function determines the maximum size for the feature vectors .

    Args:
        config (Dict[str, Any]):    Configuration dictionary.
        keys_labels_list (list):    List of keys in the labels dictionary.
        hdf5_file_path_list (list): List of HDF5 file paths.
        which_feature (str):        Which feature to use.
        model (str):                Model type.
        
    Returns:
        tuple: A tuple containing:
            - max_size (list):      A list containing the maximum number of frames and features.
            - max_size_aux (int):   The maximum size of the dataset.
    '''
    max_frames_number, max_features_number = 0, 0
    
    # Determine max sizes according to acoustic features
    if config['use_acoustic_feat']:
        for hdf5_file_path in hdf5_file_path_list:
            for audio_file in keys_labels_list:
                with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                    frames_number = hdf5_file[audio_file]['frames_number'][()]
                    features_number = hdf5_file[audio_file]['features_number'][()]

                    # Update the maximum values
                    max_frames_number = max(max_frames_number, frames_number)
                    max_features_number = max(max_features_number, features_number)
    
    # Store the maximum values
    max_size = [max_frames_number, max_features_number]
    
    # Determine the number of paralinguistic and linguistic features
    num_para_ling_feat = get_num_para_ling_feat(config = config)
    
    # Set the maximum size based on the model input size and feature type
    max_size, max_size_aux = get_max_size(config = config,
                                          max_size = max_size,
                                          which_feature = which_feature,
                                          keys_labels_list = keys_labels_list,
                                          hdf5_file_path_list = hdf5_file_path_list,
                                          num_para_ling_feat = num_para_ling_feat)

    return max_size, max_size_aux


def get_features_and_labels(labels_dict: dict,
                            keys_labels_list: list,
                            which_feature: str,
                            feature_type: list,
                            model: str, 
                            model_conf: dict[str, Any],
                            canonical_audio_order: list = None,
                            init_model: bool = True) -> tuple[torch.Tensor, 
                                                              torch.Tensor, 
                                                              list]:
    '''
    Get the features and labels from the provided dictionaries.
    
    Args:
        labels_dict (dict):          Dictionary containing the labels.
        keys_labels_list (list):     List of keys in the labels dictionary.
        which_feature (str):         Which feature to use.
        feature_type (list):         List of feature types to use.
        model (str):                 Model type.
        model_conf (dict[str, Any]): Configuration dictionary for the model.
        canonical_audio_order (list): List of canonical audio identifiers for padding missing audios.
        init_model (bool):           Initialize batch weights.

        
    Returns:
        tuple[torch.tensor, torch.Tensor, torch.Tensor]: 
            - features (torch.Tensor):   Tensor of features.
            - labels (torch.Tensor):     Tensor of labels.
            - audio_labels (np.ndarray): Array containing a tuple of labels including the provided label and additional metadata based on the dataset and feature type.
    '''
    hdf5_file_path_list = []
    for feature in feature_type:
        hdf5_file_path_list.append(model_conf['path_extracted_features'] + feature + '.h5')
        
    max_size, max_size_aux = determine_max_size(config = model_conf,
                                                keys_labels_list = keys_labels_list,
                                                hdf5_file_path_list = hdf5_file_path_list,
                                                which_feature = which_feature)    
        
    # Determine batch weights
    if init_model:
        determine_batch_weights(labels_dict = labels_dict,
                                model = model)
            
    feature, labels, audio_labels = determine_feature_and_labels(config = model_conf,
                                                                 labels_dict = labels_dict,
                                                                 keys_labels_list = keys_labels_list,
                                                                 hdf5_file_path_list = hdf5_file_path_list,
                                                                 which_feature = which_feature,
                                                                 max_size = max_size,
                                                                 max_size_aux = max_size_aux,
                                                                 canonical_audio_order = canonical_audio_order)
    
    # Apply LASSO feature selection if enabled
    if model_conf.get('use_lasso_selection', False):
        # Add feature_type to model_conf for LASSO filename matching
        model_conf_with_ft = {**model_conf, 'feature_type': feature_type}
        lasso_data = load_lasso_selected_features(model_conf_with_ft, which_feature)
        if lasso_data is not None:
            feature = apply_lasso_feature_selection(feature, lasso_data, model_conf)
       
    return feature, labels, audio_labels
              
            
def get_labels_dict(labels_dict: dict, 
                    id_index: np.ndarray, 
                    features_id: np.ndarray,
                    shuffle: bool = True) -> tuple[dict, list, list]:
    '''
    Get the labels dictionary based on the dataset name and provided indices.
    The participant with the most audio files will have all their audios placed first in the keys_list.

    Args:
        labels_dict (dict):           A dictionary where keys are file paths and values are labels.
        id_index (np.ndarray):        An array of indices or identifiers used to filter the labels.
        features_id (np.ndarray):     An array of feature identifiers corresponding to the dataset.
        shuffle (bool):               Whether to shuffle the keys in the labels dictionary.

    Returns:
        tuple[dict, list, list]:      A tuple containing:
            - selected_labels (dict): A dictionary containing the selected labels.
            - keys_list (list):       A list of keys in the selected_labels dictionary, with participant having most audios first.
            - canonical_audio_order (list): List of audio identifiers representing the canonical order.
    '''
    if shuffle:
        random.seed(42)
        random.shuffle(id_index)
    keys_list = list(labels_dict.keys())
    features_selected = features_id[id_index]
    selected_keys = [key for key in keys_list if re.match(r'^[^\-/_.,\s]+', os.path.basename(key)).group(0) in features_selected]
    selected_labels = {key: labels_dict[key] for key in selected_keys}
    
    # Reorder by participant count
    df = pd.DataFrame({'audio_path': selected_keys})
    df['file_key'] = df['audio_path'].apply(lambda x: re.match(r'^[^\-/_.,\s]+', os.path.basename(x)).group(0))
    participant_order = df['file_key'].value_counts().index
    keys_list = df.set_index('file_key').loc[participant_order].reset_index()['audio_path'].tolist()
    
    # Extract canonical audio order from the first participant (who has the most audios)
    first_participant = participant_order[0]
    first_participant_audios = df[df['file_key'] == first_participant]['audio_path'].tolist()
    canonical_audio_order = sorted([path[-6:-4] for path in first_participant_audios])
    
    return selected_labels, keys_list, canonical_audio_order


def train_model(train_index: np.ndarray, 
                features_id: np.ndarray,
                labels_dict: dict,
                which_feature: str,
                feature_type: list,
                model_name: str, 
                model: str, 
                model_conf: dict, 
                logger: BasicLogger) -> ModelBuilder:    
    '''
    Train the model with the given features and labels.

    Args:
        train_index (np.ndarray): Training features index.
        features_id (np.ndarray): Whole dataset features ID.
        labels_dict (dict):       Dictionary containing the labels.
        which_feature (str):      Which feature to use.
        feature_type (list):      List of feature types to use.
        model_name (str):         Name of the model.
        model (str):              Model type.
        model_conf: (dict):       Dictionary with the model configuration.
        logger (BasicLogger):     Logger for logging information.

    Returns:
        model_builder (ModelBuilder):  Trained model builder object.
    '''
    
    # Get labels and audio path as keys
    (labels_dict, 
    keys_labels_list,
    canonical_audio_order) = get_labels_dict(labels_dict = labels_dict, 
                                             id_index = train_index, 
                                             features_id = features_id)
    
    # Get training features and labels
    train_features, labels, _ = get_features_and_labels(labels_dict = labels_dict,
                                                        keys_labels_list = keys_labels_list,
                                                        which_feature = which_feature,
                                                        feature_type = feature_type,
                                                        model = model,
                                                        model_conf = model_conf,
                                                        canonical_audio_order = canonical_audio_order)
    
    # Get mean and standard deviation for normalization
    model_mean_std_dict = get_mean_std_dict(features = train_features,
                                            config = model_conf)
    
    # Normalize features if needed
    train_features = normalize_features(features = train_features,
                                        model_mean_std_dict = model_mean_std_dict)
            
    # Build model
    path_to_model = model_conf['model_folder']
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    model_builder = ModelBuilder(name = model,
                                 save_name = model_name,
                                 path_to_model = path_to_model,
                                 app_logger = logger,
                                 mean_std_model_dict = model_mean_std_dict)  
    model_builder.build_model()
          
    # Train the model
    model_builder.train_model(train_features, labels)
    
    return model_builder
   

def extract_train_test_data(config: Dict[str, Any]) -> None:
    '''
    Extract, train, and test data based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        None
    '''    
    
    # Adjust some config parameters
    model_conf = configure_model_cores(config = config)
    metrics_dict = initialize_metrics_dict_multiclass()
    
    # Extract and save features from raw data
    feature_combinations, labels_dict = extract_features(audioprocessor_data = config['audioprocessor_data'],
                                                         model_conf = model_conf)
    
    # Begin training and testing by feature type
    for feature_type in feature_combinations:
        feature_type_string = '_'.join(feature_type) + '_'

        # Extract features from the raw data
        for which_feature in model_conf['which_features']:
                       
            # Initialize logger
            logger = BasicLogger(model_conf['log_file']).get_logger()
            
            # Iterate over the models
            for model in model_conf['model_list']:
            
                # Train and test the model with KFold cross-validation or validation CSV
                split_by, labels_list = get_data_to_split(labels = labels_dict)
                train_test_split = create_train_test_split(model_conf = model_conf, 
                                                           model = model,
                                                           which_feature = which_feature, 
                                                           split_by = split_by,
                                                           labels_list = labels_list,
                                                           model_batch = False)
                for fold, (train_index, test_index) in enumerate(train_test_split):
                    
                    # Get model name
                    model_name = determine_model_name(model_conf = model_conf,
                                                      which_feature = which_feature, 
                                                      feature_type_string = feature_type_string,
                                                      model = model, 
                                                      fold = fold)
         
                    # Train model
                    model_builder = train_model(train_index = train_index, 
                                                features_id =  np.array(split_by), 
                                                labels_dict = labels_dict,
                                                which_feature = which_feature,
                                                feature_type = feature_type,
                                                model = model,
                                                model_name = model_name, 
                                                model_conf = model_conf,
                                                logger = logger)
                    
                    # Test the model
                    metrics_dict, detailed_pd = test_model(model_builder = model_builder, 
                                                           test_index = test_index, 
                                                           features_id =  np.array(split_by), 
                                                           labels_dict = labels_dict,
                                                           which_feature = which_feature,
                                                           feature_type = feature_type,
                                                           model = model,
                                                           model_conf = model_conf,
                                                           metrics_dict = metrics_dict)
        
                    # Save metrics to a CSV file
                    metrics_dict = save_metrics_to_csv_multiclass(metrics_dict = metrics_dict, 
                                                                  model = model,
                                                                  model_name = model_name,
                                                                  which_feature = which_feature,
                                                                  feature_type = feature_type_string,
                                                                  save_metrics_path = model_conf['save_metrics_path'], 
                                                                  detailed_pd = detailed_pd,
                                                                  detailed_metrics = model_conf['detailed_metrics'])
                      

def main(config: Dict[str, Any]) -> None:
    '''
    Main function to run the script based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        None
    '''
    if config['run']['extract_train_test']:
        extract_train_test_data(config = config)
    elif config['run']['evaluate_model']:
        evaluate_model(config = config)


if __name__ == '__main__':
    with open('./multiclass_classification_conf.yaml') as file:
        configuration = yaml.safe_load(stream = file)
    main(config = configuration)