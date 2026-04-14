import os
import re
import h5py
import umap
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skrebate import ReliefF
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from typing import Any, List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


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
      

def combine_ling_paraling_acoustic_features(feature: torch.Tensor, 
                                            audio_file: str,
                                            config: dict[str, Any]) -> torch.Tensor:
    '''
    Combine acoustic and paralinguistic features based on the configuration.

    Parameters:
        feature (torch.Tensor):     The acoustic feature tensor.
        audio_file (str):           The path to the audio file.
        config (Dict[str, Any]):    Configuration dictionary.

    Returns:
        feature (torch.Tensor): The combined feature tensor.
    '''
    if config['use_paralinguistic_feat'] or config['use_linguistic_feat']:
        if config['use_acoustic_feat']:
            if config['which_feature'] != 'frame':
                para_ling_feat = get_paraling_ling_features(audio_file = audio_file,
                                                            config = config)
                feature = torch.cat((feature, para_ling_feat), dim = 0)
        else:
            feature = get_paraling_ling_features(audio_file = audio_file, 
                                                 config = config)
    
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
    '''
    Ensure the input feature is a PyTorch tensor.

    Args:
        feature (np.array):     The input feature which may or may not be a tensor.

    Returns:
        feature (torch.Tensor): The input feature converted to a tensor if it was not already a tensor.
    '''
    if not isinstance(feature, torch.Tensor): 
        feature = torch.tensor(feature)
    return feature
    
    
def retrieve_and_process_features(hdf5_file_path: str, 
                                  audio_file: str,
                                  config: dict[str, Any]) -> torch.Tensor:
    '''
    Retrieve and process features from an HDF5 file based on the configuration.

    Parameters:
        hdf5_file_path (str):   The path to the HDF5 file.
        audio_file (str):       The path to the audio file.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        feature (torch.Tensor): The feature tensor.
    '''
    audio_file = audio_file.replace('/home/projects/dtilves/Repos/uvigo_voice', '.') 
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        if config['which_feature'] == 'frame':
            feature = hdf5_file[audio_file]['raw'][()]
            feature = check_if_tensor(feature)
            
        # Retrieve and concatenate the aggregated and raw features
        elif config['which_feature'] == 'aggre_raw' or config['which_feature'] == 'aggre_frame':
            feature_aggregated = hdf5_file[audio_file]['aggregated'][()]
            feature_aggregated = check_if_tensor(feature_aggregated)
            feature_raw = hdf5_file[audio_file]['raw'][()]
            feature_raw = check_if_tensor(feature_raw)
            feature = torch.cat((feature_aggregated, feature_raw), dim=0)
        else:
            
            # Retrieve the specified feature
            feature = hdf5_file[audio_file][config['which_feature']][()]
            feature = check_if_tensor(feature)

        if 'frame' in config['which_feature']:
            feature = pad_features(feature = feature,
                                    max_size = [config['max_size_aux'], config['max_size'][1]])
        elif 'aggregated' not in config['which_feature']:
            feature = pad_features(feature = feature, 
                                    max_size = config['max_size'])
                            
        # Reshape the feature if the feature is not frame
        if config['which_feature'] != 'frame':
            feature = feature.view(-1)

    return feature
    
    
def get_wav_files_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    '''
    Get a list of all .wav files in the specified folder along with their first four characters.

    Args:
        folder_path (str):                  Path to the folder containing .wav files.

    Returns:
        wav_files (List[Tuple[str, str]]):  List of tuples containing file paths and their first four characters.
    '''
    wav_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = root + '/' + file
                match = re.match(r'^[^\-/_.,\s]+', file)
                file_key = match.group(0) if match else file[:-4]
                wav_files.append((file_path, file_key))
    return wav_files


def create_dataset_from_files(config: dict[str, Any]) -> pd.DataFrame:
    '''
    Create a dataset from condition file and .wav files.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        dataset (pd.DataFrame): DataFrame containing the dataset.
    '''
    
    # Read the condition.csv file and create the condition dictionary
    condition_df = pd.read_csv(config['condition_path'], delimiter=';')
    condition_dict = dict(zip(condition_df.iloc[:, config['condition_id_column']], condition_df.iloc[:, config['condition_label_column']]))
    
    # Get all .wav files from the folder
    wav_files_with_keys = get_wav_files_from_folder(folder_path = config['wav_folder'])
    
    # Create a DataFrame from the wav files and their patient types
    dataset = pd.DataFrame(wav_files_with_keys, columns = ['file_path', 'file_key'])
    dataset['file_key'] = dataset['file_key'].astype(str)  # Ensure file_key is a string
    condition_dict = {str(key): value for key, value in condition_dict.items()}  # Ensure keys in condition_dict are strings
    dataset['file_key'] = dataset['file_key'].astype(type(list(condition_dict.keys())[0]))
    dataset['patient_type'] = dataset['file_key'].map(condition_dict)
    dataset = dataset.dropna(subset = ['patient_type'])
    
    # Filter the dataset by the global transcript file
    transcript_df = pd.read_csv(config['global_transcript_file'], delimiter = ';', decimal = ',')
    dataset['basefile'] = dataset['file_path'].apply(lambda x: os.path.basename(x))
    dataset = dataset[dataset['basefile'].isin(transcript_df['filename'])]
    dataset = dataset.drop(columns=['basefile'])

    # Filter the dataset by the specified filter
    if config['filter_dataset'] != 'None':
        dataset = dataset[dataset['file_path'].str.contains(config['filter_dataset'])]   

    return dataset


def get_aggregated_size(config: dict[str, Any]) -> int:
    '''
    Get the aggregated size for each HDF5 file.

    This function iterates over the list of HDF5 file paths and retrieves the size of the 'aggregated' dataset
    for a specific audio file. The size is determined by the shape of the 'aggregated' dataset in each HDF5 file.
    
    Args:
        config (Dict[str, Any]):    Configuration dictionary.
        
    Returns:
        max_aggregated_size (int): The maximum size of the 'aggregated' features in all HDF5 files.
    '''
    aggregated_size = []
    audio_file = list(dataset['file_path'])[0]
    for hdf5_path in config['hdf5_file_path']:
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            audio_file = audio_file.replace('/home/projects/dtilves/Repos/uvigo_voice', '.') 
            aggregated_size.append(hdf5_file[audio_file]['aggregated'].shape[0])
    max_aggregated_size = max(aggregated_size)
    return max_aggregated_size
    

def set_max_size_based_on_feature(config: dict[str, Any],
                                  max_size: list,
                                  max_size_aux: int) -> tuple[list, int]:
    '''
    Sets the maximum size for the feature vectors based on the feature type and model input size.
    
    Args:
        config (Dict[str, Any]):    Configuration dictionary.
        max_size (list):            A list containing the maximum number of frames and features.
        max_size_aux (int):         The maximum size of the dataset.
        
    Returns:
        tuple: A tuple containing:
            - max_size (list):      A list containing the maximum number of frames and features.
            - max_size_aux (int):   The maximum size of the dataset.
    '''    
    max_aggregated_size = get_aggregated_size(config = config)
    
    if config['which_feature'] == 'aggregated':
        max_size[0] = max_aggregated_size
    elif config['which_feature'] == 'aggre_frame':
        max_size_aux = max_size[0]
        max_size[0] = max_aggregated_size + 1
    elif config['which_feature'] == 'aggre_raw':
        max_size[0] = max_aggregated_size + max_size[0]
        
    return max_size, max_size_aux
            
            
def get_max_size(config: dict[str, Any],
                 max_size: list,
                 num_paralinguistic_feats: int = 39,
                 num_linguistic_feats: int = 63) -> tuple[list, int]:
    '''
    This function determines the maximum size for the feature vectors based on the feature type.
    
    Args:
        config (Dict[str, Any]):    Configuration dictionary.
        max_size (list):            A list containing the maximum number of frames and features.
        
    Returns:
        tuple: A tuple containing:
            - max_size (list):      A list containing the maximum number of frames and features.
            - max_size_aux (int):   The maximum size of the dataset.
    '''
    max_size_aux = 0
    if config['which_feature'] == 'frame' and config['use_acoustic_feat']:
        max_size_aux = max_size[0]
        max_size[0] = 1
    else:
        if config['use_acoustic_feat']:
            max_size, max_size_aux = set_max_size_based_on_feature(config = config,
                                                                   max_size = max_size,
                                                                   max_size_aux = max_size_aux)
        elif config['use_paralinguistic_feat'] or config['use_linguistic_feat']:
            num_para_ling_feat = 0
            if config['use_paralinguistic_feat']:
                num_para_ling_feat += num_paralinguistic_feats
            if config['use_linguistic_feat']:
                num_para_ling_feat += num_linguistic_feats
            max_size = [1, num_para_ling_feat]
            
    return  max_size, max_size_aux
                
                
def determine_max_size(config: dict[str, Any],
                       dataset: pd.DataFrame) -> tuple[list, int]:
    '''
    This function determines the maximum size for the feature vectors .

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        dataset (pd.DataFrame):      DataFrame containing the dataset.
        
    Returns:
        tuple: A tuple containing:
            - max_size (list):      A list containing the maximum number of frames and features.
            - max_size_aux (int):   The maximum size of the dataset.
    '''
    max_frames_number, max_features_number = 0, 0
    
    # Determine max sizes according to acoustic features
    if config['use_acoustic_feat']:
        for hdf5_file_path in config['hdf5_file_path']:
            for audio_file in dataset['file_path']:
                with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                    audio_file = audio_file.replace('/home/projects/dtilves/Repos/uvigo_voice', '.') 
                    frames_number = hdf5_file[audio_file]['frames_number'][()]
                    features_number = hdf5_file[audio_file]['features_number'][()]

                    # Update the maximum values
                    max_frames_number = max(max_frames_number, frames_number)
                    max_features_number = max(max_features_number, features_number)
    
    # Store the maximum values
    max_size = [max_frames_number, max_features_number]
    
    max_size, max_size_aux = get_max_size(config = config,
                                          max_size = max_size)

    return max_size, max_size_aux


if __name__ == '__main__':

    config = {'condition_path':          './data/Med_acc_ad/condition_MedPer_train_gold_calib.csv', # Ground truth file
              'condition_id_column':     0, 
              'condition_label_column':  3, 
              'wav_folder':              './data/Med_acc_ad/audios', # Path to the audio files
              'global_transcript_file':  './data/Med_acc_ad/Medper_fluency.csv', # Path to the transcript file
              'filter_dataset':          'None',
              'path_extracted_features': './Med_acc_ad_features/',  # Path to the extracted features HDF5 files
              'max_size':                [0, 0],
              'max_size_aux':            0,
              'hdf5_file_path':          [],
              'which_feature':           'aggregated',
              'use_acoustic_feat':       True,  # Set to True to use acoustic features, False
              'use_paralinguistic_feat': True,  # Set to True to use paralinguistic features
              'use_linguistic_feat':     False,  # Set to True to use linguistic features
              'compute_pca':             False,
              'compute_umap':            False,
              'mutual_info':             True,
              'ReliefF':                 True,
              'hsic':                    False,
              'hsic_kernel':             'rbf'} # Set to 'rbf' or 'linear'
    
    feature_type = ['compare_2016_energy',
                    'compare_2016_voicing',
                    'compare_2016_rasta',
                    'compare_2016_basic_spectral',
                    'spafe_mfcc',
                    'spafe_cqcc',
                    'spafe_gfcc',
                    'spafe_lfcc',
                    'spafe_plp']
        
    feature_aggregated = ['mean',
                          'std',
                          'min',
                          'max',
                          'entropy',
                          'skewness',
                          'kurtosis',
                          'q1',
                          'q2',
                          'q3',
                          'q4']
        
    feature_aggregated_deltas = ['mean',
                                 'mean_delta',
                                 'mean_delta_delta',
                                 'std',
                                 'std_delta',
                                 'std_delta_delta',
                                 'min',
                                 'min_delta',
                                 'min_delta_delta',
                                 'max',
                                 'max_delta',
                                 'max_delta_delta',
                                 'entropy',
                                 'entropy_delta',
                                 'entropy_delta_delta',
                                 'skewness',
                                 'skewness_delta',
                                 'skewness_delta_delta',
                                 'kurtosis',
                                 'kurtosis_delta',
                                 'kurtosis_delta_delta',
                                 'q1',
                                 'q1_delta',
                                 'q1_delta_delta',
                                 'q2',
                                 'q2_delta',
                                 'q2_delta_delta',
                                 'q3',
                                 'q3_delta',
                                 'q3_delta_delta',
                                 'q4',
                                 'q4_delta',
                                 'q4_delta_delta']
    
    paraling_features = ['audio_length', 
                         'silence', 
                         'silence_percentage', 
                         'num_segments', 
                         'silence_segments',
                         'num_silence_segments', 
                         'num_words', 
                         'silence_unique', 
                         'silence_percentage_unique',
                         'num_segments_unique', 
                         'silence_segments_unique', 
                         'num_silence_segments_unique',
                         'num_words_unique', 
                         'silence_word_rate', 
                         'silence_word_unique_rate', 
                         'silence_unique_word_unique_rate', 
                         'word_rate', 
                         'word_unique_rate', 
                         'silence_num_segments_rate', 
                         'silence_num_silence_segments_rate', 
                         'silence_num_segments_unique_rate', 
                         'silence_num_silence_segments_unique_rate', 
                         'silence_unique_num_segments_unique_rate',
                         'silence_unique_num_silence_segments_unique_rate', 
                         'silence_segments_audio_length_rate',
                         'silence_segments_unique_audio_length_rate', 
                         'silence_segments_num_segments_rate',
                         'silence_segments_num_silence_segments_rate', 
                         'silence_segments_num_segments_unique_rate',
                         'silence_segments_num_silence_segments_unique_rate', 
                         'silence_segments_unique_num_segments_unique_rate',
                         'silence_segments_unique_num_silence_segments_unique_rate', 
                         'silence_silence_segments_rate', 
                         'silence_silence_segments_unique_rate',
                         'silence_unique_silence_segments_unique_rate', 
                         'silence_segments_num_words_rate', 
                         'silence_segments_num_words_unique_rate', 
                         'silence_segments_unique_num_words_unique_rate',
                         'num_speakers']
    
    linguistic_features =['total_words', 
                          'unique_words', 
                          'ramification_ratio', 
                          'number_sentences',
                          'sentence_similarity',
                          'words_sentence',
                          'unique_words_sentence',
                          'utterance_length_word',
                          'utterance_length_sentence',
                          'sentence_len_word',
                          'sentence_len_sentence',
                          'structure_variety',
                          'dependency_parsing_distance',
                          'pos_tags_similarity',
                          'coherence_score_sentences',
                          'levenshtein_sentence_similarity',
                          'levens_lemmas_sentence_similarity',
                          'errors_sentence',
                          'deprel_sentence',
                          'feats_types_sentence',
                          'feats_sentence',
                          'number_of_pos_sentences',
                          'number_verbs_sentences',
                          'number_nouns_sentences',
                          'number_adjectives_sentences',
                          'number_adverbs_sentences',
                          'number_pronouns_sentences',
                          'number_prepositions_sentences',
                          'number_conjunctions_sentences',
                          'number_determiners_sentences',
                          'number_interjections_sentences',
                          'number_propernouns_sentences',
                          'number_punctuation_sentences',
                          'number_other_sentences',
                          'number_auxiliary_sentences',
                          'number_numerals_sentences',
                          'number_particles_sentences',
                          'number_subordinating_sentences',
                          'sophistication_score_words',
                          'sophistication_score_lemmas',
                          'average_dependency_distance',
                          'mood_types_sentences',
                          'tense_types_sentences',
                          'verbform_types_sentences',
                          'pronom_types_sentences',
                          'person_types_sentences',
                          'reflexive_sentences_sentences',
                          'case_types_sentences',
                          'open_class_words',
                          'open_class_words_sentences',
                          'closed_class_words',
                          'closed_class_words_sentences',
                          'content_density',
                          'content_density_sentences',
                          'nouns_to_verbs_ratio',
                          'deixis_rate',
                          'idea_density',
                          'lexical_density',
                          'ratio_pronons_nouns',
                          'continuous_repetition_ratio',
                          'number_hes_sentences',
                          'number_lau_sentences',
                          'number_uni_sentences',]
    
    acoustic_feature_dims = {'energy': 4,
                             'voicing': 6,
                             'rasta': 26,
                             'basic_spectral': 15,
                             'mfcc': 13,
                             'cqcc': 13,
                             'gfcc': 13,
                             'lfcc': 13,
                             'plp': 12}
           
    # === Step 1: Load data from HDF5 ===
    for feature in feature_type:
        config['hdf5_file_path'].append(config['path_extracted_features'] + feature + '.h5')
    dataset = create_dataset_from_files(config = config)
    (config['max_size'], 
     config['max_size_aux']) = determine_max_size(config = config,
                                                  dataset = dataset)
    
    # Determine feature names
    features_names, data, labels = [], [], [] 
    data_dict = {}
    if config['use_acoustic_feat']:
        if 'aggregated' in config['which_feature']:
            for feat, dim in acoustic_feature_dims.items():
                for agg in feature_aggregated_deltas:
                    features_names.extend([f"{agg}_{feat}_{i}" for i in range(dim)])
            if config['use_paralinguistic_feat']:
                features_names.extend(paraling_features)
            if config['use_linguistic_feat']:
                features_names.extend(linguistic_features)
        elif 'aggre' in config['which_feature']:
            for agg in feature_aggregated:
                features_names.extend([f'{agg}_{feat}' for feat in range(config['max_size'][1])])          
        else:
            features_names.extend([f'feature_{feat}' for feat in range(config['max_size'][1])])
    else:
        if config['use_paralinguistic_feat']:
            features_names.extend(paraling_features)
        if config['use_linguistic_feat']:
            features_names.extend(linguistic_features)
        
    if config['use_acoustic_feat']:
        for hdf5_file_path in tqdm(config['hdf5_file_path'], desc='HDF5 files'):
            feature_name = os.path.basename(hdf5_file_path).split('.')[0]
            for audio_file in tqdm(dataset['file_path'], desc=f'Audio files in {feature_name}', leave=False):
                feature = retrieve_and_process_features(hdf5_file_path = hdf5_file_path, 
                                                        audio_file = audio_file, 
                                                        config = config)
                feature = feature.numpy()
                if config['which_feature'] == 'aggregated':
                    if audio_file not in data_dict:
                        data_dict[audio_file] = feature
                        label = dataset.loc[dataset['file_path'] == audio_file, 'patient_type'].values[0]
                        labels.append(label)                    
                    else:
                        data_dict[audio_file] = np.concatenate((data_dict[audio_file], feature), axis=0)
                else:
                    data.append(feature)
                    label = dataset.loc[dataset['file_path'] == audio_file, 'patient_type'].values[0]
                    labels.append(label)
                
    if config['use_paralinguistic_feat'] or config['use_linguistic_feat']:
        max_size_aux = data[0].shape[0] if config['use_acoustic_feat'] and 'aggregated' not in config['which_feature'] else None
        for audio_file in tqdm(dataset['file_path'], desc='Para-linguistic features', leave=False):
            feature = get_paraling_ling_features(audio_file = audio_file, 
                                                 config = config, 
                                                 max_size = max_size_aux)
            feature = feature.numpy()
            if config['which_feature'] == 'aggregated':
                if audio_file not in data_dict:
                    data_dict[audio_file] = feature
                    label = dataset.loc[dataset['file_path'] == audio_file, 'patient_type'].values[0]
                    labels.append(label)                    
                else:
                    data_dict[audio_file] = np.concatenate((data_dict[audio_file], feature), axis=0)
            else:
                data.append(feature)
                label = dataset.loc[dataset['file_path'] == audio_file, 'patient_type'].values[0]
                labels.append(label)
    
    # Extract data from the dictionary if using aggregated features
    if data_dict:
        for key in data_dict:
            data.append(data_dict[key])

    # === Step 2: Data projection ===
    if config['compute_pca']:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        print("Transformed shape:", X_pca.shape)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        print("Number of components kept:", pca.n_components_)

        save_name = f"PCA{'_acoustic' if config['use_acoustic_feat'] else ''}" \
                    f"{'_para' if config['use_paralinguistic_feat'] else ''}" \
                    f"{'_ling' if config['use_linguistic_feat'] else ''}".strip('_') + ".csv"
        components_df = pd.DataFrame(pca.components_[:4], columns=features_names, index=[f'PC{i+1}' for i in range(4)])
        components_df.insert(0, 'explained_variance_ratio', pca.explained_variance_ratio_[:4])
        components_df.to_csv(save_name, mode = 'w', index = False, header = True, sep = ';', decimal = ',')

    if config['compute_umap']:

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42,
                            low_memory=False, verbose=True)
        embedding = reducer.fit_transform(data)
        # np.savez('umap_with_labels.npz', embedding=embedding, labels=labels)
        print(f'UMAP embedding shape: {embedding.shape}')
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=2, cmap='Spectral')
        plt.title('UMAP Projection with Labels')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.colorbar(scatter, label='Class Label')
        save_name = f"umap{'_acoustic' if config['use_acoustic_feat'] else ''}" \
                    f"{'_para' if config['use_paralinguistic_feat'] else ''}" \
                    f"{'_ling' if config['use_linguistic_feat'] else ''}".strip('_') + ".png"
                    # f"{'_ling' if config['use_linguistic_feat'] else ''}".strip('_') + "_" + feature_name + ".png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')

    if config['mutual_info']:

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # y: your target labels (shape: (n_samples,))
        mi_scores = mutual_info_classif(X_scaled, labels, random_state=42)

        # Rank features
        mi_sorted_indices = mi_scores.argsort()[::-1]
        
        # Get the feature names in the sorted order
        mi_sorted_feature_names = np.array(features_names)[mi_sorted_indices]

        # Get the MI scores in the sorted order
        mi_sorted_scores = mi_scores[mi_sorted_indices]

        # Calculate ranks based on mutual information scores (descending)
        # rankdata returns ranks, and we want 1st for highest score, so we negate mi_scores
        mi_ranks = rankdata(-mi_scores, method='ordinal')[mi_sorted_indices] # Apply sorting to ranks as well

        # Create a DataFrame with feature names, their mutual information scores, and rank
        # By building it directly with the sorted data, mi_df will be sorted by MI ranking
        mi_df = pd.DataFrame({'Feature':    mi_sorted_feature_names,
                              'MI_Score':   mi_sorted_scores,
                              'Rank':       mi_ranks})      
        save_name = f"Mutual_info{'_acoustic' if config['use_acoustic_feat'] else ''}" \
                    f"{'_para' if config['use_paralinguistic_feat'] else ''}" \
                    f"{'_ling' if config['use_linguistic_feat'] else ''}".strip('_') + ".csv"
        mi_df.to_csv(save_name, index=False, sep=';', decimal=',')

    if config['ReliefF']:
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Initialize and fit ReliefF
        # n_features_to_select should ideally be X_scaled.shape[1] if you want all importances
        relieff = ReliefF(n_neighbors=100, n_features_to_select=X_scaled.shape[1])
        relieff.fit(X_scaled, labels)
        
        # Get the feature importances
        relieff_scores = relieff.feature_importances_

        # Get the indices that would sort feature importances in descending order
        relieff_sorted_indices = relieff_scores.argsort()[::-1]

        # Get the feature names in the sorted order
        relieff_sorted_feature_names = np.array(features_names)[relieff_sorted_indices]

        # Get the ReliefF scores in the sorted order
        relieff_sorted_scores = relieff_scores[relieff_sorted_indices]

        # Calculate ranks based on ReliefF scores (descending order: highest score gets rank 1)
        # rankdata function returns ranks, so negate the scores to get rank 1 for highest value
        relieff_ranks = rankdata(-relieff_scores, method='ordinal')[relieff_sorted_indices]

        # Create a DataFrame with feature names, their ReliefF scores, and rank
        # By constructing the DataFrame with the already sorted arrays, it will be naturally sorted by rank.
        relieff_df = pd.DataFrame({'Feature':       relieff_sorted_feature_names,
                                   'ReliefF_Score': relieff_sorted_scores,
                                   'Rank':          relieff_ranks})
        save_name = f"Relieff{'_acoustic' if config['use_acoustic_feat'] else ''}" \
                    f"{'_para' if config['use_paralinguistic_feat'] else ''}" \
                    f"{'_ling' if config['use_linguistic_feat'] else ''}".strip('_') + ".csv"
        relieff_df.to_csv(save_name, index=False, sep=';', decimal=',')

    if config['hsic']:

        # 1. Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)  # (n_samples, n_features)
        y = np.array(labels).reshape(-1, 1)   # Ensure target is 2D (n_samples, 1)
        
        # 2. Compute HSIC for each feature
        n_samples = X_scaled.shape[0]
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples  # Centering matrix
        hsic_scores = []

        for i in range(X_scaled.shape[1]):
            if config['hsic_kernel'] == 'rbf':
                K = rbf_kernel(X_scaled[:, i].reshape(-1, 1), gamma=0.5)
            else:
                K = linear_kernel(X_scaled[:, i].reshape(-1, 1))
                    
            # Target kernel (linear, better for classification)
            L = linear_kernel(y)
            
            # Centered kernels
            K_centered = H @ K @ H
            L_centered = H @ L @ H
            
            # HSIC score
            hsic_score = np.trace(K_centered @ L_centered) / (n_samples ** 2)
            hsic_scores.append(hsic_score)

        hsic_scores = np.array(hsic_scores)

        # Rank features
        top_features_indices =hsic_scores.argsort()[::-1]

        # Get the feature names in the sorted order
        hsic_sorted_feature_names = np.array(features_names)[top_features_indices]

        # Get the HSIC scores in the sorted order
        hsic_sorted_scores = hsic_scores[top_features_indices]

        # Calculate ranks based on HSIC scores (descending order: highest score gets rank 1)
        # rankdata function returns ranks, so negate the scores to get rank 1 for highest value
        hsic_ranks = rankdata(-hsic_scores, method='ordinal')[top_features_indices]

        # Create a DataFrame with feature names, their HSIC scores, and rank
        # By constructing the DataFrame with the already sorted arrays, it will be naturally sorted by rank.
        hsic_df = pd.DataFrame({'Feature':       hsic_sorted_feature_names,
                                'HSIC_Score':    hsic_sorted_scores,
                                'Rank':          hsic_ranks})
        save_name = f"HSIC{'_acoustic' if config['use_acoustic_feat'] else ''}" \
                    f"{'_para' if config['use_paralinguistic_feat'] else ''}" \
                    f"{'_ling' if config['use_linguistic_feat'] else ''}" \
                    f"_{config['hsic_kernel']}".strip('_') + ".csv"
        hsic_df.to_csv(save_name, index=False, sep=';', decimal=',')