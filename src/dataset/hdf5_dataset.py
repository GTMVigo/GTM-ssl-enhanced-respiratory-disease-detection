import os
import re
import h5py
import torch
import bisect
import numpy as np
import pandas as pd
from typing import Union
from collections import Counter
from src.model.model_object_batch import DEFAULT_CONFIG


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 model_conf: dict, 
                 feature_type: list, 
                 which_feature: str,
                 features_id: np.ndarray,
                 id_index: np.ndarray,
                 labels_dict: dict,
                 model: str,
                 init_model: bool = True,
                 model_input_size: tuple = (0, 0),
                 num_paralinguistic_feats: int = 39,
                 num_linguistic_feats: int = 63,
                 model_mean_std_dict: dict = False) -> object:
        '''
        Initialize the dataset.

        Args:
            model_conf (dict):              Configuration dictionary.
            feature_type (list):            List of feature types to use.
            which_feature (str):            Which feature to use.
            features_id (np.ndarray):       Dataset features ID.
            id_index (np.ndarray):          Dataset features ID index.
            labels_dict (dict):             Labels dict.
            model (str):                    Model type.
            init_model (bool):              Initialize CNN.
            model_input_size (tuple):       Model input size.
            num_paralinguistic_feats (int): Number of paralinguistic features.
            model_mean_std_dict (dict):     Model mean and standard deviation dictionary.
            
        Returns:    
            HDF5Dataset (object):           HDF5 dataset object.
        '''
        
        # Initialize the dataset attributes
        self.filename_regex = r'^[^\-/_.,\s]+'
        self.acoustic_feat = model_conf['use_acoustic_feat']
        self.paraling_feat = model_conf['use_paralinguistic_feat']
        self.linguistic_feat = model_conf['use_linguistic_feat']
        self.transcripts_path = model_conf['global_transcript_file'] 
        self.hdf5_folder = model_conf['path_extracted_features'] 
        self.model = model
        self.which_feature = which_feature
        self.num_paralinguistic_feats = num_paralinguistic_feats
        self.num_linguistic_feats = num_linguistic_feats
        
        # Determine if is necessary to normalize the features
        self.normalize = bool(model_mean_std_dict) or \
                         ('normalize' in model_conf.keys() and model_conf['normalize'])
        
        # Get number of paralinguistic and linguistic features
        self.num_para_ling_feat = self.get_num_para_ling_feat(num_paralinguistic_feats = num_paralinguistic_feats,
                                                              num_linguistic_feats = num_linguistic_feats)   
        
        # If acoustic features are used get the HDF5 file paths and handles
        self.hdf5_file_path, self.hdf5_handles = self.open_hdf5_handles(feature_type = feature_type)
        
        # Variables that indicates if at least one HDF5 file is opened successfully
        self.has_valid_hdf5_handle = any(h is not None for h in self.hdf5_handles)
        self.acoustic_handles = self.acoustic_feat and self.has_valid_hdf5_handle
        self.if_frame_acoustic = 'frame' in self.which_feature and self.acoustic_handles
        
        # Get the labels dictionary and keys list
        (self.labels_dict, 
         self.keys_labels_list) = self.get_labels_dict(labels_dict = labels_dict, 
                                                       id_index = id_index, 
                                                       features_id = features_id)
         
        # Get the cumulative index and maximum size of the dataset
        (self.max_size,
         self.cumulative_index,
         self.cumulative_index_frame,
         self.audiofile_frame_list) = self.get_cumulative_index_and_max_size()
            
        # Get the length of features and check if the label size is equal
        self.len_features, self.equal_label_size = self.compute_len_features_and_equal_label()
            
        # Initialize the maximum size based on the which feature and model input size
        self.get_max_size(model_input_size = model_input_size)
        
        # Initialize the model
        self.initialize_model_if_needed(init_model = init_model)
        
        # Preload all non acoustic features if needed
        self.preload_all_non_acoustic_features()
        
        # Get the model mean and standard deviation dictionary
        self.model_mean_std_dict = self.get_model_mean_std_dict(model_mean_std_dict = model_mean_std_dict)


    def __del__(self):
        '''
        Destructor for the HDF5Dataset class.

        This method ensures that all open HDF5 file handles stored in `self.hdf5_handles` are properly closed
        when the HDF5Dataset object is deleted or goes out of scope. It safely attempts to close each handle,
        ignoring any exceptions that may occur during the closing process.

        Returns:
            None
        '''
        for handle in getattr(self, 'hdf5_handles', []):
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
                
                
    def get_num_para_ling_feat(self,
                               num_paralinguistic_feats: int, 
                               num_linguistic_feats: int) -> int:
        '''
        Get the number of paralinguistic and linguistic features.

        Returns:
            int: The number of paralinguistic and linguistic features.
        '''
        num_para_ling_feat = 0
        if self.paraling_feat:
            num_para_ling_feat += num_paralinguistic_feats
        if self.linguistic_feat:
            num_para_ling_feat += num_linguistic_feats
        return num_para_ling_feat


    def open_hdf5_handles(self, 
                          feature_type: list) -> tuple[list, 
                                                       list]:
        '''
        Open HDF5 file handles for each feature type in the given folder.

        Args:
            feature_type (list): List of feature type names (stems, without .h5).

        Returns:
            tuple[list, list]:
                - hdf5_file_path (list): List of HDF5 file paths.
                - hdf5_handles (list): List of opened h5py.File handles or None if opening failed.
        '''
        hdf5_file_path, hdf5_handles = [], []
        for feature_name_stem in feature_type:
            fpath = os.path.join(self.hdf5_folder, feature_name_stem + '.h5')
            hdf5_file_path.append(fpath)
            try:
                hdf5_handles.append(h5py.File(fpath, 'r'))
            except OSError:
                hdf5_handles.append(None)
        return hdf5_file_path, hdf5_handles


    def get_labels_dict(self, 
                        labels_dict: dict, 
                        id_index: np.ndarray, 
                        features_id: np.ndarray) -> tuple[dict, list]:
        '''
        Get the labels dictionary and the keys list based on the provided indices.

        Args:
            labels_dict (dict):           A dictionary where keys are file paths and values are labels.
            id_index (np.ndarray):        An array of indices or identifiers used to filter the labels.
            features_id (np.ndarray):     An array of feature identifiers corresponding to the dataset.

        Returns:
            tuple[dict, list]:            A tuple containing:
                - selected_labels (dict): A dictionary containing the selected labels based on the dataset name and provided indices.
                - keys_list (list):       A list of keys in the selected_labels dictionary.
        '''
        keys_list = list(labels_dict.keys())
        features_selected = {str(f_id) for f_id in features_id[id_index]}
        selected_keys = [key for key in keys_list
                         if re.match(self.filename_regex, os.path.basename(key)) and \
                            re.match(self.filename_regex, os.path.basename(key)).group(0) in features_selected]
        selected_labels = {key: labels_dict[key] for key in selected_keys}
        return selected_labels, selected_keys
  
   
    def get_cumulative_index_and_max_size(self) -> tuple[list, 
                                                         list,
                                                         list,
                                                         list]:
        '''
        Utility function to determine cumulative index and max size for the dataset.

        Returns:
            tuple[list, list, list, list]: A tuple containing:
                - max_size (list):               List containing the maximum number of frames and features.
                - cumulative_index (list):       List containing the cumulative index for the number of files.
                - cumulative_index_frame (list): List containing the cumulative index for frames.
                - audiofile_frame_list (list):   List containing the number of frames for each audio file.
        '''
        if self.acoustic_handles:
            (max_size,
             cumulative_index,
             cumulative_index_frame,
             audiofile_frame_list) = self.determine_cumulative_index()
        else:
            max_size = [0, 0]
            cumulative_index = [len(self.labels_dict) - 1]
            cumulative_index_frame = []
            audiofile_frame_list = []
            
        return max_size, cumulative_index, cumulative_index_frame, audiofile_frame_list
    
    
    def determine_cumulative_index(self) -> tuple[list, list, list]:
        '''
        Determine the cumulative index for frames and features, and the maximum size of the dataset.

        This function calculates the cumulative index for faster indexing of the `idx` provided in the `__getitem__` method.
        It also determines the maximum size of the dataset to reshape the data. The cumulative sum is used to index all 
        numbers from one element to the next in the cumulative index list because each of the numbers in between have the 
        same value.
        
        Returns:
            tuple: A tuple containing:
                - max_size (list):               A list containing the maximum number of frames and features.
                - cumulative_index (list):       A list containing the cumulative index for the number of files.
                - cumulative_index_frame (list): A list containing the cumulative index for frames.
        '''
        max_frames_number, max_features_number = 0, 0
        cumulative_index_frame, audiofile_frame_list = [], []
        
        # Determine max sizes according to acoustic features
        (cumulative_index_frame, 
         audiofile_frame_list, 
         max_frames_number, 
         max_features_number) = self.determine_max_size_and_frame_index()

        # Add the number of files to the cumulative index list
        cumulative_index = [] 
        len_labels_dict = len(self.labels_dict)
        for _ in range(len(self.hdf5_file_path)):
            if cumulative_index:
                cumulative_index.append(cumulative_index[-1] + len_labels_dict)
            else:
                cumulative_index.append(len_labels_dict - 1)
        
        # Store the maximum values
        max_size = [max_frames_number, max_features_number]

        return max_size, cumulative_index, cumulative_index_frame, audiofile_frame_list


    def determine_max_size_and_frame_index(self)-> tuple[list, list, int, int]:
        '''
        Process HDF5 files to extract frame and feature information.

        Returns:
            tuple: (cumulative_index_frame, audiofile_frame_list, max_frames_number, max_features_number)
                - cumulative_index_frame (list): A list containing the cumulative index for frames.
                - audiofile_frame_list (list):   A list containing the number of frames for each audio file.
                - max_frames_number (int):       The maximum number of frames across all audio files.
                - max_features_number (int):     The maximum number of features across all audio files.
        '''
        
        # Initialize lists and variables to store cumulative indices and maximum sizes
        cumulative_index_frame, audiofile_frame_list = [], []
        max_frames_number, max_features_number = 0, 0

        for hdf5_index, _ in enumerate(self.hdf5_file_path):
            
            # Gets the HDF5 file handle
            hdf5_file = self.hdf5_handles[hdf5_index]
            if hdf5_file is None: continue
            
            # Iterate through the audio files in the HDF5 file to extract frame and feature information
            for audio_file_path in self.keys_labels_list:
                if audio_file_path in hdf5_file:
                    try:
                        frames_number = hdf5_file[audio_file_path]['frames_number'][()]
                        features_number = hdf5_file[audio_file_path]['features_number'][()]

                        max_frames_number = max(max_frames_number, frames_number)
                        max_features_number = max(max_features_number, features_number)

                        # Determines the cumulative index for frames
                        (audiofile_frame_list, 
                         cumulative_index_frame) = self.update_cumulative_index_frame(frames_number = frames_number,
                                                                                      audio_file_path = audio_file_path, 
                                                                                      cumulative_index_frame = cumulative_index_frame, 
                                                                                      audiofile_frame_list = audiofile_frame_list)

                    except KeyError:
                        pass

        return cumulative_index_frame, audiofile_frame_list, max_frames_number, max_features_number


    def update_cumulative_index_frame(self,
                                      frames_number: int,
                                      audio_file_path: str,
                                      cumulative_index_frame: list,
                                      audiofile_frame_list: list) -> tuple[list,
                                                                           list]:
        '''
        Update the cumulative index for frames and the audiofile frame list.

        Args:
            frames_number (int):            Number of frames for the current audio file.
            audio_file_path (str):          Path of the current audio file.
            cumulative_index_frame (list):  List to store cumulative frame indices.
            audiofile_frame_list (list): List to store audio file paths for frames.

        Returns:
            tuple[list, list]: A tuple containing:
                - audiofile_frame_list (list): List to store audio file paths for frames.
                - cumulative_index_frame (list): List to store cumulative frame indices.
        '''
        if self.if_frame_acoustic:
            audiofile_frame_list.append(audio_file_path)
            if not cumulative_index_frame:
                cumulative_index_frame.append(frames_number - 1)
            else:
                cumulative_index_frame.append(cumulative_index_frame[-1] + frames_number)
        return audiofile_frame_list, cumulative_index_frame
            
            
    def compute_len_features_and_equal_label(self) -> tuple[int, 
                                                            bool]:
        '''
        Compute the length of features and whether the label size is equal.

        Returns:
            tuple[int, bool] containing:
                - len_features (int):      The length of features.
                - equal_label_size (bool): True if the label size is equal for the whole dataset, False otherwise.
        '''
        if self.if_frame_acoustic:
            if bool(self.cumulative_index_frame):
                len_features = self.cumulative_index_frame[-1] + 1
            else:
                len_features = 0
            equal_label_size = self.check_label_size()
        else:
            len_features = self.cumulative_index[-1] + 1
            equal_label_size = False
            
        return len_features, equal_label_size


    def check_label_size(self) -> bool:
        '''
        Check if the label size is equal for the whole dataset.

        Returns:
            bool: True if the label size is equal for the whole dataset, False otherwise.
        '''
        for file_path in self.keys_labels_list:
            base_file_path = os.path.basename(file_path)
            match = re.match(self.filename_regex, base_file_path)
            filename_no_ext, _ = os.path.splitext(base_file_path)
            if match.group(0) != filename_no_ext:
                return True
        return False


    def get_max_size(self,
                     model_input_size: Union[tuple, int]) -> None:
        '''
        This function determines the maximum size for the feature vectors based on the feature type.
        
        Args:
            model_input_size (int or tuple):   Model input size.
        '''
        if self.if_frame_acoustic:
            self.max_size_aux = self.max_size[0] if self.max_size[0] > 0 else 1
            self.max_size[0] = 1
        else:
            if self.acoustic_handles:
                self.set_max_size_based_on_feature()
            elif self.paraling_feat or self.linguistic_feat:
                 self.max_size = [self.num_para_ling_feat, 1]
            if np.max(model_input_size) > 0:
                self.set_max_size_based_on_model_input(model_input_size = model_input_size)
   
   
    def set_max_size_based_on_feature(self) -> None:
        '''
        Sets the maximum size for the feature vectors based on the feature type and model input size.
        '''       
        max_aggregated_size = self.get_aggregated_size()
        
        if self.which_feature == 'aggregated':
            self.max_size[0] = max_aggregated_size
        elif self.which_feature == 'aggre_frame':
            self.max_size_aux = self.max_size[0] if self.max_size[0] > 0 else 1
            self.max_size[0] = max_aggregated_size + 1
        elif self.which_feature == 'aggre_raw':
            original_max_raw_height = self.max_size[0] if self.max_size[0] > 0 else 1
            self.max_size[0] = max_aggregated_size + original_max_raw_height
        
    
    def get_aggregated_size(self) -> int:
        '''
        Get the aggregated size for each HDF5 file.

        This function iterates over the list of HDF5 file paths and retrieves the size of the 'aggregated' dataset
        for a specific audio file. The size is determined by the shape of the 'aggregated' dataset in each HDF5 file.

        Returns:
            max_aggregated_size (int): The maximum size of the 'aggregated' features in all HDF5 files.
        '''
        max_aggregated_size = 0
        
        # Select a audio file key to check the aggregated size in each HDF5 file
        audio_file_key = self.keys_labels_list[0]
        for hdf5_index, _ in enumerate(self.hdf5_file_path):
            
            # Gets the HDF5 file handle
            hdf5_file = self.hdf5_handles[hdf5_index]
            if hdf5_file is None or audio_file_key not in hdf5_file:
                continue
            
            # Determine the maximum size of the aggregated features
            try:
                agg_shape = hdf5_file[audio_file_key]['aggregated'].shape
                max_aggregated_size = max(max_aggregated_size, agg_shape[0])
            except Exception: 
                pass
                
        return max_aggregated_size
        

    def set_max_size_based_on_model_input(self, 
                                          model_input_size: Union[tuple, int]) -> None:
        '''
        Sets the maximum size for the feature vectors based on the model type and input size.

        Args:
            model_input_size  (int or tuple): The input size of the model.
        '''
        adjust_size = 0
        if (self.paraling_feat or self.linguistic_feat) and self.acoustic_feat:
            adjust_size = self.num_para_ling_feat 
        current_width = self.max_size[1] if self.max_size[1] > 0 else 1

        if self.model == 'MLP':
            self.set_max_size_for_mlp(model_input_size, adjust_size, current_width)
        elif self.model == 'CNN':
            self.set_max_size_for_cnn(model_input_size, adjust_size)
        else:
            self.set_max_size_for_other(model_input_size, adjust_size, current_width)

        self.max_size[0] = max(0, int(self.max_size[0]))
        self.max_size[1] = max(0, int(self.max_size[1]))
          

    def set_max_size_for_mlp(self, 
                             model_input_size: Union[tuple, int], 
                             adjust_size: int, 
                             current_width: int) -> None:
        '''
        Set the maximum size for the feature vectors for an MLP model.

        Args:
            model_input_size (int or tuple): The input size of the MLP model (usually a tuple, e.g., (input_dim,)).
            adjust_size (int):               The number of paralinguistic/linguistic features to subtract from the total input dimension.
            current_width (int):             The current width (number of features per frame) of the input tensor.
        '''
        if isinstance(model_input_size, tuple) and len(model_input_size) > 0:
            mlp_input_total_dim = model_input_size[0]
        else:
            mlp_input_total_dim = model_input_size
        acoustic_flat_dim = mlp_input_total_dim - adjust_size
        if acoustic_flat_dim > 0:
            self.max_size[0] = int(acoustic_flat_dim / current_width)


    def set_max_size_for_cnn(self, 
                             model_input_size: Union[tuple, int], 
                             adjust_size: int) -> None:
        '''
        Set the maximum size for the feature vectors for a CNN model.

        Args:
            model_input_size (int or tuple): The input size of the CNN model (usually a tuple, e.g., (height, width)).
            adjust_size (int):               The number of paralinguistic/linguistic features to subtract from the total input dimension.
        '''
        if isinstance(model_input_size, tuple) and len(model_input_size) == 2:
            self.max_size[0] = model_input_size[0] - adjust_size
        elif isinstance(model_input_size, int):
            calculated_total_height = self.calculate_height_upper_bound(model_input_size)
            self.max_size[0] = calculated_total_height - adjust_size


    def calculate_height_upper_bound(self, 
                                     flattened_size: int) -> int:
        '''
        Calculate the upper bound of the height given a flattened size.

        This method computes the upper bound of the height of an image after it has been processed
        by a convolutional neural network (CNN) layer and a pooling layer. The calculation is based
        on the flattened size of the image and the configuration parameters of the CNN.

        Args:
            flattened_size (int): The flattened size of the image after convolution.

        Returns:
            height_upper_bound (int): The upper bound of the height of the image.
        '''
        flattened_size_conv_no_filters = flattened_size/DEFAULT_CONFIG['CNN']['conv1_filters']
        height_pool_stride = flattened_size_conv_no_filters/(self.max_size[1]//DEFAULT_CONFIG['CNN']['pool1_stride'])
        height_upper_bound = int((height_pool_stride + 1)*DEFAULT_CONFIG['CNN']['pool1_stride'] - 1)
        return height_upper_bound
    

    def set_max_size_for_other(self, 
                               model_input_size: Union[tuple, int], 
                               adjust_size: int, 
                               current_width: int) -> None:
        '''
        Set the maximum size for the feature vectors for models other than MLP or CNN.

        Args:
            model_input_size (int or tuple): The input size of the model (e.g., (height, width) or a single int).
            adjust_size (int):               The number of paralinguistic/linguistic features to subtract from the total input dimension.
            current_width (int):             The current width (number of features per frame) of the input tensor.
        '''
        if isinstance(model_input_size, tuple) and len(model_input_size) > 0:
            other_input_total_dim = model_input_size[1]
        else:
            other_input_total_dim = model_input_size
        acoustic_flat_dim = other_input_total_dim - adjust_size
        if acoustic_flat_dim > 0:
            self.max_size[0] = int(acoustic_flat_dim / current_width)
                      
    
    def initialize_model_if_needed(self, 
                                   init_model: bool) -> None:
        '''
        Initialize CNN parameters or batch weights.

        Args:
            init_model (bool): Whether to initialize the model (CNN or batch weights).
        '''
        if init_model:
            if self.model == 'CNN':
                self.initialize_cnn(max_size=self.max_size)
            elif self.model != 'MLP':
                self.determine_batch_weights()
            
            
    def initialize_cnn(self, 
                       max_size: list) -> None:
        '''
        Configures the CNN model parameters based on the training features and updates the default configuration.
        Updates:
            DEFAULT_CONFIG["CNN"]["flattened_size"]: The calculated flattened size for the CNN model.
        
        Args:
            max_size (list):          The maximum size of the training features.
            DEFAULT_CONFIG (dict):    The default configuration dictionary containing CNN parameters.
        '''
        if (self.paraling_feat or self.linguistic_feat) and self.acoustic_feat: 
            max_size[0] += self.num_para_ling_feat
        elif self.paraling_feat or self.linguistic_feat:
            max_size[0] = self.num_para_ling_feat
        flattened_size_conv_no_filters = 1
        for dim_size in max_size:
            flattened_size_conv_no_filters *= dim_size//DEFAULT_CONFIG['CNN']['pool1_stride']
        flattened_size = flattened_size_conv_no_filters * DEFAULT_CONFIG['CNN']['conv1_filters']
        self.max_size[0] = self.calculate_height_upper_bound(flattened_size = flattened_size)
        if (self.paraling_feat or self.linguistic_feat) and self.acoustic_feat: 
            self.max_size[0] -= self.num_para_ling_feat
        DEFAULT_CONFIG["CNN"]["flattened_size"] = flattened_size


    def determine_batch_weights(self) -> None:
        '''
        Establish the batch weights based on the ground truth of the dataset.
        '''
        batch_weight_dict: dict[int, float] = {0: 0.5, 1: 0.5}
        label_counts = Counter(self.labels_dict.values())
        total_labels = len(self.labels_dict) 
        for label, count in label_counts.items():
            batch_weight_dict[int(label)] = count / total_labels if total_labels > 0 else 0.0
        DEFAULT_CONFIG['batch_weight_dict'] = batch_weight_dict
        if self.model == 'NaiveBayes':
            DEFAULT_CONFIG['NaiveBayes']['priors'] = [batch_weight_dict[prior] for prior in range(len(batch_weight_dict))]
        elif self.model == 'IsolationForest':
            contamination = batch_weight_dict[1]
            if contamination == 0: contamination = 0.01
            DEFAULT_CONFIG['IsolationForest']['contamination'] = min(contamination, 0.499)
               
       
    def preload_all_non_acoustic_features(self):
        '''
        Preload all non-acoustic features (paralinguistic and linguistic) if enabled in the configuration.
        '''
        self.paraling_data_cache = {}
        self.linguistic_data_cache = {}
        if self.paraling_feat:
            self.preload_non_acoustic_features('paralinguistic')
        if self.linguistic_feat:
            self.preload_non_acoustic_features('linguistic')
        
         
    def preload_non_acoustic_features(self, 
                                      feature_kind: str):
        '''
        Preload paralinguistic or linguistic features from HDF5 into a cache for fast access.

        Args:
            feature_kind (str): The kind of feature to preload ('paralinguistic' or 'linguistic').
        '''
        cache_attr = f"{feature_kind}_data_cache"
        h5_filename = f"{feature_kind}_features.h5"
        h5_key = f"{feature_kind}_features"
        hdf5_path = os.path.join(self.hdf5_folder, h5_filename)
        temp_cache = {}
        expected_dim = 0
        if feature_kind == 'paralinguistic':
            expected_dim = self.num_paralinguistic_feats
        elif feature_kind == 'linguistic':
            expected_dim = self.num_linguistic_feats

        # Check if file exists before trying to read
        if not os.path.exists(hdf5_path):
            self.fill_cache_with_zeros(cache_attr = cache_attr, 
                                       expected_dim = expected_dim)
            return

        # Try to read featured data from HDF5 file
        all_df_data = self.try_read_hdf5_dataframe(hdf5_path = hdf5_path, 
                                                   h5_key = h5_key)

        for audio_full_path in self.keys_labels_list:
            target_filename = os.path.basename(audio_full_path)

            # Get series entry 
            series_entry = self.get_series_entry_by_filename(df = all_df_data, 
                                                             target_filename = target_filename)

            # Attempt to read a single row from HDF5
            series_entry = self.read_series_entry_from_hdf5(series_entry = series_entry,
                                                            df = all_df_data, 
                                                            hdf5_path = hdf5_path,
                                                            h5_key = h5_key,
                                                            target_filename = target_filename)

            # Convert series entry to tensor
            feature = self.process_series_to_tensor(series_entry = series_entry,
                                                    expected_dim=expected_dim)

            # Add to cache
            temp_cache[target_filename] = feature
            
        # Store cache with preloaded data features
        setattr(self, cache_attr, temp_cache)


    def fill_cache_with_zeros(self, 
                              cache_attr: str, 
                              expected_dim: int) -> None:
        '''
        Fill the feature cache with zero tensors for all audio files.

        This function is used when the HDF5 file for the given feature kind does not exist.
        It creates a zero tensor of the expected dimension for each audio file and stores it in the cache.

        Args:
            cache_attr (str):   The attribute name for the cache (e.g., 'paralinguistic_data_cache').
            expected_dim (int): The expected dimension of the feature vector.
        '''
        temp_cache = {}
        for audio_full_path in self.keys_labels_list:
            target_filename = os.path.basename(audio_full_path)
            feature = torch.zeros(expected_dim, dtype=torch.float32)
            if self.model == 'CNN' and feature.ndim == 1:
                feature = feature.unsqueeze(1)
                if self.max_size[1] > 0:
                    feature = feature.expand(-1, self.max_size[1])
            temp_cache[target_filename] = feature
        setattr(self, cache_attr, temp_cache)
        
    
    def try_read_hdf5_dataframe(self,
                                hdf5_path: str, 
                                h5_key: str) -> pd.DataFrame | None:
        '''
        Try to read a DataFrame from an HDF5 file. Returns None if the file or key is missing,
        or if the DataFrame does not contain a 'filename' column or index.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            h5_key (str): Key (dataset name) in the HDF5 file.

        Returns:
            pd.DataFrame or None: The DataFrame if successful, otherwise None.
        '''
        try:
            df = pd.read_hdf(hdf5_path, key=h5_key)
            if 'filename' not in df.columns and df.index.name != 'filename':
                return None
            return df
        except Exception:
            return None
    
    
    def get_series_entry_by_filename(self,
                                     df: pd.DataFrame, 
                                     target_filename: str) -> pd.Series | None:
        '''
        Retrieve a row (as Series) from a DataFrame by filename, whether filename is a column or index.

        Args:
            df (pd.DataFrame):      The DataFrame to search.
            target_filename (str):  The filename to look for.

        Returns:
            pd.Series or None:      The matching row as a Series, or None if not found.
        '''
        if df is not None:
            try:
                if 'filename' in df.columns:
                    df_entry = df[df['filename'] == target_filename]
                elif df.index.name == 'filename':
                    df_entry = df.loc[[target_filename]]
                else:
                    df_entry = pd.DataFrame()
                if not df_entry.empty:
                    return df_entry.iloc[0]
            except Exception:
                pass
        return None


    def read_series_entry_from_hdf5(self,
                                    series_entry: pd.Series,
                                    df: pd.DataFrame,
                                    hdf5_path: str, 
                                    h5_key: str, 
                                    target_filename: str) -> pd.Series | None:
        '''
        Attempt to read a single row (as Series) from an HDF5 file by filename.

        Args:
            series_entry (pd.Series):   The Series from the DataFrame.
            df (pd.DataFrame):          The DataFrame to search.
            hdf5_path (str):            Path to the HDF5 file.
            h5_key (str):               Key (dataset name) in the HDF5 file.
            target_filename (str):      The filename to look for.

        Returns:
            series_entry (pd.Series or None): The matching row as a Series, or None if not found or on error.
        '''
        if series_entry is None and df is None:
            try:
                if not os.path.exists(hdf5_path):
                    return None
                df_entry = pd.read_hdf(hdf5_path, key=h5_key, where=f'filename == "{target_filename}"')
                if not df_entry.empty:
                    return df_entry.iloc[0]
            except Exception:
                pass
        return series_entry
    
    
    def process_series_to_tensor(self, 
                                 series_entry: pd.Series, 
                                 expected_dim: int) -> torch.Tensor:
        '''
        Convert a pandas Series of features to a torch tensor, handling 'filename' column if present,
        and adjusting for CNN models if needed.

        Args:
            series_entry (pd.Series): The feature row as a pandas Series.
            expected_dim (int):       The expected dimension of the feature vector (for fallback).

        Returns:
            torch.Tensor: The processed feature tensor.
        '''
        feature = None
        
        # Retrieve feature values into torch tensor
        if series_entry is not None:
            if 'filename' in series_entry.index:
                feature_values = series_entry.drop('filename').to_numpy()
            else:
                feature_values = series_entry.to_numpy()
            feature_values = pd.to_numeric(feature_values, errors='coerce')
            feature = torch.tensor(feature_values, dtype=torch.float32)

        # If feature is still None, create a zero tensor
        if feature is None:
            feature = torch.zeros(expected_dim, dtype=torch.float32)

        # Adjust feature shape for CNN models
        if self.model == 'CNN' and feature.ndim == 1:
            feature = feature.unsqueeze(1)
            if self.max_size is not None and len(self.max_size) > 1 and self.max_size[1] > 0:
                feature = feature.expand(-1, self.max_size[1])

        return feature

  
    def get_model_mean_std_dict(self, 
                                model_mean_std_dict: dict) -> dict:
        '''
        Set the model's mean and standard deviation dictionary.

        Args:
            model_mean_std_dict (dict): Precomputed mean and std dictionary, or False/None.

        Returns:
            model_mean_std_dict (dict): The model's mean and std dictionary.
        '''
        if model_mean_std_dict:
            return model_mean_std_dict
        elif self.normalize:
            model_mean_std_dict = self.determine_mean_std()
        elif not model_mean_std_dict:
            model_mean_std_dict = {}
        return model_mean_std_dict
        
        
    def determine_mean_std(self) -> dict:
        '''
        Determine the mean and standard deviation for each HDF5 file.

        This function calculates the mean and standard deviation for each feature type. 

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: A dictionary where keys are the base names of HDF5 file paths
                                                          and values are tuples of (mean, std) for the features in each file.
        '''

        model_mean_std_dict = {}

        # Get acoustic mean and std 
        for acoustic_feature_idx, acoustic_hdf5_path_str in enumerate(self.hdf5_file_path):
            hdf5_file_handle = self.hdf5_handles[acoustic_feature_idx]
            if hdf5_file_handle is None and self.acoustic_feat:
                continue

            (current_sum, 
             current_sum_sq, 
             total_valid) = self.process_feature_set(hdf5_file_handle = hdf5_file_handle)

            # Calculate mean and std if we have valid items
            if total_valid > 0:
                model_mean, model_std = self.compute_mean_std(current_sum = current_sum, 
                                                             current_sum_sq = current_sum_sq, 
                                                             total_valid = total_valid)
                if self.acoustic_feat:
                    model_mean_std_dict[os.path.basename(acoustic_hdf5_path_str)] = (model_mean, model_std)
                else:
                    model_mean_std_dict['non_acoustic'] = (model_mean, model_std)
                    break

        # Handle the purely non-acoustic case if the main loop didn't run (because self.hdf5_file_path was empty)
        if not self.acoustic_feat and (self.paraling_feat or self.linguistic_feat):
            (current_sum, 
             current_sum_sq, 
             total_valid) = self.process_feature_set(hdf5_file_handle = hdf5_file_handle)
            
            if total_valid > 0:
                model_mean, model_std = self.compute_mean_std(current_sum = current_sum, 
                                                              current_sum_sq = current_sum_sq, 
                                                              total_valid = total_valid)
                model_mean_std_dict['non_acoustic'] = (model_mean, model_std)

        return model_mean_std_dict


    def process_feature_set(self, 
                            hdf5_file_handle: h5py.File) -> tuple[torch.Tensor, 
                                                                  torch.Tensor, 
                                                                  int]:
        '''
        Process and accumulate features from an HDF5 file for all audio files in the dataset.

        Args:
            hdf5_file_handle (h5py.File): The HDF5 file handle to read features from.

        Returns:
            tuple:
                - sum_acc (torch.Tensor):     The sum of all valid combined feature tensors.
                - sum_sq_acc (torch.Tensor):  The sum of squares of all valid combined feature tensors.
                - count (int):                The number of valid feature tensors processed.
        '''
    
        # Initialize accumulators
        sum_acc = sum_sq_acc = None
        count = 0
        
        # Get acoustic part for the current acoustic_feature_idx
        for audio_file in self.keys_labels_list:
            acoustic_feature = torch.empty(0)
            if self.acoustic_feat and (hdf5_file_handle and audio_file in hdf5_file_handle):
                acoustic_feature = self.retrieve_acoustic_features(hdf5_file_handle = hdf5_file_handle,
                                                                   audio_file_path = audio_file)

            # Combine with paralinguistic/linguistic (from cache)
            combined_feature = self.combine_ling_paraling_acoustic_features(feature = acoustic_feature,
                                                                            audio_file = audio_file)
            combined_feature = self.replace_nan_with_zero(combined_feature)
            
            # Skip if this audio file resulted in no features
            if combined_feature.numel() == 0:
                continue

            # Update accumulators
            if sum_acc is None:
                sum_acc = torch.zeros_like(combined_feature, device = combined_feature.device)
                sum_sq_acc = torch.zeros_like(combined_feature, device = combined_feature.device)
            sum_acc += combined_feature
            sum_sq_acc += combined_feature ** 2  
            count += 1

        return sum_acc, sum_sq_acc, count
    

    def retrieve_acoustic_features(self, 
                                   hdf5_file_handle: h5py.File, 
                                   audio_file_path: str) -> torch.Tensor:        
        '''
        Retrieves acoustic features from an HDF5 file for a given audio file path.
        Args:
            hdf5_file_handle (h5py.File): The HDF5 file handle to read from.
            audio_file_path (str):        The path to the audio file within the HDF5 file.
        Returns:
            feature_tensor (torch.Tensor): The processed acoustic feature tensor for the audio file.
        '''
            
        try:
            padding_shape = list(self.max_size)
            if self.which_feature == 'frame':
                feature_data = hdf5_file_handle[audio_file_path]['raw'][()]
                padding_shape = [self.max_size_aux, self.max_size[1]]
                
            # Retrieve and concatenate the aggregated and raw features
            elif self.which_feature == 'aggre_raw' or self.which_feature == 'aggre_frame':
                feat_agg = hdf5_file_handle[audio_file_path]['aggregated'][()]
                feat_raw = hdf5_file_handle[audio_file_path]['raw'][()]
                feat_agg = self.check_if_tensor(feat_agg)
                feat_raw = self.check_if_tensor(feat_raw)
                feature_data = torch.cat((feat_agg, feat_raw), dim = 0)
            else:
                feature_data = hdf5_file_handle[audio_file_path][self.which_feature][()]

            # Check if the feature data is empty and pad it if necessary
            feature_tensor = self.check_if_tensor(feature_data)
            if feature_tensor.numel() == 0: return feature_tensor 
            feature_tensor = self.pad_features(feature = feature_tensor, 
                                               max_size = padding_shape)

            # If the model is not CNN and which feature is not frame flatten the feature
            if self.model != 'CNN' and self.which_feature != 'frame':
                feature_tensor = feature_tensor.view(-1)
            return feature_tensor
        
        except KeyError:
            return torch.empty(0)

    
    def combine_ling_paraling_acoustic_features(self, 
                                                feature: torch.Tensor, 
                                                audio_file: str) -> torch.Tensor:
        '''
        Combine acoustic and paralinguistic features based on the configuration.

        Parameters:
            feature (torch.Tensor): The acoustic feature tensor.
            audio_file (str):       The path to the audio file.

        Returns:
            feature (torch.Tensor): The combined feature tensor.
        '''
        if_feature_exists = feature is not None and feature.numel() > 0
        if if_feature_exists and self.which_feature == 'frame':
            return feature

        # Get non-acoustic features (paralinguistic and/or linguistic)
        non_acoustic_combined = self.get_non_acoustic_combined(audio_file = audio_file)
                
        if_non_acoustic = non_acoustic_combined is not None and non_acoustic_combined.numel() > 0
        if if_feature_exists and if_non_acoustic:
            try:
                return torch.cat((feature, non_acoustic_combined), dim = 0)
            except RuntimeError:
                return feature
        elif if_non_acoustic:
            return non_acoustic_combined
        elif if_feature_exists:
            return feature
        else:
            return torch.empty(0)
        
        
    def get_non_acoustic_combined(self,
                                  audio_file: str) -> torch.Tensor:
        '''
        Combine and return the non-acoustic (paralinguistic and/or linguistic) features for a given audio file.

        Args:
            audio_file (str): The path or name of the audio file for which to retrieve features.

        Returns:
            torch.Tensor or None: The combined non-acoustic feature tensor.
        '''
        para_ling_list = []
        if self.paraling_feat:
            para_ling_list.append(self.get_cached_non_acoustic_feature(audio_file = audio_file,
                                                                       feature_kind = 'paralinguistic'))
        if self.linguistic_feat:
            para_ling_list.append(self.get_cached_non_acoustic_feature(audio_file = audio_file,
                                                                       feature_kind = 'linguistic'))
        if not para_ling_list:
            return None
        if len(para_ling_list) == 1:
            return para_ling_list[0]
        try:
            return torch.cat(para_ling_list, dim=0)
        except RuntimeError:
            return para_ling_list[0]

        
    def get_cached_non_acoustic_feature(self, 
                                        audio_file: str, 
                                        feature_kind: str) -> torch.Tensor:
        '''
        Retrieve a cached non-acoustic feature (paralinguistic or linguistic) for a given audio file.

        Args:
            audio_file (str):   The path or name of the audio file for which to retrieve the feature.
            feature_kind (str): The kind of feature to retrieve ('paralinguistic' or 'linguistic').

        Returns:
            torch.Tensor: The feature tensor for the given audio file. .
        '''
        target_filename = os.path.basename(audio_file)
        cache = getattr(self, f"{feature_kind}_data_cache", {})

        if target_filename in cache:
            return cache[target_filename]
        else:
            if feature_kind == 'paralinguistic':
                expected_dim = self.num_paralinguistic_feats
            elif feature_kind == 'linguistic':
                expected_dim = self.num_linguistic_feats

            fallback_feature = torch.zeros(expected_dim, dtype=torch.float32)
            if self.model == 'CNN' and expected_dim > 0:
                fallback_feature = fallback_feature.unsqueeze(1)
                if self.max_size[1] > 0 :
                     fallback_feature = fallback_feature.expand(-1, self.max_size[1])
            return fallback_feature
        
      
    def compute_mean_std(self,
                         current_sum: torch.Tensor, 
                         current_sum_sq: torch.Tensor, 
                         total_valid: int) -> tuple[torch.Tensor, 
                                                    torch.Tensor]:
        '''
        Compute the mean and standard deviation for a set of features.

        Args:
            current_sum (torch.Tensor):      The sum of all feature tensors.
            current_sum_sq (torch.Tensor):   The sum of squared feature tensors.
            total_valid (int):               The total number of valid feature tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - model_mean: The mean tensor of the features.
                - model_std:  The standard deviation tensor of the features.
        '''
        model_mean = current_sum / total_valid
        model_var = (current_sum_sq / total_valid) - (model_mean ** 2)
        model_var = torch.clamp(model_var, min=1e-9)
        model_std = torch.sqrt(model_var)
        return model_mean, model_std
    
    
    def replace_nan_with_zero(self, 
                              feature: torch.Tensor) -> torch.Tensor:
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
    
    
    def check_if_tensor(self, 
                        feature: np.array) -> torch.Tensor:
        '''
        Ensure the input feature is a PyTorch tensor.

        Args:
            feature (np.array):     The input feature which may or may not be a tensor.

        Returns:
            feature (torch.Tensor): The input feature converted to a tensor if it was not already a tensor.
        '''
        if not isinstance(feature, torch.Tensor):
            try: 
                if isinstance(feature, np.ndarray):
                    return torch.from_numpy(feature.astype(np.float32)) 
                else: 
                    return torch.tensor(np.array(feature, dtype=np.float32)) 
            except Exception: 
                return torch.empty(0)
        return feature.to(torch.float32)
        
       
    def pad_features(self, 
                     feature: torch.Tensor, 
                     dtype: torch.dtype = torch.float32,
                     max_size: list = None) -> torch.Tensor:
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
        max_size = self.max_size if max_size is None else max_size
        if feature is None or feature.numel() == 0: return torch.empty(0)
            
        # Calculate the padding size for each dimension
        for cur_dim, max_dim in zip(feature.shape[::-1], max_size[::-1]):
            pad_width.extend([0, max_dim - cur_dim])
        
        # Apply padding
        padded_feature = torch.nn.functional.pad(feature, pad_width, mode='constant', value=0).to(dtype)
        
        return padded_feature
    
     
    def get_audio_labels(self, 
                         audio_file: str,
                         label: float) -> tuple:
        '''
        Generate a tuple of labels for a given audio file based on the dataset and feature type.

        Args:
            audio_file (str): The path to the audio file.
            label (float): The label associated with the audio file.

        Returns:
            tuple: A tuple of labels including the provided label and additional metadata based on the dataset and feature type.
        '''
        base_file_path = os.path.basename(audio_file)
        match = re.match(self.filename_regex, base_file_path)
        file_key = match.group(0) 
        if self.if_frame_acoustic and self.equal_label_size:
            target_labels = (label, file_key, base_file_path)
        else:
            target_labels = (label, file_key)
        return target_labels
    

    def normalize_feature(self, 
                          feature: torch.Tensor, 
                          feature_type_index: int) -> torch.Tensor:
        '''
        Normalize the feature tensor using the mean and standard deviation.

        Parameters:
            feature (torch.Tensor):   The feature tensor to be normalized.
            feature_type_index (int): The index to determine the feature type.

        Returns:
            feature (torch.Tensor):   The normalized feature tensor.
        '''
        if self.normalize and self.model_mean_std_dict:
            key_for_stats = ''
            if self.acoustic_feat and self.hdf5_file_path:
                key_for_stats = os.path.basename(self.hdf5_file_path[feature_type_index])
            else:
                key_for_stats = 'non_acoustic'
            model_mean, model_std = self.model_mean_std_dict[key_for_stats]
            feature = (feature - model_mean) / (model_std + 1e-7)
            feature = self.replace_nan_with_zero(feature = feature)
        return feature


    def get_feature_from_idx(self,
                             audio_file: str, 
                             feature_type_index: int) -> torch.Tensor:
        '''
        Retrieves the feature vector for a given index from the HDF5 dataset.

        Args:
            audio_file (str):             The audio file associated with the feature vector.
            feature_type_index (int):     The index of the feature to retrieve.

        Returns:
            feature (torch.Tensor): The feature vector.
        '''
        
        # Open the HDF5 file corresponding to the feature type index
        with h5py.File(self.hdf5_file_path[feature_type_index], 'r') as hdf5_file:
            if self.which_feature == 'aggre_raw':
                
                # Retrieve and concatenate the aggregated and raw features
                feature_aggregated = hdf5_file[audio_file]['aggregated'][()]
                feature_aggregated = self.check_if_tensor(feature_aggregated)
                feature_raw = hdf5_file[audio_file]['raw'][()]
                feature_raw = self.check_if_tensor(feature_raw)
                feature = torch.cat((feature_aggregated, feature_raw), dim=0)
            else:
                
                # Retrieve the specified feature
                feature = hdf5_file[audio_file][self.which_feature][()]
                feature = self.check_if_tensor(feature)
            
            # Pad the feature
            feature = self.pad_features(feature=feature)
            
            # Reshape the feature if the model is not 'CNN'
            if self.model != 'CNN': feature = feature.view(-1)
        
        return feature


    def determine_audio_file(self,
                             idx: int) -> tuple[str, int]:
        '''
        Determine the audio file based on the cumulative index and given index.

        Args:
            idx (int):       The index to determine the audio file for.

        Returns:
            tuple[str, int]: A tuple containing the audio file and the feature type index.
                - audio_file (str):         The audio file associated with the feature vector.
                - feature_type_index (int): The index of the feature type.
        '''
        # Determine the feature type index based on the cumulative index
        feature_type_index = bisect.bisect_left(self.cumulative_index, idx)
        
        # Calculate the audio file index
        audio_file_index = idx - feature_type_index * len(self.labels_dict)
        
        # Get the corresponding audio file from the list
        audio_file = self.keys_labels_list[audio_file_index]
        
        return audio_file, feature_type_index


    def get_feature_frame_from_idx(self, 
                                   idx: int) -> tuple[torch.Tensor, str, int]:
        '''
        Retrieves the feature vector for a given index from the HDF5 dataset.

        Args:
            idx (int): The index of the feature to retrieve.

        Returns:
            tuple[torch.Tensor, str]: A tuple containing the feature vector and the corresponding audio file.
                - feature (torch.Tensor): The feature vector.
                - audio_file (str): The audio file associated with the feature vector.
                - feature_type_index (int): The index of the feature in the HDF5 dataset.
        '''
        
        # Check if the HDF5 handle is valid
        if not self.has_valid_hdf5_handle: return torch.empty(0),'',0
        
        # Find the position in the cumulative index frame where the index would fit
        audio_file_global_idx_in_frame_list = bisect.bisect_left(self.cumulative_index_frame, idx)
        
        # Determine the feature type based on the position
        feature_type_index = audio_file_global_idx_in_frame_list // len(self.labels_dict)
        
        # Get the corresponding audio file from the list
        audio_file = self.audiofile_frame_list[audio_file_global_idx_in_frame_list]
        
        # Calculate the center value for the index
        index_center_value = audio_file_global_idx_in_frame_list - 1
        center_value = 0 if index_center_value < 0 else self.cumulative_index_frame[index_center_value] + 1
        
        # Calculate the actual index within the audio file
        index = idx - center_value
        
        # Check if the audio file exists in the HDF5 file for the feature type index
        hdf5_file_handle = self.hdf5_handles[feature_type_index]
        if hdf5_file_handle is None or audio_file not in hdf5_file_handle:
            return torch.empty(0), audio_file, feature_type_index
        
        feature = torch.empty(0)
        try:
            raw_frame_data = hdf5_file_handle[audio_file]['raw'][index]
            feature = self.check_if_tensor(raw_frame_data)

            if self.which_feature == 'aggre_frame':
                feature_aggregated = hdf5_file_handle[audio_file]['aggregated'][()]
                feature_aggregated = self.check_if_tensor(feature_aggregated)
                feature = torch.cat((feature_aggregated, feature.unsqueeze(0)), dim=0)
                feature = self.pad_features(feature = feature)
                
                # If the model is not 'CNN', reshape the feature
                if self.model != 'CNN': feature = feature.view(-1)
                    
            else:
                feature = self.pad_features(feature=feature)

        except Exception:
            return torch.empty(0), audio_file, feature_type_index
                                   
        return feature, audio_file, feature_type_index
    
    
    def __len__(self):
        return self.len_features if self.len_features > 0 else 0


    def __getitem__(self, idx):
        '''
        Get the feature and label corresponding to the provided index.

        Args:
            idx (int): Index provided by the DataLoader.

        Returns:
            tuple: (torch.Tensor, int) where torch.Tensor is the data loaded from the HDF5 file and int is the corresponding label.
                - feature (torch.Tensor): Feature tensor.
                - label (int): Label corresponding to the feature tensor.
                - audio_file (str): Name of the audio file.
        '''
        
        # Get the audio file, feature type and acoustic features(if needed) based on the index
        acoustic_feature = torch.empty(0)
        if self.if_frame_acoustic:
            acoustic_feature, audio_file, feature_type_index = self.get_feature_frame_from_idx(idx = idx)
        else:
            audio_file, feature_type_index = self.determine_audio_file(idx = idx)
            if self.acoustic_feat:
                acoustic_feature = self.get_feature_from_idx(audio_file = audio_file, 
                                                             feature_type_index = feature_type_index)
                
        # Get the paralinguistic features and combine them with the acoustic features(if needed)
        feature = self.combine_ling_paraling_acoustic_features(feature = acoustic_feature, 
                                                               audio_file = audio_file)
        
        # Replace NaN values with zeros and normalize the feature
        feature = self.replace_nan_with_zero(feature = feature)
        feature = self.normalize_feature(feature = feature, 
                                         feature_type_index = feature_type_index)
            
        # Get the label for the audio file and retrieve additional audio labels
        label = self.labels_dict[audio_file]
        audio_labels = self.get_audio_labels(audio_file = audio_file, 
                                             label = label)
        
        # If the model is 'CNN', adjust the feature and label dimensions
        if self.model == 'CNN': 
            feature = feature.unsqueeze(0)
            label = torch.tensor(label).unsqueeze(0)
            
        return feature, label, audio_labels
