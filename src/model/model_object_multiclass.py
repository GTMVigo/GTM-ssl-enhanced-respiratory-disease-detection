import torch
import pickle
import logging
import os.path
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from pydantic import BaseModel
from src.exceptions import ModelError
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression,SGDOneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest


DEFAULT_CONFIG = {
        'LogisticRegression': {'C': 0.1,
                               'max_iter': 5000,
                               'penalty': 'l2',
                               'random_state': 42,
                               'solver': 'lbfgs',
                               'multi_class': 'multinomial',
                               'class_weight': 'balanced',
                               'n_jobs': -1,},
        'RandomForest': {'n_estimators': 200,
                         'criterion': 'gini',
                         'max_depth': None,
                         'min_samples_split': 2,
                         'min_samples_leaf': 2,
                         'max_features': 'sqrt',
                         'class_weight': 'balanced',
                         'random_state': 42,
                         'n_jobs': -1,},
        'LinearSVM': {'C': 0.1,
                      'tol': 0.001,
                      'max_iter': 5000,
                      'class_weight': 'balanced',
                      'random_state': 42,
                      'probability': True,
                      'decision_function_shape': 'ovo'},
        'XGBoost': {'learning_rate': 0.1,
                    'n_estimators': 200,
                    'max_depth': 6,
                    'objective': 'multi:softprob',
                    'num_class': 5,
                    'random_state': 42,
                    'n_jobs': -1},
        'GaussianProcess': {'n_restarts_optimizer': 5,
                            'max_iter_predict': 100,
                            'random_state': 42},
        'NaiveBayes': {'priors': [0.022, 0.0956, 0.2096, 0.2794, 0.3934],},
        'GaussianMixture': {'n_components': 5,
                            'covariance_type': 'full',
                            'tol': 0.001,
                            'reg_covar': 1e-06,
                            'max_iter': 100,
                            'n_init': 5,
                            'init_params': 'kmeans',
                            'random_state': 42},
        'SGDOneClassSVM': {'nu': 0.25,
                           'warm_start': False,
                           'random_state': 42},
        'IsolationForest': {'n_estimators': 200,
                            'contamination': 0.25,
                            'random_state': 42,
                            'n_jobs': -1,},
    }



class ModelBuilder(BaseModel):
    name: str
    path_to_model: str
    app_logger: logging.Logger
    save_name: str = None  
    mean_std_model_dict: dict = {}

    seed: int = 42
    model: object = None
    is_trained: bool = False
    parameters: dict = None
    device: str = None
    extension_mean_std: str = '_model_mean_std.pth'
    

    class Config:
        arbitrary_types_allowed = True


    def save_as_a_serialized_object(self, 
                                    path_to_save: str = None) -> None:
        '''
        Saves the model as a serialized object to the specified path.

        Args:
            path_to_save (str, optional): The directory path where the model should be saved.
                                        If not provided, the model will be saved to the default path.

        Raises:
            IOError: If an error occurs while saving the model.
        '''
        if path_to_save:
            path_to_save_model = os.path.join(path_to_save, self.save_name  + '.pkl')
        else:
            path_to_save_model = os.path.join(self.path_to_model, self.save_name  + '.pkl')

        try:
            with open(path_to_save_model, 'wb') as file:
                pickle.dump(self.model, file)
            if self.mean_std_model_dict:
                if path_to_save:
                    path_to_save_mean_std = os.path.join(path_to_save, self.save_name  + self.extension_mean_std)
                else:
                    path_to_save_mean_std = os.path.join(self.path_to_model, self.save_name  + self.extension_mean_std)
                torch.save(self.mean_std_model_dict, path_to_save_mean_std)
        except Exception as e:
            raise IOError(f'An error occurred while saving the model: {e}')

        self.app_logger.info(f'Model saved in {path_to_save_model}')


    def load_model_from_a_serialized_object(self, 
                                            path_to_load: str = None,
                                            extension: str = '.pkl') -> object:
        '''
        Loads the model from a serialized object file.

        Args:
            path_to_load (str, optional): The directory path from where the model should be loaded.
                                        If not provided, the model will be loaded from the default path.

        Raises:
            IOError: If an error occurs while loading the model.

        Returns:
            self.model (object): The loaded model.
        '''
        if extension in self.save_name:
            save_name = self.save_name[:-4]
            extension = ''
        else:
            save_name = self.save_name
        if path_to_load:
            path_to_load_model = os.path.join(path_to_load, self.save_name  + extension)
        else:
            path_to_load_model = os.path.join(self.path_to_model, self.save_name  + extension)

        try:
            with open(path_to_load_model, 'rb') as file:
                self.model = pickle.load(file)
            if path_to_load:
                path_to_load_mean_std = os.path.join(path_to_load, save_name  + self.extension_mean_std)
            else:
                path_to_load_mean_std = os.path.join(self.path_to_model, save_name  + self.extension_mean_std)
            if os.path.exists(path_to_load_mean_std):
                self.mean_std_model_dict = torch.load(path_to_load_mean_std)
        except Exception as e:
            raise IOError(f'An error occurred while loading the model: {e}')

        self.app_logger.info(f'Model loaded from {path_to_load_model}')
        return self.model


    def build_model(self):
        '''
        Builds the model based on the specified model name and parameters.

        This method performs the following steps:
        1. Checks if the model name is supported.
        2. Retrieves the default configuration for the model.
        3. Initializes the model using the appropriate class and parameters.
        4. Handles any exceptions that occur during model initialization.

        Raises:
            ValueError: If the model name is not supported.
            ModelError: If an error occurs while loading the model.
        '''
        
        # Check if the model name is supported
        if self.name not in SUPPORTED_MODELS.keys():
            raise ValueError(f'Model {self.name} is not supported.')

        # Initialize valid_parameters with the provided parameters
        valid_parameters = self.parameters
        try:
            # Retrieve the default configuration for the model
            valid_parameters = DEFAULT_CONFIG[self.name]
            
            # Get the model class from the supported models dictionary
            model_class = SUPPORTED_MODELS[self.name]
            
            # Initialize the model based on the model name
            model = model_class(**valid_parameters)
        except Exception as e:
            raise ModelError(f'ModelBuilder - An error occurred while loading the model: {e}')
        finally:
            
            # Update the parameters with the valid parameters
            self.parameters = valid_parameters
            
        # Assign the built model to the instance variable
        self.model = model
        

    def train_model(self, 
                    inputs: torch.Tensor, 
                    labels: torch.Tensor) -> object:
        '''
        Trains the model using the provided features and labels.
        Args:
            inputs (torch.Tensor): The input features.
            labels (torch.Tensor): Tensor of labels.
        
        Returns:
            object: The trained model.
            
        Raises:
            ValueError: If the model is not built yet.
            ModelError: If an error occurs while training the model.
        '''    
            
        if self.model is None:
            raise ValueError('ModelBuilder - Model is not built yet. Please build the model first.')
        try:
            if self.name == 'GaussianMixture':
                self.model.fit(inputs)         
            elif self.name == 'NaiveBayes':
                self.model.fit(inputs, labels,)
            else: 
                batch_sample_weights = np.array([DEFAULT_CONFIG['batch_weight_dict'][int(label)] for label in labels])
                self.model.fit(inputs, labels, sample_weight = batch_sample_weights)
            self.is_trained = True
        except Exception as e:
            raise ModelError( f'ModelBuilder - An error occurred while training the model: {e}' )

        return self.model
    
    
    def evaluate_model(self, 
                       inputs: torch.Tensor) -> tuple[np.ndarray, 
                                                      np.ndarray]:
        '''
        Evaluates the model using the provided features and labels.

        Args:
            inputs (torch.Tensor):       The input features.
            labels (torch.Tensor):       Tensor of labels.
            audio_labels (torch.Tensor): Tensor containing a tuple of labels including the provided label and additional metadata based on the dataset and feature type.
        Raises:
            ValueError: If the model is not built yet.

        Returns:
            tuple[np.ndarray, np.ndarray, list]: A tuple containing:
                - test_predictions (np.ndarray): The predictions.
                - test_probabilities (np.ndarray): The probabilities of the predictions.
        '''
        if self.model is None:
            raise ValueError('ModelBuilder - Model is not built yet. Please build the model first.')

        # Generate predictions using the model
        predictions = self.model.predict(inputs)
        
        # Handle prediction probabilities based on the model type
        if self.name in ['LinearSVM', 'SGDOneClassSVM', 'IsolationForest']:
            
            # Correct the predictions for these models
            predictions[predictions == -1] = 0
            
            # For these models create a placeholder for probabilities
            probabilities = np.full((predictions.shape[0], 2), -1)
            probabilities[probabilities == -1] = 0
        else:
            
            # For other models, use the model's predict_proba method
            probabilities = self.model.predict_proba(inputs)
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Convert the lists to numpy arrays
        test_predictions = np.array(predictions)
        test_probabilities = np.array(probabilities)
        
        return test_predictions, test_probabilities
      

        
SUPPORTED_MODELS = {
        'LogisticRegression': LogisticRegression,
        'RandomForest': RandomForestClassifier,
        'LinearSVM': SVC,
        'XGBoost': xgb.XGBClassifier,
        'GaussianProcess': GaussianProcessClassifier,
        'NaiveBayes': GaussianNB,
        'GaussianMixture': GaussianMixture,
        'SGDOneClassSVM': SGDOneClassSVM,
        'IsolationForest': IsolationForest,
    }

