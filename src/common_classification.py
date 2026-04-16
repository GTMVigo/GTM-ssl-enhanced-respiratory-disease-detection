import os
import re
import h5py
import torch
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.model.model_object_multiclass import ModelBuilder
from src.features.audio_processor import AudioProcessor


def save_metrics_to_csv_multiclass(metrics_dict: Dict[str, float],
                                   model: str,
                                   model_name: str,
                                   which_feature: List[str],
                                   feature_type: str,
                                   save_metrics_path: str,
                                   detailed_pd: pd.DataFrame,
                                   detailed_metrics: bool) -> Dict[str, float]:
    '''
    Save evaluation metrics to CSV file.
    
    Args:
        metrics_dict: Dictionary containing evaluation metrics.
        model: Model used for training.
        model_name: Name of the model.
        which_feature: List of features used.
        feature_type: Feature type used.
        save_metrics_path: Path to save metrics CSV.
        detailed_pd: DataFrame with detailed predictions.
        detailed_metrics: Whether to save detailed metrics.
    
    Returns:
        Empty metrics dictionary.
    '''
    
    # Save variables to dictionary
    metrics_dict['model'].append(model)
    metrics_dict['model_name'].append(model_name)
    metrics_dict['which_features'].append(which_feature)
    metrics_dict['feature_type'].append(feature_type)
    
    # Convert to DataFrame
    metrics_pd = pd.DataFrame(metrics_dict)
    
    # Save main metrics
    if not os.path.isfile(save_metrics_path):
        metrics_pd.to_csv(save_metrics_path, mode='a', index=False,
                         header=True, sep=';', decimal=',')
    else:
        metrics_pd.to_csv(save_metrics_path, mode='a', index=False,
                         header=False, sep=';', decimal=',')
    
    metrics_dict = initialize_metrics_dict_multiclass()
    
    # Save detailed metrics if required
    if detailed_metrics:
        detailed_pd['model'] = model
        detailed_pd['model_name'] = model_name
        detailed_pd['which_features'] = str(which_feature)
        if 'no_acoustic_feature' in feature_type:
            feature_type = 'no_acoustic_feature'
        detailed_pd['feature_type'] = feature_type
        metrics_file_name = os.path.basename(save_metrics_path)[:-4]
        save_path_folder = os.path.dirname(save_metrics_path) + '/' + metrics_file_name + '/'
        if not os.path.exists(save_path_folder): os.makedirs(save_path_folder)
        save_detailed_path = save_path_folder + model_name + '.csv'
        detailed_pd.to_csv(save_detailed_path, mode='a', index=False,
                           header=True, sep=';', decimal=',')
    
    return metrics_dict


def initialize_metrics_dict_multiclass() -> Dict[str, List]:
    '''
    Initialize metrics dictionary for multiclass classification.
    
    Returns:
        Dict[str, List]: Empty metrics dictionary.
    '''
    return {'model': [],
            'model_name': [],
            'which_features': [],
            'feature_type': [],
            'accuracy_majority_voting': [],
            'f1_score_majority_voting': [],
            'recall_majority_voting': [],
            'precision_majority_voting': [],
            'confusion_matrix_majority_voting': [],
            'auc_majority_voting': [],
            'accuracy_average_prob': [],
            'f1_score_average_prob': [],
            'recall_average_prob': [],
            'precision_average_prob': [],
            'confusion_matrix_average_prob': [],
            'auc_average_prob': []}
    
    
def determine_auc_multiclass(y_true: np.ndarray, 
                             y_scores: np.ndarray,
                             num_classes: int,
                             average: str = 'macro') -> float:
    '''
    Compute multiclass AUC using one-vs-rest approach.
    
    Args:
        y_true (np.ndarray):    True labels (0, 1, 2, 3, 4).
        y_scores (np.ndarray):  Predicted probabilities for each class (shape: n_samples x num_classes).
        num_classes (int):      Number of classes.
        average (str):          Averaging method - 'macro', 'weighted', or 'micro'.
    
    Returns:
        float: Computed multiclass AUC.
    '''

    # Binarize the labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    try:
        auc = roc_auc_score(y_true_bin, y_scores, average=average, multi_class='ovr')
        return auc
    except Exception:
        return 0.0


def determine_precision_multiclass(confusion_matrix: np.ndarray,
                                   average: str = 'macro') -> float:
    '''
    Determine precision for multiclass classification.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        average (str):                 Averaging method - 'macro', 'weighted', or 'micro'.
    
    Returns:
        float: Precision score.
    '''
    if average == 'micro':
        # Micro-average: sum all TP and divide by sum of (TP + FP)
        tp = np.diag(confusion_matrix).sum()
        fp = confusion_matrix.sum(axis=0).sum() - tp
        denominator = tp + fp
        return tp / denominator if denominator > 0 else 0.0
    
    elif average == 'macro':
        # Macro-average: average of per-class precision
        precisions = []
        for i in range(len(confusion_matrix)):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            denominator = tp + fp
            precisions.append(tp / denominator if denominator > 0 else 0.0)
        return np.mean(precisions)
    
    elif average == 'weighted':
        # Weighted average by support (number of true instances per class)
        precisions = []
        supports = confusion_matrix.sum(axis=1)
        total_support = supports.sum()
        
        for i in range(len(confusion_matrix)):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            denominator = tp + fp
            precision = tp / denominator if denominator > 0 else 0.0
            precisions.append(precision * supports[i])
        
        return sum(precisions) / total_support if total_support > 0 else 0.0


def determine_recall_multiclass(confusion_matrix: np.ndarray,
                                average: str = 'macro') -> float:
    '''
    Determine recall for multiclass classification.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        average (str):                 Averaging method - 'macro', 'weighted', or 'micro'.
    
    Returns:
        float: Recall score.
    '''
    if average == 'micro':
        # Micro-average: sum all TP and divide by sum of (TP + FN)
        tp = np.diag(confusion_matrix).sum()
        fn = confusion_matrix.sum(axis=1).sum() - tp
        denominator = tp + fn
        return tp / denominator if denominator > 0 else 0.0
    
    elif average == 'macro':
        # Macro-average: average of per-class recall
        recalls = []
        for i in range(len(confusion_matrix)):
            tp = confusion_matrix[i, i]
            fn = confusion_matrix[i, :].sum() - tp
            denominator = tp + fn
            recalls.append(tp / denominator if denominator > 0 else 0.0)
        return np.mean(recalls)
    
    elif average == 'weighted':
        # Weighted average by support
        recalls = []
        supports = confusion_matrix.sum(axis=1)
        total_support = supports.sum()
        
        for i in range(len(confusion_matrix)):
            tp = confusion_matrix[i, i]
            fn = confusion_matrix[i, :].sum() - tp
            denominator = tp + fn
            recall = tp / denominator if denominator > 0 else 0.0
            recalls.append(recall * supports[i])
        
        return sum(recalls) / total_support if total_support > 0 else 0.0


def determine_f1score_multiclass(confusion_matrix: np.ndarray,
                                 average: str = 'macro') -> float:
    '''
    Determine F1 score for multiclass classification.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix.
        average (str):                 Averaging method - 'macro', 'weighted', or 'micro'.
    
    Returns:
        float: F1 score.
    '''
    precision = determine_precision_multiclass(confusion_matrix, average)
    recall = determine_recall_multiclass(confusion_matrix, average)
    
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    
    return 2 * (precision * recall) / denominator


def get_group_metrics_multiclass(metrics_dict: Dict[str, float],
                                 test_predictions: np.ndarray,
                                 test_probabilities: np.ndarray,
                                 test_labels: np.ndarray) -> Tuple[np.ndarray, 
                                                                   np.ndarray, 
                                                                   np.ndarray]:
    '''
    Calculate group metrics for multiclass test predictions and probabilities.
    
    Args:
        metrics_dict (Dict[str, float]): Dictionary to store metrics.
        test_predictions (np.ndarray):   Array of test predictions.
        test_probabilities (np.ndarray): Array of test probabilities (n_samples x num_classes).
        test_labels (np.ndarray):        Array of test labels.
    
    Returns:
        Tuple [np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - majority_array:       Array of majority voting results.
            - argmax_array:         Array of argmax predictions from average probabilities.
            - average_prob_array:   Array of average probabilities (n_groups x num_classes).
    '''
    majority_list, average_prob_list = [], []
    people_to_evaluate = np.unique(test_labels[:, 1])
    
    for person_to_evaluate in people_to_evaluate:
        # Index the current label group
        person_index = np.nonzero(np.isin(test_labels[:, 1], person_to_evaluate))[0]
        person_predictions = test_predictions[person_index]
        person_probabilities = test_probabilities[person_index]
        person_labels = test_labels[person_index]
        new_person_labels = np.delete(person_labels, 1, axis=1)
        
        # Check if there is another group in the labels to be evaluated
        if new_person_labels.shape[1] > 1:
            (person_predictions,
             _,
             average_prob) = get_group_metrics_multiclass(metrics_dict = metrics_dict,
                                                          test_predictions = person_predictions,
                                                          test_probabilities = person_probabilities,
                                                          test_labels = new_person_labels)
        else:
            # Average probability across all samples for this person
            average_prob = person_probabilities
        
        # Majority voting
        majority_list.append(majority_voting(person_predictions))
        
        # Average probability across samples (mean of probability distributions)
        average_prob_list.append(np.mean(average_prob, axis=0))
    
    # Convert lists to NumPy arrays
    majority_array = np.array(majority_list)
    average_prob_array = np.array(average_prob_list)
    
    # Get argmax predictions from average probabilities
    argmax_array = np.argmax(average_prob_array, axis=1)
    
    return majority_array, argmax_array, average_prob_array


def determine_metrics_multiclass(metrics_dict: Dict[str, float],
                                 test_predictions: np.ndarray,
                                 test_probabilities: np.ndarray,
                                 test_labels: np.ndarray,
                                 detailed_metrics: bool,
                                 average: str = 'macro') -> Tuple[Dict[str, float], 
                                                                  pd.DataFrame]:
    '''
    Determine metrics for multiclass classification with group-level evaluation.
    
    Args:
        metrics_dict (Dict[str, float]): Dictionary storing evaluation metrics.
        test_predictions (np.ndarray):   Predictions made by the model.
        test_probabilities (np.ndarray): Probabilities predicted by the model (n_samples x num_classes).
        test_labels (np.ndarray):        True labels and group identifiers.
        detailed_metrics (bool):         Whether to save detailed predictions.
        average (str):                   Averaging method for metrics ('macro', 'weighted', 'micro').
    
    Returns:
        tuple[Dict[str, float], pd.DataFrame]:
            - metrics_dict: Updated metrics dictionary.
            - detailed_pd:  DataFrame with detailed predictions.
    '''
    # Get group-level predictions
    (majority_predictions,
     argmax_predictions,
     average_prob_pred) = get_group_metrics_multiclass(metrics_dict = metrics_dict,
                                                       test_predictions = test_predictions,
                                                       test_probabilities = test_probabilities,
                                                       test_labels = test_labels)
    
    # Get the group labels
    group_unique = np.unique(test_labels[:, 1])
    group_labels = np.array([float(test_labels[test_labels[:, 1] == label][0][0]) 
                            for label in group_unique])
    
    # Infer number of classes from the labels
    num_classes = len(np.unique(group_labels))
    
    # Majority voting metrics
    majority_voting_cm = confusion_matrix(group_labels, majority_predictions, labels = range(num_classes))
    metrics_dict['accuracy_majority_voting'].append(determine_accuracy(majority_voting_cm))
    metrics_dict['f1_score_majority_voting'].append(determine_f1score_multiclass(majority_voting_cm, average))
    metrics_dict['recall_majority_voting'].append(determine_recall_multiclass(majority_voting_cm, average))
    metrics_dict['precision_majority_voting'].append(determine_precision_multiclass(majority_voting_cm, average))
    metrics_dict['confusion_matrix_majority_voting'].append(majority_voting_cm.flatten())
    metrics_dict['auc_majority_voting'].append(determine_auc_multiclass(group_labels, average_prob_pred, num_classes, average))
    
    # Average probability (argmax) metrics
    average_prob_cm = confusion_matrix(group_labels, argmax_predictions, labels = range(num_classes))
    metrics_dict['accuracy_average_prob'].append(determine_accuracy(average_prob_cm))
    metrics_dict['f1_score_average_prob'].append(determine_f1score_multiclass(average_prob_cm, average))
    metrics_dict['recall_average_prob'].append(determine_recall_multiclass(average_prob_cm, average))
    metrics_dict['precision_average_prob'].append(determine_precision_multiclass(average_prob_cm, average))
    metrics_dict['confusion_matrix_average_prob'].append(average_prob_cm.flatten())
    metrics_dict['auc_average_prob'].append(determine_auc_multiclass(group_labels, average_prob_pred, num_classes, average))
    
    # Detailed metrics
    if detailed_metrics:
        group_to_prediction = {label: (maj_pred, avg_pred) 
                               for label, maj_pred, avg_pred in zip(group_unique, 
                                                                    majority_predictions,
                                                                    argmax_predictions)}
        
        test_ids = test_labels[:, 1]
        true_labels = test_labels[:, 0].astype(float)
        majority_pred_col = np.array([group_to_prediction[id][0] for id in test_ids])
        argmax_pred_col = np.array([group_to_prediction[id][1] for id in test_ids])
        correct_majority = (majority_pred_col == true_labels).astype(int)
        correct_argmax = (argmax_pred_col == true_labels).astype(int)
        
        detailed_pd = pd.DataFrame({'id': test_ids,
                                    'true_label': true_labels,
                                    'predicted_class': test_predictions,
                                    'majority_predictions': majority_pred_col,
                                    'argmax_predictions': argmax_pred_col,
                                    'correct_majority': correct_majority,
                                    'correct_argmax': correct_argmax})
        
        # Add probability columns for each class
        for i in range(num_classes):
            detailed_pd[f'prob_class_{i}'] = test_probabilities[:, i]
    else:
        detailed_pd = pd.DataFrame()
    
    return metrics_dict, detailed_pd


def save_metrics_to_csv(metrics_dict: Dict[str, float], 
                        model: str,
                        model_name: str, 
                        which_feature: List[str], 
                        feature_type: str, 
                        save_metrics_path: str, 
                        detailed_pd: pd.DataFrame, 
                        detailed_metrics: bool,
                        threshold_results: Dict[str, List[float]] | None = None,
                        save_threshold_results: bool = False) -> Dict[str, float]:
    '''
    Save evaluation metrics to a CSV file.

    Args:
        metrics_dict (Dict[str, float]): Dictionary containing evaluation metrics.
        model (str):                     Model used for training.
        model_name (str):                Name of the model.
        which_feature (List[str]):       List of features used in training.
        feature_type (str):              Feature type used in trainig.
        save_metrics_path (str):         Path to save the metrics CSV file.
        detailed_pd (pd.DataFrame):      DataFrame containing detailed predictions and correctness checks.
        detailed_metrics (bool):         Whether to save detailed metrics.

    Returns:
        metrics_dict (Dict[str, float]): Empty dictionary to be filled with new metrics.
    '''
    
    # Save variables to dictionary
    metrics_dict['model'].append(model)
    metrics_dict['model_name'].append(model_name)
    metrics_dict['which_features'].append(which_feature)
    metrics_dict['feature_type'].append(feature_type)
                    
    # Convert metrics dictionary to DataFrame
    metrics_pd = pd.DataFrame(metrics_dict)
    
    # Define the path to save the metrics
    save_path = save_metrics_path
    
    # Check if the file exists and save accordingly
    if not os.path.isfile(save_path):
        metrics_pd.to_csv(save_path, mode='a', index=False, 
                          header=True, sep=';', decimal=',')
    else:
        metrics_pd.to_csv(save_path, mode='a', index=False, 
                          header=False, sep=';', decimal=',')
        
    metrics_dict = initialize_metrics_dict()
    
    # Check if detailed CSV is required
    if detailed_metrics:
        detailed_pd['model'] = model
        detailed_pd['model_name'] = model_name
        detailed_pd['which_features'] = which_feature
        if 'no_acoustic_feature' in feature_type: feature_type = 'no_acoustic_feature'
        detailed_pd['feature_type'] = feature_type
        metrics_file_name = os.path.basename(save_path)[:-4]
        save_path_folder = os.path.dirname(save_path) + '/' + metrics_file_name + '/'
        if not os.path.exists(save_path_folder): os.makedirs(save_path_folder)
        save_detailed_path = save_path_folder + model_name + '.csv'
        detailed_pd.to_csv(save_detailed_path, mode='a', 
                           index=False, header=True, sep=';', decimal=',')
        
    # Save threshold results if provided
    if save_threshold_results and threshold_results:
        threshold_results['model'] = model
        threshold_results['model_name'] = model_name
        threshold_results['which_features'] = which_feature
        if 'no_acoustic_feature' in feature_type: feature_type = 'no_acoustic_feature'
        threshold_results['feature_type'] = feature_type
        threshold_results_df = pd.DataFrame(threshold_results)
        metrics_file_name = os.path.basename(save_path)[:-4]
        save_path_folder = os.path.dirname(save_path) + '/' + metrics_file_name + '/'
        if not os.path.exists(save_path_folder): os.makedirs(save_path_folder)
        threshold_results_path = save_path_folder + model_name + '_threshold_results.csv'
        threshold_results_df.to_csv(threshold_results_path, mode='a', 
                                    index=False, header=True, sep=';', decimal=',')
    
    return metrics_dict


def determine_auc(y_true: np.ndarray, 
                  y_scores: np.ndarray) -> float:
    '''
    Manually compute the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.ndarray):    True binary labels (0 or 1).
        y_scores (np.ndarray):  Predicted probabilities for the positive class.

    Returns:
        float: Computed AUC.
    '''
    # Sort by predicted scores (descending)
    desc_score_indices = np.argsort(-y_scores)
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]

    # Count positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    if P == 0 or N == 0:
        return float(0)  # AUC is undefined when only one class is present

    # Initialize TPR and FPR lists
    tpr_list = [0.0]
    fpr_list = [0.0]
    
    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / P
        fpr = fp / N
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Trapezoidal rule for area under curve
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


def determine_precision(confusion_matrix: np.ndarray) -> float:
    '''
    Determine the precision based on the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.

    Returns:
        precision (float): Precision based on the confusion matrix.
    '''
    denominator = confusion_matrix[1, 1] + confusion_matrix[0, 1]
    if denominator == 0: return 0.0  
    precision = confusion_matrix[1, 1] / denominator
    return precision


def determine_recall(confusion_matrix: np.ndarray) -> float:
    '''
    Determine the recall based on the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.

    Returns:
        recall (float): Recall based on the confusion matrix.
    '''
    denominator = confusion_matrix[1, 1] + confusion_matrix[1, 0]
    if denominator == 0: return 0.0  
    recall = confusion_matrix[1, 1] / denominator
    return recall


def determine_f1score(confusion_matrix: np.ndarray) -> float:
    '''
    Determine the F1 score based on the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.

    Returns:
        f1_score (float): F1 score based on the confusion matrix.
    '''
    denominator = 2 * confusion_matrix[1, 1] + confusion_matrix[0, 1] + confusion_matrix[1, 0]
    if denominator == 0: return 0.0  
    f1_score = 2 * confusion_matrix[1, 1] / denominator
    return f1_score


def determine_accuracy(confusion_matrix: np.ndarray) -> float:
    '''
    Determine the accuracy based on the confusion matrix.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix.

    Returns:
        accuracy (float): Accuracy based on the confusion matrix.
    '''
    denominator = np.sum(confusion_matrix)
    if denominator == 0: return 0.0  
    accuracy = np.sum(np.diag(confusion_matrix)) / denominator
    return accuracy


def majority_voting(labels: np.ndarray) -> float:
    '''
    Perform majority voting on an array of labels.
    The label that appears most frequently in the array. If there is a tie, 
    the label that appears first in the list of most common labels is returned.

    Args:
        labels (np.ndarray):  A array of labels to perform majority voting on.

    Returns:
        result (float):       The label that appears most frequently in the array.
    '''
    unique_labels, counts = np.unique(labels, return_counts=True)
    result = float(unique_labels[np.argmax(counts)])
    return result


def get_group_metrics(metrics_dict: Dict[str, float], 
                      test_predictions: np.ndarray, 
                      test_probabilities: np.ndarray, 
                      predictions_threshold: float | str,
                      test_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Calculate group metrics for test predictions and probabilities.

    Args:
        metrics_dict (Dict[str, float]):     Dictionary to store metrics.
        test_predictions (np.ndarray):       Array of test predictions.
        test_probabilities (np.ndarray):     Array of test probabilities.
        predictions_threshold (float | str): Threshold for determining predictions.
        test_labels (np.ndarray):            Array of test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - majority_array:                Array of majority voting results.
            - threshold_array:               Array of threshold majority voting results.
            - average_prob_array:            Array of average probabilities.
    '''
    majority_list, threshold_list, average_prob_list = [], [], []
    people_to_evaluate = np.unique(test_labels[:, 1])
    for person_to_evaluate in people_to_evaluate:
        
        # Index the current label group
        person_index = np.nonzero(np.isin(test_labels[:, 1], person_to_evaluate))[0]
        person_predictions = test_predictions[person_index]
        person_probabilities = test_probabilities[person_index]
        person_labels = test_labels[person_index]
        new_person_labels = np.delete(person_labels, 1, axis=1)
        
        # Check if there is another group in the labels to be evaluated
        if new_person_labels.shape[1] > 1:
            (person_predictions, 
             person_threshold, 
             average_prob) = get_group_metrics(metrics_dict = metrics_dict, 
                                               test_predictions = person_predictions,
                                               test_probabilities = person_probabilities,
                                               test_labels = new_person_labels, 
                                               predictions_threshold = predictions_threshold)
        else:
            
            # Determination of threshold and probabilities to be evaluated
            mask_threshold = person_probabilities[:, 0] > predictions_threshold
            person_threshold = np.where(mask_threshold, 0, 1)
            average_prob = person_probabilities[:, 1]
            
        # Majority voting
        majority_list.append(majority_voting(person_predictions))
        
        # Threshold majority voting predictions
        threshold_list.append(majority_voting(person_threshold))
        
        # Average probability
        average_prob_list.append(np.mean(average_prob))
    
    # Convert lists to NumPy arrays
    majority_array = np.array(majority_list)
    threshold_array = np.array(threshold_list)
    average_prob_array = np.array(average_prob_list)
            
    return majority_array, threshold_array, average_prob_array


def determine_metrics(metrics_dict: Dict[str, float], 
                      test_predictions: np.ndarray, 
                      test_probabilities: np.ndarray, 
                      predictions_threshold: float | str,
                      test_labels: np.ndarray, 
                      detailed_metrics: bool) -> tuple[Dict[str, float], 
                                                   pd.DataFrame]:
    '''
    Determine the metrics dictionary with group specific evaluation metrics.
    Majority voting, threshold majority voting and average probability.

    Args:
        metrics_dict (Dict[str, float]):       Dictionary storing evaluation metrics.
        test_predictions (np.ndarray):         Predictions made by the model.
        test_probabilities (np.ndarray):       Probabilities predicted by the model.
        test_labels (np.ndarray):              True labels of the test data and its original file path.
        predictions_threshold (float | str):   Threshold for succesful predictions.
        detailed_metrics (bool):               Whether to save predictions and correctness checks to a DataFrame.

    Returns:
        tuple[Dict[str, float], pd.DataFrame]:
            - metrics_dict (Dict[str, float]): Updated metrics dictionary.
            - detailed_pd (pd.DataFrame):      DataFrame containing detailed predictions and correctness checks.
    '''
    (majority_predictions, 
     threshold_predictions, 
     average_prob_pred) = get_group_metrics(metrics_dict = metrics_dict, 
                                            test_predictions = test_predictions,
                                            test_probabilities = test_probabilities,
                                            test_labels = test_labels, 
                                            predictions_threshold = predictions_threshold)
    
    # Change probabilities into predictions
    probabilities_pred = np.round(average_prob_pred)
    
    # Get the group labels from the individual labels
    group_unique = np.unique(test_labels[:, 1])
    group_labels = np.array([float(test_labels[test_labels[:, 1] == label][0][0]) for label in group_unique])
        
    # Majority voting metrics
    majority_voting_cm = confusion_matrix(group_labels, majority_predictions)
    metrics_dict['accuracy_majority_voting'].append(determine_accuracy(majority_voting_cm))
    metrics_dict['f1_score_majority_voting'].append(determine_f1score(majority_voting_cm))
    metrics_dict['recall_majority_voting'].append(determine_recall(majority_voting_cm))
    metrics_dict['precision_majority_voting'].append(determine_precision(majority_voting_cm))
    metrics_dict['confusion_matrix_majority_voting'].append(majority_voting_cm.flatten())
    metrics_dict['auc_majority_voting'].append(determine_auc(y_true = group_labels,
                                                             y_scores = majority_predictions))
    
    # Threshold majority voting metrics
    threshold_cm = confusion_matrix(group_labels, threshold_predictions)
    metrics_dict['threshold'].append(predictions_threshold)
    metrics_dict['accuracy_threshold'].append(determine_accuracy(threshold_cm))
    metrics_dict['f1_score_threshold'].append(determine_f1score(threshold_cm))
    metrics_dict['recall_threshold'].append(determine_recall(threshold_cm))
    metrics_dict['precision_threshold'].append(determine_precision(threshold_cm))
    metrics_dict['confusion_matrix_threshold'].append(threshold_cm.flatten())
    metrics_dict['auc_threshold'].append(determine_auc(y_true = group_labels,
                                                       y_scores = threshold_predictions))    
    
    # Average probability metrics
    average_prob_cm = confusion_matrix(group_labels, probabilities_pred)    
    metrics_dict['accuracy_average_prob'].append(determine_accuracy(average_prob_cm))
    metrics_dict['f1_score_average_prob'].append(determine_f1score(average_prob_cm))
    metrics_dict['recall_average_prob'].append(determine_recall(average_prob_cm))
    metrics_dict['precision_average_prob'].append(determine_precision(average_prob_cm))
    metrics_dict['confusion_matrix_average_prob'].append(average_prob_cm.flatten())
    metrics_dict['auc_average_prob'].append(determine_auc(y_true = group_labels,
                                                          y_scores = probabilities_pred))
    
    # Check if detailed CSV is required
    if detailed_metrics:
        
        # Map group predictions to individual labels
        group_to_prediction = {label: (maj_pred, thr_pred, avg_pred) 
                               for label, maj_pred, thr_pred, avg_pred in zip(group_unique, 
                                                                              majority_predictions, 
                                                                              threshold_predictions, 
                                                                              probabilities_pred)}
        # Determine correctness for group-level predictions for each sample
        test_ids = test_labels[:, 1]
        true_labels = test_labels[:, 0].astype(float)
        majority_pred_col = np.array([group_to_prediction[id][0] for id in test_ids])
        threshold_pred_col = np.array([group_to_prediction[id][1] for id in test_ids])
        avg_prob_pred_col = np.array([group_to_prediction[id][2] for id in test_ids])
        correct_majority = (majority_pred_col == true_labels).astype(int)
        correct_threshold = (threshold_pred_col == true_labels).astype(int)
        correct_avg_prob = (avg_prob_pred_col == true_labels).astype(int)
        
        # Create detailed DataFrame
        mask_threshold = test_probabilities[:, 0] > predictions_threshold
        threshold_true_pred_col = np.where(mask_threshold, 0, 1)
        detailed_pd = pd.DataFrame({'id': test_ids,
                                    'true_label': true_labels,
                                    'average_prob': test_probabilities[:, 0],
                                    'majority_predictions': test_predictions,
                                    'threshold': threshold_true_pred_col,
                                    'correct_majority': correct_majority,
                                    'correct_threshold': correct_threshold,
                                    'correct_avg_prob': correct_avg_prob})
    else:
        detailed_pd = pd.DataFrame()
        
    return metrics_dict, detailed_pd


def find_optimal_threshold(test_predictions: np.ndarray,
                          test_probabilities: np.ndarray,
                          test_labels: np.ndarray,
                          threshold_range: np.ndarray = None,
                          optimization_metric: str = 'f1_score',
                          voting_method: str = 'threshold') -> Tuple[float, 
                                                                     Dict[str, 
                                                                     List[float]]]:
    '''
    Find the optimal predictions_threshold by testing different values.
    
    Args:
        test_predictions (np.ndarray):   Array of test predictions.
        test_probabilities (np.ndarray): Array of test probabilities.
        test_labels (np.ndarray):        Array of test labels.
        threshold_range (np.ndarray):    Range of thresholds to test. If None, uses np.arange(0.1, 1.0, 0.05).
        optimization_metric (str):       Metric to optimize ('f1_score', 'accuracy', 'recall', 'precision', 'auc').
        voting_method (str):             Which voting method to optimize ('threshold', 'majority_voting', 'average_prob').
    
    Returns:
        Tuple[float, Dict[str, List[float]]]: Best threshold and metrics for all tested thresholds.
            - best_threshold (float): The best threshold found.
            - threshold_results (Dict[str, List[float]]): Dictionary containing metrics for all tested thresholds.
    '''
    
    if threshold_range is None:
        threshold_range = np.arange(0.1, 1.0, 0.05)
    
    # Storage for results
    threshold_results = {'thresholds': [],
                         'accuracy': [],
                         'f1_score': [],
                         'recall': [],
                         'precision': [],
                         'auc': [],
                         'confusion_matrix': []}
    
    best_threshold = threshold_range[0]
    best_score = -np.inf
    
    for threshold in threshold_range:
        # Initialize metrics dict for this threshold
        metrics_dict = initialize_metrics_dict()
        
        # Calculate metrics using current threshold
        metrics_dict, _ = determine_metrics(metrics_dict = metrics_dict,
                                            test_predictions = test_predictions,
                                            test_probabilities = test_probabilities,
                                            predictions_threshold = threshold,
                                            test_labels = test_labels,
                                            detailed_metrics = False)
        
        # Extract the metric for the chosen voting method
        metric_key = f'{optimization_metric}_{voting_method}'
        current_score = metrics_dict[metric_key][0]  # First (and only) element
        
        # Store results
        threshold_results['thresholds'].append(threshold)
        threshold_results['accuracy'].append(metrics_dict[f'accuracy_{voting_method}'][0])
        threshold_results['f1_score'].append(metrics_dict[f'f1_score_{voting_method}'][0])
        threshold_results['recall'].append(metrics_dict[f'recall_{voting_method}'][0])
        threshold_results['precision'].append(metrics_dict[f'precision_{voting_method}'][0])
        threshold_results['auc'].append(metrics_dict[f'auc_{voting_method}'][0])
        threshold_results['confusion_matrix'].append(metrics_dict[f'confusion_matrix_{voting_method}'][0])

        # Check if this is the best threshold so far
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
    
    return best_threshold, threshold_results


def determine_metrics_with_optimal_threshold(test_predictions: np.ndarray,
                                             test_probabilities: np.ndarray,
                                             test_labels: np.ndarray,
                                             detailed_metrics: bool = False,
                                             threshold_range: np.ndarray = None,
                                             optimization_metric: str = 'f1_score',
                                             voting_method: str = 'threshold') -> Tuple[Dict[str, float], 
                                                                                        pd.DataFrame, 
                                                                                        Dict[str, List[float]]]:
    '''
    Wrapper function that finds optimal threshold and returns metrics using that threshold.
    
    Args:
        test_predictions (np.ndarray): Array of test predictions.
        test_probabilities (np.ndarray): Array of test probabilities.
        test_labels (np.ndarray): Array of test labels.
        detailed_metrics (bool): Whether to return detailed metrics DataFrame.
        threshold_range (np.ndarray): Range of thresholds to test.
        optimization_metric (str): Metric to optimize.
        voting_method (str): Which voting method to optimize.
    
    Returns:
        Tuple[Dict[str, float], pd.DataFrame, float]: Metrics dict, detailed DataFrame, optimal threshold.
            - metrics_dict (Dict[str, float]): Dictionary containing evaluation metrics.
            - detailed_pd (pd.DataFrame): DataFrame containing detailed predictions and correctness checks.
            - threshold_results (Dict[str, List[float]]): Dictionary containing metrics for all tested thresholds.
    '''
    
    # Find optimal threshold
    optimal_threshold, threshold_results = find_optimal_threshold(test_predictions = test_predictions,
                                                                  test_probabilities = test_probabilities,
                                                                  test_labels = test_labels,
                                                                  threshold_range = threshold_range,
                                                                  optimization_metric = optimization_metric,
                                                                  voting_method = voting_method)
    
    # Calculate final metrics with optimal threshold
    metrics_dict = initialize_metrics_dict()
    metrics_dict, detailed_pd = determine_metrics(metrics_dict = metrics_dict,
                                                  test_predictions = test_predictions,
                                                  test_probabilities = test_probabilities,
                                                  predictions_threshold = optimal_threshold,
                                                  test_labels = test_labels,
                                                  detailed_metrics = detailed_metrics)
    
    return metrics_dict, detailed_pd, threshold_results


def get_model_input_size(model_builder: ModelBuilder) -> tuple:
    '''
    Get the model input size.

    Args:
        model_builder (ModelBuilder): Model builder object. 

    Returns:
        model_input_size (tuple):    Model input size.
    '''
    if model_builder.name == 'LogisticRegression' or model_builder.name == 'LinearSVM':
        model_input_size = model_builder.model.coef_.shape
    elif model_builder.name == 'MLP':
        model_input_size = model_builder.model.coefs_[0].shape
    elif model_builder.name == 'CNN':
        model_input_size = model_builder.model.fc1.in_features
    else:
        model_input_size = model_builder.model.n_features_in_
    return model_input_size


def determine_model_name(model_conf: dict,
                         which_feature: str,
                         feature_type_string: str,
                         model: str,
                         fold: str) -> str:
    '''
    Generate a model name based on the given parameters.

    Args:
        model_conf (dict):         The model configuration.
        which_feature (str):       The feature type.
        feature_type_string (str): A string of feature types.
        model (str):               The model name.
        fold (int):                The fold number.

    Returns:
        model_name (str):          The generated model name.
    '''
    model_name = which_feature + '_'
    if model_conf.get('use_acoustic_feat', False):
        model_name += feature_type_string
    if model_conf.get('use_paralinguistic_feat', False):
        model_name += 'paralinguistic_'
    if model_conf.get('use_linguistic_feat', False):
        model_name += 'linguistic_'
    if model_conf.get('use_ssl_hubert', False):
        model_name += 'ssl_hubert_'
    if model_conf.get('use_ssl_wavlm', False):
        model_name += 'ssl_wavlm_'
    if model_conf.get('use_ssl_wav2vec', False):
        model_name += 'ssl_wav2vec_'
    model_name += model + '_' + str(fold)
    return model_name


def create_train_test_split(model_conf: Dict[str, Any],
                            model: str,
                            which_feature: str,
                            split_by: List[str],
                            model_batch: bool,
                            stratified: bool = False,
                            labels_list: List[str] = []) -> List[Tuple[np.ndarray, np.ndarray]] | KFold:
    '''
    Create train/test split indices based on model configuration.

    Args:
        model_conf (Dict[str, Any]):  Model configuration dictionary.
        model (str):                  Name of the model type.
        which_feature (str):          Feature type used in training.
        split_by (List[str]):         List of sample IDs to split.
        model_batch (bool):           Whether the model is batch-based.
        stratified (bool):            Whether to use stratified KFold.  
        labels_list (List[str]):      List of labels for applying stratified KFold.
        
    Returns:
        Empty list, List[Tuple[np.ndarray, np.ndarray]] or KFold object:
            Empty list:                          No split required
            List[Tuple[np.ndarray, np.ndarray]]: A list of train/test index tuples.
                - train_indices (np.ndarray):    Training indices
                - val_indices (np.ndarray):      Validation indices 
            Kfold object:                        The KFold object generator
    '''

    # Case 1: Model is batched and CNN model is paired with frame feature or model training is disabled
    if model_batch and ((model == 'CNN' and which_feature == 'frame') or not model_conf['train_model']): 
        train_test_split = []
        
    # Case 2: Model is not batched and CNN/MLP models or model training is disabled
    elif not model_batch and ((model == 'CNN' or model == 'MLP') or not model_conf['train_model']): 
        train_test_split = []

    # Case 3: Use validation CSV
    elif model_conf.get('use_val_csv_no_kfold', False):
        val_data = pd.read_csv( model_conf['validation_csv_path'], delimiter=';', dtype={'ID': str} )['ID'].tolist()
        val_indices = np.array([i for i, id_ in enumerate(split_by) if id_ in val_data])
        train_indices = np.array([i for i, id_ in enumerate(split_by) if id_ not in val_data])
        train_test_split = [(train_indices, val_indices)]

    # Case 4: Use all indices for both training and testing (no split)
    elif model_conf.get('kfold_splits', 1) <= 1:
        all_indices = np.arange(len(split_by))
        train_test_split = [(all_indices, all_indices)]

    # Case 5: Use stratified KFold cross-validation
    elif stratified:
        kf = StratifiedKFold(n_splits = model_conf['kfold_splits'], shuffle = True, random_state = 42)
        train_test_split = kf.split(split_by, labels_list)
    
    # Case 6: Use KFold cross-validation
    else:
        kf = KFold(n_splits = model_conf['kfold_splits'], shuffle = True, random_state = 42)
        train_test_split = kf.split(split_by)

    return train_test_split


def get_data_to_split(labels: dict) -> Tuple[list, list]:
    '''
    Prepares data for splitting based on the model configuration and labels.

    Args:
        labels (dict):     A dictionary where keys are file paths and values are labels.

    Returns:
        labels_list(list): A list of labels from the labels dictionary, potentially modified based on the dataset name in the model configuration.
            - split_by(list): A list of keys from the labels dictionary.
            - labels_list(list): A list of labels from the labels dictionary.
    '''
    new_labels = {re.match(r'^[^\-/_.,\s]+', os.path.basename(key)).group(0): value for key, value in labels.items()}
    split_by = list(new_labels.keys())
    labels_list = list(new_labels.values())
    return split_by, labels_list


def process_labels(dataset: pd.DataFrame) -> dict:
    '''
    Process the labels according to the dataset. Also change to 0 labels that are less than zero

    Args:
        dataset (pd.DataFrame): The input dataset containing 'patient_type' and 'file_path' columns.
        model_config (dict):    The model configuration dictionary.

    Returns:
        labels (dict):          A dictionary containing the ID and labels.
    '''
    dataset['patient_type'] = dataset['patient_type'].astype(float).apply(lambda x: 0 if x < 0 else x)
    labels = dataset.set_index('file_path')['patient_type'].to_dict()
    return labels


def generate_feature_combinations(config: Dict[str, Any]) -> List[List[str]]:
    '''
    Generate all possible combinations of feature types.

    Args:
        config (Dict[str, Any]):            Configuration dictionary.

    Returns:
        all_combinations (List[List[str]]): List of feature type combinations.
    '''
    
    # Get all features types and generate all possible combinations
    feature_types = config['feature_type']
    all_combinations = []
    for r in range(1, len(feature_types) + 1):
        all_combinations.extend(combinations(feature_types, r))
    
    # Convert each combination from tuple to list
    all_combinations = [list(comb) for comb in all_combinations]
    
    return all_combinations


def select_features_to_train(model_conf: Dict[str, Any],
                             audioprocessor_data: AudioProcessor) -> List[list]:
    '''
    Select features to be trained based on the model configuration.

    Args:
        model_conf (Dict):                      The model configuration dictionary.
        audioprocessor_data (AudioProcessor):   The audio processor data dictionary.

    Returns:
        feature_combinations(List[list]):       A list of feature combinations to be trained.
    '''
    if_train_comb_in_config = 'train_each_feat_combination' in model_conf.keys()
    if_train_acoustic_feat = model_conf['use_acoustic_feat']    
    if if_train_comb_in_config and model_conf['train_each_feat_combination'] and if_train_acoustic_feat:
        feature_combinations = generate_feature_combinations(config = audioprocessor_data)
    elif not if_train_acoustic_feat and (model_conf['use_linguistic_feat'] or model_conf['use_paralinguistic_feat']):
        feature_combinations = [['no_acoustic_feature']]
    else:
        feature_combinations = [audioprocessor_data['feature_type']]
    
    return feature_combinations


def determine_speaking_silence_timeline(speaking_times: np.ndarray, 
                                        silent_times: np.ndarray,
                                        audio_length: int) -> np.ndarray:
    '''
    Generates a timeline array based on given time intervals. Where each frame in a speaking segment adds 1 to the timeline array.
    Where each frame in a silent segment adds -1 to the timeline array.

    Args:
        speaking_times (np.ndarray): An array of speaking intervals where each interval is represented as a tuple (start, end).
        silent_times (np.ndarray):   An array of silent intervals where each interval is represented as a tuple (start, end).
        audio_length (int):          The total length of the audio.

    Returns:
        timeline (np.ndarray): A timeline array where each speaking segment adds 1 and each silent segment adds -1.
    '''
    
    # Initialize the timeline array with zeros
    timeline = np.zeros(audio_length)
    
    # Combine starts into one array
    starts_speaking = speaking_times[0]
    starts_silent = silent_times[0]
    starts = np.concatenate((starts_speaking, starts_silent))
    
    # Combine ends into one array
    ends_speaking = speaking_times[1]
    ends_silent = silent_times[1]
    ends = np.concatenate((ends_speaking, ends_silent))
    
    # Initialize the positive (speaking) and negative (silent) cumsum modifier
    cumsum_modifier = np.concatenate((np.ones(len(starts_speaking)), -1 * np.ones(len(starts_silent))))
    
    # Sort the speaking and silent values
    sorted_indices = np.argsort(starts)
    starts = starts[sorted_indices]
    ends = ends[sorted_indices]
    cumsum_modifier = cumsum_modifier[sorted_indices]
    
    for index, (start, end, sum_modifier) in enumerate(zip(starts, ends, cumsum_modifier)):
        
        # If the end is greater than the audio length, set it to the audio length
        if end >= audio_length: end = audio_length - 1
        
        # Create the array to add 1 each in frame of the segment
        cumsum = sum_modifier*(np.arange(end - start) + 1)
    
        # Accumulate over previous value in the array
        timeline[start:end] = timeline[start - 1] + cumsum
               
        # Fill the gaps with the previous value
        if index + 1 <= len(starts) - 1:
            timeline[end:starts[index + 1]] = timeline[end - 1]
        else:
            timeline[end:] = timeline[end - 1]
    
    return timeline


def determine_timeline(times: np.ndarray, 
                       audio_length: int) -> np.ndarray:
    '''
    Generates a timeline array based on given time intervals. Where each frame in a segment adds 1 to the timeline array.

    Args:
        times (np.ndarray):    An array of time intervals where each interval is represented as a tuple (start, end).
        audio_length (int):    The total length of the audio.

    Returns:
        timeline (np.ndarray): A timeline array where each segment between start and end is filled with cumulative sums.
    '''
    
    # Initialize the timeline array with zeros
    timeline = np.zeros(audio_length)
    starts = times[0]
    ends = times[1]

    for index, (start, end) in enumerate(zip(starts, ends)):
        
        # If the end is greater than the audio length, set it to the audio length
        if end >= audio_length: end = audio_length - 1
        
        # Create the array to add 1 each in frame of the segment
        cumsum = np.arange(end - start) + 1
        
        # Accumulate over previous value in the array
        timeline[start:end] = timeline[start - 1] + cumsum
        
        # Fill the gaps with the previous value
        if index + 1 <= len(starts) - 1:
            timeline[end:starts[index + 1]] = timeline[end - 1]
        else:
            timeline[end:] = timeline[end - 1]
    
    return timeline
        

def calculate_silent_segments(speaking_times: np.ndarray, 
                              audio_length: int) -> np.ndarray:
    '''
    Calculate the silent segments in an audio file given the speaking times.

    Args:
        speaking_times (np.ndarray): A 2D numpy array where the first row contains the start times of speaking words and the second row contains the end times of speaking words.
        audio_length (int):          The total length of the audio file in the same units as speaking_times.

    Returns:
        silent_times (np.ndarray):   A 2D numpy array where the first row contains the start times of silent segments and the second row contains the end times of silent segments.
    '''
    # Initialize the lists
    start_silence, end_silence = [], []
    
    # Calculate silence before the first segment
    first_segment_start_time = speaking_times[0][0]
    if first_segment_start_time > 0:
        start_silence.append(0)
        end_silence.append(first_segment_start_time)

    # Calculate silence between segments
    for i in range(len(speaking_times[0]) - 1):
        current_segment_end_time = speaking_times[1][i]
        next_segment_start_time = speaking_times[0][i + 1]
        silence_duration = next_segment_start_time - current_segment_end_time
        if silence_duration > 0:
            start_silence.append(current_segment_end_time)
            end_silence.append(next_segment_start_time)

    # Calculate silence after the last segment
    last_segment_end_time = speaking_times[1][-1]
    last_silence_duration = audio_length - last_segment_end_time
    if last_silence_duration > 0:
        start_silence.append(last_segment_end_time)
        end_silence.append(audio_length)

    # Vertically stack the lists
    silent_times = np.vstack((np.array(start_silence), np.array(end_silence)))
    
    return silent_times


def determine_timelines(transcript_path: str) -> Tuple[np.ndarray, 
                                                       np.ndarray, 
                                                       np.ndarray,
                                                       np.ndarray, 
                                                       np.ndarray, 
                                                       np.ndarray]:
    '''
    Determine the speaking and silence timelines based on the transcriptions file.

    Args:
        transcript_path (str): The path to the transcriptions file.

    Returns:
        speaking_timeline (np.ndarray):               The speaking timeline.
        silent_timeline (np.ndarray):                 The silence timeline.
        speaking_timeline_unique (np.ndarray):        The unique speaking timeline.
        silent_timeline_unique (np.ndarray):          The unique silence timeline.
        speaking_silent_timeline (np.ndarray):        The speaking-silence timeline.
        speaking_silent_timeline_unique (np.ndarray): The unique speaking-silence timeline.
    '''
    
    # Read the transcript and select relevant columns
    transcript = pd.read_csv(transcript_path, delimiter = ';', decimal = ',')
    
    # Check if the transcript is empty
    if transcript.empty:
        print(f"Transcript file {transcript_path} is empty.")
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    # Change the variables to miliseconds
    audio_length = int(np.ceil(transcript['audio_length'].iloc[0]*1000))
    start_speaking = (transcript['start'] * 1000).astype(int).to_numpy()
    end_speaking = (transcript['end'] * 1000).astype(int).to_numpy()
    speaking_times = np.vstack((start_speaking, end_speaking))
    
    # Get the unique values of the start and end speaking times
    is_unique = transcript['is_unique'].to_list()
    start_speaking_unique = start_speaking[is_unique]
    end_speaking_unique = end_speaking[is_unique]
    speaking_times_unique = np.vstack((start_speaking_unique, end_speaking_unique))
    
    # Calculate silent segments based on speaking times and audio length
    silent_times = calculate_silent_segments(speaking_times = speaking_times,
                                                audio_length = audio_length)
    silent_times_unique = calculate_silent_segments(speaking_times = speaking_times_unique,
                                                    audio_length = audio_length)

    # Generate timelines for speaking and silent segments
    speaking_timeline = determine_timeline(times = speaking_times,
                                           audio_length = audio_length)
    silent_timeline = determine_timeline(times = silent_times,
                                         audio_length = audio_length)
    speaking_timeline_unique = determine_timeline(times = speaking_times_unique,
                                                  audio_length = audio_length)  
    silent_timeline_unique = determine_timeline(times = silent_times_unique,
                                                audio_length = audio_length) 
    
    # Generate combined speaking and silence timelines
    speaking_silence_timeline = determine_speaking_silence_timeline(speaking_times = speaking_times,
                                                                    silent_times = silent_times,
                                                                    audio_length = audio_length)
    speaking_silence_timeline_unique = determine_speaking_silence_timeline(speaking_times = speaking_times_unique,
                                                                            silent_times = silent_times_unique,
                                                                            audio_length = audio_length)  
    
    # Return all generated timelines
    return(speaking_timeline,
           silent_timeline,
           speaking_timeline_unique,
           silent_timeline_unique,
           speaking_silence_timeline,
           speaking_silence_timeline_unique)
    

def delete_nan_features(features: List[Dict[str, torch.tensor]]) -> List[Dict[str, torch.tensor]]:
    '''
    Remove features that contain all NaN values and ensure both dictionaries have the same keys.

    This function takes a list of two dictionaries containing raw and aggregated features, respectively.
    It removes any features (keys) that contain all NaN values and ensures that both dictionaries have
    the same set of keys by keeping only the common keys.

    Args:
        features (List[Dict[str, torch.Tensor]]): A list containing two dictionaries. The first dictionary contains raw features and the second dictionary contains aggregated features.

    Returns:
        features_extracted_cleaned (List[Dict[str, torch.Tensor]]): A list containing two cleaned dictionaries with the same set of keys.
    '''
    
    # Extract raw and aggregated features, removing any that contain all NaN values
    features_extracted_raw_cleaned = {key: value for key, value in features[0].items() if not torch.isnan(value).all()}
    features_extracted_aggre_cleaned = {key: value for key, value in features[1].items() if not torch.isnan(value).all()}

    # Find the intersection of keys
    common_keys = set(features_extracted_raw_cleaned.keys()).intersection(features_extracted_aggre_cleaned.keys())

    # Filter dictionaries to only include common keys
    features_extracted_raw_cleaned = {key: features_extracted_raw_cleaned[key] for key in common_keys}
    features_extracted_aggre_cleaned = {key: features_extracted_aggre_cleaned[key] for key in common_keys}
    features_extracted_cleaned = [features_extracted_raw_cleaned, features_extracted_aggre_cleaned]
    
    return features_extracted_cleaned


def replace_nan_features(features: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    '''
    Replace NaN values with 0 in features and ensure both dictionaries have the same keys.

    Args:
        features (List[Dict[str, torch.Tensor]]): A list containing two dictionaries. The first dictionary contains raw features, and the second contains aggregated features.

    Returns:
        features_cleaned (List[Dict[str, torch.Tensor]]): A list containing two dictionaries with NaNs replaced by 0 and the same set of keys.
    '''

    # Replace NaNs with 0
    features_extracted_raw_cleaned = {key: torch.nan_to_num(value, nan=0.0) for key, value in features[0].items()}
    features_extracted_aggre_cleaned = {key: torch.nan_to_num(value, nan=0.0) for key, value in features[1].items()}

    # Find the intersection of keys
    common_keys = set(features_extracted_raw_cleaned.keys()).intersection(features_extracted_aggre_cleaned.keys())

    # Filter both dictionaries to only include common keys
    features_extracted_raw_cleaned = {key: features_extracted_raw_cleaned[key] for key in common_keys}
    features_extracted_aggre_cleaned = {key: features_extracted_aggre_cleaned[key] for key in common_keys}

    return [features_extracted_raw_cleaned, features_extracted_aggre_cleaned]


def extract_and_save_features(audio_processor: AudioProcessor, 
                              raw_data_matrix: Dict[str, torch.tensor], 
                              num_cores: int, 
                              hdf5_file_name: str,
                              transcripts_path: str) -> None:
    '''
    Extracts and saves features based on the specified feature type.

    Args:
        audio_processor (AudioProcessor):          An instance of the AudioProcessor class.
        raw_data_matrix (Dict[str, torch.tensor]): Dictionary containing raw audio data.
        num_cores (int):                           Number of cores to use for multiprocessing.
        hdf5_file_name (str):                      Name of the HDF5 file to save the features.
        transcripts_path (str):                    Path to the transcriptions file.

    Returns:
        None
    '''
    
    # Extract features with AudioProcessor
    aggregated_features, raw_features = audio_processor.extract_and_aggregate_features(raw_data_matrix = raw_data_matrix, 
                                                                                       num_cores = num_cores)
    features_extracted = [raw_features, aggregated_features]
        
    # Check for nan values and remove them
    features_extracted = replace_nan_features(features = features_extracted)

    # Save features in HDF5 format to disk
    with h5py.File(hdf5_file_name, 'w') as hdf:
        for audio_path, raw_features in features_extracted[0].items():
            
            # Determine timelines
            transcript_path = transcripts_path + '/' + os.path.basename(audio_path)[:-4] + '.csv'
            if os.path.isfile(transcript_path):
                timelines = determine_timelines(transcript_path = transcript_path)
            
            # Save data to HDF5
            audio_group = hdf.create_group(audio_path)
            audio_group.create_dataset('raw', data = raw_features, compression='gzip', compression_opts=4)
            audio_group.create_dataset('aggregated', data = features_extracted[1][audio_path], compression='gzip', compression_opts=4)
            audio_group.create_dataset('frames_number', data = raw_features.shape[0])
            audio_group.create_dataset('features_number', data = raw_features.shape[1])
            if os.path.isfile(transcript_path):
                audio_group.create_dataset('speaking_timeline', data = timelines[0], compression='gzip', compression_opts=4)
                audio_group.create_dataset('silent_timeline', data = timelines[1], compression='gzip', compression_opts=4)
                audio_group.create_dataset('speaking_timeline_unique', data = timelines[2], compression='gzip', compression_opts=4)
                audio_group.create_dataset('silent_timeline_unique', data = timelines[3], compression='gzip', compression_opts=4)
                audio_group.create_dataset('speaking_silence_timeline', data = timelines[4], compression='gzip', compression_opts=4)
                audio_group.create_dataset('speaking_silence_timeline_unique', data = timelines[5], compression='gzip', compression_opts=4)
            

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


def create_dataset_from_files(model_conf: dict[str, Any]) -> pd.DataFrame:
    '''
    Create a dataset from condition file and .wav files.

    Args:
        model_conf (Dict[str, Any]): Configuration dictionary.

    Returns:
        dataset (pd.DataFrame): DataFrame containing the dataset.
    '''
    
    # Read the condition.csv file and create the condition dictionary
    condition_df = pd.read_csv(model_conf['condition_path'], delimiter=';')
    condition_dict = dict(zip(condition_df.iloc[:, model_conf['condition_id_column']], condition_df.iloc[:, model_conf['condition_label_column']]))
    
    # Get all .wav files from the folder
    wav_files_with_keys = get_wav_files_from_folder(folder_path = model_conf['wav_folder'])
    
    # Create a DataFrame from the wav files and their patient types
    dataset = pd.DataFrame(wav_files_with_keys, columns = ['file_path', 'file_key'])
    dataset['file_key'] = dataset['file_key'].astype(str)  # Ensure file_key is a string
    condition_dict = {str(key): value for key, value in condition_dict.items()}  # Ensure keys in condition_dict are strings
    dataset['file_key'] = dataset['file_key'].astype(type(list(condition_dict.keys())[0]))
    dataset['patient_type'] = dataset['file_key'].map(condition_dict)
    dataset = dataset.dropna(subset = ['patient_type'])
    
    # Filter the dataset by the global transcript file
    transcript_df = pd.read_csv(model_conf['global_transcript_file'], delimiter = ';', decimal = ',')
    dataset['basefile'] = dataset['file_path'].apply(lambda x: os.path.basename(x))
    dataset = dataset[dataset['basefile'].isin(transcript_df['filename'])]
    dataset = dataset.drop(columns=['basefile'])

    # Filter the dataset by the specified filter
    if model_conf['filter_dataset'] != 'None':
        dataset = dataset[dataset['file_path'].str.contains(model_conf['filter_dataset'])]   

    # Sort dataset by file_path
    dataset = dataset.sort_values(by='file_path').reset_index(drop=True)
    
    # Secondary sort: for each unique file_key
    dataset['last_2_char'] = dataset['file_path'].str[-6:-4]
    dataset = dataset.sort_values(by=['file_key', 'last_2_char']).reset_index(drop=True)
    dataset = dataset.drop(columns=['last_2_char'])
    
    return dataset


def extract_features(audioprocessor_data: Dict, 
                     model_conf: Dict) -> Tuple[List[str], Dict[str, float]]:    
    '''
    Extract features from raw data using the specified dataset, feature type and save them to disk.
    If saved features do not exist or load_extracted_features is False, the features are extracted and saved.

    Args:
        audioprocessor_data (Dict):                Dictionary with the audio processor configuration.
        model_conf (Dict):                         Dictionary with the model configuration.

    Returns:
        Tuple[List[str], Dict[str, float]]:        Tuple containing the following:
            - feature_combinations (List[str]):    List of feature combinations.
            - labels (Dict[str, float]):           Dictionary containing labels for each file path.
    '''
    # Get dataset from files
    dataset = create_dataset_from_files(model_conf = model_conf)
    
    # Initialize audio processor to load all .wav files from the dataset
    raw_data_matrix = None  
    save_features_path = model_conf['path_extracted_features']
    feature_type_list = audioprocessor_data['feature_type'].copy()
    for feature_type in feature_type_list:
        audioprocessor_data['feature_type'] = [feature_type]
        audio_processor = AudioProcessor(arguments = audioprocessor_data)
        hdf5_file_name = save_features_path + feature_type + '.h5'
        if not model_conf['load_extracted_features'] or not os.path.exists(hdf5_file_name):
            os.makedirs(save_features_path, exist_ok=True)
            if raw_data_matrix is None:
                raw_data_matrix = audio_processor.load_all_wav_files_from_dataset(dataset = dataset, 
                                                                                  name_column_with_path = 'file_path', 
                                                                                  num_cores = model_conf['num_cores'])               
            extract_and_save_features(audio_processor = audio_processor,
                                      raw_data_matrix = raw_data_matrix, 
                                      num_cores = model_conf['num_cores'],
                                      hdf5_file_name = hdf5_file_name,
                                      transcripts_path = model_conf['transcript_folder']) 
        else:
            continue
    audioprocessor_data['feature_type'] = feature_type_list
    
    # Select features to be trained
    feature_combinations = select_features_to_train(model_conf = model_conf,
                                                    audioprocessor_data = audioprocessor_data)
    # feature_combinations = feature_combinations[262:]        
        
    # Extract labels from the dataset
    labels = process_labels(dataset = dataset)  
    
    return feature_combinations, labels


def initialize_metrics_dict() -> Dict[str, List[float]]:
    '''
    Initialize the metrics dictionary.

    Args:
        None

    Returns:
        metrics_dict (Dict[str, Any]):  Metrics dictionary.
    '''
    metrics_dict = {'model_name': [], 'feature_type': [],  
                    'which_features': [], 'model': [], 'threshold': [], 
                    'accuracy_threshold': [], 'f1_score_threshold': [], 
                    'recall_threshold': [], 'precision_threshold': [],
                    'confusion_matrix_threshold': [], 'auc_majority_voting': [],
                    'accuracy_majority_voting': [], 'f1_score_majority_voting': [], 
                    'recall_majority_voting': [], 'precision_majority_voting': [],
                    'confusion_matrix_majority_voting': [], 'auc_threshold': [],
                    'accuracy_average_prob': [], 'f1_score_average_prob': [], 
                    'recall_average_prob': [], 'precision_average_prob': [],
                    'confusion_matrix_average_prob': [], 'auc_average_prob': []}
    return metrics_dict


def configure_model_cores(config: dict) -> dict:
    '''
    Configure the number of cores for the model based on the provided configuration.

    Args:
        config (dict):      The configuration dictionary containing model settings.

    Returns:
        model_conf (dict):  The updated configuration dictionary with the number of cores set appropriately.
    '''
    model_conf = config['model_extract_train_test']
    if model_conf['num_cores'] == 'None':
        model_conf['num_cores'] = None
    return model_conf

                