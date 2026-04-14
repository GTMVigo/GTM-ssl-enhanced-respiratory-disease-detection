# -*- coding: utf-8 -*-
"""
Feature extraction for non-interpretable approaches
(x-vector, TRILLsson, wav2vec2/HuBERT)

Input recordings assumed to be under rec_path.
Output feature files (.npy) will be saved under feat_path

@author: yiting
Modiffied by @ldocio
"""
import numpy as np
import numpy.matlib
import math
import os
import sys
from numpy import save
import pickle

import librosa
import torch
from speechbrain.pretrained import EncoderClassifier
import tensorflow as tf
import tensorflow_hub as hub
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification, Wav2Vec2FeatureExtractor


def dynamic_chunk_segmentation(Batch_data, m, C, n=1):
    """
    @author: winston lin

    Implementation function of the proposed Dynamic-Chunk-Segmentation as a general data preprocessing step.
    The proposed approach can always map originally different length inputs into fixed size and fix number of data chunks.
    
    Expected I/O:
        Input: batch data list contains different length (time-dim) of 2D feature maps (i.e., 2D numpy matrix)
        Output: fixed 3D dimension numpy array with the shape= (batch-size*C, m, feat-dim)
    
    *** Note *** This function can't process sequence length that is less than the given m!
                 Please make sure all your input data's lengths are always greater then the given m.
    
    Args:
          Batch_data$ (list): list of different length 2D numpy array for batch training data
                   m$ (int) : chunk window length (i.e., number of frames within a chunk),
                              e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec
                   C$ (int) : number of data chunks splitted for each sentence
                   n$ (int) : scaling factor to increase number of chunks splitted in a sentence
    """
    num_shifts = n*C-1
    Split_Data = []
    for i in range(len(Batch_data)):
        data = Batch_data[i]
        # checking valid lenght of the input data
        if len(data)<m:
            raise ValueError("input data length is less than the given m, please decrease m!")
        # chunk-shifting size is depending on the length of input data => dynamic step size
        step_size = int(int(len(data)-m)/num_shifts)      
        # calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # output split data
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )    
    return np.array(Split_Data) # stack as fixed size 3D output


def get_all_sub_segment_inds(x, fs=16e3, dur=10):
    """
        get the range of indices that can be used to run get_sub_segment()
        from the given audio signal
        
        - dur: number of seconds in a segment
    """
    N = x.shape[0] # number of samples in input signal
    N_seg = dur*fs # number of samples in a segment with the duration we want 
    ind_range = math.ceil(N/N_seg) # possible indices: 0:ind_range exclusive
    return ind_range


def get_sub_segment(x, fs=16e3, dur=10, index=0, pad=True):
    """
        Get a segment of the input signal x
        
        - dur: number of seconds in a segment
        - index: index of the segment counted from the whole signal
        - pad: if padding segments of length < dur seconds
    """
    # check if segment out of input range
    N = x.shape[0] # number of samples in input signal
    start_pt = int(index*dur*fs)
    end_pt = int(start_pt + dur*fs)
    if end_pt > N:
        end_pt = N
    
    # get segment
    seg = x[start_pt:end_pt]
    if pad:
        # zero padding at the end to dur if needed
        if seg.shape[0] < (dur*fs):
            pad_len = int((dur*fs)-seg.shape[0])
            seg = np.pad(seg, ((0,pad_len)), 'constant')

    return seg

 
def trillsson_extraction(x,m,dur,pad):
    """
    extract Paralinguistic speech embeddings using TRILLsson model (TRILLsson: Distilled Universal Paralinguistic Speech Representations https://arxiv.org/abs/2203.00236)
    get trillsson embeddings from one audio
    x: input audio (16khz)
    m = trillsson model
    dur: number of seconds in a segment. If the length of the audio is longer than dur, it is divided into segments of that length.
    pad: if padding when segment duration is smaller than dur
    """
    # normalize input
    x = x / (max(abs(x))) 

    # divide into sub segments
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=dur) # dur sec segments
    embeddings = np.zeros(shape=(ind_range, 1024))
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=dur, index=spec_ind, pad=pad)
        seg = tf.expand_dims(seg, 0) # -> tf.size [1, 160000]
        embedding = m(seg)['embedding'] # 1x1024
        embeddings[spec_ind,:] = embedding.numpy()

    # average across embeddings of all sub-specs
    features_tmp = np.mean(embeddings, axis=0) # (1024,)
    features_tmp = features_tmp.reshape((1,1024)) # (1,1024)

    return features_tmp


def xvector_extraction(x,classifier):
    # x-vector extraction, given x sampled at 16kHz. Uses SpeechBrain library
    # normalize input
    x = x / (max(abs(x))) 
    x = torch.tensor(x[np.newaxis,]) # (459203,) -> torch.Size([1, 459203])

    # extract x-vectors using speechbrain
    embeddings = classifier.encode_batch(x)                        # torch.Size([1, 1, 512])
    features_tmp = embeddings.squeeze().numpy()
    features_tmp = np.reshape(features_tmp,(1, features_tmp.size)) # 1x512
    
    return features_tmp


def wav2vec2_hubert_extraction(x,feature_extractor,model,dur,pad, filename_path):
    """
    wav2vec2-base-superb-sid or hubert-base-superb-sid mean pooled hidden states extraction, 
    given x sampled at 16kHz
    dur: number of seconds in a segment. If the length of the audio is longer than dur, it is divided into segments of that length.
    pad: if padding when segment duration is smaller than dur
    """
    # normalize input
    x = x / (max(abs(x))) 
    
    # divide input into segments
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=dur) # dur sec segments
    embeddings = np.zeros(shape=(ind_range, 13, 768)) # 5 layers features x 768 dim
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=dur, index=spec_ind, pad=pad)
        inputs = feature_extractor(seg, sampling_rate=16000, padding=True, return_tensors="pt")
        try:
            hidden_states = model(**inputs).hidden_states # tuple of 13 [1, frame#, 768] tensors
        except Exception as e:
            error_message = f"Failed to determine hidden states of {filename_path}: {e}"
            print(error_message)
            continue
        for layer_num in range(13): # layer_num 0:12
            embeddings[spec_ind,layer_num,:] = hidden_states[layer_num].squeeze().mean(dim=0).detach().numpy() # [768,]

    # average across embeddings of all sub-specs
    hidden_states_list = []
    for layer_num in range(13): # layer_num 0:12
        hidden_layer_avg = np.mean(embeddings[:,layer_num,:], axis=0) # (768,)
        hidden_layer_avg = hidden_layer_avg.reshape((1,768)) # (1,768)
        hidden_states_list.append(hidden_layer_avg)

    return hidden_states_list


def extract_embeddings(sdir, out_dir = 'feats/xvector/', trill = 0, task_inds = [1], id_inds = [0],  db_name = 'celia'):
    """
    Feature extraction for celia db audios, based on feature_extraction_db_extra, also
    return features as arrays of feats, and store features at the same time 
    Input 
    - sdir: source directory of both PD and HC data or list of data
    - out_dir: output directory to store features of each wav file, include part of file name
    - trill: if using trillsson instead of x-vector (modified: 0-xvector, 1-trillsson, 2-wav2vec, 3-hubert)
    - task_inds: list[int], which index or indices of the split filename (split by '_') indicates the task name
    - id_inds: list[int], which index or indices of the split filename (split by '_') indicates the subject id
    - db_name: str of the db name, used for filename when saving all feats
    
    (different way to check labels for db_name = nls)
    eg. filename = PD_xx_task_language.wav, then task_inds = [2], id_inds = [1]
    
    Output
    - features: dict of { subjectID(str)  ->  dict of {task(str) -> xvector[np 1x512]} }, or 1x1024 
            trillsson instead of xvector
    """
    
    # Set the directory for TensorFlow Hub modules
    os.environ['TFHUB_CACHE_DIR'] = out_dir
    
    # Lists all wav filenames in the selected folder
    if not isinstance(sdir, list) and os.path.isdir(sdir):
        filenames_path_list = [os.path.join(sdir, f) for f in os.listdir(sdir)]
    else:
        filenames_path_list = sdir
    
    # Create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Variables para hacer una segmentacion de audios en segmentos. Quizas no sea necesario cuando lo audios no son demasiado grandes
    segmentsdur = 3 # Segmentos de 3 segundos. Cambiuarlo para pasarlo como variable
    padding = False # No se hace padding con ceros. Pensar si hacerlo

    # pretrained model
    if trill == 1:
        m = hub.KerasLayer('https://tfhub.dev/google/trillsson1/1') # select trillsson1~5
    elif trill == 0: # xvector
        classifier = EncoderClassifier.from_hparams(source  = "speechbrain/spkrec-xvect-voxceleb", 
                                                    savedir = "pretrained_models/spkrec-xvect-voxceleb")
    elif trill == 2: # wav2vec2
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
    elif trill == 3: # hubert
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")
    print('Extracting features from all target train/test data...')

    # Store features and categories
    features = {}
    tasks = set()
    for filename_path in filenames_path_list:
        # get info of the to-be-extracted file 
        filename = os.path.basename(filename_path)
        print('Processing file: ', filename)
        file_split = filename.split('_')
        # task/session
        task = [file_split[i] for i in task_inds] # get the list of str that indicate task name
        task = '_'.join(task) # list to str
        tasks.add(task)
        # id
        ID = [file_split[i] for i in id_inds] # get the list of str that indicate id name
        ID = '_'.join(ID) # list to str
         
        # get audio data, record duration
        x, fs = librosa.load(filename_path, sr = 16000) 
        # get x-vector / other embeddings
        if trill == 0:
            features_tmp = xvector_extraction(x,classifier)
            # save individual feature vectors
            save(out_dir + filename[:-4] + '.npy', features_tmp) # exclude '.wav'
        elif trill == 1:
            features_tmp = trillsson_extraction(x, m, segmentsdur, padding)
            # save individual feature vectors
            save(out_dir + filename[:-4] + '.npy', features_tmp) # exclude '.wav'
        else:
            hidden_states_list = wav2vec2_hubert_extraction(x, feature_extractor, model, segmentsdur, padding, filename_path)
            features_tmp = hidden_states_list[-1]
            # save individual feature vectors
            for layer_num in range(13):
                if not os.path.exists(out_dir + 'hidden' + str(layer_num)):
                    os.makedirs(out_dir + 'hidden' + str(layer_num))
                save(out_dir + 'hidden' + str(layer_num) + '/'+ filename[:-4] + '.npy', hidden_states_list[layer_num]) # exclude '.wav'
         
        if ID in features:
            features[ID][task] = features_tmp
        else:
            features[ID] = {}
            features[ID][task] = features_tmp
             
    # save features vars 
    with open(out_dir + db_name + '_features.pkl', 'wb') as file:
        pickle.dump(features, file)
    with open(out_dir + db_name + '_tasks.pkl', 'wb') as file:
        pickle.dump(tasks, file)
        
    return features, tasks


def main():

    # path to the directory that contains all input data sets 
    rec_path = sys.argv[1] #'/almacen/bdd/laura/celia-dataset/' 
    
    # path to the directory to save the extracted features
    feat_path = sys.argv[2] #'/home/temporal2/ldocio/EXPERIMENTOS/celia-dataset/features/' 

    # extract wav2vec2 ---
    features, tasks = extract_embeddings(sdir =       rec_path + 'celia_audios', 
                                         task_inds =  [1], 
                                         id_inds =    [0], 
                                         out_dir =    feat_path + 'wav2vec2/', 
                                         db_name =    'celia', 
                                         trill =      2)
    
    # extract hube
    features, tasks = extract_embeddings(sdir =       rec_path + 'celia_audios', 
                                         task_inds =  [1], 
                                         id_inds =    [0], 
                                         out_dir =    feat_path + 'hubert/', 
                                         db_name =    'celia', 
                                         trill =      3)
    
    # extract trillsson representation
    features, tasks = extract_embeddings(sdir =       rec_path + 'celia_audios', 
                                         task_inds =  [1], 
                                         id_inds =    [0], 
                                         out_dir =    feat_path + 'trill/', 
                                         db_name =    'celia', 
                                         trill =      1)


if __name__ == "__main__":
    main()


