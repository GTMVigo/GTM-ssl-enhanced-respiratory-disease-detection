import concurrent.futures
import multiprocessing
import os.path
from typing import List, Callable, Dict
from scipy.stats import skew, kurtosis

import librosa
import numpy as np
import opensmile
import pandas as pd
import torch
import torchaudio
import wave
from numpy import ndarray
from pydub import AudioSegment
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp
from tqdm import tqdm

SUPPORTED_FEATS = [
    "compare_2016_energy",
    "compare_2016_llds",
    "compare_2016_voicing",
    "compare_2016_spectral",
    "compare_2016_mfcc",
    "compare_2016_rasta",
    "compare_2016_basic_spectral",
    "spafe_mfcc",
    "spafe_imfcc",
    "spafe_cqcc",
    "spafe_gfcc",
    "spafe_lfcc",
    "spafe_lpc",
    "spafe_lpcc",
    "spafe_msrcc",
    "spafe_ngcc",
    "spafe_pncc",
    "spafe_psrcc",
    "spafe_plp",
    "spafe_rplp",
]


class MultiProcessor:
    def __init__(self, num_cores: int = multiprocessing.cpu_count()):
        self.num_cores = num_cores

        if not isinstance(self.num_cores, int):
            raise ValueError("The `num_cores` argument must be an integer.")
        elif self.num_cores < 1 or self.num_cores > multiprocessing.cpu_count():
            raise ValueError(
                "The `num_cores` argument must be between 1 and the number of CPUs."
            )

    @staticmethod
    def _parameters_validation_for_multiprocessing(
        raw_data_paths: List[str], process_func: Callable
    ):
        if not isinstance(raw_data_paths, list):
            raise TypeError("The `raw_data_paths` argument must be a list.")
        elif not callable(process_func):
            raise TypeError("The `process_func` argument must be a callable function.")
        elif not raw_data_paths:
            raise ValueError("The `raw_data_paths` argument must not be empty.")
        elif not all(os.path.isfile(path) for path in raw_data_paths):
            raise FileNotFoundError("One or more paths do not exist.")

    def _process_with_progress(
        self, raw_data_paths: List[str], process_func: Callable
    ) -> Dict[str, np.ndarray]:
        results = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_cores
        ) as executor:
            future_to_data = {
                executor.submit(process_func, path): path for path in raw_data_paths
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_data),
                total=len(future_to_data),
                desc="Processing data",
                unit="item",
            ):
                path = future_to_data[future]
                try:
                    result = future.result()
                    results.update(result)
                except Exception as e:
                    raise RuntimeError(
                        f"An error occurred during processing {path}: {e}"
                    )
        return results

    def process_with_multiprocessing(
        self, raw_data_paths: List[str], process_func: Callable
    ) -> Dict[str, np.ndarray]:
        try:
            self._parameters_validation_for_multiprocessing(
                raw_data_paths, process_func
            )
            dict_with_id_and_features_from_raw_data = self._process_with_progress(
                raw_data_paths, process_func
            )
            return dict_with_id_and_features_from_raw_data

        except (TypeError, ValueError, FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(
                f"An error occurred during multiprocessing: {str(e)}"
            ) from e


class AudioProcessor:
    def __init__(self, arguments: dict):
        self.supported_feats: list = SUPPORTED_FEATS
        self.arguments = arguments

        try:
            self.feature_type: str = self.arguments["feature_type"]
            self.resampling_rate = int(self.arguments["resampling_rate"])
            self.top_db = float(self.arguments["top_db"])
            self.resampling_rate = int(self.arguments["resampling_rate"])
            self.pre_emphasis_coefficient = float(
                self.arguments["pre_emphasis_coefficient"]
            )

            self.f_min = int(self.arguments["f_min"])
            self.f_max = int(self.arguments["f_max"])
            self.window_size = int(self.arguments["window_size"])
            self.nfft = int(float(self.window_size) * 1e-3 * self.resampling_rate)
            self.hop_length = int(
                float(int(self.arguments["hop_length"])) * 1e-3 * self.resampling_rate
            )

            self.n_mels = int(self.arguments["n_mels"])
            self.n_mfcc = int(self.arguments["n_mfcc"])

            self.plp_order = int(self.arguments["plp_order"])
            self.conversion_approach = self.arguments["conversion_approach"]

            self.normalize = self.arguments["normalize"]
            self.use_energy = bool(self.arguments["use_energy"])
            self.apply_mean_norm = bool(self.arguments["apply_mean_norm"])
            self.apply_vari_norm = bool(self.arguments["apply_vari_norm"])

            self.compute_deltas = bool(self.arguments["compute_deltas_feats"])
            self.compute_deltas_deltas = bool(
                self.arguments["compute_deltas_deltas_feats"]
            )
            self.compute_opensmile_extra_features = (
                False
                if "compare_2016" in self.feature_type
                else bool(self.arguments["compute_opensmile_extra_features"])
            )

            self.feature_transform = self._create_feature_transformer()
        except Exception as e:
            raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")

    def _create_feature_transformer(self):
        feature_transformers = {}
        for feature_type in self.feature_type:
            if "compare_2016" in feature_type.lower():
                feature_transformers[feature_type] = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.ComParE_2016,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                    sampling_rate=self.resampling_rate,
                )
            elif "spafe_" in feature_type.lower() or feature_type.lower() == "mfcc":
                spafe_feature_transformers = {
                    "spafe_mfcc": mfcc,
                    "spafe_imfcc": imfcc,
                    "spafe_bfcc": bfcc,
                    "spafe_cqcc": cqcc,
                    "spafe_gfcc": gfcc,
                    "spafe_lfcc": lfcc,
                    "spafe_lpc": lpc,
                    "spafe_lpcc": lpcc,
                    "spafe_msrcc": msrcc,
                    "spafe_ngcc": ngcc,
                    "spafe_pncc": pncc,
                    "spafe_psrcc": psrcc,
                    "spafe_plp": plp,
                    "spafe_rplp": rplp,
                }
                feature_transformers[feature_type] = spafe_feature_transformers[feature_type]
            else:
                raise ValueError(f"Feature type {feature_type} not implemented")
        return feature_transformers

    def _do_feature_extraction(self, s: np.ndarray, sr: int):
        """Feature preparation
        Steps:
        1. Apply feature extraction to waveform
        2. Convert amplitude to dB if required
        3. Append delta and delta-delta features
        """
        features = []
        for feature_type, transformer in self.feature_transform.items():
            matrix_with_feats = None

            if feature_type.lower() == "mfcc":
                matrix_with_feats = transformer(
                    s,
                    sr,
                    num_ceps=self.n_mfcc,
                    low_freq=self.f_min,
                    high_freq=int(sr // 2),
                    nfilts=self.n_mels,
                    nfft=self.nfft,
                    use_energy=self.use_energy,
                )
                
            opensmile_extra_feats_set = {"subset": []}

            if "compare_2016" in feature_type.lower():
                s = s[None, :]
                matrix_with_feats = transformer.process_signal(s, sr)

                # feature subsets
                if feature_type.lower() == "compare_2016_voicing":
                    opensmile_extra_feats_set["subset"] = [
                        "F0final_sma",
                        "voicingFinalUnclipped_sma",
                        "jitterLocal_sma",
                        "jitterDDP_sma",
                        "shimmerLocal_sma",
                        "logHNR_sma",
                    ]

                if feature_type.lower() == "compare_2016_energy":
                    opensmile_extra_feats_set["subset"] = [
                        "audspec_lengthL1norm_sma",
                        "audspecRasta_lengthL1norm_sma",
                        "pcm_RMSenergy_sma",
                        "pcm_zcr_sma",
                    ]

                if feature_type.lower() == "compare_2016_spectral":
                    opensmile_extra_feats_set["subset"] = [
                        "audSpec_Rfilt_sma[0]",
                        "audSpec_Rfilt_sma[1]",
                        "audSpec_Rfilt_sma[2]",
                        "audSpec_Rfilt_sma[3]",
                        "audSpec_Rfilt_sma[4]",
                        "audSpec_Rfilt_sma[5]",
                        "audSpec_Rfilt_sma[6]",
                        "audSpec_Rfilt_sma[7]",
                        "audSpec_Rfilt_sma[8]",
                        "audSpec_Rfilt_sma[9]",
                        "audSpec_Rfilt_sma[10]",
                        "audSpec_Rfilt_sma[11]",
                        "audSpec_Rfilt_sma[12]",
                        "audSpec_Rfilt_sma[13]",
                        "audSpec_Rfilt_sma[14]",
                        "audSpec_Rfilt_sma[15]",
                        "audSpec_Rfilt_sma[16]",
                        "audSpec_Rfilt_sma[17]",
                        "audSpec_Rfilt_sma[18]",
                        "audSpec_Rfilt_sma[19]",
                        "audSpec_Rfilt_sma[20]",
                        "audSpec_Rfilt_sma[21]",
                        "audSpec_Rfilt_sma[22]",
                        "audSpec_Rfilt_sma[23]",
                        "audSpec_Rfilt_sma[24]",
                        "audSpec_Rfilt_sma[25]",
                        "pcm_fftMag_fband250-650_sma",
                        "pcm_fftMag_fband1000-4000_sma",
                        "pcm_fftMag_spectralRollOff25.0_sma",
                        "pcm_fftMag_spectralRollOff50.0_sma",
                        "pcm_fftMag_spectralRollOff75.0_sma",
                        "pcm_fftMag_spectralRollOff90.0_sma",
                        "pcm_fftMag_spectralFlux_sma",
                        "pcm_fftMag_spectralCentroid_sma",
                        "pcm_fftMag_spectralEntropy_sma",
                        "pcm_fftMag_spectralVariance_sma",
                        "pcm_fftMag_spectralSkewness_sma",
                        "pcm_fftMag_spectralKurtosis_sma",
                        "pcm_fftMag_spectralSlope_sma",
                        "pcm_fftMag_psySharpness_sma",
                        "pcm_fftMag_spectralHarmonicity_sma",
                        "mfcc_sma[1]",
                        "mfcc_sma[2]",
                        "mfcc_sma[3]",
                        "mfcc_sma[4]",
                        "mfcc_sma[5]",
                        "mfcc_sma[6]",
                        "mfcc_sma[7]",
                        "mfcc_sma[8]",
                        "mfcc_sma[9]",
                        "mfcc_sma[10]",
                        "mfcc_sma[11]",
                        "mfcc_sma[12]",
                        "mfcc_sma[13]",
                        "mfcc_sma[14]",
                    ]

                if feature_type.lower() == "compare_2016_mfcc":
                    opensmile_extra_feats_set["subset"] = [
                        "mfcc_sma[1]",
                        "mfcc_sma[2]",
                        "mfcc_sma[3]",
                        "mfcc_sma[4]",
                        "mfcc_sma[5]",
                        "mfcc_sma[6]",
                        "mfcc_sma[7]",
                        "mfcc_sma[8]",
                        "mfcc_sma[9]",
                        "mfcc_sma[10]",
                        "mfcc_sma[11]",
                        "mfcc_sma[12]",
                        "mfcc_sma[13]",
                        "mfcc_sma[14]",
                    ]

                if feature_type == "compare_2016_rasta":
                    opensmile_extra_feats_set["subset"] = [
                        "audSpec_Rfilt_sma[0]",
                        "audSpec_Rfilt_sma[1]",
                        "audSpec_Rfilt_sma[2]",
                        "audSpec_Rfilt_sma[3]",
                        "audSpec_Rfilt_sma[4]",
                        "audSpec_Rfilt_sma[5]",
                        "audSpec_Rfilt_sma[6]",
                        "audSpec_Rfilt_sma[7]",
                        "audSpec_Rfilt_sma[8]",
                        "audSpec_Rfilt_sma[9]",
                        "audSpec_Rfilt_sma[10]",
                        "audSpec_Rfilt_sma[11]",
                        "audSpec_Rfilt_sma[12]",
                        "audSpec_Rfilt_sma[13]",
                        "audSpec_Rfilt_sma[14]",
                        "audSpec_Rfilt_sma[15]",
                        "audSpec_Rfilt_sma[16]",
                        "audSpec_Rfilt_sma[17]",
                        "audSpec_Rfilt_sma[18]",
                        "audSpec_Rfilt_sma[19]",
                        "audSpec_Rfilt_sma[20]",
                        "audSpec_Rfilt_sma[21]",
                        "audSpec_Rfilt_sma[22]",
                        "audSpec_Rfilt_sma[23]",
                        "audSpec_Rfilt_sma[24]",
                        "audSpec_Rfilt_sma[25]",
                    ]

                if feature_type == "compare_2016_basic_spectral":
                    opensmile_extra_feats_set["subset"] = [
                        "pcm_fftMag_fband250-650_sma",
                        "pcm_fftMag_fband1000-4000_sma",
                        "pcm_fftMag_spectralRollOff25.0_sma",
                        "pcm_fftMag_spectralRollOff50.0_sma",
                        "pcm_fftMag_spectralRollOff75.0_sma",
                        "pcm_fftMag_spectralRollOff90.0_sma",
                        "pcm_fftMag_spectralFlux_sma",
                        "pcm_fftMag_spectralCentroid_sma",
                        "pcm_fftMag_spectralEntropy_sma",
                        "pcm_fftMag_spectralVariance_sma",
                        "pcm_fftMag_spectralSkewness_sma",
                        "pcm_fftMag_spectralKurtosis_sma",
                        "pcm_fftMag_spectralSlope_sma",
                        "pcm_fftMag_psySharpness_sma",
                        "pcm_fftMag_spectralHarmonicity_sma",
                    ]

                if feature_type == "compare_2016_llds":
                    opensmile_extra_feats_set["subset"] = list(matrix_with_feats.columns)

                matrix_with_feats = matrix_with_feats[
                    opensmile_extra_feats_set["subset"]
                ].to_numpy()
                matrix_with_feats = np.nan_to_num(matrix_with_feats)
                matrix_with_feats = torch.from_numpy(matrix_with_feats).T
                s = s[0]

            if "spafe_" in feature_type:
                if feature_type in [
                    "spafe_mfcc",
                    "spafe_imfcc",
                    "spafe_gfcc",
                    "spafe_lfcc",
                    "spafe_msrcc",
                    "spafe_ngcc",
                    "spafe_psrcc",
                ]:
                    matrix_with_feats = transformer(
                        s,
                        sr,
                        num_ceps=self.n_mfcc,
                        low_freq=self.f_min,
                        high_freq=int(sr // 2),
                        nfilts=self.n_mels,
                        nfft=self.nfft,
                        use_energy=self.use_energy,
                    )

                elif feature_type in ["spafe_pncc"]:
                    matrix_with_feats = transformer(
                        s,
                        sr,
                        nfft=self.nfft,
                        nfilts=self.n_mels,
                        low_freq=self.f_min,
                        num_ceps=self.n_mfcc,
                        high_freq=int(sr // 2),
                    )

                elif feature_type in ["spafe_cqcc"]:
                    matrix_with_feats = transformer(
                        s,
                        sr,
                        num_ceps=self.n_mfcc,
                        low_freq=self.f_min,
                        high_freq=(sr // 2),
                        nfft=self.nfft,
                    )

                elif feature_type in [
                    "spafe_lpc",
                    "spafe_lpcc",
                ]:
                    try:
                        matrix_with_feats = transformer(s, sr, order=self.plp_order)
                        if isinstance(matrix_with_feats, tuple):
                            matrix_with_feats = matrix_with_feats[0]
                    except (np.linalg.LinAlgError, ValueError) as e:
                        matrix_with_feats = np.zeros([int(s.shape[0]/160), self.plp_order])
                        
                elif feature_type in ["spafe_plp", "spafe_rplp"]:
                    try:
                        normalize ="mvn" if self.normalize else None
                        matrix_with_feats = transformer(s, 
                                                        sr, 
                                                        order=self.plp_order, 
                                                        conversion_approach=self.conversion_approach, 
                                                        low_freq=self.f_min, 
                                                        high_freq=int(sr // 2), 
                                                        normalize=normalize, 
                                                        nfilts=self.n_mels, 
                                                        nfft=self.nfft, )
                    except (np.linalg.LinAlgError, ValueError) as e:
                        matrix_with_feats = np.zeros([int(s.shape[0]/160), self.plp_order])
                matrix_with_feats = np.nan_to_num(matrix_with_feats)
                matrix_with_feats = torch.from_numpy(matrix_with_feats).T

            if self.compute_deltas:
                if matrix_with_feats.ndim == 2 and matrix_with_feats.shape[1] > 0:
                    matrix_with_feats_deltas = torchaudio.functional.compute_deltas(matrix_with_feats)
                    matrix_with_feats = torch.cat((matrix_with_feats, matrix_with_feats_deltas), dim=0)
                else:
                    raise ValueError(f"Invalid shape for feature matrix: {matrix_with_feats.shape}")

            if self.compute_deltas_deltas:
                if matrix_with_feats.ndim == 2 and matrix_with_feats.shape[1] > 0:
                    matrix_with_feats_deltas_deltas = torchaudio.functional.compute_deltas(matrix_with_feats_deltas)
                    matrix_with_feats = torch.cat((matrix_with_feats, matrix_with_feats_deltas_deltas), dim=0)
                else:
                    raise ValueError(f"Invalid shape for feature matrix: {matrix_with_feats.shape}")

            if self.apply_mean_norm:
                matrix_with_feats = matrix_with_feats - torch.mean(matrix_with_feats, dim=0)

            if self.apply_vari_norm:
                matrix_with_feats = matrix_with_feats / torch.std(matrix_with_feats, dim=0)

            # own feature selection
            if self.compute_opensmile_extra_features and ("compare_2016" not in feature_type):
                s = s[None, :]

                # Config OpenSMILE
                opensmile_extra_feats_set = {
                    "subset": [
                        # Voicing
                        "F0final_sma",
                        "voicingFinalUnclipped_sma",
                        "jitterLocal_sma",
                        "jitterDDP_sma",
                        "shimmerLocal_sma",
                        "logHNR_sma",
                        # Energy
                        "audspec_lengthL1norm_sma",
                        "audspecRasta_lengthL1norm_sma",
                        "pcm_RMSenergy_sma",
                        "pcm_zcr_sma",
                        # Spectral
                        "pcm_fftMag_fband250-650_sma",
                        "pcm_fftMag_fband1000-4000_sma",
                        "pcm_fftMag_spectralRollOff25.0_sma",
                        "pcm_fftMag_spectralRollOff50.0_sma",
                        "pcm_fftMag_spectralRollOff75.0_sma",
                        "pcm_fftMag_spectralRollOff90.0_sma",
                        "pcm_fftMag_spectralFlux_sma",
                        "pcm_fftMag_spectralCentroid_sma",
                        "pcm_fftMag_spectralEntropy_sma",
                        "pcm_fftMag_spectralVariance_sma",
                        "pcm_fftMag_spectralSkewness_sma",
                        "pcm_fftMag_spectralKurtosis_sma",
                        "pcm_fftMag_spectralSlope_sma",
                        "pcm_fftMag_psySharpness_sma",
                        "pcm_fftMag_spectralHarmonicity_sma",
                    ]
                }
                extra_transform = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.ComParE_2016,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                    sampling_rate=self.resampling_rate,
                )

                # Extract features
                matrix_with_extra_feats = extra_transform.process_signal(s, sr)
                matrix_with_extra_feats = matrix_with_extra_feats[
                    opensmile_extra_feats_set["subset"]
                ].to_numpy()
                matrix_with_extra_feats = np.nan_to_num(matrix_with_extra_feats)
                matrix_with_extra_feats = torch.from_numpy(matrix_with_extra_feats).T

                # Concatenate the features
                common_shape = min(
                    matrix_with_feats.shape[1], matrix_with_extra_feats.shape[1]
                )
                matrix_with_feats = torch.cat(
                    (
                        matrix_with_feats[:, :common_shape],
                        matrix_with_extra_feats[:, :common_shape],
                    ),
                    dim=0,
                )

            # Apply the transpose to have the features in the columns
            matrix_with_feats = matrix_with_feats.T
            features.append(matrix_with_feats)
            
        features_combined = self._concatenate_features(features)  
          
        return features_combined

    def _concatenate_features(self,features: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenates a list of features along the first dimension. If the features
        have different second dimensions, the smaller feature is padded with zeros.

        Args:
            features (List[torch.Tensor]): List of feature tensors to concatenate.

        Returns:
            torch.Tensor: Concatenated feature tensor.
        """
        # Check the second dimension of each feature
        max_dim = max(feature.shape[1] for feature in features)
        
        # Pad features if necessary
        padded_features = []
        for feature in features:
            if feature.shape[1] < max_dim:
                padding = (0, max_dim - feature.shape[1])
                padded_feature = torch.nn.functional.pad(feature, padding, "constant", 0)
                padded_features.append(padded_feature)
            else:
                padded_features.append(feature)
        
        # Concatenate features along the first dimension
        concatenated_features = torch.cat(padded_features, dim=0)
        
        return concatenated_features


    def _convert_to_wav_and_replace(self, audio_file: str) -> str:
        """
        Converts any audio file to WAV format and replaces the original file.
        Returns the path to the converted file (same as original but with .wav extension if needed).
        """
        # Check if file is already a valid WAV
        if audio_file.lower().endswith('.wav'):
            try:
                with wave.open(audio_file, 'rb') as audio_wave:
                    # Additional check: verify it has proper WAV structure
                    if audio_wave.getnchannels() > 0 and audio_wave.getframerate() > 0:
                        return audio_file
            except (wave.Error, EOFError, OSError):
                pass  # Not a valid wav, will convert below
        
        # Determine the output file path (replace extension with .wav if needed)
        if not audio_file.lower().endswith('.wav'):
            output_file = os.path.splitext(audio_file)[0] + '.wav'
        else:
            output_file = audio_file
        
        try:
            # Load and convert the audio
            audio = AudioSegment.from_file(audio_file)
            
            # Convert to mono 16kHz (common for speech processing)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export as WAV with PCM encoding
            audio.export(output_file, format="wav", parameters=["-acodec", "pcm_s16le"])
            
            # Verify the converted file is valid
            with wave.open(output_file, 'rb') as audio_wave:
                if audio_wave.getnchannels() > 0 and audio_wave.getframerate() > 0:
                    # Remove the original file if it's different from the output
                    if output_file != audio_file and os.path.exists(audio_file):
                        os.remove(audio_file)
                    return output_file
                else:
                    raise ValueError("Converted file is not valid")
                    
        except Exception as e:
            # Clean up if conversion fails
            if os.path.exists(output_file) and output_file != audio_file:
                os.remove(output_file)
            raise RuntimeError(f"Failed to convert {audio_file} to WAV: {str(e)}")
    
    
    def _read_a_wav_file(self, wav_path: str) -> tuple[ndarray, int]:
        if os.path.getsize(wav_path) <= 44:
            raise ValueError(f"File {wav_path} is too small to be a valid wav file.")

        try:
            # load the audio file
            converted_file = self._convert_to_wav_and_replace(wav_path)
            s, sr = librosa.load(converted_file, sr=None, mono=True)
            
            # resample the audio file
            if (sr != self.resampling_rate) and (0 < self.resampling_rate < sr):
                sr = self.resampling_rate
                s = librosa.resample(y=s, orig_sr=sr, target_sr=self.resampling_rate)

            # apply speech activity detection
            speech_indices = librosa.effects.split(s, top_db=self.top_db)
            s = np.concatenate([s[start:end] for start, end in speech_indices])

            # apply a pre-emphasis filter
            s = librosa.effects.preemphasis(s, coef=self.pre_emphasis_coefficient)

            # normalize
            s /= np.max(np.abs(s))

            return s, sr
        except Exception as e:
            raise RuntimeError(f"An error occurred during feature extraction: {str(e)}")

    def simple_thread_wav_2_dict_with_path_and_data(
        self, wav_path: str
    ) -> dict[str, ndarray]:
        """
        The code above implements SAD, a pre-emphasis filter with a coefficient of 0.97, and normalization.
        :param wav_path: Path to the audio file
        :return: Audio samples
        """
        s, sr = self._read_a_wav_file(wav_path)

        # Check if the audio file is empty
        if not s.size or not np.any(s) or np.nan_to_num(s).sum() == 0:
            raise ValueError(f"File {wav_path} is empty.")

        return {wav_path: s}

    def simple_thread_extract_features_from_raw_data(
        self, id_data: str, raw_data: ndarray, sampling_rate: int
    ) -> dict[str, ndarray]:
        features = self._do_feature_extraction(raw_data, sampling_rate)

        return {id_data: features}

    def load_all_wav_files_from_dataset(
        self,
        dataset: pd.DataFrame,
        name_column_with_path: str,
        num_cores: int = None,
    ) -> dict:
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        try:
            multi_processor = MultiProcessor(num_cores=num_cores)

            raw_data_paths = dataset[name_column_with_path].drop_duplicates().tolist()
            raw_data_matrix = multi_processor.process_with_multiprocessing(
                raw_data_paths, self.simple_thread_wav_2_dict_with_path_and_data
            )

        except Exception as e:
            message = f"An error occurred during feature extraction: {str(e)}"
            raise RuntimeError(message)

        if len(raw_data_matrix) != len(raw_data_paths):
            missing_files = list(set(raw_data_paths) - set(raw_data_matrix.keys()))

            for missing_file in missing_files:
                missing_file_raw_data = (
                    self.simple_thread_wav_2_dict_with_path_and_data(missing_file)
                )
                raw_data_matrix.update(missing_file_raw_data)
        return raw_data_matrix

    def extract_features_from_raw_data(
        self,
        raw_data_matrix: dict[str, np.ndarray],
        num_cores: int = None,
    ) -> dict[str, np.ndarray]:
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        def worker(id_data, raw_data):
            return self.simple_thread_extract_features_from_raw_data(
                id_data, raw_data, self.resampling_rate
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_cores
        ) as executor:
            future_to_id_data = {
                executor.submit(worker, id_data, raw_data): id_data
                for id_data, raw_data in raw_data_matrix.items()
            }

            features = {}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_id_data),
                total=len(future_to_id_data),
            ):
                id_data = future_to_id_data[future]
                result = future.result()
                features.update(result)

        return features

    def aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features by computing the mean, standard deviation, and other statistics.
        :param features: Tensor of shape (num_frames, num_features)
        :return: Aggregated features of shape (num_aggregated_features,)
        """
        # Check if the input is a tensor, if not convert it to one
        if not isinstance(features, torch.Tensor): features = torch.tensor(features)
        mean_features = torch.mean(features, dim=0)
        std_features = torch.std(features, dim=0)
        min_features = torch.min(features, dim=0).values
        max_features = torch.max(features, dim=0).values
        clamped_features = torch.clamp(features, min=1e-10)
        entropy = -torch.sum(clamped_features * torch.log2(clamped_features + 1e-10), dim=0)
        skewness = skew(features.cpu().numpy(), axis=0)
        kurt = kurtosis(features.cpu().numpy(), axis=0)
        q1 = np.percentile(features.cpu().numpy(), 25, axis=0)
        q2 = np.percentile(features.cpu().numpy(), 50, axis=0)
        q3 = np.percentile(features.cpu().numpy(), 75, axis=0)
        q4 = np.percentile(features.cpu().numpy(), 90, axis=0)
        aggregated_features = torch.stack([mean_features, std_features, min_features, max_features,
                                           entropy, torch.tensor(skewness), torch.tensor(kurt), torch.tensor(q1),
                                           torch.tensor(q2), torch.tensor(q3), torch.tensor(q4)])

        return aggregated_features

    def extract_and_aggregate_features(self, raw_data_matrix: dict[str, np.ndarray], num_cores: int = None) -> dict[str, np.ndarray]:
        """
        Extract and aggregate features from raw data.
        :param raw_data_matrix: Dictionary with raw audio data
        :param num_cores: Number of cores to use for multiprocessing
        :return: Dictionary with aggregated features
        """
        features = self.extract_features_from_raw_data(raw_data_matrix, num_cores)
        aggregated_features = {id_data: self.aggregate_features(feat_matrix) for id_data, feat_matrix in features.items()}
        return aggregated_features, features