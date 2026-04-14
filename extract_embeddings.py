"""
SSL4PR Fine-tuned Model Loader
Load the SSL4PR fine-tuned checkpoint and extract embeddings from the second-to-last layer
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, HubertModel, Wav2Vec2Model
from pathlib import Path
from typing import Union, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


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
                wav_files.append(file_path)
    return wav_files


class AttentionPoolingLayer(nn.Module):
    """
    This layer implement attention pooling.
    Given a sequence of vectors (bs, seq_len, embed_dim), it:
    1. Applies a linear layer to each vector (bs, seq_len, 1)
    2. Applies a softmax to each vector (bs, seq_len, 1)
    3. Applies a weighted sum to the sequence (bs, embed_dim)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: The input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            The output tensor of shape (batch_size, embed_dim).
        """
        # Linear layer (bs, seq_len, embed_dim) -> (bs, seq_len, 1)
        weights = self.linear(x)
        # Softmax (bs, seq_len, 1) -> (bs, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        # Weighted sum (bs, seq_len, 1) * (bs, seq_len, embed_dim) -> (bs, embed_dim)
        x = torch.sum(weights * x, dim=1)
        return x


class SSL4PRClassificationModel(nn.Module):
    """
    SSL4PR Model Architecture for Parkinson's Disease Detection.
    Based on the original model structure from the provided code.
    """
    
    def __init__(
        self,
        model_type: str = "hubert",  # "wavlm" or "hubert"
        num_classes: int = 2,
        hidden_size: int = 768,
        num_layers: int = 12,
        use_all_layers: bool = True,
        classifier_type: str = "attention_pooling",
        classifier_num_layers: int = 2,
        classifier_hidden_size: int = 768,  # Changed to 768 to match checkpoint
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model_type = model_type.lower()
        self.use_all_layers = use_all_layers
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Load base SSL model
        if self.model_type == "wavlm":
            self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base", output_hidden_states=use_all_layers)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
            self.is_whisper = False
        elif self.model_type == "hubert":
            self.ssl_model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=use_all_layers)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.is_whisper = False
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Layer weighting for all layers (if used)
        if use_all_layers:
            self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers + 1)])
            self.softmax = nn.Softmax(dim=-1)
        
        # Pooling layer - matching the exact structure from checkpoint
        if classifier_type == "attention_pooling":
            self.pooling_layer = nn.Sequential()
            self.pooling_layer.add_module(
                "attention_pooling_head", 
                AttentionPoolingLayer(hidden_size)
            )
        else:
            self.pooling_layer = nn.Identity()
        
        # Classification head - EXACTLY matching the checkpoint structure
        self.classifier = nn.Sequential()
        
        # Hidden layers - using 768 hidden size to match checkpoint
        for layer_idx in range(classifier_num_layers):
            if layer_idx == 0:
                input_size = hidden_size
            else:
                input_size = classifier_hidden_size
                
            self.classifier.add_module(
                f"layer_{layer_idx}",
                nn.Linear(input_size, classifier_hidden_size)
            )
            self.classifier.add_module(
                f"layer_{layer_idx}_activation",
                nn.ReLU()
            )
            self.classifier.add_module(
                f"layer_{layer_idx}_dropout",
                nn.Dropout(dropout)
            )
        
        # Final layer
        if num_classes == 2:
            # Binary classification
            self.classifier.add_module(
                "final_layer",
                nn.Linear(classifier_hidden_size, 1)
            )
            self.classifier.add_module(
                "final_layer_activation",
                nn.Sigmoid()
            )
        else:
            # Multi-class classification
            self.classifier.add_module(
                "final_layer",
                nn.Linear(classifier_hidden_size, num_classes)
            )
        
    def get_ssl_features(self, input_values):
        """Extract SSL features with layer weighting if enabled."""
        if self.use_all_layers:
            outputs = self.ssl_model(input_values=input_values, return_dict=True)
            ssl_hidden_states = outputs.hidden_states
            
            ssl_hidden_state = torch.zeros_like(ssl_hidden_states[-1])
            weights = self.softmax(self.layer_weights)
            for i in range(len(ssl_hidden_states)):
                ssl_hidden_state += weights[i] * self.layer_norms[i](ssl_hidden_states[i])
        else:
            outputs = self.ssl_model(input_values=input_values, return_dict=True)
            ssl_hidden_state = outputs.last_hidden_state
        
        return ssl_hidden_state
    
    def forward(self, input_values, return_embeddings=False):
        """
        Forward pass through the model.
        
        Args:
            input_values: [batch_size, sequence_length]
            return_embeddings: If True, return embeddings before classification
            
        Returns:
            If return_embeddings=False: logits [batch_size, num_classes]
            If return_embeddings=True: (embeddings, logits)
        """
        # Get SSL features
        ssl_features = self.get_ssl_features(input_values)
        
        # Apply pooling
        if self.classifier_type == "average_pooling":
            embeddings = torch.mean(ssl_features, dim=1)
        else:  # attention_pooling
            # Access the attention pooling head within the sequential
            embeddings = self.pooling_layer.attention_pooling_head(ssl_features)
        
        # Classification
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return embeddings, logits
        return logits
    
    def extract_embeddings_before_classifier(self, input_values):
        """
        Extract embeddings right before the classifier (after pooling).
        
        Args:
            input_values: [batch_size, sequence_length]
            
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        with torch.no_grad():
            ssl_features = self.get_ssl_features(input_values)
            
            if self.classifier_type == "average_pooling":
                embeddings = torch.mean(ssl_features, dim=1)
            else:
                embeddings = self.pooling_layer.attention_pooling_head(ssl_features)
                
        return embeddings
    
    def extract_ssl_layer_embeddings(self, input_values, layer_index=-2):
        """
        Extract embeddings from a specific SSL encoder layer.
        
        Args:
            input_values: [batch_size, sequence_length]
            layer_index: Which layer to extract from (-1 = last, -2 = second-to-last)
            
        Returns:
            layer_embeddings: [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            outputs = self.ssl_model(input_values=input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            return hidden_states[layer_index]
    
    def extract_temporal_embeddings(self, input_values, layer_index=-2):
        """
        Extract temporal embeddings from a specific SSL encoder layer.
        Returns frame-level embeddings without pooling.
        
        Args:
            input_values: [batch_size, sequence_length]
            layer_index: Which layer to extract from (-1 = last, -2 = second-to-last)
            
        Returns:
            temporal_embeddings: [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            outputs = self.ssl_model(input_values=input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            if self.use_all_layers:
                # Apply layer weighting to get weighted temporal embeddings
                temporal_embeddings = torch.zeros_like(hidden_states[-1])
                weights = self.softmax(self.layer_weights)
                for i in range(len(hidden_states)):
                    temporal_embeddings += weights[i] * self.layer_norms[i](hidden_states[i])
            else:
                # Use specific layer
                temporal_embeddings = hidden_states[layer_index]
            
            return temporal_embeddings


class SSL4PREmbeddingExtractor:
    """
    High-level interface for loading SSL4PR checkpoints and extracting embeddings.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "hubert",
        device: str = None
    ):
        """
        Initialize the extractor with a fine-tuned checkpoint.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file (e.g., model_best.pt)
            model_type: Type of base model ("wavlm" or "hubert")
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type.lower()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.model = self._load_checkpoint()
        self.model.eval()
        
        print(f"✓ Loaded SSL4PR {model_type.upper()} model from {checkpoint_path}")
        print(f"✓ Device: {self.device}")
        
    def _load_checkpoint(self):
        """Load the checkpoint and reconstruct the model."""
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize model with the EXACT architecture from checkpoint
        model = SSL4PRClassificationModel(
            model_type=self.model_type,
            num_classes=2,  # Binary classification (PD vs Control)
            hidden_size=768,
            num_layers=12,
            use_all_layers=True,  # Based on the original code
            classifier_type="attention_pooling",  # Based on the original code
            classifier_num_layers=2,
            classifier_hidden_size=768,  # MUST be 768 to match checkpoint
            dropout=0.1
        )
        
        # Load state dict - handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Try loading directly
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load the state dict
        model.load_state_dict(state_dict)
        
        model = model.to(self.device)
        return model
    
    def load_audio(self, audio_path: Union[str, Path], target_sr: int = 16000):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sampling rate (default: 16000 Hz)
            
        Returns:
            Preprocessed audio waveform
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy and flatten
        waveform = waveform.squeeze().numpy()
        
        return waveform
    
    def extract_embeddings(
        self,
        audio_path: Union[str, Path, List[str], List[Path]],
        embedding_type: str = "pre_classifier",  # "pre_classifier", "ssl_layer", "pooled", "temporal"
        layer_index: int = -2,
        return_numpy: bool = True
    ):
        """
        Extract embeddings from audio file(s).
        
        Args:
            audio_path: Path(s) to audio file(s)
            embedding_type: Type of embeddings to extract:
                - "pre_classifier": Embeddings before the classification head (recommended)
                - "ssl_layer": Embeddings from specific SSL encoder layer (frame-level, then averaged)
                - "pooled": Pooled embeddings (same as pre_classifier but more explicit)
                - "temporal": Temporal embeddings (T x D) without pooling
            layer_index: Which SSL layer to use if embedding_type="ssl_layer" or "temporal"
            return_numpy: If True, return numpy array. If False, return torch tensor
            
        Returns:
            Embeddings array/tensor
        """
        # Handle single file or batch
        if isinstance(audio_path, (str, Path)):
            audio_paths = [audio_path]
            single_file = True
        else:
            audio_paths = audio_path
            single_file = False
        
        all_embeddings = []
        
        for audio_file in audio_paths:
            print(f"Processing: {audio_file}")
            
            # Load and preprocess audio
            waveform = self.load_audio(audio_file)
            
            # Extract features
            inputs = self.model.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            # Extract embeddings based on type
            with torch.no_grad():
                if embedding_type == "pre_classifier" or embedding_type == "pooled":
                    # Get embeddings before classification head
                    embeddings = self.model.extract_embeddings_before_classifier(input_values)
                    
                elif embedding_type == "ssl_layer":
                    # Get frame-level embeddings from SSL layer
                    layer_output = self.model.extract_ssl_layer_embeddings(input_values, layer_index)
                    # Average pool over time
                    embeddings = layer_output.mean(dim=1)
                    
                elif embedding_type == "temporal":
                    # Get temporal embeddings (T x D) without pooling
                    embeddings = self.model.extract_temporal_embeddings(input_values, layer_index)
                    # Remove batch dimension for single file
                    if single_file:
                        embeddings = embeddings.squeeze(0)
                    
                else:
                    raise ValueError(f"Unknown embedding_type: {embedding_type}")
                
                # Convert to numpy if requested
                if return_numpy and not isinstance(embeddings, tuple):
                    embeddings = embeddings.cpu().numpy()
                    if single_file and embedding_type != "temporal":
                        embeddings = embeddings.squeeze()  # Remove batch dim for single file (except temporal)
                
                all_embeddings.append(embeddings)
        
        # Stack or return single embedding
        if single_file:
            return all_embeddings[0]
        else:
            if return_numpy and embedding_type != "temporal":
                return np.vstack([e if e.ndim == 1 else e for e in all_embeddings])
            else:
                # For temporal embeddings, return as list since they have different lengths
                return all_embeddings
    
    def extract_temporal_embeddings_batch(
        self,
        audio_paths: List[Union[str, Path]],
        layer_index: int = -2,
        return_numpy: bool = True
    ):
        """
        Extract temporal embeddings for multiple audio files.
        Returns a list of temporal embeddings (each T x D).
        
        Args:
            audio_paths: List of paths to audio files
            layer_index: Which SSL layer to extract from
            return_numpy: If True, return numpy arrays. If False, return torch tensors
            
        Returns:
            List of temporal embeddings, each of shape [T, D]
        """
        temporal_embeddings = []
        
        for audio_path in audio_paths:
            print(f"Processing: {audio_path}")
            
            # Load and preprocess audio
            waveform = self.load_audio(audio_path)
            
            # Extract features
            inputs = self.model.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            # Extract temporal embeddings
            with torch.no_grad():
                embeddings = self.model.extract_temporal_embeddings(input_values, layer_index)
                embeddings = embeddings.squeeze(0)  # Remove batch dimension
                
                if return_numpy:
                    embeddings = embeddings.cpu().numpy()
                
                temporal_embeddings.append(embeddings)
        
        return temporal_embeddings
    
    def predict(self, audio_path: Union[str, Path]):
        """
        Get prediction (PD vs Control) for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction, probability, and logits
        """
        waveform = self.load_audio(audio_path)
        
        inputs = self.model.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values)
            
            # Handle binary vs multi-class classification
            if self.model.num_classes == 2:
                # Binary classification with sigmoid
                probs = torch.sigmoid(logits)
                # For binary, we need to create a 2-class probability vector
                probs_2class = torch.cat([1 - probs, probs], dim=1)
                prediction = (probs > 0.5).long()
            else:
                # Multi-class classification
                probs_2class = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)
        
        return {
            'prediction': 'Parkinson\'s Disease' if prediction.item() == 1 else 'Control',
            'prediction_idx': prediction.item(),
            'probabilities': probs_2class.cpu().numpy()[0],
            'logits': logits.cpu().numpy()[0]
        }


class XLRSEmbeddingExtractor:
    """
    Extract embeddings from XLR-S (Wav2Vec2-XLSR) 300M model.
    Supports both utterance-level (averaged) and temporal (T x D) embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        device: str = None,
        layer_index: int = -1  # -1 for last layer (before classification)
    ):
        """
        Initialize the XLR-S embedding extractor.
        
        Args:
            model_name: Hugging Face model name for XLR-S
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None
            layer_index: Which layer to extract embeddings from
                        -1 = last layer (24th layer, right before classification head)
                        -2 = second-to-last layer (23rd layer)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer_index = layer_index
        
        print(f"Loading XLR-S model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model and feature extractor
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model info
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        print(f"✓ Model loaded successfully!")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Number of layers: {self.num_layers}")
        print(f"  - Extracting from layer: {self.layer_index} (layer {self.num_layers + self.layer_index + 1 if self.layer_index < 0 else self.layer_index})")
        
    def load_audio(self, audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sampling rate (default: 16000 Hz)
            
        Returns:
            Preprocessed audio waveform as numpy array
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy and flatten
        waveform = waveform.squeeze().numpy()
        
        return waveform
    
    def extract_embeddings(
        self,
        audio_path: Union[str, Path, List[str], List[Path]],
        embedding_type: str = "utterance",  # "utterance" or "temporal"
        return_numpy: bool = True,
        show_progress: bool = True
    ) -> Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]:
        """
        Extract embeddings from audio file(s).
        
        Args:
            audio_path: Path(s) to audio file(s). Can be single path or list of paths
            embedding_type: Type of embeddings to extract:
                - "utterance": Averaged embeddings [D] - one vector per audio
                - "temporal": Frame-level embeddings [T x D] - sequence of vectors
            return_numpy: If True, return numpy array(s). If False, return torch tensor(s)
            show_progress: If True, show progress bar for batch processing
            
        Returns:
            For utterance embeddings:
                - Single file: array of shape [D] or [hidden_size]
                - Multiple files: array of shape [N, D] where N is number of files
            For temporal embeddings:
                - Single file: array of shape [T, D]
                - Multiple files: list of arrays, each of shape [T_i, D]
        """
        # Handle single file or batch
        if isinstance(audio_path, (str, Path)):
            audio_paths = [audio_path]
            single_file = True
        else:
            audio_paths = audio_path
            single_file = False
        
        all_embeddings = []
        
        # Use tqdm for progress bar if processing multiple files
        iterator = tqdm(audio_paths, desc="Extracting embeddings") if (show_progress and len(audio_paths) > 1) else audio_paths
        
        for audio_file in iterator:
            # Load and preprocess audio
            waveform = self.load_audio(audio_file)
            
            # Extract features using the feature extractor
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                # Get all hidden states
                outputs = self.model(
                    input_values,
                    output_hidden_states=True
                )
                
                # Get hidden states from all layers
                # hidden_states is a tuple of (num_layers + 1) tensors
                # hidden_states[0] = input embeddings
                # hidden_states[1] to hidden_states[-1] = layer outputs
                hidden_states = outputs.hidden_states
                
                # Extract from specified layer
                layer_output = hidden_states[self.layer_index]
                # Shape: [batch_size, sequence_length, hidden_size]
                
                if embedding_type == "utterance":
                    # Average pooling over time dimension
                    embeddings = layer_output.mean(dim=1)  # [batch_size, hidden_size]
                    embeddings = embeddings.squeeze(0)  # Remove batch dimension
                    
                elif embedding_type == "temporal":
                    # Keep temporal dimension: [T, D]
                    embeddings = layer_output.squeeze(0)  # [sequence_length, hidden_size]
                    
                else:
                    raise ValueError(f"Unknown embedding_type: {embedding_type}. Use 'utterance' or 'temporal'")
                
                # Convert to numpy if requested
                if return_numpy:
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
        
        # Return format depends on single/batch and embedding type
        if single_file:
            return all_embeddings[0]
        else:
            if embedding_type == "utterance" and return_numpy:
                # Stack utterance embeddings into single array [N, D]
                return np.vstack(all_embeddings)
            elif embedding_type == "utterance" and not return_numpy:
                # Stack utterance embeddings into single tensor
                return torch.stack(all_embeddings, dim=0)
            else:
                # For temporal embeddings, return list (different lengths)
                return all_embeddings
            
    def save_embeddings_to_h5(
        self,
        audio_paths: List[str],
        output_file: str,
        embedding_type: str = "temporal",
        compression: str = "gzip",
        compression_opts: int = 4
    ):
        """
        Extract embeddings and save directly to HDF5 file.
        
        Args:
            audio_paths: List of paths to audio files
            output_file: Path to output HDF5 file
            embedding_type: "utterance" or "temporal"
            compression: Compression type for HDF5
            compression_opts: Compression level (0-9)
        """
        print(f"\nExtracting {embedding_type} embeddings and saving to {output_file}")
        print(f"Processing {len(audio_paths)} audio files...")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as hdf:
            for audio_path in tqdm(audio_paths, desc="Processing and saving"):
                # Extract embeddings for this file
                embedding = self.extract_embeddings(
                    audio_path,
                    embedding_type=embedding_type,
                    return_numpy=True,
                    show_progress=False
                )
                
                # Create group for this audio file
                audio_group = hdf.create_group(audio_path)
                
                # Save embedding
                audio_group.create_dataset(
                    'raw',
                    data=embedding,
                    compression=compression,
                    compression_opts=compression_opts
                )
        
        print(f"✓ Embeddings saved to {output_file}")
        
        # Print file info
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  - File size: {file_size_mb:.2f} MB")
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        info = {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "extracting_from_layer": self.layer_index,
            "actual_layer_number": self.num_layers + self.layer_index + 1 if self.layer_index < 0 else self.layer_index,
            "device": self.device
        }
        return info
    
    
# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("SSL4PR Fine-tuned Model - Embedding Extraction")
    print("=" * 60)
    
    # Path to your downloaded checkpoint and audio folder
    model_type = "wavlm" # "hubert" or "wavlm" or "xlsr_300m"
    wav_folder = '/home/temporal2/avera/coperia_raw/train'
    embedding_type = "pre_classifier"  # "utterance", "temporal", "pre_classifier", "ssl_layer", "pooled", "temporal"
    
    # Initialize extractor
    if model_type == "xlsr_300m":
        print("\nInitializing XLR-S Embedding Extractor...")
        extractor = XLRSEmbeddingExtractor(model_name="facebook/wav2vec2-xls-r-300m",
                                           device = "cuda",  # or "cpu"
                                           layer_index = -1)
    else:
        checkpoint_path = "/home/temporal2/dtilves/sand_challenge/task1/task1/SSL4PR-" + model_type + "-base-full/model_best.pt"
        print("\nInitializing model...")
        extractor = SSL4PREmbeddingExtractor(checkpoint_path = checkpoint_path,
                                            model_type = model_type,  
                                            device="cuda") # or "cpu"
    
   
    # Uncomment below to test with actual audio file
    wav_files = get_wav_files_from_folder(folder_path = wav_folder)
    # audio_file = "./training/ID000_phonationA.wav"
    # audio_files = ["./training/ID000_phonationA.wav", "./training/ID000_phonationE.wav", "./training/ID000_phonationO.wav"]
    
    # To get embeddings you can use "temporal", "utterance", "pre_classifier", "ssl_layer", "pooled"
    embeddings = extractor.extract_embeddings(wav_files, embedding_type = embedding_type, return_numpy=True)
    with h5py.File("./dicoperia_features2/ssl_" + model_type + "_" + embedding_type + ".h5", 'w') as hdf:
        for embedding, audio_path in zip(embeddings, wav_files):
            audio_group = hdf.create_group(audio_path)
            audio_group.create_dataset('raw', data = embedding, compression='gzip', compression_opts=4)
            # print(embedding.shape)
    
            
    # prediction = extractor.predict(audio_file)
    # print(f"\nPrediction: {prediction['prediction']}")
    # print(f"Probabilities: Control={prediction['probabilities'][0]:.3f}, PD={prediction['probabilities'][1]:.3f}")
    

    # print("\n" + "=" * 60)
    # print("Usage Examples")
    # print("=" * 60)
    
    # # Example 1: Extract embeddings before classifier (RECOMMENDED)
    # print("\n1. Extract embeddings before classifier (after pooling):")
    # print("   embeddings = extractor.extract_embeddings('audio.wav', embedding_type='pre_classifier')")
    # print("   Shape: (768,) for single file")
    
    # # Example 2: Extract from SSL encoder layer
    # print("\n2. Extract from specific SSL encoder layer:")
    # print("   embeddings = extractor.extract_embeddings('audio.wav', embedding_type='ssl_layer', layer_index=-2)")
    # print("   Shape: (768,) - averaged over time")
    
    # # Example 3: Extract temporal embeddings (T x D)
    # print("\n3. Extract temporal embeddings (frame-level):")
    # print("   temporal_emb = extractor.extract_embeddings('audio.wav', embedding_type='temporal')")
    # print("   Shape: (T, 768) where T depends on audio length")
    
    # # Example 4: Batch processing
    # print("\n4. Process multiple files:")
    # print("   audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']")
    # print("   embeddings = extractor.extract_embeddings(audio_files)")
    # print("   Shape: (3, 768)")
    
    # # Example 5: Extract from wav2vec encoder layer
    # print("\n5. Get prediction:")
    # print("   result = extractor.predict('audio.wav')")
    # print("   result['prediction']  # 'Parkinson's Disease' or 'Control'")
    # print("   result['probabilities']  # [prob_control, prob_pd]")
    
    # # Example 6: Extract from SSL wav2vec encoder layer
    # print("\n2. Extract from specific SSL encoder layer:")
    # print("   embeddings = extractor.extract_embeddings('audio.wav', embedding_type='utterance', layer_index=-1)")
    # print("   Shape: (1024,) - averaged over time")
    
    # # Example 6: Extract from SSL wav2vec temporal embeddings (T x D)
    # print("\n2. Extract from specific SSL temporal embeddings (T x D):")
    # print("   embeddings = extractor.extract_embeddings('audio.wav', embedding_type='temporal')")
    # print("   Shape: [T, 1024] - where T varies per audio")
    
    # print("\n" + "=" * 60)
    # print("Ready to extract embeddings!")
    # print("=" * 60)