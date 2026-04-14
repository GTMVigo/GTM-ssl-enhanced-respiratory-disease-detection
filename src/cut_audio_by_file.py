import os
import yaml
import librosa
import numpy as np
import soundfile as sf
from datetime import datetime


def get_configuration_and_audio_files() -> tuple[dict, dict]:
    """
    Reads the configuration file and returns a dict with the audio files and their corresponding timing files.

    Returns:
        tuple[dict, dict]: A tuple containing the configuration dictionary and a list of audio file paths.
            - configuration (dict):     The configuration dictionary.
            - audio_timing_dict (dict): Dictionary containing the audio files and their corresponding timing files.
    """
    
    # Reads the configuration file
    with open('./conf/cut_audio_by_file_conf.yaml') as file:
        configuration = yaml.safe_load(file)
        
    # Gets the audio files into a list
    audios_path = configuration['audios_path']
    if os.path.isdir(audios_path):
        audio_files = [os.path.join(audios_path, f) for f in os.listdir(audios_path)]
    else:
        audio_files = [audios_path]
        
    # Extract the audio path column of metadata_path into a list
    if os.path.isfile(configuration["metadata_path"]):
        metadata_list = []
        with open(configuration["metadata_path"], "r") as metadata_file:
            for line in metadata_file:
                columns = line.strip().split("\t")
                metadata_list.append(columns[configuration["audio_column"]])
        audio_files = list(filter(lambda f: os.path.basename(f) in metadata_list, audio_files))
        
    # Extract the timing files and relate them to the audio_files
    timing_path = configuration['timings_path']
    if os.path.isdir(timing_path) and configuration['timings_in_filename']:
        audio_timing_dict = {}
        timing_files = [os.path.join(timing_path, f) for f in os.listdir(timing_path)]       
        audio_timing_dict = {audio_file: timing_file
                             for audio_file in audio_files
                              for timing_file in timing_files
                               if os.path.basename(timing_file)[:-4] in {
                                  os.path.basename(audio_file)[:-4], 
                                  os.path.basename(audio_file)[:-4] + "_timings"}}
    else:
        audio_timing_dict = {audio_files[0]: timing_path}
           
    return configuration, audio_timing_dict


def timestamp_to_seconds(timestamp: str) -> datetime:
    """
    Convert a timestamp in the format 'HH:MM:SS,fff' to total seconds.

    Args:
        timestamp (str): A string representing time in 'HH:MM:SS,fff' format.

    Returns:
        total_seconds (datetime): The total time represented by the timestamp in seconds.
    """
    dt = datetime.strptime(timestamp, "%H:%M:%S,%f")
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds


if __name__ == "__main__":
    
    # Get configuration and audio_files
    conf, audio_timing_dict = get_configuration_and_audio_files()
    if not audio_timing_dict: raise Exception("No audio timing files found")

    # Iterate over the dictionary
    for audio_file, timing_file in audio_timing_dict.items():
        
        # Load the audio file
        audio, sr = librosa.load(audio_file, sr=None)

        # Open the timing file and read the timestamps
        with open(timing_file, 'r') as f:
            lines = f.readlines()

        # Initialize variables to keep track of the remaining audio parts
        remaining_audio_parts = []
        prev_end_sample = 0
        
        # For each line in the timing file
        for i, line in enumerate(lines):
            
            # Split the line into start and end timestamps
            start, end = map(timestamp_to_seconds, line.split())

            # Cut the audio
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # Append the audio part before the cut to the list
            remaining_audio_parts.append(audio[prev_end_sample:start_sample])

            # Update the end sample for the next iteration
            prev_end_sample = end_sample

        # Append the last part of the audio after the last cut
        remaining_audio_parts.append(audio[prev_end_sample:])

        # Concatenate all the remaining parts of the audio
        remaining_audio = np.concatenate(remaining_audio_parts)

        # Save the cut audio
        remaining_audio_path = conf['output_path'] + os.path.basename(audio_file)
        sf.write(remaining_audio_path, remaining_audio, sr)
  


