

from Filters.audio import Audio
from Filters.save_to_wav import save_to_wav

import noisereduce as nr
from pathlib import Path
import numpy as np


def noise_reduction_filter(audio_object, noise_audio_object, std_threshold=2.5, stationary = True):

    reduced_noise_data = np.zeros_like(audio_object.data)
    freq_mask_smooth_hz = 1000
    time_mask_smooth_ms = 500

    if noise_audio_object is not None:
        print('Analyzing Noise Profile')
        for channel in range(audio_object.data.shape[0]):
            reduced_noise_data[channel, :] = nr.reduce_noise(
                y=audio_object.data[channel, :],
                sr=audio_object.sample_rate,
                y_noise=noise_audio_object.data[channel, :],
                stationary=stationary, # stationary noise reduction
                freq_mask_smooth_hz=freq_mask_smooth_hz, # default is 500Hz
                time_mask_smooth_ms=time_mask_smooth_ms, # default is 50ms
                use_tqdm=True, # show terminal progress bar
                n_std_thresh_stationary = std_threshold, # default is 1.5
                n_jobs = -1 # use all available cores
            )

    else:
        print('No Noise Profile Included')
        if audio_object.num_channels == 1:
            reduced_noise_data = nr.reduce_noise(
                y=audio_object.data,
                sr=audio_object.sample_rate,
                stationary=stationary,  # stationary noise reduction
                freq_mask_smooth_hz=freq_mask_smooth_hz,  # default is 500Hz
                time_mask_smooth_ms=time_mask_smooth_ms,  # default is 50ms
                use_tqdm=True,  # show terminal progress bar
                n_std_thresh_stationary=std_threshold,  # default is 1.5
                n_jobs=-1  # use all available cores
            )
        else:
            for channel in range(audio_object.data.shape[0]):
                reduced_noise_data[channel, :] = nr.reduce_noise(
                    y=audio_object.data[channel, :],
                    sr=audio_object.sample_rate,
                    stationary=stationary,  # stationary noise reduction
                    freq_mask_smooth_hz=freq_mask_smooth_hz,  # default is 500Hz
                    time_mask_smooth_ms=time_mask_smooth_ms,  # default is 50ms
                    use_tqdm=True,  # show terminal progress bar
                    n_std_thresh_stationary=std_threshold,  # default is 1.5
                    n_jobs=-1  # use all available cores
                )

    reduced_noise_data = np.clip(reduced_noise_data, -1.0, 1.0)

    return reduced_noise_data


if __name__ == '__main__':

    # filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 FOSSN/Data/Tests/5_outdoor_testing/07-12-2024_02-49-21_chunk_1.wav'
    # audio = Audio(filepath=filepath, num_channels=48)

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/5 Angel Boom Mount/data/flight_ch1.wav'
    audio = Audio(filepath=filepath, num_channels=1)

    print(audio)

    # filepath_noise = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Analysis/angel_noise_red/angel_noise.wav'
    # angel_noise = Audio(filepath=filepath, num_channels=1)
    #
    # filtered_data = noise_reduction_filter(audio, angel_noise, std_threshold=2.5)

    filtered_data = noise_reduction_filter(audio, noise_audio_object=None, std_threshold=2.5)

    print(f'Max: {np.max(filtered_data)}')
    print(f'Min: {np.min(filtered_data)}')

    # Ensure the filtered data shape matches the original
    assert filtered_data.shape == audio.data.shape, "Filtered data shape does not match original data shape"

    # Create the new filename with "_HPF" suffix
    # original_path = Path(filepath)
    # new_filename = original_path.stem + "_NR2" + original_path.suffix
    # new_filepath = str(original_path.parent / new_filename)

    # Save the filtered audio to the new file
    # save_to_wav(filtered_data, audio.sample_rate, audio.num_channels, new_filepath)

    path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Analysis/angel_noise_red'
    name = f'{audio.name}_NR_2.5.wav'
    audio.data = filtered_data
    audio.export(filepath=path, name=name)
