from audio.audio import Audio
from audio.normalize import normalize
from pathlib import Path

def audio_generator(data_path='../data'):
    """
    Generator that yields audio objects and folder labels.
    :param data_path: Path to the main data directory containing subfolders of audio files.
    """
    for path in Path(data_path).iterdir():
        if not path.stem.startswith('.') and path.is_dir():
            folder_label = path.stem
            for filepath in path.rglob('*wav'):
                audio = Audio(filepath=filepath)
                audio.data = normalize(audio)
                yield audio, folder_label
