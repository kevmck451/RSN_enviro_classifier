from audio.audio import Audio
from audio.normalize import normalize
from audio import process



import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path


# for path in Path('data').rglob('*wav'):
#     print(path.parent.stem)
#     label = path.parent.stem
#     audio = Audio(filepath=path)
#     # audio.waveform(display=True)
#     audio.data = normalize(audio)
#     # audio.waveform(display=True)
#     average_spectrum, frequency_bins = process.average_spectrum(audio, display=True)




for path in Path('../data').iterdir():

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_title(f'Spectral Plot: {path.stem}')
    # fig.tight_layout(pad=1)

    if not path.stem.startswith('.'):
        for filepath in path.rglob('*wav'):
            label = path.parent.stem
            audio = Audio(filepath=filepath)
            # audio.waveform(display=True)
            audio.data = normalize(audio)
            # audio.waveform(display=True)
            average_spectrum, frequency_bins = process.average_spectrum(audio, norm=False)

            ax.plot(frequency_bins, average_spectrum)
            ax.set_xscale('symlog')
            ax.set_xlim([10, 10000])
            ax.set_xlabel('Frequency (Hz)', fontweight='bold')
            ax.set_ylabel('Magnitude', fontweight='bold')

            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
            ax.grid(True, which='both')

        plt.savefig(f'SpecPlot_{path.stem}')
        plt.close()