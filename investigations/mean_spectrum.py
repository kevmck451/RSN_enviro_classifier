from audio.audio import Audio
from audio.normalize import normalize
from audio import process

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_title(f'Spectral Plot: Mean of Each')
for path in Path('../data').iterdir():
    if not path.stem.startswith('.'):
        # fig.tight_layout(pad=1)
        average_spectrums = []
        av_temp = []
        x_bins = None

        for filepath in path.rglob('*wav'):
            label = path.parent.stem
            audio = Audio(filepath=filepath)
            # audio.waveform(display=True)
            audio.data = normalize(audio)
            # audio.waveform(display=True)
            average_spectrum, x_bins = process.average_spectrum(audio, norm=False, convert_to_dB=False, frequency_range=(10, 2000))
            av_temp.append(average_spectrum)

        mean_spectrum = np.mean(np.stack(av_temp), axis=0)
        average_spectrums.append(mean_spectrum)
        ax.plot(x_bins, mean_spectrum, label=path.stem)

        av_temp = []

ax.set_xscale('symlog')
ax.set_xlim([10, 2000])
ax.set_xlabel('Frequency (Hz)', fontweight='bold')
ax.set_ylabel('Magnitude', fontweight='bold')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.grid(True, which='both')
plt.legend()
plt.savefig(f'SpecPlot_Mean_Normed_1')
plt.close()
