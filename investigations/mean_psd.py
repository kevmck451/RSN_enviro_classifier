from audio.audio import Audio
from audio.normalize import normalize
from audio import process

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

freq_range = (10, 2000)
nperseg = 2 ** 14 # 10: 1024 / 11: 2048 / 12: 4096 / 13: 8132 / 14: 16, / 15: 32,

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_title(f'PSD Plot: Mean of Each')
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

            x_bins, psd = process.power_spectral_density(audio, freq_range=freq_range, nperseg=nperseg, norm=True)
            av_temp.append(psd)

        mean_spectrum = np.mean(np.stack(av_temp), axis=0)
        average_spectrums.append(mean_spectrum)
        ax.plot(x_bins, mean_spectrum, label=path.stem)

        av_temp = []

ax.set_xscale('symlog')
ax.set_xlim([freq_range[0], freq_range[1]])
ax.set_xlabel('Frequency (Hz)', fontweight='bold')
ax.set_ylabel('Magnitude', fontweight='bold')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
ax.grid(True, which='both')
plt.legend()
plt.savefig(f'PSD_Mean_2')
plt.close()
