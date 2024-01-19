import logging

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.filt import filter_signal_fir
from bycycle.features import compute_features

logger = logging.getLogger('runtime')

def metaname(metadata):
    return '_'.join(metadata.values())

def get_episodes(df_features, threshold_episode=2500):
    """
    Determine episodes by connecting the cycles.

    Parameters
    ----------
    df_features: pandas.DataFrame
        Dataframe containing 'sample_last_trough' and 'sample_next_trough' features.
    threshold_episode: int, optional
        Threshold value for connecting consecutive episodes.

    Returns:
        list: A list of tuples representing episodes [(start1, end1), (start2, end2), ...].
    """
    l1 = df_features["sample_last_trough"].to_numpy()
    l2 = df_features["sample_next_trough"].to_numpy()

    # List to store extracted episodes
    episodes = []

    # Assign the start of the first episode
    start = l1[0]
    
    # Initialize the end of the first episode
    end = 0

    for i, l1_i in enumerate(l1):
        if (l1_i - l2[i - 1]) > threshold_episode:
            # Set the end of the current episode
            end = l2[i - 1]
            # Append the current episode to the list
            episodes.append((start, end))
            # Update the start for the next episode
            start = l1_i
    # Set the end of the last episode
    end = l2[-1]
    # Append the last episode to the list
    episodes.append((start, end))  

    return episodes

def get_segments(signal, timestamps):
    """
    Get segments of a signal based on timestamps.
    
    Parameters
    ----------
    signal: 1D array
        Time series
    timestamps: a list of tuples
        A list of tuples representing timestamps [(start1, end1), (start2, end2), ...].   
    """
    segments = []
    for period in timestamps:
        start, end = period
        segments.append(signal[start:end])
    return segments

###########################################################################
#Signal Class

class BaseSignal:
    """
    Base class for representing signals.

    """
    def __init__(self, sig: np.ndarray, fs: float):
        """
        Initialize a Signal instance.
        """
        if sig.ndim != 1:
            raise ValueError('Signal must be 1-dimensional.')

        self.raw = sig
        self.fs = fs
        
        self.metadata = {}
        self.filtered = {}
        self.metaname = ''
        self.duration = len(sig) / fs

        logger.debug("Signal of shape {0}, {1} Hz.".format(self.raw.shape, self.fs))

    def set_metadata(self, metadata):
        for key, value in metadata.items():
            setattr(self, key, value)  # Update instance attributes
        self.metadata = metadata  # Update the metadata dictionary
        self.metaname = metaname(metadata)

        logger.info("Metadata: {0}".format(self.metaname))
    
    def filter(self, pass_type, f_range, n_seconds):
        if pass_type not in self.filtered:
            logger.info("Filterting the signal with {0} filter.".format(pass_type))
            logger.debug("n_seconds = {0}, frequency = {1}".format(n_seconds, f_range))
            self.filtered[pass_type] = filter_signal_fir(self.raw, self.fs, pass_type, f_range,
                                                         n_seconds=n_seconds, remove_edges=False)
    
    def plot(self, xlim=(0, 10), figsize=(16, 4)):
        _, ax = plt.subplots(figsize=figsize)
        times = np.arange(0, self.duration, 1 / self.fs)

        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Voltage (uV)', fontsize=14)

        if self.metaname:
            ax.set_title(self.metaname, fontsize=18)
            
        ax.plot(times, self.raw, color='k', label='raw')

        if self.filtered:
            for pass_type, sig_filt in self.filtered.items():
                ax.plot(times, sig_filt, label=pass_type, color='r')
            ax.legend()

        plt.xlim(xlim)

    def summary(self):
        """
        Print a statistical summary of the signal.
        """
        header = ""
        if self.metaname:
            header = self.metaname
        message = f"""
        -Sampling rate: {self.fs} Hz
        -Duration: {self.duration:.2f} seconds
        -Max value: {self.raw.max():.2f} uV, Min value: {self.raw.min():.2f} uV
        -Mean value: {self.raw.mean():.2f} uV
        -Standard deviation: {self.raw.std():.2f} uV
        -Signal range: {self.raw.max() - self.raw.min():.2f} uV
        -Signal shape: {self.raw.shape}
                """
        print(header + message)

    def __len__(self):
        return len(self.raw)

    def __str__(self):
        message = f"Sampling rate: {self.fs} Hz\nSignal: "
        return message + str(self.raw)
    
##############################
#NeuralSignal Class
class NeuralSignal(BaseSignal):
    """
    Class for representing neural signals.
    """
    def __init__(self, sig: np.ndarray, fs: float):
        """
        Initialize a NeuralSignal instance.
        Parameters
        ----------
        sig : 1D array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        """
        logger.info("Initializing NeuralSignal.")
        super().__init__(sig, fs)
        self.phasic = []
        self.tonic = []    

    def segment(self, f_range, threshold_episode, threshold_bycycle):
        """
        Segment the neural signal into phasic and tonic episodes using the Cycle-by-Cycle algorithm.
        
        This function uses the cycle-by-cycle algorithm [1] to identify phasic and tonic episodes in a neural signal.
        The algorithm identifies cycles part of a bursting oscillation based on the amplitude fraction and minimum 
        cycle length.

        Parameters
        ----------
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        threshold_episode: int
            Threshold parameter for connecting consecutive phasic episodes.
        threshold_bycycle: dict
            Threshold parameters for the cycle-by-cycle algorithm.
        
        References:
        -----------
        [1] Cole, S. R. and Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. 
        Journal of Neurophysiology, 122(2), 849-861. https://doi.org/10.1152/jn.00273.2019
        """
        logger.info("STARTED: Segmenting the signal into phasic and tonic episodes.")
        
        if threshold_episode < 1:
            raise ValueError("Invalid value for `threshold_episode` parameter. Should be a positive integer.")

        # Run cycle-by-cycle algorithm for burst detection
        df = compute_features(self.filtered["lowpass"], 
                              self.fs,
                              f_range=f_range, 
                              center_extrema='peak',
                              burst_method='cycles',
                              threshold_kwargs=threshold_bycycle)
        
        # Extract the timestamps and burst detection results
        df = df[["sample_last_trough", "sample_next_trough", "is_burst"]]
        phasic_df = df[df["is_burst"] == True]
        tonic_df = df[df["is_burst"] == False]

        logger.info("Found {0} phasic cycles in the signal".format(len(phasic_df)))
        logger.info("Found {0} tonic cycles in the signal".format(len(tonic_df)))

        if len(phasic_df) != 0:
            self.phasic = get_episodes(phasic_df, threshold_episode=threshold_episode)
            self.tonic = get_episodes(tonic_df, threshold_episode=0)
        else:
            self.tonic = [(tonic_df["sample_last_trough"].iloc[0], tonic_df["sample_next_trough"].iloc[-1])]
        
        logger.debug("Phasic episodes: {0}".format(self.phasic))
        logger.debug("Tonic episodes: {0}".format(self.tonic))
        
        logger.info("COMPLETED: Segmenting the signal into phasic and tonic episodes.")
    
    def get_phasic(self, filter_type):
        if filter_type == "raw":
            return get_segments(self.raw, self.phasic)
        elif filter_type in self.filtered:
            return get_segments(self.filtered[filter_type], self.phasic)
    
    def get_tonic(self, filter_type):
        if filter_type == "raw":
            return get_segments(self.raw, self.tonic)
        elif filter_type in self.filtered:
            return get_segments(self.filtered[filter_type], self.tonic)
