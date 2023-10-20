import matplotlib.pyplot as plt
import numpy as np
from neurodsp.filt import filter_signal

class Signal:
    """
    A class for working with signal data.

    Args:
        array (np.ndarray): The raw signal data.
        sampling_rate (int): The sampling rate in Hz.
    """

    def __init__(self, array: np.ndarray, sampling_rate: int):
        """
        Initialize a Signal object with raw signal data and sampling rate.

        Args:
            array (np.ndarray): The raw signal data.
            sampling_rate (int): The sampling rate in Hz.
        """
        self.raw_signal = array
        self.sampling_rate = sampling_rate

        self.shape = array.shape
        self.duration = len(array) / sampling_rate
        self.filtered = None
        self.filter_type = None

    def filter(self, filter_type: str, f_range: any, n_seconds: int, remove_edges: bool = False):
        """
        Apply a filter to the signal.

        Args:
            filter_type (str): The type of filter to apply.
            f_range (any): The filter's cutoff frequency range.
            n_seconds (int): The filter order in seconds.
            remove_edges (bool, optional): Whether to remove edges. Defaults to False.

        Raises:
            ValueError: If filter_type is not a valid filter type.
        """
        valid_filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        if filter_type not in valid_filter_types:
            raise ValueError("Invalid filter_type. Supported types: 'lowpass', 'highpass', 'bandpass', 'bandstop'.")

        try:
            sig_filt = filter_signal(sig=self.raw_signal,
                                     fs=self.sampling_rate, 
                                     pass_type=filter_type,
                                     f_range=f_range,
                                     n_seconds=n_seconds,
                                     remove_edges=remove_edges)
            self.filtered = sig_filt
            self.filter_type = filter_type
        except Exception as e:
            raise RuntimeError(f"Filtering failed: {str(e)}") from e

    def plot(self, xlim):
        """
        Plot the signal.
        """
        _, ax = plt.subplots(figsize=(16, 4))
        times = np.arange(0, self.duration, 1 / self.sampling_rate)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (uV)')

        ax.plot(times, self.raw_signal, color='k', label='raw')

        if self.filtered is not None:
            ax.plot(times, self.filtered, label=self.filter_type, color='r')
            ax.legend()

        plt.xlim(xlim)
        plt.show()

    def summary(self):
        """
        Print a summary of the signal, including sampling rate, duration, max, and min values.
        """
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Max value: {self.raw_signal.max():.2f} uV")
        print(f"Min value: {self.raw_signal.min():.2f} uV")
        print(f"Mean value: {self.raw_signal.mean():.2f} uV")
        print(f"Standard Deviation: {self.raw_signal.std():.2f} uV")
        print(f"Signal Range: {self.raw_signal.max() - self.raw_signal.min():.2f} uV")
        print(f"Number of Data Points: {len(self.raw_signal)}")

    def get_raw_signal(self) -> np.ndarray:
        return self.raw_signal

    def get_filtered(self) -> np.ndarray:
        return self.filtered

    def __len__(self) -> int:
        return len(self.raw_signal)

    def __str__(self) -> str:
        """
        Get a string representation of the Signal object with sampling rate information.
        """
        sampling_rate_info = f"Sampling rate: {self.sampling_rate} Hz\n"
        array_str = str(self.raw_signal)  # Get the string representation of the NumPy array
        return sampling_rate_info + array_str
