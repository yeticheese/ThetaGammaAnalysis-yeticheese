import numpy as np
import emd.sift as sift
import emd.cycles as cycles
from src.functions import get_rem_states, tg_split
from dataclasses import dataclass, field
@dataclass
class Signal:
    """
    A data class representing a signal with various attributes.

    Attributes:
    - signal (np.ndarray): The signal data.
    - sample_rate (float): The sampling rate of the signal.
    - freq_range (tuple): The theta frequency range of the signal.
    - duration (np.ndarray): An array of the time period duration in milliseconds
    - _imf (np.ndarray): Private attribute for the intrinsic mode function (IMF).
    - _mask_freq (float): Private attribute for the mask frequency.
    - _cycles (int): Private attribute for the cycles index data. (To be added)

    Methods:
    - iter.sift(): Uses to the iterated mask sift from the emd library to retrive IMFs and their mask frequencies.
    """

    signal: np.ndarray
    sample_rate: float
    freq_range: tuple
    duration: np.ndarray = field(init=False)
    _imf: np.ndarray = field(init=False)
    _mask_freq: float = field(init=False)

    def __post_init__(self):
        """
        Initializes the calculated attributes after the main initialization.
        """
        self.iter_sift()
        self.duration = np.linspace(0,
                                    len(self.signal)/2500,
                                    len(self.signal))*1000 #in milliseconds 

    @property
    def imf(self):
        """
        Getter for the intrinsic mode function (IMF).
        """
        return self._imf

    @property
    def mask_freq(self):
        """
        Getter for the mask frequency.
        """
        return self._mask_freq

    @property
    def cycles(self):
        """
        Getter for the number of cycles.
        """
        return self._cycles

    @imf.setter
    def imf(self, value):
        """
        Setter for the intrinsic mode function (IMF).
        """
        self._imf = value

    @mask_freq.setter
    def mask_freq(self, value):
        """
        Setter for the mask frequency.
        """
        self._mask_freq = value

    def iter_sift(self, mask='zc')-> tuple:
        self._imf,self._mask_freq =sift.iterated_mask_sift(self.signal,
                                                        mask_0=mask,
                                                        sample_rate=self.sample_rate,
                                                        ret_mask_freq=True)
        return self._imf,self._mask_freq    

