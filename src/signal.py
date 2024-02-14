import numpy as np
import emd.sift as sift
import emd.cycles as cycles
import emd.spectra as spectra
from src.functions import get_rem_states, tg_split, extrema, get_cycles
from dataclasses import dataclass, field
from neurodsp.filt import filter_signal_fir
from neurodsp.filt import filter_signal
from bycycle.features import compute_features

class SignalProcessor:
    def __init__(self,signal: np.ndarray, sample_rate: float,freq_range: tuple ):
        self.signal = signal
        self.sample_rate = sample_rate
        self.freq_range = freq_range

        # Front-End Attributes
        self.theta: np.ndarray = None
        self.cycles = None
        self.imf: np.ndarray = None
        self.mask_freq: np.ndarray = None

        #Back-End attributes
        self._theta: np.ndarray = None
        self._cycles = None
        self._imf: np.ndarray = None
        self._mask_freq: np.ndarray = None
        self._phasic: None
        self.tonic: None

    
    def get_duration(self) -> np.ndarray:
        duration = np.linspace(0,len(self.signal)/self.sample_rate,len(self.signal))*1000
        return duration
    
    # TODO: Create a dynamic EMD sifting method for the signal processor
    # def sift(self,func=None,**kwargs):
    #     if func == None:
    #         self.imf,self.mask_freq=sift.iterated_mask_sift(self.signal,
    #                                                         mask_0=mask,
    #                                                         sample_rate=self.sample_rate,
    #                                                         ret_mask_freq=True)
    #     else:


    

    def iter_sift(self,**kwargs) -> tuple:
        self._imf,self._mask_freq =sift.iterated_mask_sift(self.signal,
                                                        sample_rate=self.sample_rate,
                                                        ret_mask_freq=True)
        return self._imf,self._mask_freq

    def split_signals(self) -> np.ndarray:
        if(getattr(self,'imf') is None) or (getattr(self,'mask_freq') is None):
            if  (getattr(self,'_imf') is None) or (getattr(self,'_mask_freq') is None):
                self._imf,self._mask_freq = self.iter_sift()
                imf = self._imf
                mask_freq = self._mask_freq
            else:
                imf = self._imf
                mask_freq = self._mask_freq
        else:
            imf = self.imf
            mask_freq = self.mask_freq
        

        sub,theta,gamma = tg_split(mask_freq,self.freq_range)
        split_signals = np.empty((3,self.signal.shape[0]))
        split_signals[2] = np.sum(imf.T[sub], axis=0)
        split_signals[1] = np.sum(imf.T[theta], axis=0)
        split_signals[0] = np.sum(imf.T[gamma], axis=0)

        return split_signals

    def get_theta(self) -> np.ndarray:
        self._theta = self.split_signals()[1]
        return self._theta

    def get_cycles(self,mode='peak'):
        cycles = get_cycles(self.get_theta(),mode)
        self._cycles = cycles
        return cycles
    
    def spike_df(self):
        filtered_signal = filter_signal(sig=self.signal, 
                                        fs=self.sample_rate, 
                                        pass_type='lowpass',
                                        f_range=25, 
                                        n_seconds = 0.5, 
                                        remove_edges = False)
        
        # threshold_bycycle:{'amp_fraction_threshold':0,
        #                    'amp_consistency_threshold':0,
        #                    'period_consistency_threshold':0,
        #                    'monotonicity_threshold':0,
        #                    'min_n_cycles':8}


        df = compute_features (filtered_signal,
                                2500,f_range=(4,12),center_extrema ='peak',burst_method= 'cycles',
                                threshold_kwargs={'amp_fraction_threshold':0,
                                                  'amp_consistency_threshold':0,
                                                  'period_consistency_threshold':0,
                                                  'monotonicity_threshold':0,
                                                  'min_n_cycles':8})

        spike_df = df[["sample_last_trough", "sample_next_trough", "is_burst"]]

        
        return spike_df

    def get_phasic(self):
        df = self.spike_df()
        self._phasic = df[df['is_burst'] == True]
        return self._phasic

    def get_tonic(self):
        df = self.spike_df()
        self._tonic = df[df['is_burst'] == False]
        return self._tonic

    def apply_duration_threshold(self,duration_length:float or tuple = None):
        if duration_length is None:
            duration_length = 1000/np.array(self.freq_range)
            print(duration_length)
        if getattr(self,'cycles') is None:
            if getattr(self,'_cycles') is None:
                print('Back-end cycles attribute is missing')
            else:
                cycles = self._cycles
        else:
            cycles = self.cycles

        duration_check = np.diff(cycles[:,[0,-1]],axis=1)*(1000/self.sample_rate)
        duration_check_mask = np.squeeze(np.logical_and(duration_check <= duration_length[0], duration_check > duration_length[1]))
        print(duration_check_mask.shape)

        self._cycles = cycles[duration_check_mask]

    @property
    def imf(self):
        """
        Getter for the imf array.
        """
        return self._imf

    @imf.setter
    def imf(self,value):
        """
        Getter for the imf array.
        """
        self._imf = value

    @property
    def mask_freq(self):
        """
        Getter for the mask_freq array.
        """
        return self._mask_freq

    @mask_freq.setter
    def mask_freq(self,value):
        """
        Getter for the imf array.
        """
        self._mask_freq = value

    @property
    def theta(self):
        """
        Getter for the theta signal.
        """
        return self._theta

    @mask_freq.setter
    def theta(self,value):
        """
        Getter for the imf array.
        """
        self._theta = value

    @property
    def cycles(self):
        """
        Getter for the cycles array (theta).
        """
        return self._cycles

    @cycles.setter
    def cycles(self, value):
        """
        Setter for the cycles array.
        """
        self._cycles = value
    
    @property
    def phasic(self):
        """
        Getter for phasic periods.
        """
        return self._phasic

    @phasic.setter
    def phasic(self, value):
        """
        Setter for the phasic periods.
        """
        self._phasic = value

