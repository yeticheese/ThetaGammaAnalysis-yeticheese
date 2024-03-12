import numpy as np
import emd.sift as sift
import emd.cycles as cycles
import emd.spectra as spectra
import pandas as pd
import os
import scipy.io as sio
import scipy
import sails
from src.functions import get_rem_states, tg_split, extrema, get_cycles,get_states, morlet_wt, bin_tf_to_fpp, peak_cog
from dataclasses import dataclass, field
from neurodsp.filt import filter_signal_fir
from neurodsp.filt import filter_signal
from bycycle.features import compute_features
from icecream import ic


class SignalProcessor:
    def __init__(self,signal: np.ndarray, sample_rate: float,freq_range: tuple):
        self.signal = signal
        self.sample_rate = sample_rate
        self.freq_range = freq_range

        # Front-End Attributes
        self.imf: np.ndarray = None
        self.mask_freq: np.ndarray = None
        self.IP = None
        self.IF = None
        self.IA = None
        self.theta: np.ndarray = None
        self.cycles = None
        self.phasic = None
        self.tonic = None


        #Back-End attributes
        self._imf: np.ndarray = None
        self._mask_freq: np.ndarray = None
        self._IP = None
        self._IF = None
        self._IA = None
        self._theta: np.ndarray = None
        self._cycles = None
        self._spike_df = None
        self._phasic = None
        self._tonic = None

    
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

    def frequency_transform(self):
        if  (getattr(self,'_imf') is None) or (getattr(self,'_mask_freq') is None):
            self._imf,self._mask_freq = self.iter_sift()
        self._IP, self._IF, self._IA = spectra.frequency_transform(self._imf,self.sample_rate,'nht')
        
        return self._IP, self._IP, self._IA

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
        


        threshold_bycycle={'amp_fraction_threshold':0.8,
                           'amp_consistency_threshold':0,
                           'period_consistency_threshold':0,
                           'monotonicity_threshold':0,
                           'min_n_cycles':8}



        df = compute_features (filtered_signal,
                                2500,f_range=(4,12),center_extrema ='peak',burst_method= 'cycles',
                                threshold_kwargs=threshold_bycycle)

        self._spike_df = df[["sample_last_trough", "sample_next_trough", "is_burst"]]

        return self._spike_df

    def get_phasic_states(self):
        if (getattr(self,'_spike_df') is None) or (hasattr(self,'_spike_df') is False):
            self._spike_df = self.spike_df()
            df = self._spike_df
        else:
            df = self._spike_df
        try:
            split_states = get_states(df['is_burst'].to_numpy(),True,1)
        except IndexError as e:
            print('No phasic states detected')
            split_states = np.empty((0,2))
        if split_states.ndim == 3:
            split_states=np.squeeze(split_states,0)
        phasic_states = np.empty((0,2)).astype(int)
        for state in split_states:
            phasic_state = np.array([df['sample_last_trough'].iloc[state[0]],df['sample_next_trough'].iloc[state[1]]])
            phasic_states = np.vstack([phasic_states, phasic_state])

        self._phasic = phasic_states
        return phasic_states



    def get_tonic_states(self):
        if (getattr(self,'_spike_df') is None) or (hasattr(self,'_spike_df') is False):
            self._spike_df = self.spike_df()
            df = self._spike_df
        else:
            df = self._spike_df
        try:
            split_states = get_states(df['is_burst'].to_numpy(),False,1)
        except IndexError as e:
            print('No tonic states detected')
            split_states = np.empty((0,2))
        if split_states.ndim == 3:
            split_states=np.squeeze(split_states,0)
        tonic_states = np.empty((0,2)).astype(int)
        for state in split_states:
            tonic_state = np.array([df['sample_last_trough'].iloc[state[0]],df['sample_next_trough'].iloc[state[1]]])
            tonic_states = np.vstack([tonic_states, tonic_state])

        self._tonic = tonic_states
        return tonic_states


        
    
    #TODO: Fix duration length data type adjustability
    def apply_duration_threshold(self,duration_length:float or tuple = None):
        print(self._cycles.shape)
        if duration_length is None:
            duration_length = 1000/np.array(self.freq_range)
        if getattr(self,'cycles') is None:
            if getattr(self,'_cycles') is None:
                print('Back-end cycles attribute is missing')
            else:
                cycles = self._cycles
        else:
            cycles = self.cycles

        duration_check = np.diff(cycles[:,[0,-1]],axis=1)*(1000/self.sample_rate)
        duration_check_mask = np.squeeze(np.logical_and(duration_check <= duration_length[0], duration_check > duration_length[1]))

        self._cycles = cycles[duration_check_mask]
     

    def peak_center_of_gravity(self):
        frequencies = np.arange(20,141,1)
        angles = angles=np.linspace(-180,180,19)
        sub_theta = tg_split(self.mask_freq, self.freq_range)[2]
        power = morlet_wt(np.sum(self.imf.T[sub_theta], axis=0),
                                           self.sample_rate,
                                           frequencies,
                                           mode='power')
        power = scipy.stats.zscore(power, axis=0)
        shifted_zscore_power = power + 2*np.abs(power)
        fpp_cycles = bin_tf_to_fpp(self.cycles[:,[0,-1]], shifted_zscore_power, 19)
        cog_values = peak_cog(frequencies,fpp_cycles,0.95)

        return cog_values

        


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

    @theta.setter
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

    @property
    def tonic(self):
        """
        Getter for tonic periods.
        """
        return self._tonic

    @tonic.setter
    def tonic(self, value):
        """
        Setter for the tonic periods.
        """
        self._tonic = value

class SegmentSignalProcessor(SignalProcessor):
    def __init__(self, signal: np.ndarray, period: np.ndarray, sample_rate: float, freq_range: tuple, ):
        super().__init__(signal, sample_rate, freq_range)
        self.period = period

    def get_duration(self) -> np.ndarray:
        time = self.period/self.sample_rate
        duration = np.linspace(time[0],time[1],len(self.signal))*1000
        return duration

    def get_cycles(self,mode='peak'):
        cycles = super().get_cycles(mode = mode) + self.period[0]
        self._cycles = cycles
        return cycles

    def get_phasic_states(self):
        phasic_states = super().get_phasic_states() + self.period[0]
        self._phasic = phasic_states
        return phasic_states

    def get_tonic_states(self):
        tonic_states = super().get_tonic_states() + self.period[0]
        self._tonic = tonic_states
        return tonic_states

    def spike_df(self):
        spike_df = super().spike_df()
        spike_df[['sample_last_trough', 'sample_next_trough']] += self.period[0]
        self._spike_df = spike_df
        return spike_df

    def peak_center_of_gravity(self):
        frequencies = np.arange(20,141,1)
        angles = angles=np.linspace(-180,180,19)
        sub_theta = tg_split(self.mask_freq, self.freq_range)[2]
        power = morlet_wt(np.sum(self.imf.T[sub_theta], axis=0),
                                           self.sample_rate,
                                           frequencies,
                                           mode='power')
        power = scipy.stats.zscore(power, axis=0)
        shifted_zscore_power = power + 2*np.abs(power)
        fpp_cycles = bin_tf_to_fpp(self.cycles[:,[0,-1]]-self.period[0], shifted_zscore_power, 19)
        cog_values = peak_cog(frequencies,angles,fpp_cycles,0.95)

        return cog_values


        
@dataclass
class SleepSignal(SignalProcessor):
    """
    A data class representing a signal with various attributes.

    Attributes:
    - signal (np.ndarray): The signal data.
    - sample_rate (float): The sampling rate of the signal.
    - freq_range (tuple): The theta frequency range of the signal.
    - cycles (int): Private attribute for the cycles index data. (To be added)

    Methods:
    - To be determined
    """
    signal: np.ndarray
    rem_states: np.ndarray
    sample_rate: float
    freq_range: tuple
    REM: list = field(default_factory=list)  # Initialize _REM as an empty list using default_factory
    cycles: np.ndarray = field(default_factory=lambda: np.empty((0, 5)).astype(int))  # Initialize cycles with empty array using default_factory

    def __post_init__(self):
        if not self.REM:
            ic('REM List is empty')
            actual_rem_states = []
            for i,rem_period in enumerate(self.rem_states):
                signal = self.signal[rem_period[0]:rem_period[1]]
                try:
                    REM = REM_Segment(signal, rem_period ,self.sample_rate, self.freq_range)
                    actual_rem_states.append(i)
                    self.REM.append(REM)
                except Exception as e:
                    f'Error processing REM period {rem_period} '
                    continue
            self.rem_states = self.rem_states[actual_rem_states]
        else:
            ic('REM List provided')
        self.get_cycles()
        self.apply_duration_threshold()
        self.apply_amplitude_threshold()
        self.spike_df()


    def get_cycles(self,mode='peak'):
        cycles = np.empty((0,5)).astype(int)
        for rem in self.REM:
            if hasattr(rem, 'cycles') is None:
                cycles = np.vstack([cycles, rem.get_cycles(mode = mode)])
            else:
                cycles = np.vstack([cycles, rem.cycles])
        self.cycles = cycles
        return cycles

    def apply_duration_threshold(self, duration_length: float or tuple = None):
        for rem in self.REM:
            rem.apply_duration_threshold(duration_length = duration_length)
        cycles = super().apply_duration_threshold(duration_length= duration_length)

    def apply_amplitude_threshold(self):
        sub_theta = np.array([])
        theta_peak_amp = np.array([])
        for rem in self.REM:
            sub_theta = np.append(sub_theta,np.sum(rem.imf.T[tg_split(rem.mask_freq)[0]],axis=0))
            theta_sig = rem.get_theta()
            theta_peak_amp = np.append(theta_peak_amp,theta_sig[rem.cycles[:,2]-rem.period[0]])
        amp_threshold =2*sub_theta.std()
        amp_threshold_mask = theta_peak_amp > amp_threshold
        self.cycles = self.cycles[amp_threshold_mask]

        for rem in self.REM:
            theta_sig = rem.get_theta()
            cycles_mask = theta_sig[rem.cycles[:,2]-rem.period[0]] > amp_threshold
            rem.cycles = rem.cycles[cycles_mask]
            
    def spike_df(self):
        spike_df = pd.DataFrame()
        for rem in self.REM:
            spike_df = pd.concat([spike_df,rem._spike_df],axis = 0,ignore_index= True)
        self._spike_df = spike_df
        return spike_df

    def build_dataset(self):
        df = pd.DataFrame(self.cycles)
        df = df.rename(columns={0:'first_trough',1:'first_zero_x',2:'peak',3:'last_zero_x',4:'last_trough'})
        df['sample_rate']=self.sample_rate
        df['peak_amplitude']=self.signal[self.cycles[:,2]]
        df['phasic/tonic'] = None
        for phasic_state in self.get_phasic_states():
            phasic_in_range_df = (df['first_trough'].between(*phasic_state) | df['last_trough'].between(*phasic_state))
            df.loc[phasic_in_range_df, 'phasic/tonic'] = 'phasic'
        for tonic_state in self.get_tonic_states():
            tonic_in_range_df = (df['first_trough'].between(*tonic_state) | df['last_trough'].between(*tonic_state))
            df.loc[tonic_in_range_df, 'phasic/tonic'] = 'tonic'
        cog_df = pd.DataFrame()
        for rem in self.REM:
            cog_df = pd.concat([cog_df,pd.DataFrame(rem.peak_center_of_gravity())], axis=0,ignore_index=True)
        df[['cog_freq','cog_phase']]= cog_df

        return df

@dataclass
class REM_Segment(SegmentSignalProcessor):
    signal: np.ndarray
    period: np.ndarray
    sample_rate: float
    freq_range: tuple
    imf: np.ndarray = field(default_factory=lambda: np.array([]))
    mask_freq: float = field(default_factory=lambda: np.array([]))
    IP: np.ndarray = field(default_factory=lambda: np.array([]), metadata= 'Instaneous Power')
    IF: np.ndarray = field(default_factory=lambda: np.array([]), metadata= 'Instantaneous Frequency')
    IA: np.ndarray = field(default_factory=lambda: np.array([]), metadata= 'Instantaneous Amplitude')
    cycles: np.ndarray = field(default_factory=lambda: np.empty((0, 5)).astype(int))  # Initialize cycles with empty array using default_factory
    _spike_df: pd.DataFrame = None
    tonic: np.ndarray = None
    phasic: np.ndarray = None

    def __post_init__(self):
        REM = SegmentSignalProcessor(self.signal,self.period,self.sample_rate,self.freq_range)
        if (self.imf.size == 0) or (self.mask_freq.size == 0):
            ic('No imf data, generating imfs....')
            self.imf,self.mask_freq = REM.iter_sift()
            self.IP,self.IF,self.IA = REM.frequency_transform()
        if self.cycles.size == 0:
            ic('No cycle data, extracting cycles....')
            REM.get_cycles()
            self.cycles = REM._cycles
        REM.spike_df()
        self.phasic = self.get_phasic_states()
        self.tonic = self.get_tonic_states()
    
    # def __eq__(self, other):
    #     if not isinstance(other, REM_Segment):
    #         return False
    #     attrs = ['signal', 'period', 'sample_rate', 'freq_range', 'imf', 'mask_freq', 'IP', 'IF', 'IA', 'cycles']
    #     for attr in attrs:
    #         if not np.array_equal(getattr(self, attr), getattr(other, attr)):
    #             return False
    #     return True





    