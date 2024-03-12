from dataclasses import dataclass, field
from src.signal import * 
from src.utils import get_file_dict
from src.functions import get_rem_states
import scipy.io as sio
import numpy as np
import os
import re

def generate_class_dict(data_dict:dict,signal:np.ndarray,sample_rate:float,freq_range:tuple):
    class_dict={}
    class_dict['signal'] = signal
    class_dict['sample_rate'] = sample_rate
    class_dict['freq_range'] = freq_range
    class_dict['REM']= []
    class_dict['rem_states'] = np.empty((0,2)).astype(int)
    for key,value in data_dict.items():
        rem_dict={}
        if re.search(r'REM (\d{1,2})',key):
            rem_dict['period'] = value['start-end']
            class_dict['rem_states'] = np.vstack([class_dict['rem_states'], rem_dict['period']])
            rem_dict['signal'] = class_dict['signal'][rem_dict['period'][0]:rem_dict['period'][1]+1]
            rem_dict['imf'] = value['IMFs']
            rem_dict['mask_freq'] = value['IMF_Frequencies']
            rem_dict['IP'] = value['Instantaneous Phases']
            rem_dict['IF'] = value['Instantaneous Frequencies']
            rem_dict['IA'] = value['Instantaneous Amplitudes']
            rem_dict['cycles'] = value['Cycles']
            rem_dict['sample_rate'] = class_dict['sample_rate']
            rem_dict['freq_range'] = class_dict['freq_range']
            class_dict['REM'].append(REM_Segment(**rem_dict))
        elif key == 'header_info':
            class_dict['rat'] = value['rat']
            class_dict['study_day'] = value['study_day']
            class_dict['condition'] = value['condition']
            class_dict['trial']= value['trial']

    return class_dict


@dataclass
class Trial(SleepSignal):
    rat: int = field(default_factory=None)
    study_day: int = field(default_factory=None)
    condition: str = field(default_factory="")
    trial: str = field(default_factory="")

    def __post_init__(self):
        super().__post_init__()

    def build_dataset(self):
        df = super().build_dataset()
        if self.rat is not None:
            df['rat']=self.rat
            df['study_day']=self.study_day
            df['condition']=self.condition
            df['trial']=self.trial
        return df


