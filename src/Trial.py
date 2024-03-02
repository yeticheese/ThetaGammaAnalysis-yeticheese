from dataclasses import dataclass, field
from src.Signal import * 
from src.utils import get_file_dict
from src.functions import get_rem_states
import scipy.io as sio
import numpy as np
import os

@dataclass
class Trial(SleepSignal):
    rat: int = field(init=False)
    study_day: int = field(init=False)
    condition: str = field(init=False)
    trial: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()  # Call parent's __post_init__ to initialize common fields
        file_dict = get_file_dict(self.root)
        self.rat = file_dict['rat']
        self.study_day = file_dict['study_day']
        self.condition = file_dict['condition']
        self.trial = file_dict['trial']

    def build_dataset(self):
        df = super().build_dataset()
        df['rat']=self.rat
        df['study_day']=self.study_day
        df['condition']=self.condition
        df['trial']=self.trial
        return df