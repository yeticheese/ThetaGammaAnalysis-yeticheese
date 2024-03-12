from dataclasses import dataclass, field
from src.signal import * 
from src.utils import get_file_dict
from src.functions import get_rem_states
import scipy.io as sio
import numpy as np
import os

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


