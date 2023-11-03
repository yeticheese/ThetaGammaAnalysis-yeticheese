from src.Signal import Signal
from src.pipeline import pipeline
import numpy as np

class ThetaSignal(Signal):
    def __init__(self, array: np.ndarray, sampling_rate: int, phasic=None, tonic=None):
        super().__init__(array, sampling_rate)
        
        self.phasic = phasic
        self.tonic = tonic
    
    def segment_cycles(self):
        self.phasic = pipeline(self, phasic=True)
        self.tonic = pipeline(self, phasic=False)
    
    def get_phasic(self):
        segments = []
        for period in self.phasic:
            start, end = period
            segments.append(self.filtered[start:end])
        return segments
    
    def get_tonic(self):
        segments = []
        for period in self.tonic:
            start, end = period
            segments.append(self.filtered[start:end])
        return segments