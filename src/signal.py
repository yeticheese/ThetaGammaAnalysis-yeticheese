import numpy
import emd.sift as sift
class signal():
    def __init__(self,array:np.ndarray, sampling_rate:int):
        self.signal = array
        self.duration= len(array)/sampling_rate

    def iter_sift(self,sample_rate)