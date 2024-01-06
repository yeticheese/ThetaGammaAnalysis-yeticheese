import numpy as np
import emd.sift as sift
class signal():
    def __init__(self,array:np.ndarray, sample_rate:int):
        self.signal = array
        self.sample_rate = sample_rate
        self.duration= np.linspace(0,len(array)/sample_rate,len(array))*1000
        self.imf= None
        self.mask_freq = None
        
    def iter_sift(self, mask='zc')-> tuple:
        self.imf,self.mask_freq =sift.iterated_mask_sift(self.signal,
                                                  mask_0=mask,
                                                  sample_rate=self.sample_rate,
                                                  ret_mask_freq=True)
    