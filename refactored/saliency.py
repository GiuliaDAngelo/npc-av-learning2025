import numpy as np
import pySaliencyMap


# superclass for any saliency method
class Saliency(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_saliency_map(self, img):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_next_fixation(self, img):
        raise NotImplementedError("Subclasses should implement this!")



class IttiKochNieburSaliency(Saliency):
    def get_saliency_map(self, img):
        # For a given sequence, create fixation sequence using Itti-Koch-Niebur
        sm = pySaliencyMap.pySaliencyMap(self.width, self.height)
        # computation
        saliency_map = sm.SMGetSM(img) 
        #binarized_map = sm.SMGetBinarizedSM(img)
        #salient_region = sm.SMGetSalientRegion(img)
        return saliency_map

    def get_next_fixation(self, img):
        # Implement Itti-Koch-Niebur fixation selection
        sm = pySaliencyMap.pySaliencyMap(self.width, self.height)
        # computation
        saliency_map = sm.SMGetSM(img)
        # get the location (x,y) of the maximum pixel of the saliency_map
        max_loc = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
        # max_loc is in x,y convert to y,x
        max_loc = (max_loc[1], max_loc[0])
        # get the value at that location
        max_val = saliency_map[max_loc]
        return max_loc, max_val



class Config:
    MAX_X, MAX_Y = 128, 128

    OMS_PARAMS = {
        'size_krn_center': 8,
        'sigma_center': 1,
        'size_krn_surround': 8,
        'sigma_surround': 4,
        'threshold': 0.96,
        'tau_memOMS': 0.3,
        'sc': 1,
        'ss': 1
    }

    ATTENTION_PARAMS = {
        'size_krn': 16,
        'r0': 14,
        'rho': 0.05,
        'theta': np.pi * 3 / 2,
        'thetas': np.arange(0, 2 * np.pi, np.pi / 4),
        'thick': 3,
        'fltr_resize_perc': [2, 2],
        'offsetpxs': 0,
        'offset': (0, 0),
        'num_pyr': 6,
        'tau_mem': 0.3,
        'stride': 1,
        'out_ch': 1
    }
config = Config()

