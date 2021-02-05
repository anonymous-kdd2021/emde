import sys
import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from ..utils import codes_to_sketch


class SessionRecoDataset(Dataset):
    def __init__(self, data, product2codes, sketch_dim, n_sketches, n_modalities, alpha, W):
        """
        :param pd.DataFrame data: Data frame with examples. 
                Consists of 4 columns: 
                history - product ids in user history, separated by space, e.g. 219718 117889 351798 130121 450404 227688
                target - single next product in session, e.g. 246589
                target_multi - ale next product in session, e.g. 246589, 287323
                time_diff - time difference between products in history, e.g. 0 107 40 43 31 105
        :param dict product2codes: dictionary that maps product to codes
        :param int sketch_dim: sketch width
        :param int n_sketches: sketch depth
        :param int n_modalities: number of input modalities
        :param float alpha:
        :param float W:
        """
        self.data = data
        self.product2codes = product2codes
        self.sketch_dim = sketch_dim
        self.n_sketches = n_sketches
        self.n_modalities = n_modalities
        self.alpha = alpha
        self.W = W

    def __len__(self):
        return len(self.data)
    
    def time_decay_sketches(self, history):
        """
        Create input history sketch with time decay.
        Sketch doesn't include the last item in history (another sketch is used to keep last item)
        Sketch is l2 normalized
        """
        history_sketch = np.zeros((self.n_sketches * self.sketch_dim * self.n_modalities,))
        if len(history) > 1:
            for h_idx, h in enumerate(history[:-1]):
                c_sketch = codes_to_sketch(self.product2codes[h][None], self.sketch_dim, self.n_sketches, self.n_modalities)
                history_sketch = pow(self.alpha, self.W * self.timestamp_diff[h_idx]) * history_sketch + c_sketch
        history_sketch = normalize(history_sketch.reshape(-1, self.sketch_dim), 'l2').reshape((self.n_sketches * self.sketch_dim * self.n_modalities,))
        return history_sketch

    def __getitem__(self, idx):
        row = dict(self.data.iloc[idx, :])
        history = row['history'].split()
        target = row['target']
        self.timestamp_diff = [float(i) for i in row['time_diff'].split()]
        history_sketch = self.time_decay_sketches(history)

        history_sketch_last = codes_to_sketch(self.product2codes[history[-1]][None], self.sketch_dim, self.n_sketches, self.n_modalities)
        history_sketch_last = normalize(history_sketch_last.reshape(-1, self.sketch_dim), 'l2').reshape((self.n_sketches * self.sketch_dim * self.n_modalities,))
        target_sketch = codes_to_sketch(self.product2codes[target][None], self.sketch_dim, self.n_sketches, self.n_modalities)
        return {
            'input': history_sketch,
            'input_last': history_sketch_last,
            'output': target_sketch,
            'target': target,
            'target_multi': row['target_multi']
        }
    