from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class VCoder(object):
  def __init__(self, n_sketches, sketch_dim, input_dim):
    self.n_sketches = n_sketches
    self.sketch_dim = sketch_dim
    self.input_dim = input_dim
    self.standard_scaler = StandardScaler()
    if self.input_dim < 10000:
        self.random_projection = GaussianRandomProjection(n_components = 16*n_sketches)
    else:
        self.random_projection = SparseRandomProjection(n_components = 16*n_sketches, density = 1/3.0)

  def fit(self, v):
    self.standard_scaler = self.standard_scaler.fit(v)
    v = self.standard_scaler.transform(v)
    self.random_projection = self.random_projection.fit(v)
    v = self.random_projection.transform(v)
    self.init_biases(v)

  def transform(self, v):
    v = self.standard_scaler.transform(v)
    v = self.random_projection.transform(v)
    v = self.discretize(v)
    v = np.packbits(v, axis=-1)
    v = np.frombuffer(np.ascontiguousarray(v), dtype=np.uint16).reshape(v.shape[0], -1) % self.sketch_dim
    return v


class DLSH(VCoder):
    """
    Density-dependent Local Sensitive Hashing.
    Described in section 3.2 in the paper https://arxiv.org/pdf/2006.01894.pdf.
    Example of use:
    ```
    embeddings - numpy array with upstream representatnion, shape num_items x embedding_dimension
    sketch_dim = 128
    n_sketches = 10
    coder = dlsh.DLSH(n_sketches, sketch_dim, emb_dim)
    coder.fit(embeddings)
    codes = coder.transform(embeddings)
    ```
    """
    def __init__(self, n_sketches, sketch_dim, input_dim):
        """
        param int n_sketches: sketch depth
        param int sketch_dim: sketch width
        param int input_dim:  dimension of input upstream representatnion
        """
        super(DLSH, self).__init__(n_sketches, sketch_dim, input_dim)

    def init_biases(self, v):
        """
        Drawing the bias value from data-dependent quantile function.
        """
        self.biases = np.array([np.percentile(v[:, i], q=100*np.random.rand(), axis=0) for i in range(v.shape[1])])


    def discretize(self, v):
        """
        Hash function that specifies the cutoff point
        """
        return ((np.sign(v - self.biases)+1)/2).astype(np.uint8)
