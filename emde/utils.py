import torch
import numpy as np


def multi_query(concatenated_sketches, absolute_codes):
    """
    Decodes output of the model into probabilities for every product using variant of count-Min sketch retrieval procedure.
    Scores are computed as a geometric mean of the values from relevant buckets instead of min value in Count-Min sketch algorithm.
    :param np.array concatenated_sketches: output of the model (batch_size x output_size)
    :param np.array absolute_codes: abosulte codes for products (num_products x num_codes)
    :return: probability of decoded products (batch_size x num_products)
    """
    x = concatenated_sketches[:, absolute_codes]
    op_geom = np.log(1e-6+x)
    op_geom = op_geom.mean(-1)
    op_geom = np.exp(op_geom)
    return op_geom


def codes_to_sketch(codes, sketch_dim, n_sketches, n_modalities):
    """
    Convert codes into sketch sparse vector
    :param np.array codes: represent codes of buckets (num_products x num_codes)
    :param int sketch_dim: sketch width
    :param n_sketches sketch depth
    :param n_modalities: number of input upstream representations
    """
    pos_index = np.array([i*sketch_dim for i in range(n_sketches*n_modalities)], dtype=np.int_)
    index = codes + pos_index
    x = np.zeros(n_sketches*sketch_dim*n_modalities)
    for ind in index:
        x[ind] += 1

    return x


def categorical_cross_entropy(y_pred, y_true):
    """
    Computing categorical cross entropy loss
    """
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()