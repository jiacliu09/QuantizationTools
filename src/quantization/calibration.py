import os
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from scipy import stats
from ..reconstructor import TFReconstructor

def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist

def threshold_distribution(distribution, target_bin=128):
    """
    Return the best threshold value.
    Ref: https://github.com//apache/incubator-mxnet/blob/master
                    /python/mxnet/contrib/quantization.py
    Args:
        distribution: list, activations has been processed by
        histogram and normalize,size is 2048
        target_bin: int, the num of bin that is used by quantize,
        Int8 default value is 128
    Returns:
        target_threshold: int, num of bin with the minimum KL
    """
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = deepcopy(distribution[:threshold])
        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        #
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate
        # quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin

        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        # q[p == 0] = 0
        p = _smooth_distribution(p)
        q = _smooth_distribution(q)
        # p[p == 0] = 0.0001
        # q[q == 0] = 0.0001

        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value

spec = lambda x, i : x + ':' + str(i)

class Calibration(object):

    def __init__(self, graph, params, dataset):
        super(Calibration, self).__init__()
        self.graph = graph
        self.model = TFReconstructor(graph, params)
        self.dataset = dataset
        self.model._execute()

    def run(self):
        sess = tf.Session(graph=self.model.tf_graph)
        node_names = [node['name'] for node in self.graph]
        tensors =dict((spec(k, 0), self.model.node_dict[spec(k, 0)]) for k in node_names)
        input_tsrs = [self.model.node_dict[spec(name, 0)] for name in self.model.input_node_ids]

        table = dict()
        for idx in tqdm(range(len(self.dataset))):
            input_arr = self.dataset[idx]
            results = sess.run(tensors, feed_dict={input_tsrs[0]: input_arr})
            for name in results:
                if name not in table:
                    table[name] = dict(min=100000, max=-100000)
                table[name]['min'] = min(table[name]['min'], float(results[name].min()))
                table[name]['max'] = max(table[name]['max'], float(results[name].max()))

        for name in self.model.params:
            table[spec(name, 0)] = dict()
            if self.model.params[name].ndim != 4:
                table[spec(name, 0)]['max'] = float(np.absolute(self.model.params[name]).max())
            else:
                weight = self.model.params[name]
                weight_max = np.absolute(weight).reshape(-1, weight.shape[3]).max(axis=0)
                table[spec(name, 0)]['max'] = weight_max.tolist()
        sess.close()
        return table

    def run_kl(self):
        init_table = self.run()

        sess = tf.Session(graph=self.model.tf_graph)
        node_names = [node['name'] for node in self.graph]
        tensors =dict((spec(k, 0), self.model.node_dict[spec(k,0)]) for k in node_names)
        input_tsrs = [self.model.node_dict[spec(name,0)] for name in self.model.input_node_ids]

        table = dict((spec(n,0), dict(min=100000, max=-100000)) for n in node_names)
        distubution = dict((spec(n,0), np.zeros(2048, dtype=np.float32)) for n in node_names)

        for idx in tqdm(range(len(self.dataset))):
            input_arr = self.dataset[idx]
            results = sess.run(tensors, feed_dict={input_tsrs[0]: input_arr})
            for name in results:
                th = max(1e-5, -init_table[name]['min'], init_table[name]['max'])
                blob_data = np.absolute(results[name])
                hist, hist_edge = np.histogram(blob_data, bins=2048, range=(0,th))
                distubution[name] += hist

        for name in distubution:
            threshold_value = threshold_distribution(distubution[name])
            th = max(1e-5, -init_table[name]['min'], init_table[name]['max'])
            T = (threshold_value + 0.5) * th / 2048
            table[name]['min'] = -T
            table[name]['max'] = T

        for name in self.model.params:
            table[spec(name, 0)] = dict()
            if self.model.params[name].ndim != 4:
                table[spec(name, 0)]['max'] = float(np.absolute(self.model.params[name]).max())
            else:
                weight = self.model.params[name]
                weight_max = np.absolute(weight).reshape(-1, weight.shape[3]).max(axis=0)
                table[spec(name, 0)]['max'] = weight_max.tolist()
        sess.close()
        return table

