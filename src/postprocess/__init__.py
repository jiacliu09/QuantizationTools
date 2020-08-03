from .tf.tf_centernet_postprocess import postprocess as tf_centernet_postprocess
from .tf.tf_centerpose_postprocess import postprocess  as tf_centerpose_postprocess

postprocessor_factory = {
        'tf_centernet_postprocess': tf_centernet_postprocess,
        'tf_centerpose_postprocess': tf_centerpose_postprocess,
}
