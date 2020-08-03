import tensorflow as tf

def postprocess( heatmap, wh, reg, kth=10):
    inshape= tf.shape(heatmap)
    N, out_h, out_w, out_c = inshape[0], inshape[1], inshape[2], inshape[3]
    scores, indices = tf.math.top_k(tf.reshape(heatmap, (N, -1)), k=kth)
    indices = tf.cast(indices, tf.int32)
    topk_classes = tf.floormod(indices, out_c, name='classes')
    topk_indices = tf.floordiv(indices, out_c)
    topk_ys = tf.floordiv(indices, out_w * out_c)
    topk_xs = tf.floordiv(tf.floormod(indices, out_w * out_c), out_c)
    center = tf.reshape(reg, (N, -1, 2))
    wh_tt  = tf.reshape(wh, (N, -1, 2))
    b_range = tf.range(0, 256)
    batch_indices = tf.tile(tf.expand_dims(tf.slice(b_range, [0], [N]), -1), [1, kth])
    batch_indices = tf.reshape(batch_indices, (-1, 1))
    topk_indices = tf.reshape(topk_indices, (-1, 1))
    reg_xs_indices = tf.zeros_like(batch_indices)
    reg_ys_indices = tf.ones_like(batch_indices)
    reg_xs = tf.concat([batch_indices, topk_indices, reg_xs_indices], axis=1)
    reg_ys = tf.concat([batch_indices, topk_indices, reg_ys_indices], axis=1)
    xs = tf.cast(tf.reshape(tf.gather_nd(center, reg_xs), (-1, kth)), tf.float32)
    ys = tf.cast(tf.reshape(tf.gather_nd(center, reg_ys), (-1, kth)), tf.float32)
    topk_xs = tf.math.add(tf.cast(topk_xs, tf.float32) , xs)
    topk_ys = tf.math.add(tf.cast(topk_ys, tf.float32) , ys)
    w = tf.cast(tf.reshape(tf.gather_nd(wh_tt, reg_xs), (-1, kth)), tf.float32)
    h = tf.cast(tf.reshape(tf.gather_nd(wh_tt, reg_ys), (-1, kth)), tf.float32)
    half_w = w / 2
    half_h = h / 2
    results = [tf.math.subtract(topk_xs, half_w), tf.math.subtract(topk_ys, half_h), w, h]
    results = [tf.expand_dims(dim, -1) for dim in results]
    boxes = tf.concat(results, axis=2)
    feat_shape = tf.cast(tf.stack([out_w, out_h, out_w, out_h]), tf.float32)
    boxes = tf.divide(boxes, feat_shape, name='boxes')
    bcs = tf.concat([
            boxes,
            tf.expand_dims(scores, -1),
            tf.cast(tf.expand_dims(topk_classes, -1), tf.float32),
        ],
        axis=2, name='detection')
    return bcs
