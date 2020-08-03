import tensorflow as tf

def postprocess(heat, hm_hp, hp_offset, kps, reg, wh, K=100, thresh=0.1):

    heat = tensorflow.nn.sigmoid(heat)
    hm_hp = tensorflow.nn.sigmoid(hm_hp)
    batch, height, width, cat = heat.shape.as_list()
    num_joints = kps.shape.as_list()[3] // 2
    temp = tensorflow.pad(heat, [[0, 0], [1, 1], [1, 1], [0, 0]])
    hmax = tensorflow.nn.max_pool2d(temp, 3, 1, 'VALID')
    keep = tensorflow.cast(tensorflow.equal(hmax, heat), tensorflow.float32)
    heat = heat * keep

    heat = tensorflow.transpose(heat, [0, 3, 1, 2])
    topk_scores, inds = tensorflow.math.top_k(tensorflow.reshape(heat, [batch, cat, -1]), K)
    inds = inds % (height * width)
    ys = tensorflow.cast(inds // width, tensorflow.float32)
    xs = tensorflow.cast(inds % width, tensorflow.float32)
    scores, topk_ind = tensorflow.math.top_k(tensorflow.reshape(topk_scores, [batch, -1]), K)
    clses = topk_ind // K
    tempf = tensorflow.reshape(inds, [batch, -1])
    inds = tensorflow.gather(tempf, topk_ind, axis=1, batch_dims=1)
    tempf = tensorflow.reshape(ys, [batch, -1])
    ys = tensorflow.gather(tempf, topk_ind, axis=1, batch_dims=1)
    tempf = tensorflow.reshape(xs, [batch, -1])
    xs = tensorflow.gather(tempf, topk_ind, axis=1, batch_dims=1)

    kps = tensorflow.reshape(kps, [kps.shape.as_list()[0], -1, kps.shape.as_list()[3]])
    kps = tensorflow.gather(kps, inds, axis=1, batch_dims=1)
    kps = kps + tensorflow.tile(tensorflow.stack([xs, ys], -1), [1, 1, num_joints])

    reg = tensorflow.reshape(reg, [reg.shape.as_list()[0], -1, reg.shape.as_list()[3]])
    reg = tensorflow.gather(reg, inds, axis=1, batch_dims=1)
    xs = tensorflow.expand_dims(xs, -1) + reg[:, :, 0:1]
    ys = tensorflow.expand_dims(ys, -1) + reg[:, :, 1:2]

    wh = tensorflow.reshape(wh, [wh.shape.as_list()[0], -1, wh.shape.as_list()[3]])
    wh = tensorflow.gather(wh, inds, axis=1, batch_dims=1)
    clses = tensorflow.cast(tensorflow.expand_dims(clses, -1), tensorflow.float32)
    scores = tensorflow.expand_dims(scores, -1)

    bboxes = tensorflow.concat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], 2)
    temp = tensorflow.pad(hm_hp, [[0, 0], [1, 1], [1, 1], [0, 0]])
    hmax = tensorflow.nn.max_pool2d(temp, 3, 1, 'VALID')
    keep = tensorflow.cast(tensorflow.equal(hmax, hm_hp), tensorflow.float32)
    hm_hp = hm_hp * keep
    kps = tensorflow.transpose(tensorflow.reshape(kps, [batch, K, num_joints, 2]), [0, 2, 1, 3])
    reg_kps = tensorflow.repeat(tensorflow.expand_dims(kps, 3), K, 3)

    batch, height, width, cat = hm_hp.shape.as_list()
    hm_hp = tensorflow.transpose(hm_hp, [0, 3, 1, 2])
    hm_score, hm_inds = tensorflow.math.top_k(tensorflow.reshape(hm_hp, [batch, cat, -1]), K)
    hm_inds = hm_inds % (height * width)
    hm_ys = tensorflow.cast(hm_inds // width, tensorflow.float32)
    hm_xs = tensorflow.cast(hm_inds % width, tensorflow.float32)

    tempi = tensorflow.reshape(hm_inds, [batch, -1])
    hp_offset = tensorflow.reshape(hp_offset, [hp_offset.shape.as_list()[0], -1, hp_offset.shape.as_list()[3]])
    hp_offset = tensorflow.gather(hp_offset, tempi, axis=1, batch_dims=1)
    hp_offset = tensorflow.reshape(hp_offset, [batch, num_joints, K, 2])
    hm_xs = hm_xs + hp_offset[:, :, :, 0]
    hm_ys = hm_ys + hp_offset[:, :, :, 1]

    # thresh = 0.1
    mask = tensorflow.cast(tensorflow.greater(hm_score, thresh), tensorflow.float32)
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = tensorflow.repeat(tensorflow.expand_dims(tensorflow.stack([hm_xs, hm_ys], -1), 2), K, 2)
    dist = tensorflow.reduce_sum((reg_kps - hm_kps) ** 2, 4) ** 0.5
    min_dist = tensorflow.reduce_min(dist, 3)
    min_ind = tensorflow.arg_min(dist, 3)
    hm_score = tensorflow.gather(hm_score, min_ind, axis=2, batch_dims=2)
    hm_score = tensorflow.expand_dims(hm_score, -1)
    min_dist = tensorflow.expand_dims(min_dist, -1)
    min_ind = tensorflow.repeat(tensorflow.reshape(min_ind, [batch, num_joints, K, 1, 1]), 2, 4)
    hm_kps = tensorflow.gather(tensorflow.transpose(hm_kps, [0, 1, 2, 4, 3]), tensorflow.transpose(min_ind, [0, 1, 2, 4, 3]), axis=4, batch_dims=4)
    hm_kps = tensorflow.transpose(hm_kps, [0, 1, 2, 4, 3])
    hm_kps = tensorflow.reshape(hm_kps, [batch, num_joints, K, 2])
    l = tensorflow.repeat(tensorflow.reshape(bboxes[:, :, 0], [batch, 1, K, 1]), num_joints, 1)
    t = tensorflow.repeat(tensorflow.reshape(bboxes[:, :, 1], [batch, 1, K, 1]), num_joints, 1)
    r = tensorflow.repeat(tensorflow.reshape(bboxes[:, :, 2], [batch, 1, K, 1]), num_joints, 1)
    b = tensorflow.repeat(tensorflow.reshape(bboxes[:, :, 3], [batch, 1, K, 1]), num_joints, 1)
    t1 = tensorflow.cast(hm_kps[..., 0:1] < l, tensorflow.float32)
    t2 = tensorflow.cast(hm_kps[..., 0:1] > r, tensorflow.float32)
    t3 = tensorflow.cast(hm_kps[..., 1:2] < t, tensorflow.float32)
    t4 = tensorflow.cast(hm_kps[..., 1:2] > b, tensorflow.float32)
    t5 = tensorflow.cast(hm_score < thresh, tensorflow.float32)
    t6 = tensorflow.cast(min_dist > (tensorflow.maximum(b - t, r - l) * 0.3), tensorflow.float32)
    mask = t1 + t2 + t3 + t4 + t5 + t6
    mask = tensorflow.repeat(tensorflow.cast(mask > 0, tensorflow.float32), 2, 3)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = tensorflow.reshape(tensorflow.transpose(kps, [0, 2, 1, 3]), [batch, K, num_joints * 2])
    dets = tensorflow.concat([bboxes, scores, kps, clses], 2)

    bbox = dets[:, :, :4] * 4
    pts = dets[:, :, 5:39] * 4
    top_preds = tensorflow.concat([bbox, dets[:, :, 4:5], pts], 2)
    return top_preds
