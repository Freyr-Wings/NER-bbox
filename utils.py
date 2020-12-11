import torch


def xw2xx(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 1] / 2
    y[..., 1] = x[..., 0] + x[..., 1] / 2
    return y


def bbox_iou(box1, box2, two_point_repr=False):
    if two_point_repr:
        b1_x1, b1_x2 = box1[..., 0], box1[..., 1]
        b2_x1, b2_x2 = box2[..., 0], box2[..., 1]
    else:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 1] / 2, box1[..., 0] + box1[..., 1] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 1] / 2, box2[..., 0] + box2[..., 1] / 2

    inter_x = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    union_x = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)

    iou = inter_x / (union_x + 1e-16)

    return iou


def non_max_suppression(bbox, conf_threshold=0.5, nms_threshold=0.25):
    # bbox: num_token * num_anchor * [conf, start, end, p0, p1, p2, p3]
    # 0<=conf<=1, 0<=p<=1, 0<=threshold<=1

    num_token = bbox.size(0)
    num_anchor = bbox.size(1)
    num_classes = bbox.size(-1) - 3

    valid_bbox = bbox[bbox[bbox[..., 0] > conf_threshold]]
    candidates = []
    for span in valid_bbox:
        candidates.append(span)

    outputs = []
    while candidates:
        score_max = 0.0
        bbox_max = []
        for span in candidates:
            if span[0] > score_max:
                score_max = span[0]
                bbox_max = span
        outputs.append(bbox)
        for m in range(len(candidates) - 1, -1, -1):
            score, bbox = candidates[m]
            if candidates[m] >= nms_threshold:
                candidates.pop(m)
    return outputs


def flat_ner_f1(bbox, labels, offset_mapping):
    # bbox: output from NMS, should be a dict as below:
    # {0:[tensor([p0, p1, p2, p3, c, start, end]), ...], ...}
    # labels: [[cls, x_ori, w_ori, x_processed, w_processed], ...]
    # offset_mapping: [0, 0, 1, 2, 2, 2, 3, 3, 4, ...]

    num_token = bbox.size(0)
    num_anchor = bbox.size(1)
    num_classes = bbox.size(-1) - 3

    pred_tokens = torch.zeros(num_token, dtype=torch.long)
    label_tokens = torch.zeros(num_token, dtype=torch.long)

    pred_conf = bbox[..., 0]
    pred_obj = pred_conf > 0.5

    if torch.sum(pred_obj) > 0:
        pred_span = bbox[..., 1:3][pred_obj]
        pred_cls = torch.argmax(bbox[..., 3:][pred_obj], dim=1).view(-1)
        pred_dec_span = offset_mapping[torch.round(pred_span).long()].view(-1, 2)

        for i, span in enumerate(pred_dec_span):
            for j in range(span[0], span[1]):
                pred_tokens[j] = pred_cls[i]+1

    if len(labels) > 0:
        for i, span in enumerate(labels):
            for j in range(span[3], span[3] + span[4]):
                label_tokens[j] = span[0]+1

    tp = torch.sum((pred_tokens > 0) * (label_tokens == pred_tokens))
    fp = torch.sum((pred_tokens > 0) * (label_tokens != pred_tokens))
    fn = torch.sum((pred_tokens != label_tokens) * (label_tokens > 0))
    return tp, fp, fn


def nested_ner_f1(bbox, labels, offset_mapping):
    # bbox: output from NMS, should be a dict as below:
    # {0:[tensor([p0, p1, p2, p3, c, start, end]), ...], ...}
    # labels: [[cls, x_ori, w_ori, x_processed, w_processed], ...]
    # offset_mapping: [0, 0, 1, 2, 2, 2, 3, 3, 4, ...]

    num_token = bbox.size(0)
    num_anchor = bbox.size(1)
    num_classes = bbox.size(-1) - 3

    pred_tokens = torch.zeros(num_classes, num_token, dtype=torch.long)
    label_tokens = torch.zeros(num_classes, num_token, dtype=torch.long)

    pred_conf = bbox[..., 0]
    pred_obj = pred_conf > 0.5

    if torch.sum(pred_obj) > 0:
        pred_span = bbox[..., 1:3][pred_obj]
        pred_cls = torch.argmax(bbox[..., 3:][pred_obj], dim=1).view(-1)
        pred_dec_span = offset_mapping[torch.round(pred_span).long()].view(-1, 2)

        for i, span in enumerate(pred_dec_span):
            for j in range(span[0], span[1]):
                pred_tokens[pred_cls[i], j] = 1

    if len(labels) > 0:
        for i, span in enumerate(labels):
            for j in range(span[3], span[3] + span[4]):
                label_tokens[span[0], j] = 1

    tp = torch.sum((pred_tokens == 1) * (label_tokens == 1))
    fp = torch.sum((pred_tokens == 1) * (label_tokens == 0))
    fn = torch.sum((pred_tokens == 0) * (label_tokens == 1))
    return tp, fp, fn


def pred2bboxes(pred_cls, pred_conf, pred_boxes):
    # pred_cls: batch_size * num_token * num_anchor * num_classes
    # pred_conf: batch_size * num_token * num_anchor
    # pred_boxes: batch_size * num_token * num_anchor (width)
    # output: batch_size * (num_token * num_anchor) * [conf, start, end, p0, p1, p2, p3]

    device = pred_cls.device
    batch_size, num_token, num_anchor, num_classes = pred_cls.shape
    pred_conf = pred_conf.view(batch_size, num_token, num_anchor, 1)
    pred_boxes = pred_boxes.view(batch_size, num_token, num_anchor, 1)
    start = torch.tensor(range(num_token), dtype=torch.float).to(device)
    start = start.repeat(num_anchor, 1).transpose(0, 1).unsqueeze(0)
    # start = torch.stack((start, start), dim=1)
    # start = start.repeat(num_anchor)
    start = start.repeat(batch_size, 1, 1).view(batch_size, num_token, num_anchor, 1)
    end = start + pred_boxes
    bboxes = torch.cat((pred_conf, start, end, pred_cls), dim=-1)

    return bboxes


if __name__ == '__main__':
    aa = torch.rand(13, 5)
    bb = torch.rand(5)

    iou = bbox_iou(aa, bb, two_point_repr=True)






