
import time
import torch

from collections import defaultdict, Counter

def IOU(s1, e1, s2, e2):
    if s1 > e2 or s2 > e1:
        return torch.tensor(0)
    I = min(e1, e2) - max(s1, s2) + 1
    U = max(e1, e2) - min(s1, s2) + 1
    return torch.div(I.float(), U.float())

def pred2bboxes(pred_cls, pred_conf, pred_boxes):
    # pred_cls: batch_size * num_token * num_anchor * num_classes
    # pred_conf: batch_size * num_token * num_anchor
    # pred_boxes: batch_size * num_token * num_anchor (width)
    # output: batch_size * (num_token * num_anchor) * [p0, p1, p2, p3, conf, start, end]

    batch_size, num_token, num_anchor, num_classes = pred_cls.shape
    pred_conf = pred_conf.view(batch_size, num_token, num_anchor, 1)
    pred_boxes = pred_boxes.view(batch_size, num_token, num_anchor, 1)
    start = torch.tensor(range(num_token)).to(device)
    start = torch.stack((start, start), dim=1)
    start = start.repeat(batch_size, 1).view(batch_size, num_token, num_anchor, 1)
    end = start + pred_boxes
    bboxes = torch.cat((pred_cls, pred_conf, start, end), dim=-1).view(batch_size, -1, 7)
    return bboxes

def NMS(bboxes, IOU_threshold, score_threshold):
    # bboxes: (num_token * num_anchor) * [p0, p1, p2, p3, conf, start, end]
    # 0<=conf<=1, 0<=p<=1, 0<=threshold<=1

    n = len(bboxes)         # number of bboxes
    k = len(bboxes[0]) - 3  # number of entity classes
    outputs = defaultdict(list)

    for i in range(k):
        # calculate NMS outputs for kth class
        candidates = []
        for j in range(n):
            conf = bboxes[j][-3]
            score = bboxes[j][i] * conf
            if score > score_threshold:
                candidates.append([score, bboxes[j]])
        while candidates:
            score_max = 0.0
            bbox_max = []
            for score, bbox in candidates:
                if score > score_max:
                    score_max = score
                    bbox_max = bbox
            outputs[i].append(bbox_max)
            for m in range(len(candidates) - 1, -1, -1):
                score, bbox = candidates[m]
                if IOU(bbox_max[-2], bbox_max[-1], bbox[-2], bbox[-1]) >= IOU_threshold:
                    candidates.pop(m)
    return outputs

def processed2ori(start_processed, end_processed, offset_mapping):
    offset_mapping = list(offset_mapping)
    valid_tokens = len(offset_mapping) - offset_mapping[::-1].index(max(offset_mapping)) - 1
    counter = Counter(offset_mapping[:valid_tokens + 1])
    print("start_processed: ", start_processed)
    start_ori = offset_mapping[start_processed]
    end_ori = offset_mapping[end_processed]
    pred_counter = Counter(offset_mapping[start_processed:end_processed + 1])
    # if the predicted range connot cover more than half of the processed tokens
    # then discard that original token
    if pred_counter[start_ori] < counter[start_ori] * 0.5:
        start_ori += 1
    if pred_counter[end_ori] < counter[end_ori] * 0.5:
        end_ori -= 1
    return start_ori, end_ori


def yoro_span_f1(output_bboxes, labels, offset_mapping):
    # output_bboxes: output from NMS, should be a dict as below:
    # {0:[tensor([p0, p1, p2, p3, c, start, end]), ...], ...}
    # labels: [[cls, x_ori, w_ori, x_processed, w_processed], ...]
    # offset_mapping: [0, 0, 1, 2, 2, 2, 3, 3, 4, ...]

    seq_len = max(offset_mapping)
    # calculate tp, fp, fn for each class
    tp, fp, fn = 0.0, 0.0, 0.0
    num_classes = len(output_bboxes.keys())  # number of entity classes in prediction
    if not num_classes or not labels:
        return tp, fp, fn

    for c in output_bboxes.keys():
        match_pred = [0] * seq_len
        match_true = [0] * seq_len
        bboxes = output_bboxes[c]
        for bbox in bboxes:
            start_processed = int(bbox[-2])
            end_processed = int(bbox[-1])
            start_ori, end_ori = processed2ori(start_processed, end_processed, offset_mapping)
            for i in range(start_ori, end_ori):
                match_pred[i] = 1
        for label in labels:
            if label[0] == c:
                for i in range(label[1], label[2] + 1):
                    match_true[i] = 1
        for i in range(seq_len):
            p = match_pred[i]
            t = match_true[i]
            if p == t:
                tp += 1
            if p and not t:
                fp += 1
            if not p and t:
                fn += 1

    tp /= num_classes
    fp /= num_classes
    fn /= num_classes
    return tp, fp, fn


def evaluate(dataloader, IOU_threshold, score_threshold, model):
    start_time = time.time()
    tp, fp, fn = 0.0, 0.0, 0.0
    for sentence_ids_batch, attention_mask_batch, span_labels_batch, offset_mapping_batch in dataloader:
        sentence_ids_batch = sentence_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        span_labels_batch = span_labels_batch.to(device)
        offset_mapping_batch = offset_mapping_batch.to(device)

        pred_cls, pred_conf, pred_boxes = model(input_ids=sentence_ids_batch, targets=None)
        bboxes = pred2bboxes(pred_cls, pred_conf, pred_boxes)
        for i in range(len(bboxes)):
            output_bboxes = NMS(bboxes[i], IOU_threshold, score_threshold)
            span_labels_batchi = []
            for label in span_labels_batch:
                if label[0] == i:
                    span_labels_batchi.append(label[1:])
            tp_sentence, fp_sentence, fn_sentence = yoro_span_f1(output_bboxes, span_labels_batchi,
                                                                 offset_mapping_batch[i])
            tp += tp_sentence
            fp += fp_sentence
            fn += fn_sentence
    precision = tp / (tp + fn + 1e-10)
    recall = tp / (tp + fp + 1e-10)
    f1score = precision * recall * 2 / (precision + recall + 1e-10)
    end_time = time.time()

    return f1score








