import torch
import numpy as np
from torch import nn


def visualize(tokenizer):
    example = "CRICKET - ENGLISH COUNTY CHAMPIONSHIP SCORES ."
    example_enc = tokenizer.encode_plus(example, return_tensors='pt', return_attention_mask=True, padding='max_length',
                                        max_length=156)
    # print(example_enc['attention_mask'])
    example_tokens = tokenizer.convert_ids_to_tokens(example_enc['input_ids'][0])
    # print(type(example_tokens))
    # print(example_tokens)

    cls_name = ["LOC", "ORG", "PER", "MISC"]
    num_cls = len(cls_name)
    num_token = torch.sum(example_enc['attention_mask']).long()

    example_tokens = [t.replace('▁', '#') for t in example_tokens]

    example_ids = example_enc['input_ids'].to(device)
    example_attention = example_enc['attention_mask'].to(device)

    pred_cls, pred_conf, pred_boxes = model(
        input_ids=example_ids,
        targets=None,
        attention_mask=example_attention
    )

    bboxes = pred2bboxes(pred_cls, pred_conf, pred_boxes)
    pred_bbox = bboxes[0]

    pred_tokens = torch.zeros(num_cls, num_token, dtype=torch.long)
    pred_conf = bbox[..., 0]
    pred_obj = pred_conf > 0.5

    if torch.sum(pred_obj) > 0:
        pred_span = bbox[..., 1:3][pred_obj]
        pred_span = torch.clip(pred_span, 1, num_token)
        pred_span = torch.round(pred_span).long().view(-1, 2)
        pred_cls = torch.argmax(bbox[..., 3:][pred_obj], dim=1).view(-1)
        # pred_dec_span = offset_mapping[torch.round(pred_span).long()].view(-1, 2)

        for i, span in enumerate(pred_span):
            for j in range(span[0] - 1, span[1] - 1):
                pred_tokens[pred_cls[i], j] = 1

    row_format = "{:>15}" * (num_token + 1)
    print(row_format.format("", *example_tokens))
    for i in range(num_cls):
        print(row_format.format(cls_name[i], *pred_tokens[i].numpy()))



if __name__ == '__main__':
    # hidden_size = 3000
    # num_classes = 10
    # num_bbox_labels = 3
    # ignore_threshold = 0.5
    #
    # bbox_trans = nn.Linear(hidden_size, num_bbox_labels + num_classes)
    #
    # hidden_output = torch.rand((3, 64, hidden_size))
    #
    # predictions = bbox_trans(hidden_output)
    #
    # tx = predictions[:, :, 0].unsqueeze(2)
    # tw = predictions[:, :, 1].unsqueeze(2)
    #
    # confidence = predictions[:, :, 2].unsqueeze(2)
    # labels = predictions[:, :, 3:].unsqueeze(2)
    #
    # print(tx.shape)


    # print(hidden_output[])

    padded_w = 500
    padded_h = 300
    h_factor, w_factor = (1, 1)
    boxes = torch.from_numpy(np.loadtxt('train.txt').reshape(-1, 5))
    # Extract coordinates for unpadded + unscaled image
    x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

    # Returns (x, y, w, h)
    boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    boxes[:, 3] *= w_factor / padded_w
    boxes[:, 4] *= h_factor / padded_h

    targets = torch.zeros((len(boxes), 6))
    targets.round()
    targets[:, 1:] = boxes

    print(targets)
    print(targets.shape)

    # target_boxes = targets[:, 2:6] * nG
    #
    # gxy = target_boxes[:, :2]
    # gwh = target_boxes[:, 2:]
    # # Get anchors with best iou
    # ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # best_ious, best_n = ious.max(0)
    # # Separate target values
    # b, target_labels = targets[:, :2].long().t()
    # gx, gy = gxy.t()
    # gw, gh = gwh.t()
    # gi, gj = gxy.long().t()
    # # Set masks
    # obj_mask[b, best_n, gj, gi] = 1
    # noobj_mask[b, best_n, gj, gi] = 0
    #
    # # Set noobj mask to zero where iou exceeds ignore threshold
    # for i, anchor_ious in enumerate(ious.t()):
    #     noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    #
    # # Coordinates
    # tx[b, best_n, gj, gi] = gx - gx.floor()
    # ty[b, best_n, gj, gi] = gy - gy.floor()
    # # Width and height
    # tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    # th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # # One-hot encoding of label
    # tcls[b, best_n, gj, gi, target_labels] = 1
    # # Compute label correctness and iou at best anchor
    # class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    #
    # tconf = obj_mask.float()
    # return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    # sentences = []
    # classes = []
    # cur_sentence = ''
    # len_sentence = 0
    # with open('conll2003/valid.txt') as f:
    #     for l in f:
    #         line = l[:-1]
    #         if len(line) == 0 and len_sentence > 0:
    #             sentences.append(cur_sentence)
    #             print(len_sentence)
    #             cur_sentence = ''
    #             len_sentence = 0
    #             continue
    # 
    #         tokens = line.split(' ')
    #         cur_sentence += tokens[0] + ' '
    #         len_sentence += 1


    wh1 = torch.tensor([1., 2.4])
    wh2 = torch.rand((7, 2))
    ret = bbox_wh_iou(wh1, wh2)
    print(ret)
    print(ret.shape)


    # ious = torch.stack([bbox_wh_iou(torch.rand((2)), wh2) for _ in range(5)])
    #
    # best_ious, best_n = ious.max(0)
    # print(ious)
    # print(best_ious, best_n)

    ByteTensor = torch.ByteTensor
    FloatTensor = torch.FloatTensor

    pred_width = FloatTensor().fill_(0)

    predef_widths = [1., 2.4]

    bbox_width = targets[:, 1]
    bbox_scores = torch.stack([predef_width_score(bbox_width, width) for width in predef_widths])
    best_scores, best_n = bbox_scores.min(0)
    print(bbox_scores)
    print(best_scores, best_n)

    data = {'text': 'hello world', 'label': ''}

    pred_idx = torch.rand((5, 10, 2))

    for i in pred_idx:
        print(i.shape)
        break
    # pred = torch.rand((5, 10, 2, 4))
    # pred_cls = torch.argmax(pred, dim=-1)
    # print(pred_cls)
    # print(pred[pred_idx > 0.5])
    #
    # start = torch.tensor(range(156), dtype=torch.float)
    # # start = torch.stack((start, start), dim=1)
    # start = start.repeat(4, 1).transpose(0, 1).unsqueeze(0)
    # print(start.shape)
    # print(start)
    # # start.repeat()
    # start = start.repeat(5, 1, 1).view(5, 156, 4, 1).contiguous()
    # print(start[:, :, 0])
    # print(start.shape)

    num_batch = 6
    seq_len = 156
    hidden_size = 738
    start_outputs = torch.nn.Linear(hidden_size, 1)
    end_outputs = torch.nn.Linear(hidden_size, 1)
    span_embedding = torch.nn.Linear(hidden_size*2, 1)

    sequence_heatmap = torch.rand(num_batch, seq_len, hidden_size)
    start_logits = start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
    end_logits = end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

    # for every position $i$ in sequence, should concate $j$ to
    # predict if $i$ and $j$ are start_pos and end_pos for an entity.
    # [batch, seq_len, seq_len, hidden]
    start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
    # [batch, seq_len, seq_len, hidden]
    end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
    # [batch, seq_len, seq_len, hidden*2]
    span_matrix = torch.cat([start_extend, end_extend], 3)
    # [batch, seq_len, seq_len]
    span_logits = span_embedding(span_matrix).squeeze(-1)

    print(start_logits.shape)

    attentions = torch.ones(16)
    tokens = ['[CLS]', '▁eu', '▁reject', 's', '▁german', '▁call', '▁to', '▁boycott', '▁british', '▁lamb', '▁', '.', '[SEP]']
    ids = [2, 2898, 12170, 18, 548, 645, 20, 16617, 388, 8624, 13, 9, 3]

    tokens = [t.replace('▁', '#') for t in tokens]

    row_format = "{:>15}" * (len(tokens)-10)
    print(row_format.format(*tokens))
    for id in ids:
        print(row_format.format(*ids))
