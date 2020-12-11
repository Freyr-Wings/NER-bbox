import torch
import torch.utils.checkpoint
from torch import nn
from transformers import BertModel, BertPreTrainedModel, AlbertModel, AlbertPreTrainedModel


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = torch.nn.functional.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class AlbertForNERBox(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.anchors = range(1, 5)  # [1.36, 4.00]
        self.num_anchor = len(self.anchors)
        self.num_classes = 4
        self.obj_scale = 40
        self.noobj_scale = 400

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.bbox_trans = MultiNonLinearClassifier(
            self.hidden_size,
            self.num_anchor * (2 + self.num_classes),
            0.1
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.cre_loss = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        targets=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        predictions = self.bbox_trans(sequence_output)

        num_batch = predictions.size(0)
        num_token = predictions.size(1)

        predictions = predictions.view(
            num_batch, num_token, self.num_anchor, 2 + self.num_classes
        )

        pred_rel_w = predictions[..., 0]
        pred_conf = torch.sigmoid(predictions[..., 1])
        pred_cls = predictions[..., 2:]

        anchors = torch.tensor(self.anchors, device=predictions.device, dtype=torch.float)
        # pred_boxes = FloatTensor(pred_w.shape)
        pred_boxes = torch.exp(pred_rel_w) * anchors

        if targets is None:
            return pred_cls, pred_conf, pred_boxes

        obj_mask, noobj_mask, label_w, label_conf, label_cls = build_targets(
            pred_boxes=pred_boxes,
            targets=targets,
            anchors=self.anchors,
        )

        loss_w = self.mse_loss(pred_boxes[obj_mask], label_w)
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], label_conf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], label_conf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.cre_loss(pred_cls[obj_mask].view(-1, self.num_classes), label_cls)
        total_loss = loss_w + loss_conf + loss_cls

        return total_loss


class BertForNERBox(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.anchors = [1.26, 2.42, 4.30]
        self.num_anchor = len(self.anchors)
        self.num_classes = 4
        self.obj_scale = 1
        self.noobj_scale = 100

        self.bert = BertModel(config, add_pooling_layer=False)
        self.bbox_trans = MultiNonLinearClassifier(
            self.hidden_size,
            self.num_anchor * (2 + self.num_classes),
            0.1
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.cre_loss = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        targets=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # output_attention:
        # list of ``torch.FloatTensor``(one for each layer)
        # each with shape (batch_size, num_heads, sequence_length, sequence_length)

        sequence_output = outputs[0]
        predictions = self.bbox_trans(sequence_output)

        num_batch = predictions.size(0)
        num_token = predictions.size(1)

        predictions = predictions.view(
            num_batch, num_token, self.num_anchor, 2 + self.num_classes
        )

        pred_rel_w = predictions[..., 0]
        pred_conf = torch.sigmoid(predictions[..., 1])
        pred_cls = predictions[..., 2:]

        anchors = torch.tensor(self.anchors, device=predictions.device, dtype=torch.float)
        # pred_boxes = FloatTensor(pred_w.shape)
        pred_boxes = torch.exp(pred_rel_w) * anchors

        if targets is None:
            return pred_cls, pred_conf, pred_boxes

        obj_mask, noobj_mask, label_w, label_conf, label_cls = build_targets(
            pred_boxes=pred_boxes,
            targets=targets,
            anchors=self.anchors,
        )

        label_w_noobj = torch.zeros_like(pred_boxes[obj_mask], device=predictions.device)

        loss_w = self.mse_loss(pred_boxes[obj_mask], label_w)
        loss_w_noobj = self.mse_loss(pred_boxes[noobj_mask], label_w_noobj)
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], label_conf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], label_conf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.cre_loss(pred_cls[obj_mask].view(-1, self.num_classes), label_cls)
        total_loss = loss_w + loss_w_noobj + loss_conf + loss_cls

        return total_loss


def build_targets(pred_boxes, targets, anchors, ignore_thres=0.01):
    device = pred_boxes.device
    nB, nT, nA = pred_boxes.shape

    # labels: [batch_idx, cls, x, w, x_ori, w_ori]

    obj_mask = torch.zeros((nB, nT, nA), device=device, dtype=torch.bool)
    noobj_mask = torch.ones((nB, nT, nA), device=device, dtype=torch.bool)

    # label_rel_w = torch.zeros((nB, nT, nA), device=device, dtype=torch.float)

    batch_idx = targets[:, 0].long()
    label_cls = targets[:, 1].long()
    label_pos = targets[:, 2].long()
    label_w = targets[:, 3].float()

    bbox_scores = torch.stack([torch.square(label_w - width) for width in anchors])
    best_scores, best_n = bbox_scores.min(0)

    obj_mask[batch_idx, label_pos, best_n] = 1
    noobj_mask[batch_idx, label_pos, best_n] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_score in enumerate(bbox_scores.t()):
        noobj_mask[batch_idx[i], label_pos[i], anchor_score < ignore_thres] = 0

    # label_rel_w[batch_idx, label_pos, best_n] = torch.log(label_w / anchors[best_n])
    label_conf = obj_mask.float()

    return obj_mask, noobj_mask, label_w, label_conf, label_cls



# if __name__ == '__main__':
#     num_batch = 32
#     num_token = 156
#     num_anchor = 2
#     num_classes = 4
#
#     predictions = torch.rand(num_batch, num_token, num_anchor * (2 + num_classes))
#     predictions[:, 100:] = 0
#
#     predictions = predictions.view(
#         num_batch, num_token, num_anchor, 2 + num_classes
#     )
#
#     pred_rel_w = predictions[..., 0]
#     print(pred_rel_w.shape)
#     pred_conf = torch.sigmoid(predictions[..., 1])
#     pred_cls = predictions[..., 2:]
#     print(pred_cls.shape)
#
#     anchors = torch.FloatTensor([1., 2.1])
#     print(anchors)
#     # pred_boxes = FloatTensor(pred_w.shape)
#     pred_boxes = torch.exp(pred_rel_w) * anchors
#     print(pred_boxes.shape)
#
#     targets = torch.FloatTensor(10, 6).fill_(0)
#     targets[:, 0] = torch.tensor([0, 1, 1, 3, 4, 6, 9, 10, 11, 11])
#     targets[:, 1] = torch.randint(0, 4, (10,))
#     targets[:, 2] = torch.randint(0, 99, (10,))
#     targets[:, 3] = torch.randint(0, 10, (10,))
#     # targets[]
#     print(targets)
#
#     obj_mask, noobj_mask, label_rel_w, label_conf, label_cls = build_targets(
#         pred_boxes=pred_boxes,
#         targets=targets,
#         anchors=anchors,
#     )
#
#     loss_rel_w = nn.MSELoss()(pred_boxes[obj_mask], label_rel_w)
#     loss_conf_obj = nn.BCELoss()(pred_conf[obj_mask], label_conf[obj_mask])
#     loss_conf_noobj = nn.BCELoss()(pred_conf[noobj_mask], label_conf[noobj_mask])
#     loss_conf = loss_conf_obj + 30 * loss_conf_noobj
#
#     loss_cls = nn.CrossEntropyLoss()(pred_cls[obj_mask].view(-1, num_classes), label_cls)
#     total_loss = loss_rel_w + loss_conf + loss_cls
#
#     print(pred_cls[obj_mask].shape)
#     print(loss_rel_w, loss_conf_obj, loss_conf_noobj, loss_conf, loss_cls, total_loss)
#
#     # print(obj_mask)
#     # print(noobj_mask)
#     # print(label_rel_w)
#     # print(total_loss)
