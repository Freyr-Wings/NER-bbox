import numpy as np
import random
import time
import torch

from torch import nn
from transformers import \
    BertTokenizer, BertModel, \
    AutoTokenizer, AutoModelForMaskedLM, \
    AdamW, get_linear_schedule_with_warmup, BertForTokenClassification

from data import get_dataloaders
from model import AlbertForNERBox
from utils import pred2bboxes, nested_ner_f1

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
train_dataloader, val_dataloader, test_dataloader = get_dataloaders('conll2003', tokenizer, batch_size=6)

model = AlbertForNERBox.from_pretrained(
    "albert-base-v2",
    output_attentions=False,
    output_hidden_states=False,
)

optimizer = AdamW(
    model.parameters(),
    lr=3e-5,
    eps=1e-8,
)

num_epochs = 100

total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model.to(device)

for epoch in range(num_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')
    total_train_loss = 0
    model.train()
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        if batch[2] is None:
            continue

        sentence_ids_batch = batch[0].to(device)
        attention_mask_batch = batch[1].to(device)
        span_labels_batch = batch[2].to(device)

        model.zero_grad()

        loss = model(
            input_ids=sentence_ids_batch,
            targets=span_labels_batch,
            attention_mask=attention_mask_batch
        )
        total_train_loss += loss.item()
        loss.backward()

        # prevent the "exploding gradients" problem
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        print(loss)

    duration = int(time.time() - start_time)
    avg_train_loss = total_train_loss / len(train_dataloader)

    print("")
    print("Time cost: {} minutes, {} seconds".format(duration // 60, duration % 60))
    print("Average training loss: {0:.6f}".format(avg_train_loss))
    print("")
    print("Running Training f1...")

    model.eval()

    tp, fp, fn = 0., 0., 0.
    for step, batch in enumerate(train_dataloader):
        sentence_ids_batch = batch[0].to(device)
        attention_mask_batch = batch[1].to(device)
        span_labels_batch = batch[2]
        offset_mapping_batch = batch[3].to(device)

        if span_labels_batch is not None:
            span_labels_batch = span_labels_batch.to(device)

        with torch.no_grad():
            pred_cls, pred_conf, pred_boxes = model(
                input_ids=sentence_ids_batch,
                targets=None,
                attention_mask=attention_mask_batch
            )

        # bboxes: batch_size * (num_token * num_anchor) * [cls, conf, start, end]
        bboxes = pred2bboxes(pred_cls, pred_conf, pred_boxes)

        for i, bbox in enumerate(bboxes):
            labels = []
            if span_labels_batch is not None:
                for label in span_labels_batch:
                    if label[0] == i:
                        labels.append(label[1:])
            tp_sentence, fp_sentence, fn_sentence = nested_ner_f1(bbox, labels, offset_mapping_batch[i])
            tp += tp_sentence
            fp += fp_sentence
            fn += fn_sentence

        if step > 1000:
            break

    precision = tp / (tp + fn + 1e-10)
    recall = tp / (tp + fp + 1e-10)
    f1score = precision * recall * 2 / (precision + recall + 1e-10)

    print("F1 score: {0:.6f}".format(f1score))
    print("Precision: {0:.6f}".format(precision))
    print("Recall: {0:.6f}".format(recall))
    print("")
    print("Running Validation...")

    model.eval()

    tp, fp, fn = 0., 0., 0.
    for step, batch in enumerate(val_dataloader):
        sentence_ids_batch = batch[0].to(device)
        attention_mask_batch = batch[1].to(device)
        span_labels_batch = batch[2]
        offset_mapping_batch = batch[3].to(device)

        if span_labels_batch is not None:
            span_labels_batch = span_labels_batch.to(device)

        with torch.no_grad():
            pred_cls, pred_conf, pred_boxes = model(
                input_ids=sentence_ids_batch,
                targets=None,
                attention_mask=attention_mask_batch
            )

        # bboxes: batch_size * num_token * num_anchor * [conf, start, end, p0, p1, p2, p3]
        bboxes = pred2bboxes(pred_cls, pred_conf, pred_boxes)

        for i, bbox in enumerate(bboxes):
            labels = []
            if span_labels_batch is not None:
                for label in span_labels_batch:
                    if label[0] == i:
                        labels.append(label[1:])
            tp_sentence, fp_sentence, fn_sentence = nested_ner_f1(bbox, labels, offset_mapping_batch[i])
            tp += tp_sentence
            fp += fp_sentence
            fn += fn_sentence

    precision = tp / (tp + fn + 1e-10)
    recall = tp / (tp + fp + 1e-10)
    f1score = precision * recall * 2 / (precision + recall + 1e-10)

    print("F1 score: {0:.6f}".format(f1score))
    print("Precision: {0:.6f}".format(precision))
    print("Recall: {0:.6f}".format(recall))
    print("")
    print("Running Testing...")

    model.eval()

    tp, fp, fn = 0., 0., 0.
    for step, batch in enumerate(test_dataloader):
        sentence_ids_batch = batch[0].to(device)
        attention_mask_batch = batch[1].to(device)
        span_labels_batch = batch[2]
        offset_mapping_batch = batch[3].to(device)

        if span_labels_batch is not None:
            span_labels_batch = span_labels_batch.to(device)

        with torch.no_grad():
            pred_cls, pred_conf, pred_boxes = model(
                input_ids=sentence_ids_batch,
                targets=None,
                attention_mask=attention_mask_batch
            )

        # bboxes: batch_size * (num_token * num_anchor) * [cls, conf, start, end]
        bboxes = pred2bboxes(pred_cls, pred_conf, pred_boxes)

        for i, bbox in enumerate(bboxes):
            labels = []
            if span_labels_batch is not None:
                for label in span_labels_batch:
                    if label[0] == i:
                        labels.append(label[1:])
            tp_sentence, fp_sentence, fn_sentence = nested_ner_f1(bbox, labels, offset_mapping_batch[i])
            tp += tp_sentence
            fp += fp_sentence
            fn += fn_sentence

    precision = tp / (tp + fn + 1e-10)
    recall = tp / (tp + fp + 1e-10)
    f1score = precision * recall * 2 / (precision + recall + 1e-10)

    print("F1 score: {0:.6f}".format(f1score))
    print("Precision: {0:.6f}".format(precision))
    print("Recall: {0:.6f}".format(recall))



