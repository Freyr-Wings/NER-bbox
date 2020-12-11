import torch

from transformers import AutoTokenizer

from model import AlbertForNERBox
from utils import pred2bboxes, nested_ner_f1

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = AlbertForNERBox.from_pretrained(
    "./pretrain",
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)
model.eval()
cls_name = ["LOC", "ORG", "PER", "MISC"]


def visualize(example):
    # example = "CRICKET - ENGLISH COUNTY CHAMPIONSHIP SCORES ."
    example_enc = tokenizer.encode_plus(example, return_tensors='pt', return_attention_mask=True, padding='max_length',
                                        max_length=156)
    example_tokens = tokenizer.convert_ids_to_tokens(example_enc['input_ids'][0])

    num_cls = len(cls_name)
    num_token = torch.sum(example_enc['attention_mask']).long().item()

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
    pred_conf = pred_bbox[..., 0]
    pred_obj = pred_conf > 0.4

    if torch.sum(pred_obj) > 0:
        pred_span = pred_bbox[..., 1:3][pred_obj]
        pred_span = torch.clip(pred_span, 1, num_token)
        pred_span = torch.round(pred_span).long().view(-1, 2)
        pred_cls = torch.argmax(pred_bbox[..., 3:][pred_obj], dim=1).view(-1)
        # pred_dec_span = offset_mapping[torch.round(pred_span).long()].view(-1, 2)

        for i, span in enumerate(pred_span):
            for j in range(span[0], span[1]):
                pred_tokens[pred_cls[i], j] = 1

    row_format = "{:>15}" * (num_token + 1)
    print(row_format.format("", *example_tokens))
    for i in range(num_cls):
        print(row_format.format(cls_name[i], *pred_tokens[i].numpy()))









