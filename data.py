import torch

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from transformers import BertTokenizer


def get_dataloaders(base_path, tokenizer, batch_size=32):
    train_dataset = CONLLDataset(base_path + '/train.txt', tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=2,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataset = CONLLDataset(base_path + '/valid.txt', tokenizer)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        num_workers=2,
        collate_fn=val_dataset.collate_fn,
    )

    test_dataset = CONLLDataset(base_path + '/test.txt', tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        num_workers=2,
        collate_fn=test_dataset.collate_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader


def prepare_block_labels(block, tokenizer, max_sentence_len=156):
    # block is a list containing strings as below:
    # EU NNP B-NP B-ORG\n
    # rejects VBZ B-VP O\n
    # German JJ B-NP B-MISC\n
    # call NN I-NP O\n

    class_mapping = {"LOC": 0, "ORG": 1, "PER": 2, "MISC": 3}
    document_begin = tokenizer.encode('[CLS]', return_tensors='pt', add_special_tokens=False,
                                      return_attention_mask=False)
    document_end = tokenizer.encode('[SEP]', return_tensors='pt', add_special_tokens=False, return_attention_mask=False)

    block = [line[:-1].split() for line in block]
    n = len(block)  # number of tokens

    sentence = [line[0] for line in block]
    labels = [line[3] for line in block]

    sentence_ids = list()
    span_labels = list()
    offset_mapping = list()

    sentence_ids.append(document_begin)
    offset_mapping.append(torch.ones_like(document_begin, dtype=torch.long) * (-1))

    next_ori_idx = 0
    next_enc_idx = 1

    entity_ori_begin = -1
    entity_enc_begin = -1
    entity_type = "none"

    for i in range(n):
        if labels[i][0] == "B":
            if entity_enc_begin >= 0:
                # save
                label = torch.tensor(
                    [0, class_mapping[entity_type],
                     entity_enc_begin, next_enc_idx - entity_enc_begin,
                     entity_ori_begin, next_ori_idx - entity_ori_begin, ],
                    dtype=torch.long
                )  # [batch_idx, cls, x, w, x_ori, w_ori]
                span_labels.append(label)

            entity_ori_begin = next_ori_idx
            entity_enc_begin = next_enc_idx
            entity_type = labels[i][2:]

        elif labels[i][0] == "I":
            pass
        else:
            if entity_enc_begin >= 0:
                label = torch.tensor(
                    [0, class_mapping[entity_type],
                     entity_enc_begin, next_enc_idx - entity_enc_begin,
                     entity_ori_begin, next_ori_idx - entity_ori_begin, ],
                    dtype=torch.long
                )  # [batch_idx, cls, x, w, x_ori, w_ori]
                span_labels.append(label)
                entity_ori_begin, entity_enc_begin = -1, -1
            pass

        enc = tokenizer.encode(sentence[i], return_tensors='pt', add_special_tokens=False,
                               return_attention_mask=False, )
        sentence_ids.append(enc)

        offset = torch.ones_like(enc, dtype=torch.long) * next_ori_idx
        offset_mapping.append(offset)

        next_ori_idx += 1
        next_enc_idx += enc.shape[1]

    if entity_enc_begin >= 0:
        label = torch.tensor(
            [0, class_mapping[entity_type],
             entity_enc_begin, next_enc_idx - entity_enc_begin,
             entity_ori_begin, next_ori_idx - entity_ori_begin, ],
            dtype=torch.long
        )  # [batch_idx, cls, x, w, x_ori, w_ori]
        span_labels.append(label)

    sentence_ids.append(document_end)
    offset_mapping.append(torch.ones_like(document_end, dtype=torch.long) * next_ori_idx)

    sentence_ids = torch.cat(sentence_ids, dim=1)
    attention_mask = torch.ones_like(sentence_ids, dtype=torch.long)
    offset_mapping = torch.cat(offset_mapping, dim=1)

    padding_fct = nn.ConstantPad1d((0, max_sentence_len - sentence_ids.shape[1]), 0)

    sentence_ids = padding_fct(sentence_ids)  # auto truncate
    attention_mask = padding_fct(attention_mask)
    offset_mapping = padding_fct(offset_mapping)

    if len(span_labels) > 0:
        span_labels = torch.stack(span_labels)
    else:
        span_labels = None

    return sentence_ids, attention_mask, span_labels, offset_mapping


def gen_data(infile, tokenizer):
    f = open(infile, 'r')
    sentence_ids_list = []
    attention_mask_list = []
    span_labels_list = []
    offset_mapping_list = []
    block = []

    for line in f:
        if line == '-DOCSTART- -X- -X- O\n':
            continue
        if line == '\n':
            if len(block) > 0:
                sentence_ids, attention_mask, span_labels, offset_mapping = prepare_block_labels(block, tokenizer)
                sentence_ids_list.append(sentence_ids)
                attention_mask_list.append(attention_mask)
                span_labels_list.append(span_labels)
                offset_mapping_list.append(offset_mapping)

                block = []
            continue
        block.append(line)
    if len(block) > 0:
        sentence_ids, attention_mask, span_labels, offset_mapping = prepare_block_labels(block, tokenizer)
        sentence_ids_list.append(sentence_ids)
        attention_mask_list.append(attention_mask)
        span_labels.append(span_labels)
        offset_mapping_list.append(offset_mapping)

    f.close()

    return sentence_ids_list, attention_mask_list, span_labels_list, offset_mapping_list


class CONLLDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.sentence_ids_list, self.attention_mask_list, \
            self.span_labels_list, self.offset_mapping_list = gen_data(path, tokenizer)

    def __getitem__(self, i):
        return self.sentence_ids_list[i], self.attention_mask_list[i], \
               self.span_labels_list[i], self.offset_mapping_list[i]

    def __len__(self):
        return len(self.sentence_ids_list)

    @staticmethod
    def collate_fn(batch):
        sentence_ids_batch, attention_mask_batch, span_labels_batch, offset_mapping_batch = list(zip(*batch))
        for i, boxes in enumerate(span_labels_batch):
            if boxes is not None:
                boxes[:, 0] = i
        span_labels_batch = [boxes for boxes in span_labels_batch if boxes is not None]

        if len(span_labels_batch) == 0:
            span_labels_batch = None
        else:
            span_labels_batch = torch.cat(span_labels_batch, dim=0)
        sentence_ids_batch = torch.cat(sentence_ids_batch, dim=0)
        attention_mask_batch = torch.cat(attention_mask_batch, dim=0)
        offset_mapping_batch = torch.cat(offset_mapping_batch, dim=0)
        return sentence_ids_batch, attention_mask_batch, span_labels_batch, offset_mapping_batch


# if __name__ == '__main__':
#     sentences = [
#         torch.rand((64, 300)),
#         torch.rand((64, 300))
#     ]
#     targets = [
#         torch.zeros((4, 6)),  # idx, cls, x, w
#         torch.zeros((2, 6)),
#     ]
#     # Remove empty placeholder targets
#     targets = [boxes for boxes in targets if boxes is not None]
#     # Add sample index to targets
#     for i, boxes in enumerate(targets):
#         boxes[:, 0] = i
#     targets = torch.cat(targets, 0)
#     sentences = torch.stack(sentences)
#     print(targets, targets.shape)
#
#     b, target_labels = targets[:, :2].long().t()
#
#     print(b)
#     print(target_labels)
