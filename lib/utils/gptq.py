import torch
from torch import LongTensor
from typing import List,Dict
import copy

def collate_data(blocks: List[Dict[str, List[List[int]]]], pad_token_id: int) -> Dict[str, LongTensor]:
    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1)

    input_ids_blocks = [LongTensor(block["input_ids"]) for block in blocks]
    attention_mask_blocks = [LongTensor(block["attention_mask"]) for block in blocks]
    label_blocks = [LongTensor(block["labels"]) for block in blocks]

    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    label_max_len = max([block.size(-1) for block in label_blocks])

    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        block_label_len = label_blocks[i].shape[-1]
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    return {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
        "labels": torch.cat(label_blocks, dim=0).long(),
    }

# qwen
def prepare_examples_for_quantization(
        examples,
        model,
        batch_size=128,
    ):
        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_examples = []
        for example in examples:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])
            if "labels" in example:
                labels = _convert_tensor_to_list(example["labels"])
            elif "label" in example:
                labels = _convert_tensor_to_list(example["label"])
            elif "label_ids" in example:
                labels = _convert_tensor_to_list(example["label_ids"])
            else:
                labels = copy.deepcopy(input_ids)
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        pad_token_id = model.config.pad_token_id
        if not pad_token_id:
            pad_token_id = model.config.eos_token_id

        new_examples = [
            collate_data(new_examples[start : start + batch_size], pad_token_id)
            for start in range(0, len(new_examples), batch_size)
        ]
        for new_example in new_examples:
            del new_example["labels"]

        return new_examples

def get_device(obj):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device

def move_to_device(obj, device):
    if obj is None:
        return obj
    else:
        if get_device(obj) != device:
            obj = obj.to(device)
        return obj

def nested_move_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to_device(e, device) for e in v])
    else:
        return v