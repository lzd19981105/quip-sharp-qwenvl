import sys
sys.path.append("../../quip-sharp/")
import argparse
import copy
import datetime
import gc
import math
import os
import time

from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import glog
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune, quip
from lib.utils.unsafe_import import model_from_hf_path

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=64, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--base_model', type=str)
parser.add_argument('--hf_path', type=str)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--ft_lr', default=1e-5, type=float)
parser.add_argument('--ft_susv_lr', default=1e-4, type=float)
parser.add_argument('--ft_bs', default=8, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=1, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--ft_nshards', default=-1, type=int)


def get_qwen_save_fn(args):

    def save_fn(shard_model):
        ct = 0
        for i in range(len(shard_model.shards)):
            for j in range(len(shard_model.shards[i].layers)):
                shard = shard_model.shards[i].layers[j]
                utils.save_susv(shard.attn.c_attn,
                                f'{args.ckpt_path}/{ct}_c_attn.pt')
                utils.save_susv(shard.attn.c_proj,
                                f'{args.ckpt_path}/{ct}_attn_c_proj.pt')
                utils.save_susv(shard.mlp.upgate_proj,
                                f'{args.ckpt_path}/{ct}_w1.pt')
                utils.save_susv(shard.mlp.c_proj,
                                f'{args.ckpt_path}/{ct}_mlp_c_proj.pt')
                torch.save(
                    {
                        'ln_1':
                        shard.ln_1.weight,
                        'ln_2':
                        shard.ln_2.weight,
                    }, f'{args.ckpt_path}/{ct}_layernorm.pt')
                glog.info(f'wrote layer {ct}')
                ct += 1
        torch.save(
            {
                'lm_head': shard_model.output_layer[1].weight,
                'norm': shard_model.output_layer[0].weight,
            }, f'{args.ckpt_path}/lmhead.pt')

    return save_fn


def get_qwen_save_fn_quant(args):

    def save_fn(quant_model):
        ct = 0
        for j in range(len(quant_model.transformer.h)):
            shard = quant_model.transformer.h[j]
            utils.save_susv(shard.attn.c_attn,
                            f'{args.ckpt_path}/{ct}_c_attn.pt')
            utils.save_susv(shard.attn.c_proj,
                            f'{args.ckpt_path}/{ct}_attn_c_proj.pt')
            utils.save_susv(shard.mlp.upgate_proj,
                            f'{args.ckpt_path}/{ct}_w1.pt')
            utils.save_susv(shard.mlp.c_proj,
                            f'{args.ckpt_path}/{ct}_mlp_c_proj.pt')
            torch.save(
                {
                    'ln_1':
                    shard.ln_1.weight,
                    'ln_2':
                    shard.ln_1.weight,
                }, f'{args.ckpt_path}/{ct}_layernorm.pt')
            glog.info(f'wrote layer {ct}')
            ct += 1
        torch.save(
            {
                'lm_head': quant_model.lm_head.weight,
                'norm': quant_model.transfomrer.ln_f.weight,
            }, f'{args.ckpt_path}/lmhead.pt')

    return save_fn


def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    return args[0]


def main(args):
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc,model="qwen")

    orig_model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                      torch_dtype=torch.bfloat16,
                                                      device_map='auto',
                                                      low_cpu_mem_usage=True,
                                                      trust_remote_code=True)
    orig_logits = utils.calculate_logits(orig_model, devset, args.batch_size,model_type="qwen")
    # orig_logits = orig_logits[:, :-1].contiguous().softmax(dim=-1).float()
    orig_logits = orig_logits.contiguous().softmax(dim=-1).float()

    del orig_model
    utils.clean()

    quant_model = model_from_hf_path(args.hf_path,
                                     use_cuda_graph=False,
                                     use_flash_attn=False,
                                     device_map=None)[0].cpu()

    # torch.set_grad_enabled(True)
    # finetune.finetune_susv_e2e_qwen_2(quant_model, orig_logits, devset,args,get_qwen_save_fn_quant(args))

    # 在qwen.transformer.h的第一层插入hook，统计输入，参照GPTQ
    dev_emb = []
    attention_masks = []
    rotary_pos_emb = []
    layer_input_kwargs = []
    data_device = "cpu" # cpu
    def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(utils.move_to_device(inp, data_device))
            dev_emb.append(layer_input)

            # Keyword arguments.
            if kwargs["attention_mask"] is not None:
                attention_masks.append(kwargs["attention_mask"].to(data_device))
            else:
                attention_masks.append(None)

            rotary_pos = kwargs.get("rotary_pos_emb", None)
            if rotary_pos is not None:
                rotary_pos_emb.append(utils.nested_move_to_device(rotary_pos, data_device))
            one_kwargs = {}
            for (
                k,
                v,
            ) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "rotary_pos_emb"]:
                    one_kwargs[k] = utils.nested_move_to_device(v, data_device)
            layer_input_kwargs.append(one_kwargs)
            raise ValueError
    handle = quant_model.transformer.h[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    try:
        for k, v in devset.items():
            # v = v.to("cuda:0")
            v = v.to("cpu")
            devset[k] = v
        quant_model(**devset)
    except ValueError:
        pass
    handle.remove()
    # attention_mask和layer_input_kwargs中的部分参数需要和batch_size对齐
    attention_masks = [attention_masks[0][:args.ft_bs]]

    # construct shards
    nshards = torch.cuda.device_count(
    ) if args.ft_nshards < 0 else args.ft_nshards
    nlayers = len(quant_model.transformer.h)
    shards = [nn.ModuleList([]) for _ in range(nshards)]
    for i in range(nshards):
        for j in range(int(nlayers * i / nshards),
                       int(nlayers * (i + 1) / nshards)):
            shards[i].append(quant_model.transformer.h[j])
        shards[i] = {'device': i, 'arg_fn': llama_arg_fn, 'shard': shards[i]}
    output_layer = {
        'layer': nn.Sequential(quant_model.transformer.ln_f, quant_model.lm_head),
        'fn': get_emb
    }


    shard_model = utils.ShardTransformer(shards, output_layer,
                                         args.ft_grad_ckpt, args.ft_train_mode)
    rotary_pos_emb = utils.nested_move_to_device(rotary_pos_emb,"cuda:0")
    shard_model.manifest(dev_emb[0][0][:args.ft_bs],
                         rotary_pos_emb=rotary_pos_emb[0],
                         attention_mask=attention_masks[0])
    utils.clean()

    torch.set_grad_enabled(True)
    # 不对lmhead的参数进行修改
    # shard_model.output_layer[1].weight.required_grad = False
    finetune.finetune_susv_e2e_qwen(shard_model, orig_logits, dev_emb[0][0], rotary_pos_emb[0],
                               attention_masks[0], get_qwen_save_fn(args), args)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
