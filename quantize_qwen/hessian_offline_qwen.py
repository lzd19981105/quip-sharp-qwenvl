# 针对QwenVL，重构一下这个计算方式
import sys
sys.path.append("../../quip-sharp/")
import argparse
import datetime
import os
import random
from copy import deepcopy

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
import torch.multiprocessing as mp
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast)
import copy

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--devset_size', default=256, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model',
                    default='meta-llama/Llama-2-70b-hf',
                    type=str)
parser.add_argument('--save_path', default='hessians/llama2_7b', type=str)
parser.add_argument('--scratch_path', default=None, type=str)
parser.add_argument('--chunk_size', default=256, type=int)
parser.add_argument('--async_copy_speed', default=-1, type=int)
parser.add_argument('--act_save_rate', default=4, type=int)
parser.add_argument('--save_activations', action='store_true')
parser.add_argument('--sample_proc', default=4, type=int)



def move_fn(in_q, async_copy_speed):
    # async copy to avoid slow disk
    while True:
        item = in_q.get()
        if item is None:
            return
        src, tgt = item
        if async_copy_speed > 0:
            os.system(f'rsync --bwlimit={async_copy_speed} {src} {tgt}')
        else:
            os.system(f'rsync {src} {tgt}')
        os.system(f'rm {src}')
        print(f'moved {src} to {tgt}')


def forward_layer(layer, rotary_pos_emb, attention_mask, bs, device, in_q,
                  out_q,extra_kwargs):
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    rotary_pos_emb = utils.nested_move_to_device(rotary_pos_emb,device)
    attention_mask = attention_mask.to(device)
    done_c_attn = utils.register_H_hook(layer.attn.c_attn, device)
    done_attn_c_proj = utils.register_H_hook(layer.attn.c_proj, device)
    done_w1 = utils.register_H_hook(layer.mlp.w1, device)
    done_w2 = utils.register_H_hook(layer.mlp.w2, device)
    done_mlp_c_proj = utils.register_H_hook(layer.mlp.c_proj, device)

    while True:
        dev_emb = in_q.get()
        if dev_emb is None:
            layer = layer.cpu()
            rotary_pos_emb = utils.nested_move_to_device(rotary_pos_emb,"cpu")
            attention_mask = attention_mask.cpu()
            out_q.put({
                'c_attn': done_c_attn(),
                'attn.c_proj': done_attn_c_proj(),
                'w1': done_w1(),
                'w2': done_w2(), # 在之后用不上
                "mlp.c_proj":done_mlp_c_proj(),
            })
            return

        assert len(dev_emb) % bs == 0
        for i in range(len(dev_emb) // bs):
            dev_emb[i * bs:(i + 1) * bs] = layer(
                dev_emb[i * bs:(i + 1) * bs].to(device),
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                )[0].cpu()


def accumulate(in_q, move_q, ngpus, args, transformer_layer_index):
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape,
                                      dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape,
                                       dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    keys = list(Hs.keys())

    for key in Hs:
        mus[key].div_(cts[key])
        Hs[key].div_(cts[key])
        Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))
        save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" if args.scratch_path is not None else f"{args.save_path}/{transformer_layer_index}_{key}.pt"
        torch.save(
            {   
                # 因为H是对称矩阵，所以只保留下三角的数值以压缩矩阵
                'flatH': utils.sym_to_flat(Hs[key].to(torch.float32)),
                'mu': mus[key].to(torch.float32),
                'n': Hs[key].shape[0],
                'ct': cts[key]
            }, save_path)
        if args.scratch_path is not None:
            move_q.put(
                (f"{args.scratch_path}/{transformer_layer_index}_{key}.pt",
                 f"{args.save_path}/{transformer_layer_index}_{key}.pt"))

    del Hs, mus, cts, out


def main(args):
    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype="auto",
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    model.to("cuda:0")
    print("loaded model!")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    
    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("loading cached dataset...")
        loaded_dev_activations = torch.load(
            f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(
            f"loaded cached dataset from {loaded_dev_activations['timestamp']}"
        )
    else:
        print("loading dataset...")
        devset = utils.sample_rp1t(tokenizer,
                                   args.devset_size,
                                   args.ctx_size,
                                   nproc=args.sample_proc,
                                   model="qwen")
        after_layer = -1
        print("loaded dataset!")
    
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
    handle = model.transformer.h[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    try:
        for k, v in devset.items():
            v = v.to("cuda:0")
            devset[k] = v
        model(**devset)
    except ValueError:
        pass
    handle.remove()


    if args.scratch_path is not None:
        move_q = mp.Queue()
        move_p = mp.Process(target=move_fn,
                            args=(move_q, args.async_copy_speed))
        move_p.start()
    else:
        move_q = None

    # attention_mask和layer_input_kwargs中的部分参数需要和batch_size对齐
    attention_masks = [attention_masks[0][:args.batch_size]]

    for transformer_layer_index in range(len(model.transformer.h)):
        if (transformer_layer_index <= after_layer):
            print(
                f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}"
            )
            continue

        transformer_layer = model.transformer.h[transformer_layer_index]
        # check that there are four layers, as expected
        assert (len([
            m for m in transformer_layer.modules()
            if isinstance(m, torch.nn.Linear)
        ]) == 5)

        chunk_size = min(args.chunk_size, dev_emb[0][0].shape[0])
        ngpus = min(torch.cuda.device_count(), dev_emb[0][0].shape[0] // chunk_size)

        manager = mp.get_context('spawn').Manager()
        in_q = manager.Queue()
        out_q = manager.Queue()

        accumulate_proc = mp.Process(target=accumulate,
                                     args=(out_q, move_q, ngpus, args,
                                           transformer_layer_index))
        accumulate_proc.start()

        forward_procs = []
        for i in range(ngpus):
            p = mp.Process(target=forward_layer,
                           args=(transformer_layer, rotary_pos_emb[0],
                                 attention_masks[0], args.batch_size, i, in_q,
                                 out_q,layer_input_kwargs[0]))
            p.start()
            forward_procs.append(p)

        assert dev_emb[0][0].shape[0] % args.batch_size == 0 and chunk_size % args.batch_size == 0
        i = 0
        while i < len(dev_emb):
            next = min(i + chunk_size, dev_emb[0][0].shape[0])
            in_q.put(dev_emb[0][0][i:next])
            i = next

        for i in range(ngpus):
            in_q.put(None)

        for p in forward_procs:
            p.join()

        accumulate_proc.join()

        transformer_layer.cpu()
        model.transformer.h[transformer_layer_index] = None
        utils.clean()

        if args.save_activations and (
                transformer_layer_index % args.act_save_rate == 0 or \
                transformer_layer_index == len(model.model.layers) - 1):
            if args.scratch_path is not None:
                if os.path.exists(f'{args.scratch_path}/dev_activations.pt'):
                    print('not saving layer since disk is too slow')
                else:
                    torch.save(
                        {
                            'dev_emb': dev_emb,
                            'after_layer': transformer_layer_index,
                            'timestamp': str(datetime.datetime.now())
                        }, f'{args.scratch_path}/dev_activations.pt')
                    move_q.put((f'{args.scratch_path}/dev_activations.pt',
                                f'{args.save_path}/dev_activations.pt'))
            else:
                torch.save(
                    {
                        'dev_emb': dev_emb,
                        'after_layer': transformer_layer_index,
                        'timestamp': str(datetime.datetime.now())
                    }, f'{args.save_path}/dev_activations.pt')

        print(f"done processing layer {transformer_layer_index}")

    if args.scratch_path is not None:
        move_q.put(None)
        move_p.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
