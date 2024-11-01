import sys
sys.path.append("../../quip-sharp/")
import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune, quip
from lib.linear import FusedLinear
from model.qwen.modeling_qwen import QWenBlock

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--hessian_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--incoh_mode',
                    default='had',
                    type=str,
                    choices=['had', 'kron'])
parser.add_argument('--lora_rank',
                    default=0,
                    type=int,
                    help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--resid_scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=5e-5, type=float)
parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
parser.add_argument('--ft_bs', default=1, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')


def check_exist(idx, args):
    suffix = ['c_attn', 'attn_c_proj', "update_proj", "mlp_c_proj",'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_qwen_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb,
                         model_config,rotary_pos_emb,attention_mask):
    if check_exist(idx, args):
        return

    mixed_layer = QWenBlock(model_config).cpu()
    with torch.no_grad():

        mixed_layer.attn.c_attn = layer.attn.c_attn

        mixed_layer.attn.c_proj = layer.attn.c_proj

        weights = [layer.mlp.w1.weight, layer.mlp.w2.weight]
        fused_upgate_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                        weights[0].shape[1],
                                        sum([_.shape[0] for _ in weights]),
                                        bias=False)
        cur = 0
        for w in weights:
            fused_upgate_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]
        mixed_layer.mlp.upgate_proj = fused_upgate_proj

        mixed_layer.mlp.c_proj = layer.mlp.c_proj

        mixed_layer.ln_1.weight.copy_(layer.ln_1.weight)
        mixed_layer.ln_2.weight.copy_(layer.ln_2.weight)

    finetune.quantize_finetune_decoder_layer(mixed_layer,
                                             [('attn.c_attn', 'c_attn'),
                                              ('attn.c_proj', 'attn_c_proj'),
                                              ('mlp.upgate_proj', 'w1'),
                                              ('mlp.c_proj', 'mlp_c_proj')], idx,
                                             cb, args, device, pre_orig_emb,
                                             orig_emb,rotary_pos_emb,attention_mask,mode="qwen")

    torch.save(
        {
            'ln_1':
            mixed_layer.ln_1.weight,
            'ln_2':
            mixed_layer.ln_2.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')
    del mixed_layer


def main(args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = codebook.get_codebook(args.codebook)

    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    model.to("cuda:0")

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'codesz': cb.codesz,
        'idx_dtype': str(cb.idx_dtype),
        'packsz': cb.packsz,
        'resid_scale_override': args.resid_scale_override,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model,trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    glog.info('loaded model')

    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc,model="qwen")
    glog.info('loaded dataset and devset')

    nproc = torch.cuda.device_count()

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
    # attention_mask和layer_input_kwargs中的部分参数需要和batch_size对齐
    attention_masks = [attention_masks[0][:args.batch_size]]

    orig_emb_cache = [dev_emb[0][0]]
    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                        dtype=orig_emb_cache[0].dtype,
                        device=orig_emb_cache[0].device))

    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(model.transformer.h)):
        glog.info(f'layer {i} gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device].join()
            if cur_device == 0:
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            proc_list[cur_device + 1].join()
        utils.clean()

        if args.ft_epochs > 0:
            st = time.time()
            # position_ids = position_ids.to(cur_device)
            rotary_pos_emb = utils.nested_move_to_device(rotary_pos_emb,cur_device)
            attention_masks = utils.nested_move_to_device(attention_masks,cur_device)
            model.transformer.h[i].to(cur_device)
            for j in range(args.devset_size // args.batch_size):
                orig_emb_cache[cur_device + 1][
                    args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.transformer.h[i](
                        orig_emb_cache[cur_device][
                            args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        rotary_pos_emb=rotary_pos_emb[0],
                        attention_mask=attention_masks[0],
                        use_cache=False,
                        output_attentions=False)[0].cpu()
            model.transformer.h[i].cpu()
            orig_msv = orig_emb_cache[cur_device].float().norm(
            )**2 / orig_emb_cache[cur_device].numel()
            target_msv = orig_emb_cache[cur_device + 1].float().norm(
            )**2 / orig_emb_cache[cur_device + 1].numel()
            rotary_pos_emb = utils.nested_move_to_device(rotary_pos_emb,"cpu")
            attention_masks = utils.nested_move_to_device(attention_masks,"cpu")
            utils.clean()
            glog.info(
                'computed original embedding for layer {} in {}s, pre msv {}, post msv {}'
                .format(i,
                        time.time() - st, orig_msv, target_msv))

        proc_list[cur_device] = mp.Process(target=quantize_qwen_layer,
                                           args=(
                                               model.transformer.h[i],
                                               i,
                                               cb,
                                               args,
                                               cur_device,
                                               orig_emb_cache[cur_device],
                                               orig_emb_cache[cur_device + 1],
                                               all_config['model_config'],
                                               rotary_pos_emb[0],
                                               attention_masks[0],
                                           ))
        proc_list[cur_device].start()

        cur_device = (cur_device + 1) % nproc

    for p in proc_list:
        p.join()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
