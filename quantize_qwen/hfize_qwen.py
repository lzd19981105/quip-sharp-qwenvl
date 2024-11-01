import sys
sys.path.append("../../quip-sharp/")
import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from lib.utils.model_version import MODEL_VERSION
from model.qwen.modeling_qwen import QWenLMHeadModel
from model.qwen.configuration_qwen import QWenConfig
from model.qwen.modeling_qwen import QWenModel

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']

    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path,trust_remote_code=True)

    # model_config.quip_params['model_version'] = MODEL_VERSION
    # model = QWenLMHeadModel.from_pretrained(model_config._name_or_path,
    #                                          torch_dtype='auto',
    #                                          low_cpu_mem_usage=True,
    #                                          config=model_config).half()
    # cpu = torch.device('cpu')
    # if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
    #     lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
    #                              map_location=cpu)
    #     model.lm_head.weight.copy_(lmhead_data['lm_head'])
    #     model.transformer.ln_f.weight.copy_(lmhead_data['norm'])

    # for ii in range(len(model.transformer.h)):
    #     layer = model.transformer.h[ii]

    #     if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
    #         ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
    #                              map_location=cpu)
    #         layer.ln_1.weight.copy_(ln_data['ln_1'])
    #         layer.ln_2.weight.copy_(ln_data['ln_2'])

    #     saved_layer = torch.load(f'{args.quantized_path}/{ii}_c_attn.pt',
    #                              map_location=cpu)
    #     utils.unpack_quip(layer.attn.c_attn, saved_layer, codebook_id,
    #                       codesz)

    #     saved_layer = torch.load(f'{args.quantized_path}/{ii}_attn_c_proj.pt',
    #                              map_location=cpu)
    #     utils.unpack_quip(layer.attn.c_proj, saved_layer, codebook_id,
    #                       codesz)

    #     saved_layer = torch.load(f'{args.quantized_path}/{ii}_w1.pt',
    #                              map_location=cpu)
    #     for i in range(len(saved_layer['scales'])):
    #         layer.mlp.upgate_proj.fuse_scales[i].copy_(
    #             saved_layer['scales'][i])
    #     utils.unpack_quip(layer.mlp.upgate_proj, saved_layer, codebook_id,
    #                       codesz)

    #     saved_layer = torch.load(f'{args.quantized_path}/{ii}_mlp_c_proj.pt',
    #                              map_location=cpu)
    #     utils.unpack_quip(layer.mlp.c_proj, saved_layer, codebook_id,
    #                       codesz)
    #     glog.info(f'loaded layer {ii} down')

    # glog.info(f'saving model...')
    # model.save_pretrained(args.hf_output_path, safe_serialization=True)

    # del model

    model, _ = model_from_hf_path(args.hf_output_path, use_cuda_graph=False)
    glog.info('successfully loaded hfized model')

    glog.info('generating some text...')
    
    start = time.time()
    prompt = 'It is a truth universally acknowledged that'
    # inputs = tokenizer(prompt, return_tensors='pt').to("cuda:0")
    model.generation_config = GenerationConfig.from_pretrained(
        model_config._name_or_path, trust_remote_code=True, resume_download=True, revision='master',
    )
    # out = model.generate(**inputs, max_new_tokens=256)
    # output_str = tokenizer.decode(out[0], skip_special_tokens=True)
    output_str,_ = model.chat(tokenizer,prompt,history=None)
    glog.info(output_str)
    glog.info(f'elapsed: {time.time() - start}')


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("",
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    del model
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
