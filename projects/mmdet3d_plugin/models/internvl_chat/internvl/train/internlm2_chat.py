import gc
import json
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers

from internvl.dist_utils import init_dist
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.ao.quantization.utils import weight_dtype
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)

from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from mmdet.models import HEADS
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

from ipdb import set_trace
import warnings

# Apply necessary patches for the transformers library
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default="./pretrained/InternVL2-2B",
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    llm_path: Optional[str] = field(
        default="./pretrained/internlm2-chat-1_8b",
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM decoder.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default="./shell/data/coco_caption.json",
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )


def wrap_llm_lora(llm, r=128, lora_alpha=256, lora_dropout=0.05, target_modules=None):
    # Determine the target modules based on the architecture of the language model
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type='CAUSAL_LM'
    )
    llm = get_peft_model(llm, lora_config)
    llm.enable_input_require_grads()
    llm.print_trainable_parameters()

def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

@HEADS.register_module()
class MLPV2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MLPV2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            nn.GELU(),
            nn.Linear(output_channel, output_channel),
        )

    def forward(self, x):
        if not self.training:
            x_dtype = x.dtype
            weight_dtype = self.fc1[0].weight.dtype
            x = x.to(weight_dtype)
        x = self.fc1(x)
        if not self.training:
            x = x.to(x_dtype)
        return x


@HEADS.register_module()
class HERMESLLM(nn.Module):
    def __init__(self, model_path="./pretrained/InternVL2-2B",
                 load_weight=True,
                 set_lora=True,
                 is_pretraining=False,
                 chat_cfg=None,
                 input_dim=1024,
                 attention_type='flash_attention_2',
                 img_length=60,
                 num_learnable_query=0,
                 ):
        super(HERMESLLM, self).__init__()
        self.tokenizer_path = model_path
        # self.llm_path = llm_path
        self.model_path = model_path

        self.template_name = 'internlm2-chat'
        self.ds_name = 'sth'
        self.load_weight = load_weight
        self.img_length = img_length
        self.chat_cfg = chat_cfg

        self.model_name = model_path.split("/")[-1]
        if self.model_name == "InternVL2-1B":
            self.template_name = "Hermes-2"
            self.preprocess_func = preprocess_mpt
            self.target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                                   'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
            llm_channel = 896
        elif self.model_name == "InternVL2-2B":
            self.template_name = "internlm2-chat"
            self.preprocess_func = preprocess_internlm
            self.target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2',
                                   'feed_forward.w3']
            llm_channel = 2048
        elif self.model_name == "InternVL2-4B":
            self.template_name = "phi3-chat"
            self.preprocess_func = preprocess_phi3
            self.target_modules = ['self_attn.o_proj', 'self_attn.qkv_proj', 'mlp.gate_up_proj', 'mlp.down_proj']
            llm_channel = 3072
        elif self.model_name == "InternVL2-8B":
            self.template_name = "internlm2-chat"
            self.preprocess_func = preprocess_internlm
            self.target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2',
                                   'feed_forward.w3']
            llm_channel = 4096
        else:
            raise ValueError('LLM must be InternVL2-1B, 2B, 4B or 8B')

        self.in_mlp = MLPV2(input_dim, llm_channel)
        self.out_mlp = MLPV2(llm_channel, input_dim)
        self.attention_type = attention_type

        self.lora = set_lora
        self.is_pretraining = is_pretraining
        if set_lora:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.num_learnable_query = num_learnable_query

    def create_llm(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, add_eos_token=False,
                                                       trust_remote_code=True, use_fast=False)
        self.tokenizer.tokenizer_path = self.tokenizer_path
        self.tokenizer.model_max_length = 4096
        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                      QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                      REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
        self.num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.learnable_query_token_id = -10

        config = InternVLChatConfig.from_pretrained(self.model_path)

        config.vision_config.drop_path_rate = 0.
        config.template = self.template_name
        config.select_layer = -1  # -1
        config.dynamic_image_size = True  # True
        config.use_thumbnail = True  # True
        config.ps_version = 'v2'  # 'v2'
        config.min_dynamic_patch = 1  # 1
        config.max_dylisnamic_patch = 6  # 6

        if self.model_name == "InternVL2-1B":
            config.llm_config._attn_implementation = self.attention_type  # qwen2
        config.llm_config.attn_implementation = self.attention_type  # flash attn / attn

        if self.load_weight:
            model = InternVLChatModel.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, config=config)
        else:
            model = InternVLChatModel._from_config(config).to(self.torch_dtype)

        model.language_model.config.use_cache = False
        self.llm = model.language_model

        if self.is_pretraining:
            _freeze_params(self.llm)
            if self.lora:
                warnings.warn("Should not use lora during pretraining!")
                self.set_lora(model)
        else:
            if self.lora:
                self.set_lora(model)
            else:
                self.set_lora(model, freeze_llm=False, freeze_mlp=False, use_llm_lora=0)

        del model
        torch.cuda.empty_cache()

    def set_lora(self,
                 freeze_llm=True,
                 unfreeze_lm_head=False,
                 use_llm_lora=128,
                 freeze_mlp=True,
                 ):
        if freeze_llm:
            self.llm = self.llm.eval()
            _freeze_params(self.llm)
        if unfreeze_lm_head:
            self.llm.lm_head.requires_grad = True
        if use_llm_lora:
            wrap_llm_lora(self.llm, r=use_llm_lora, lora_alpha=2 * use_llm_lora, target_modules=self.target_modules)


    def preprocess(self, conversations=None, img_length=3600):
        if conversations == None:
            conversations = self.conv_default[0]

        ret = self.preprocess_func(self.template_name, [deepcopy(conversations)],
                                   self.tokenizer, [img_length],
                                   group_by_length=True, ds_name=self.ds_name)

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
        )
        return ret

    def forward(self,
                bev=None,
                text=None,
                learn_query=None,
                device=torch.device("cuda:0"),
                position_ids=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=True
                ):


        if bev == None:
            bev = torch.rand(1, 60, 60, 1024).to(torch.float32).to(device)  # used for debug
        else:
            device = bev.device

        try:
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            labels = text['labels'].to(device)
        except Exception as e:
            print('Load conversation in forward.')
            processed_text = self.preprocess(text, img_length=self.img_length ** 2)
            input_ids = processed_text['input_ids'].unsqueeze(0).to(device)
            attention_mask = processed_text['attention_mask'].unsqueeze(0).to(device)
            labels = processed_text['labels'].unsqueeze(0).to(device)
        input_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        _, H, W, _ = bev.shape
        input_embeds = input_embeds.reshape(B * N, C)
        bev = bev.reshape(B, H * W, -1)

        bev_embeds = self.in_mlp(bev).to(self.torch_dtype)
        input_ids = input_ids.reshape(B * N)
        if self.num_learnable_query > 0 and learn_query is not None:
            assert learn_query is not None
            if learn_query.size(1) != self.num_learnable_query:
                self.num_learnable_query = learn_query.size(1)
            if learn_query.shape[-1] != C:
                learn_query = self.in_mlp(learn_query).to(self.torch_dtype)
            query_ids = torch.zeros(self.num_learnable_query, device=input_ids.device) + self.learnable_query_token_id
            input_ids = torch.cat((input_ids, query_ids), dim=0)

            selected_query = (input_ids == self.learnable_query_token_id)
            for_query = torch.zeros((self.num_learnable_query, input_embeds.size(1)), device=input_embeds.device,
                                    dtype=input_embeds.dtype)
            input_embeds = torch.cat((input_embeds, for_query), dim=0)

            try:
                input_embeds[selected_query] = input_embeds[selected_query] * 0.0 + learn_query.to(input_embeds.dtype)
                ignore_flag = False
            except Exception as e:
                print(f'warning: {e}')
                input_embeds[selected_query] = input_embeds[selected_query] * 0.0
                # the self.query_embeds must specify when self.num_learnable_query > 0
                assert False
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + bev_embeds.reshape(-1, C).to(input_embeds.dtype)
            ignore_flag = False
        except Exception as e:
            vit_embeds = bev_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token].to(input_embeds.dtype)
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, -1, C).to(torch.bfloat16)
        # input_embeds = input_embeds.reshape(B, -1, C).to(self.torch_dtype)
        if self.template_name == "internlm2-chat":
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_select=selected  # internlm only
            )
        else:
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # img_select=selected  # internlm only
            )

        hidden_states = outputs.hidden_states
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]
        hidden_states = hidden_states.reshape(B, -1, C)

        hidden_bev = hidden_states[:, selected, :].to(torch.float32)
        text_emb = hidden_states[:, ~selected, :].to(torch.float32)
        if self.num_learnable_query > 0 and learn_query is not None:
            hidden_query = hidden_states[:, selected_query, :].to(torch.float32)
            logits = outputs.logits[:, ~selected_query, :]
            hidden_query = self.out_mlp(hidden_query)
            text_emb = text_emb[:, :-selected_query.sum(), :]
        else:
            hidden_query = None
            logits = outputs.logits


        loss = None
        if labels is not None and self.is_pretraining:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        output_bev = self.out_mlp(hidden_bev).reshape(B, H, W, -1)

        return {
            'out_bev': output_bev,
            'logits': logits,
            'chat_loss': loss,
            'out_query': hidden_query,
            'text_emb': text_emb,
        }

    def chat(self, tokenizer, pixel_values, question, conv=None, generation_config=None, history=None,
             return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):
        if generation_config == None:
            generation_config = self.chat_cfg
        if tokenizer == None:
            tokenizer = self.tokenizer
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template_name)
        template.system_message = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            self.num_image_token = self.img_length ** 2
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            visual_features=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            if conv is not None:
                output = self(bev=pixel_values.clone().to(torch.float32), text=conv)
                return response, output
            else:
                return response, None

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.llm.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            _, H, W, _ = vit_embeds.shape
            vit_embeds = vit_embeds.reshape(B, H * W, -1)
            vit_embeds = self.in_mlp(vit_embeds.to(torch.float32)).to(self.torch_dtype)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.llm.get_input_embeddings()(input_ids)

        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs


def main():
    # # Parse input arguments
    # # See all possible arguments in src/transformers/training_args.py
    # # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    # launcher = os.environ.get('LAUNCHER', 'slurm')
    # init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
    #     # If we pass only one argument to the script, and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    model_args, training_args = parser.parse_args_into_dataclasses()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = 4096
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # debug
    b_sentences = ["hello, I'm a sentences", "And another sentence", "And the very very last one"]
    b_value = tokenizer(b_sentences)

    logger.info('Loading LLaMA...')
    llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
    if llm_config.model_type == 'internlm2':
        model_type = InternLM2ForCausalLM
        llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        logger.info('Using flash_attention_2 for InternLM')
    else:
        model_type = AutoModelForCausalLM
        llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        logger.info('Using flash_attention_2 for LLaMA')
    llm = model_type.from_pretrained(
        model_args.llm_path, torch_dtype=torch.bfloat16,
        config=llm_config, trust_remote_code=True)

    logger.info('Finished')

    llm.config.use_cache = False

    llm._set_gradient_checkpointing()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_llm:
        llm = llm.eval()
        _freeze_params(llm)

    if model_args.unfreeze_lm_head:
        llm.lm_head.requires_grad = True

    if model_args.use_llm_lora:
        wrap_llm_lora(llm, r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)

    #
    # # print trainable parameters
    # if dist.get_rank() == 0:
    #     for name, param in llm.named_parameters():
    #         if param.requires_grad:
    #             logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    device = torch.device("cuda:0")
    debug_embeds = torch.rand(1, 4000, 2048).to(torch.bfloat16).to(device)
    debug_attn_mask = torch.ones_like(debug_embeds).to(torch.bool).to(device)
    llm.to(device)
    position_ids = None
    past_key_values = None
    use_cache = None
    output_attentions = None
    output_hidden_states = True
    return_dict = True

    outputs = llm(
        inputs_embeds=debug_embeds,
        attention_mask=debug_attn_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    return 0


if __name__ == '__main__':
    pass
