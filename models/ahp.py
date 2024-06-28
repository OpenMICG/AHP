'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, ViTModel, AutoTokenizer, CLIPVisionModel
# from transformers.adapters import BertAdapterModel, AutoAdapterModel, ViTAdapterModel
from models.ahp_adapter.ahp_adapter import AHPAdapter

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
from einops import rearrange

class AHP_Decoder(nn.Module):
    def __init__(self,
                 args,
                 med_config_root = 'configs/med_config.json',
                 prompt = '',
                 queue_size=57600,
                 momentum=0.995,
                 embed_dim=256,
                 datasets='caption_iu_xray',
                 tokenizer=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.args = args


        self.visual_encoder = AHPAdapter(embed_dim=768, patch_size=16, depth=12, num_heads=12, conv_inplane=64,
                                         n_points=4, deform_num_heads=12, cffn_ratio=0.25, deform_ratio=0.5,
                                         interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]] )
        vision_width = 768

        self.visual_encoder_m = AHPAdapter(embed_dim=768, patch_size=16, depth=12, num_heads=12, conv_inplane=64,
                                         n_points=4, deform_num_heads=12, cffn_ratio=0.25, deform_ratio=0.5,
                                         interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]] )


        self.tokenizer = tokenizer
        # self.tokenizer = init_tokenizer()


        med_config = BertConfig.from_json_file(med_config_root)
        med_config.encoder_width = vision_width

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt))-1

        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)


        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(self.tokenizer.vocab_size + 3)
        text_width = self.text_encoder.config.hidden_size

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)
        self.text_encoder_m.resize_token_embeddings(self.tokenizer.vocab_size + 3)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)



        self.itm_head = nn.Linear(text_width, 2)

        self.datasets = datasets


        if self.datasets == 'caption_iu_xray':
            self.iu_proj = nn.Linear(768 * 2, 768)
            self.iu_proj_m = nn.Linear(768 * 2, 768)
            # queue_size = 1380
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.vision_proj, self.vision_proj_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                [self.iu_proj, self.iu_proj_m]]
        else:
            self.mimic_proj = nn.Linear(768, 768)

            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.vision_proj, self.vision_proj_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                ]
        self.copy_params()


        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.queue_size = queue_size


        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config_root)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.text_decoder.resize_token_embeddings(self.tokenizer.vocab_size + 3)
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')


    def forward(self, image, reports_ids, reports_masks, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        if self.datasets == 'caption_iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)
            image_embeds = self.mimic_proj(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder(reports_ids.clone(), attention_mask=reports_masks.clone(),
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            if self.datasets == 'caption_iu_xray':
                image_embeds0_m = self.visual_encoder_m(image[:, 0])
                image_embeds1_m = self.visual_encoder_m(image[:, 1])
                image_embeds_m = torch.cat((image_embeds0_m, image_embeds1_m), dim=2)
                image_embeds_m = self.iu_proj_m(image_embeds_m)
            else:
                image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()],dim=1)

            text_output_m = self.text_encoder_m(reports_ids, attention_mask = reports_masks,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###============== Image-text Matching ===================###
        encoder_input_ids = reports_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.bos_idx

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = reports_masks,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                       return_dict = True,
                                      )
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(reports_masks[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg,dim=0)
        text_atts_neg = torch.stack(text_atts_neg,dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)
        text_atts_all = torch.cat([reports_masks, text_atts_neg],dim=0)

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,
                                       return_dict = True,
                                      )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

    ##================= LM ========================##
        decoder_input_ids = reports_ids.clone()
        decoder_input_ids[:,0] = self.tokenizer.bos_idx
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_idx, -100)

        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask = reports_masks,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           labels = decoder_targets,
                                           return_dict = True,
                                          )
        loss_lm = decoder_output.loss
        return loss_ita + loss_itm + loss_lm

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        if self.datasets == 'caption_iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)
            image_embeds = self.mimic_proj(image_embeds)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        input_ids = torch.tensor([[1]] * image.size(0)).to(image.device)
        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate( input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.eos_idx,
                                                  pad_token_id=self.tokenizer.pad_idx,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate( input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.eos_idx,
                                                  pad_token_id=self.tokenizer.pad_idx,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)

        return outputs

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        '''
        :param embeddings: bz, 1, 112, 768
        :param caption_ids: bz, 112
        :param last_layer_attn: bz, 111
        '''
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def ahp_decoder(pretrained='', **kwargs):
    model = AHP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print(msg.missing_keys)

    return model

def ahp_feature_extractor(pretrained='',**kwargs):
    model = ahp_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):

    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate)
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate)
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    print("model:", len(model.state_dict().keys()))
    print("state_dict:", len(state_dict.keys()))
    print("difference", len(set(state_dict.keys()).difference(set(model.state_dict().keys()))))

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)

    print('load checkpoint from %s'%url_or_filename)
    return model, msg

from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights = uninitialized_encoder_weights + list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)