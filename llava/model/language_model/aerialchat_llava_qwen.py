#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import re
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from llava.constants import IGNORE_INDEX, DEFAULT_ACTION_TOKEN, DEFAULT_QUESTION_TOKEN

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
from llava.model.llava_arch import unpad_image
import random


class AerialChatLlavaQwenConfig(Qwen2Config):
    model_type = "aerialchat_llava_qwen"


class AerialChatLlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = AerialChatLlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(AerialChatLlavaQwenModel, self).__init__(config)


class AerialChatLlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = AerialChatLlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "aerialchat_llava_qwen"
        config.rope_scaling = None

        self.model = AerialChatLlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        
        self.num_embeds = 4
        # TODO: init with require grad
        self.emb_embeddings_action = nn.Embedding(num_embeddings=self.num_embeds + 1, embedding_dim=config.hidden_size)
        self.emb_embeddings_question = nn.Embedding(num_embeddings=self.num_embeds + 1, embedding_dim=config.hidden_size)

        self.use_grid_action = config.use_grid_action
        self.grid_offset_size = config.grid_offset_size
        self.grid_altitude_size = config.grid_altitude_size
        if self.use_grid_action:
            self.grid_decoder = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size),
            )
            self.grid_image_decoder = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size),
            )
            self.altitude_decoder = nn.Sequential(
                nn.Linear(config.hidden_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, self.grid_altitude_size),
            )
        else:
            self.waypoint_decoder = nn.Sequential(
                nn.Linear(config.hidden_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
            self.altitude_decoder = nn.Sequential(
                nn.Linear(config.hidden_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        self.iou_progress_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.distance_progress_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.use_action = True
        self.post_init()

    def get_model(self):
        return self.model
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if "unpad" in mm_patch_merge_type:
                            # image_feature = image_feature.permute(2, 0, 1).contiguous()
                            # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            # image_feature = image_feature.permute(1, 2, 0).contiguous()
                            image_feature = image_feature.flatten(0, 1)
                            image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)

                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        metas=None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        input_ids_copy = input_ids
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
          
            if labels is not None:
                emb_embeddings_action = self.emb_embeddings_action.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
                emb_embeddings_question = self.emb_embeddings_question.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
                emb_ids = torch.tensor([x for x in range(self.emb_token_id, self.emb_token_id + self.num_embeds)], dtype=torch.long).to(inputs_embeds.device)
                # add action prediction embeddings
                new_input_embeds = []
                new_labels = []
                new_attention_mask = []
                for cur_input_embeds, cur_labels, cur_attention_mask, cur_emb_embeddings_action, cur_emb_embeddings_question in zip(inputs_embeds, labels, attention_mask, emb_embeddings_action, emb_embeddings_question):
                    emb_start_pos_action = torch.where(cur_labels==self.action_token_id)[0] # +1 beacause use output embed
                    emb_start_pos_question = torch.where(cur_labels==self.question_token_id)[0]

                    for i, _start_pos in enumerate(emb_start_pos_action):
                        # concat emb ids
                        cur_labels = torch.cat(
                            [
                                cur_labels[: _start_pos + 1],
                                emb_ids,
                                cur_labels[_start_pos + 1 :]
                            ], dim=0
                        )
                        cur_attention_mask = torch.cat(
                            [
                                cur_attention_mask[: _start_pos + 1],
                                torch.ones_like(emb_ids).bool(),
                                cur_attention_mask[_start_pos + 1 :]
                            ], dim=0
                        )
                        # repalce with emb embeddings
                        cur_input_embeds = torch.cat(
                            [
                                cur_input_embeds[: _start_pos],
                                cur_emb_embeddings_action,
                                cur_input_embeds[_start_pos + 1 :]
                            ], dim=0
                        ).contiguous()  # replace with self.emb_embeddings
                    
                    for i, _start_pos in enumerate(emb_start_pos_question):
                        # concat emb ids
                        cur_labels = torch.cat(
                            [
                                cur_labels[: _start_pos + 1],
                                emb_ids,
                                cur_labels[_start_pos + 1 :]
                            ], dim=0
                        )
                        # repalce with emb embeddings
                        cur_attention_mask = torch.cat(
                            [
                                cur_attention_mask[: _start_pos + 1],
                                torch.ones_like(emb_ids).bool(),
                                cur_attention_mask[_start_pos + 1 :]
                            ], dim=0
                        )
                        # repalce with emb embeddings
                        cur_input_embeds = torch.cat(
                            [
                                cur_input_embeds[: _start_pos],
                                cur_emb_embeddings_question,
                                cur_input_embeds[_start_pos + 1 :]
                            ], dim=0
                        ).contiguous()  # replace with self.emb_embeddings

                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(cur_labels)
                    new_attention_mask.append(cur_attention_mask)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
                labels = torch.stack(new_labels, dim=0)
                attention_mask = torch.stack(new_attention_mask, dim=0)
            else:
                if input_ids_copy is not None and input_ids_copy.shape[1] != 1:
                    emb_embeddings_action = self.emb_embeddings_action.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
                    emb_embeddings_question = self.emb_embeddings_question.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
                    emb_ids = torch.tensor([x for x in range(self.emb_token_id, self.emb_token_id + self.num_embeds)], dtype=torch.long).to(inputs_embeds.device)
                    new_input_embeds = []
                    # new_attention_mask = []
                    new_input_ids = []
                    for cur_input_ids, cur_input_embeds, cur_emb_embeddings_action, cur_emb_embeddings_question in zip(input_ids_copy, inputs_embeds, emb_embeddings_action, emb_embeddings_question):
                        emb_start_pos_action = torch.where(cur_input_ids==self.action_token_id)[0]
                        emb_start_pos_question = torch.where(cur_input_ids==self.question_token_id)[0]

                        for i, _start_pos in enumerate(emb_start_pos_action):
                            # repalce with emb embeddings
                            _start_pos_emb = _start_pos + cur_input_embeds.shape[0] - cur_input_ids.shape[0] 
                            cur_input_embeds = torch.cat(
                                [
                                    cur_input_embeds[: _start_pos_emb],
                                    cur_emb_embeddings_action,
                                    cur_input_embeds[_start_pos_emb + 1 :]
                                ], dim=0
                            ).contiguous()  # replace with self.emb_embeddings
                            cur_input_ids = torch.cat(
                                [
                                    cur_input_ids[: _start_pos + 1],
                                    emb_ids,
                                    cur_input_ids[_start_pos + 1 :]
                                ], dim=0
                            )
                            # cur_attention_mask = torch.cat(
                            #     [
                            #         cur_attention_mask[: _start_pos + 1],
                            #         torch.ones_like(emb_ids).bool(),
                            #         cur_attention_mask[_start_pos + 1 :]
                            #     ], dim=0
                            # )
                        for i, _start_pos in enumerate(emb_start_pos_question):
                            # repalce with emb embeddings
                            _start_pos_emb = _start_pos + cur_input_embeds.shape[0] - cur_input_ids.shape[0] 
                            cur_input_embeds = torch.cat(
                                [
                                    cur_input_embeds[: _start_pos_emb],
                                    cur_emb_embeddings_question,
                                    cur_input_embeds[_start_pos_emb + 1 :]
                                ], dim=0
                            ).contiguous()  # replace with self.emb_embeddings
                            cur_input_ids = torch.cat(
                                [
                                    cur_input_ids[: _start_pos + 1],
                                    emb_ids,
                                    cur_input_ids[_start_pos + 1 :]
                                ], dim=0
                            )
                            # cur_attention_mask = torch.cat(
                            #     [
                            #         cur_attention_mask[: _start_pos + 1],
                            #         torch.ones_like(emb_ids).bool(),
                            #         cur_attention_mask[_start_pos + 1 :]
                            #     ], dim=0
                            # )
                        new_input_embeds.append(cur_input_embeds)
                        new_input_ids.append(cur_input_ids)
                        # new_attention_mask.append(cur_attention_mask)
                    inputs_embeds = torch.stack(new_input_embeds, dim=0)
                    new_input_ids = torch.stack(new_input_ids, dim=0)
                    # attention_mask = torch.stack(new_attention_mask, dim=0)

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            hidden_states = outputs.hidden_states[-1]
            
            # get action loss
            if self.use_action and labels is not None:
                bs, _, embd_dim = hidden_states.shape
                emb_select = (labels >= self.emb_token_id) & (labels <= self.emb_token_id + self.num_embeds - 1)  # [bs, seq_len]
                action_embeddings = hidden_states[emb_select].reshape(bs, -1, embd_dim)
                
                pred_actions = self.get_action(action_embeddings, image_features)
                action_loss = self.compute_action_loss(pred_actions, metas)
                outputs["pred_actions"] = pred_actions
                
            elif self.use_action and labels is None and input_ids_copy is not None and input_ids_copy.shape[1] != 1:
                bs, _, embd_dim = hidden_states.shape
                emb_select = (new_input_ids >= self.emb_token_id) & (new_input_ids <= self.emb_token_id + self.num_embeds - 1)  # [bs, seq_len]
                emb_select = torch.cat([torch.zeros(bs, inputs_embeds.shape[1] - new_input_ids.shape[1]).bool().to(emb_select.device),emb_select], dim=1)
                action_embeddings = hidden_states[emb_select].reshape(bs, -1, embd_dim)
                
                pred_actions = self.get_action(action_embeddings, image_features)
                outputs["pred_actions"] = pred_actions

            loss = 0.
            if labels is not None:
                logits = outputs.logits            
                # ignore the emb_tokens for labels
                emb_select = (labels >= self.emb_token_id) & (labels <= self.emb_token_id + self.num_embeds - 1)  # [B, L]
                labels[emb_select] = IGNORE_INDEX
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                
                # action losses 
                outputs.loss = loss + action_loss

            return outputs
        
    def get_action(self, action_embeddings, image_features):

        if self.use_grid_action:
            current_image_features = [image_feature for i, image_feature in enumerate(image_features) if i % 2 == 1]
            current_image_features = torch.stack(current_image_features)
            
            grid_action_embeddings = self.grid_decoder(action_embeddings[:, 0])
            current_image_features = self.grid_image_decoder(current_image_features)
            grid_score = self.get_grid_point(grid_action_embeddings, current_image_features, grid_size=self.grid_offset_size)

            pred_altitude = self.altitude_decoder(action_embeddings[:, 1])
            iou_progress = self.iou_progress_decoder(action_embeddings[:, 2])
            distance_progress = self.distance_progress_decoder(action_embeddings[:, 3])

            return {
                "offset": ((torch.stack([grid_score.argmax(-1) % self.grid_offset_size, grid_score.argmax(-1) // self.grid_offset_size], dim=-1) + 0.5 )/ self.grid_offset_size) * 2 - 1,
                "grid": grid_score,
                "grid_altitude": pred_altitude,
                "altitude": ((pred_altitude.argmax(-1) + 0.5) / pred_altitude.shape[-1]).unsqueeze(-1),
                "iou_progress": iou_progress.sigmoid(),
                "distance_progress": distance_progress.sigmoid(),
            }

        else:
            pred_waypoint = self.waypoint_decoder(action_embeddings[:, 0])
            pred_altitude = self.altitude_decoder(action_embeddings[:, 1])
            iou_progress = self.iou_progress_decoder(action_embeddings[:, 2])
            distance_progress = self.distance_progress_decoder(action_embeddings[:, 3])
        
        
            return {
                "offset": pred_waypoint,
                "altitude": pred_altitude,
                "iou_progress": iou_progress.sigmoid(),
                "distance_progress": distance_progress.sigmoid(),
            }
    
    def get_grid_point(self, action_embeddings, image_features, grid_size=9):
        image_features = image_features[:, :-1, :]
        origin_grid_size = int((image_features.shape[1])**0.5)
        bs, _, c = image_features.shape
        image_features = image_features.reshape(bs, origin_grid_size, origin_grid_size, c)
        
        image_features = image_features.permute(0, 3, 1, 2)
        image_features = F.interpolate(image_features, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.reshape(bs, -1, c)
        
        score = (image_features @ action_embeddings.unsqueeze(-1)) 
        score = score.squeeze(-1) / c**0.5
        return score
        
    def compute_action_loss(self, pred_actions, metas):
        dtype = pred_actions['iou_progress'].dtype
        loss_mse = nn.MSELoss(reduction='sum')
        loss_fct = CrossEntropyLoss()
        loss_bce = nn.BCELoss()
        
        loss = 0.
        target = metas['target']
        if self.use_grid_action:
            loss += loss_bce(pred_actions['grid'].sigmoid(), target['grid_onehot'])
            loss += loss_bce(pred_actions['grid_altitude'].sigmoid(), target['grid_altitude_onehot'])
        else:
            loss += loss_mse(pred_actions['offset'], target['offset'].to(dtype))
            loss += loss_mse(pred_actions['altitude'], target['altitude'].unsqueeze(-1).to(dtype))

        loss += loss_bce(pred_actions['iou_progress'], target['progress'].unsqueeze(-1).to(dtype))
        # loss += loss_bce(pred_actions['distance_progress'], target['distance_progress'].unsqueeze(-1).to(dtype))
        
        # loss /= pred_actions['altitude'].shape[0]
        return loss

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        inputs_copy = inputs
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        if self.use_action:
            emb_embeddings_action = self.emb_embeddings_action.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
            emb_embeddings_question = self.emb_embeddings_question.weight.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)    # [bs, num_embeds, c]
            emb_ids = torch.tensor([x for x in range(self.emb_token_id, self.emb_token_id + self.num_embeds)], dtype=torch.long).to(inputs_embeds.device)

            new_input_embeds = []
            new_attention_mask = []
            for cur_input_ids, cur_input_embeds, cur_emb_embeddings_action, cur_emb_embeddings_question in zip(inputs_copy, inputs_embeds, emb_embeddings_action, emb_embeddings_question):
                emb_start_pos_action = torch.where(cur_input_ids==self.action_token_id)[0]
                emb_start_pos_question = torch.where(cur_input_ids==self.question_token_id)[0]

                for i, _start_pos in enumerate(emb_start_pos_action):
                    # repalce with emb embeddings
                    cur_input_embeds = torch.cat(
                        [
                            cur_input_embeds[: _start_pos + 1],
                            cur_emb_embeddings_action,
                            cur_input_embeds[_start_pos + 1 :]
                        ], dim=0
                    ).contiguous()  # replace with self.emb_embeddings
                    # cur_attention_mask = torch.cat(
                    #     [
                    #         cur_attention_mask[: _start_pos + 1],
                    #         torch.ones_like(emb_ids).bool(),
                    #         cur_attention_mask[_start_pos + 1 :]
                    #     ], dim=0
                    # )
                for i, _start_pos in enumerate(emb_start_pos_question):
                    # repalce with emb embeddings
                    cur_input_embeds = torch.cat(
                        [
                            cur_input_embeds[: _start_pos + 1],
                            cur_emb_embeddings_question,
                            cur_input_embeds[_start_pos + 1 :]
                        ], dim=0
                    ).contiguous()  # replace with self.emb_embeddings
                    # cur_attention_mask = torch.cat(
                    #     [
                    #         cur_attention_mask[: _start_pos + 1],
                    #         torch.ones_like(emb_ids).bool(),
                    #         cur_attention_mask[_start_pos + 1 :]
                    #     ], dim=0
                    # )
                new_input_embeds.append(cur_input_embeds)
                # new_attention_mask.append(cur_attention_mask)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            # attention_mask = torch.stack(new_attention_mask, dim=0)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    def initialize_added_tokenizer(self, tokenizer):
        tokens = [DEFAULT_ACTION_TOKEN, DEFAULT_QUESTION_TOKEN]

        tokenizer.add_tokens(tokens)
        # self.resize_token_embeddings(len(tokenizer))
        self.action_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN)
        self.question_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_QUESTION_TOKEN)
        
        self.emb_token_id = self.action_token_id + 20
        
    def unfreeze_part_params(self):

        if self.use_grid_action:
            for name, param in self.grid_decoder.named_parameters():
                param.requires_grad_(True)
            for name, param in self.grid_image_decoder.named_parameters():
                param.requires_grad_(True)

        if not self.use_grid_action:
            for name, param in self.waypoint_decoder.named_parameters():
                param.requires_grad_(True)
            for name, param in self.altitude_decoder.named_parameters():
                param.requires_grad_(True)

        for name, param in self.iou_progress_decoder.named_parameters():
            param.requires_grad_(True)
        for name, param in self.distance_progress_decoder.named_parameters():
            param.requires_grad_(True)
        
        for name, param in self.emb_embeddings_action.named_parameters():
            param.requires_grad_(True)
        for name, param in self.emb_embeddings_question.named_parameters():
            param.requires_grad_(True)
        


AutoConfig.register("aerialchat_llava_qwen", AerialChatLlavaQwenConfig)
AutoModelForCausalLM.register(AerialChatLlavaQwenConfig, AerialChatLlavaQwenForCausalLM)
