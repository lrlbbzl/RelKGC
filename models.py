from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")

import json
import os
import time

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask, construct_mask_with_ids
from doc import entity_dict
from logger_config import logger

def build_model(args) -> nn.Module:
    return CustomBertModel(args)

def get_entities_length(args):
    fp = open(os.path.join(os.path.dirname(args.train_path), 'entities.json'), 'r', encoding='utf-8')
    file = json.load(fp)
    fp.close()
    return len(file)

@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=None)

        self.entity_num = get_entities_length(args)
        self.initial = False

    def _encode(self, encoder, token_ids, mask, token_type_ids, mode):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        # if mode != 'hr':
        #     return cls_output
        # else:
        #     bs = token_ids.size(0)
        #     h_in_hr, r_in_hr = torch.rand(*(bs, cls_output.size(1))), torch.rand(*(bs, cls_output.size(1)))
        #     for i in range(bs):
        #         single = last_hidden_state[i, :, :] # (len, 768)
        #         idx = (token_type_ids[i, :] == 1).nonzero()
        #         head_ed, rel_ed = idx[0][0] - 1, idx[-1][0] 
        #         head_vec, rel_vec = single[1 : head_ed + 1, :].mean(dim=0), single[head_ed + 1 : rel_ed + 1, :].mean(dim=0)
        #         h_in_hr[i, :], r_in_hr[i, :] = head_vec, rel_vec
        #     return cls_output, h_in_hr, r_in_hr
        return cls_output

    def score_func(self, head_vec, rel_vec, tail_vec):
        return 9 - 0.5 * (((head_vec + rel_vec - tail_vec) ** 2).sum(dim=-1) / head_vec.size(-1) ** 0.5)

    def create_neg_examples_loss(self, head_vec, rel_vec, tail_vec, neg_num):
        batch_size = tail_vec.size(0)
        head_vec, rel_vec, tail_vec = head_vec.repeat(3, 1), rel_vec.repeat(3, 1), tail_vec.repeat(3, 1)
        temp = np.arange(batch_size)
        corrupted_head = np.random.choice(range(batch_size), batch_size)
        while (True in (corrupted_head == temp)):
            corrupted_head = np.random.choice(range(batch_size), batch_size)
        corrupted_tail = np.random.choice(range(batch_size), batch_size)
        while (True in (corrupted_tail == temp)):
            corrupted_tail = np.random.choice(range(batch_size), batch_size)
        # index = np.random.choice(range(batch_size), neg_num * batch_size)
        tail_vec[batch_size : 2 * batch_size, :] = tail_vec[corrupted_tail, :]
        head_vec[2 * batch_size : 3 * batch_size] = head_vec[corrupted_head, :]
        triple_scores = self.score_func(head_vec.to(tail_vec.device), rel_vec.to(tail_vec.device), tail_vec)
        labels = torch.Tensor([1] * batch_size + [0] * (batch_size * 2)).to(tail_vec.device)
        return self.loss_func(triple_scores, labels.float())

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids,
                                 mode='hr')
                                      
        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   mode='tail')

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids,
                                   mode='head')

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector,
                # 'h_in_hr': h_in_hr,
                # 'r_in_hr': r_in_hr
            }

    def cal_rince(self, logits):
        exp_logits = torch.exp(logits)
        q, l = self.args.rince_q, self.args.rince_lambda
        part_2 = torch.sum(torch.pow(torch.sum(exp_logits, dim=1) * l, q) / q)
        part_1 = -1. * torch.sum(torch.pow(exp_logits.diagonal(), q)) / q
        return part_1 + part_2

    def compute_logits(self, output_dict: dict, batch_dict: dict, mode: str) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)
        logits = hr_vector.mm(tail_vector.t()) # cosine similarity
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4) 

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)
        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        # logits: (bs, bs + pre_batch_num * bs + 1)
        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach(),
                }

    def _compute_similar_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        start_step, siz = 3, 2
        st = set()
        for vector in tail_vector:
            similarity = F.cosine_similarity(vector, self.entity_embedding)
            idx = torch.argsort(similarity, descending=True)[start_step : start_step + siz]
            for v in idx:
                st.add(int(v))
        st = torch.LongTensor(list(st))
        batch_exs = batch_dict['batch_data']
        similar_vectors = self.entity_embedding[st].detach()
        siz = similar_vectors.size(1)
        assert not (siz in torch.tensor(torch.eq(torch.full([siz], 0).cuda(), similar_vectors), dtype=torch.int32).sum(dim=1))
        similar_batch_logits = hr_vector.mm(similar_vectors.t())
        similar_batch_logits *= self.log_inv_t.exp()
        similar_mask = construct_mask_with_ids(batch_exs, st).to(hr_vector.device)
        similar_batch_logits.masked_fill_(~similar_mask, -1e4)
        return similar_batch_logits
        

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            # false negative samples which probably be neighbors will be masked
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   mode='tail')
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
