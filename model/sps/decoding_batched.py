import torch
import copy
import warnings

from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput, ModelOutput
from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
    CandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
import torch.distributed as dist
import os, time
FUNC_MAP = {}
CONFIG_MAP = {}
COLOR_PRINT = int(os.environ.get("COLOR_PRINT", 0))

def assisted_decoding_batched_step(
    input_ids,
    att_mask,
    model,
    assistant_model,
    draft_tokens=4,
    do_sample=False, 
    temperature=0,
):
    if temperature == 0:
        do_sample = False
    if do_sample == False:
        temperature = 0

    bs = input_ids.shape[0]

    drafted = generate_with_pad(input_ids, att_mask, assistant_model, do_sample, temperature, draft_tokens, stop_on_eos=False)
    
    candidate_input_ids = drafted["sequences"]
    candidate_logits = drafted["scores"][:, -draft_tokens:]

    verifier_att_mask=torch.cat([att_mask, torch.ones((bs, draft_tokens), device=att_mask.device)], 1)
    
    verifier_logits = model.forward(candidate_input_ids, attention_mask=make_padded_mask_batched(verifier_att_mask).to(dtype=model.dtype), position_ids=make_padded_pos_ids(verifier_att_mask))["logits"]
    max_matches = draft_tokens - 1

    if do_sample:
        verifier_logits = verifier_logits / temperature
        new_logits = verifier_logits[:, -draft_tokens-1:-1]
        
        candidate_length = draft_tokens

        new_tokens, matches = _speculative_sampling(candidate_input_ids, candidate_logits, draft_tokens, new_logits, False, max_matches)
    else:
        new_verifier_tokens = verifier_logits[:, -draft_tokens-1:-1].argmax(-1)
        new_drafter_tokens = drafted["sequences"][:, -draft_tokens:]
        
        is_accepted = (new_verifier_tokens == new_drafter_tokens)
        matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(-1)
        matches = torch.minimum(matches, torch.ones(bs, device=matches.device, dtype=torch.long) * max_matches)

        new_token = torch.take_along_dim(new_verifier_tokens, matches.reshape(bs, 1), dim=1)

        if (matches > 0).any():
            valid_tokens = torch.cat((new_drafter_tokens[:, :matches.max()], new_token), dim=-1)
        else:
            valid_tokens = new_token

        new_tokens = valid_tokens
        
    new_att_mask = (torch.arange(matches.max().item(), device=matches.device).repeat((bs,1)) < matches.reshape(bs, 1)).to(dtype=torch.long)
    new_att_mask = torch.cat([new_att_mask, torch.ones(bs, 1, device=new_att_mask.device, dtype=torch.long)], -1)
    new_tokens = torch.where(new_att_mask == 1, new_tokens, model.generation_config.pad_token_id)
 
    return {"sequences" : torch.cat([input_ids, new_tokens], -1),
            "attention_mask" : torch.cat([att_mask, new_att_mask], -1), 
            "verified" : matches + 1}

def generate_assisted_batched(inputs, model, assistant_model, tokenizer, draft_tokens=4, iters=5, do_sample=False, temperature=1, stop_on_eos=True):
    input_ids = inputs.input_ids
    att_mask = inputs.attention_mask
    bs = input_ids.shape[0]

    accept_length_list = []
    all_iters = iters
    for i in range(iters):
        generated = assisted_decoding_batched_step(input_ids, att_mask, model, assistant_model, draft_tokens, do_sample, temperature)
        input_ids = generated["sequences"]
        att_mask = generated["attention_mask"]

        accept_length_list.append(generated["verified"])
        if stop_on_eos and ((input_ids == tokenizer.eos_token_id).any(-1).sum() == bs):
            all_iters = i
            break

    accept_length_list = torch.stack(accept_length_list).permute(1,0).tolist()

    return input_ids, all_iters, accept_length_list, att_mask


def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    last_assistant_token_is_eos,
    max_matches,
):
    bs = candidate_input_ids.shape[0]

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    q = candidate_logits.softmax(dim=-1)
    q_i = torch.gather(q, -1, new_candidate_input_ids[:, :, None]).squeeze(-1)

    p = new_logits.softmax(dim=-1)
    p_i = torch.gather(p, -1, new_candidate_input_ids[:, :, None]).squeeze(-1)

    probability_ratio = p_i / q_i
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum(-1)
    n_matches = torch.minimum(n_matches, torch.ones(bs, device=n_matches.device, dtype=torch.long) * max_matches)

    gamma = min(candidate_logits.shape[1], max_matches)
    p_n_plus_1 = torch.take_along_dim(p, n_matches.resize(bs, 1, 1), 1)
    q_n_plus_1 = torch.take_along_dim(q, n_matches.resize(bs, 1, 1), 1)
    
    p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1 * (n_matches < gamma).reshape(bs, 1, 1)), min=0).squeeze(1)

    new_token = torch.multinomial(p_prime, 1).to(candidate_input_ids.device)
    
    if (n_matches > 0).any():
        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches.max()], new_token), dim=-1)
    else:
        valid_tokens = new_token

    return valid_tokens, n_matches

def make_casual_mask(n):
    return (torch.tril(torch.ones(n,n)) + torch.triu(torch.ones(n,n) * float("-inf"), 1))[None, None, :]

def make_padded_mask(att_mask):
    att_mask = torch.Tensor(att_mask)
    # bs = att_mask.shape[0]
    n = att_mask.size(-1)
    mask = make_casual_mask(n)
    mask[0, 0, :, att_mask==0] = float("-inf")
    return mask.to(att_mask.device)

def make_padded_mask_batched(att_mask):
    return torch.cat([make_padded_mask(mask) for mask in att_mask])

def make_padded_pos_ids(att_mask):
    att_mask = torch.Tensor(att_mask)
    return ((att_mask.cumsum(-1) - 1) * att_mask).to(dtype=torch.long)

@torch.no_grad()
def generate_with_pad(input_ids, att_mask, model, do_sample=False, temperature=1, max_new_tokens=1, stop_on_eos=True):
    bs = input_ids.shape[0]
    logits = None

    for i in range(max_new_tokens):
        logits = model.forward(input_ids, attention_mask=make_padded_mask_batched(att_mask).to(dtype=model.dtype), position_ids=make_padded_pos_ids(att_mask))["logits"]
        if do_sample:
            logits = logits / temperature

        if do_sample:
            probs = logits[:, -1].softmax(-1)
            new_token = torch.multinomial(probs, 1).to(input_ids.device)
        else:
            new_token = logits[:, -1].argmax(-1, keepdim=True).to(input_ids.device)
        
        input_ids = torch.cat([input_ids, new_token], dim=-1)
        att_mask = torch.cat([att_mask, torch.ones(bs, 1, device=att_mask.device)], dim=-1)

        if stop_on_eos and ((input_ids == model.generation_config.eos_token_id).any(-1).sum() == bs):
            break

    return {"sequences" : input_ids, 
            "scores" : logits, 
            "attention_mask" : att_mask}
