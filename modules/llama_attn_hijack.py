import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers.models.llama.modeling_llama

import modules.shared as shared
from modules.logging_colors import logger

#_attention_type = "random"
_attention_type = "default"
_coherence_json = "{}"

if shared.args.xformers:
    try:
        import xformers.ops
    except Exception:
        logger.error("xformers not found! Please install it before trying to use it.", file=sys.stderr)


def hijack_llama_attention():
    if shared.args.xformers:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = xformers_forward
        logger.info("Replaced attention with xformers_attention")
    elif shared.args.sdp_attention:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = sdp_attention_forward
        logger.info("Replaced attention with sdp_attention")


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # We only apply xformers optimizations if we don't need to output the whole attention matrix
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(query_states, key_states, value_states, attn_bias=None)
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(query_states, key_states, value_states, attn_bias=xformers.ops.LowerTriangularMask())
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


def sdp_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # We only apply sdp attention if we don't need to output the whole attention matrix
    if not output_attentions:
#        print('q.shape: '+str(query_states.shape)+', k.shape: '+str(key_states.shape)+', attn_mask.shape: '+str(attention_mask.shape))
#        print('query_states: '+str(query_states)+', key_states: '+str(key_states)+', attention_mask: '+str(attention_mask))
#        print('attention_mask: '+str(attention_mask))
        global _coherence_json
        global _attention_type
        _attention_type = shared.settings['attention_type']
        _coherence_json = shared.settings['coherence_json']
#        print('_attention_type: '+str(_attention_type))
        
        if _attention_type == "default":
          attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)
        else:
#          print('query_states: '+str(query_states)+', key_states: '+str(key_states)+', attention_mask: '+str(attention_mask))
#          print('attention_mask: '+str(attention_mask))
          if query_states.shape[2] == key_states.shape[3]:
            attention_mask = _coherence_attention_mask(query_states, key_states)
          else:
            attention_mask = _coherence_attention_mask(attention_mask, key_states)
#          print('q.shape: '+str(query_states.shape)+', k.shape: '+str(key_states.shape)+', attn_mask.shape: '+str(attention_mask.shape))
#          print('attention_mask: '+str(attention_mask))
          attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value
    
def _coherence_attention_mask(query: torch._C.Value, key: torch._C.Value
) -> torch._C.Value:
    global _coherence_json
    global _attention_type

    L = query.shape[2]
    S = key.shape[2]
    print('L: '+str(L)+', S: '+str(S))
#    L = 400
#    S = 400
    mask = torch.ones(L, S, device=query.device, dtype=torch.bool).tril(diagonal=0)
#    attn_mask = torch.ones(L, S, device=query.device, dtype=query.dtype).tril(diagonal=0)
#    attn_mask = torch.zeros(L, S, device=query.device, dtype=query.dtype)
#    attn_mask = torch.ones(L, S, device=query.device, dtype=query.dtype)

#    print(' '.join(globals()).split(' '))
    
    if _attention_type == "random":
#    if True:
#    if False:
#      attn_mask = torch.rand(L, S, device=query.device, dtype=query.dtype)/100
      attn_mask = torch.rand(L, S, device=query.device, dtype=query.dtype)/1
#      attn_mask = torch.zeros(L, S, device=query.device, dtype=query.dtype)
      attn_mask = attn_mask.masked_fill(mask==False, -65504.)
      attn_mask = attn_mask.reshape([1, 1, L, S])
      if L == 1:
#      attn_mask = torch.zeros(1, 1, L, S, device=query.device, dtype=query.dtype)
#      attn_mask = torch.rand(1, 1, L, S, device=query.device, dtype=query.dtype)/100
        attn_mask = torch.rand(1, 1, L, S, device=query.device, dtype=query.dtype)/1
    if _attention_type == "coherence":
#    if True:
#    if False:
      import json
      import numpy as np
      coherence_tril = json.loads(_coherence_json)
      coherence_tril_tensor = torch.Tensor(coherence_tril).to(query.device).type(query.dtype)
      L_max = shared.settings['truncation_length']-1
      mask_max = torch.ones(L_max, L_max, device=query.device, dtype=torch.bool).tril(diagonal=0)
      lower_indices = np.tril_indices(L_max, k = -1)
      attn_mask_max = torch.zeros(L_max, L_max, device=query.device, dtype=query.dtype)
#      print('lower_indices: '+str(lower_indices))
      attn_mask_max[lower_indices] = coherence_tril_tensor
#      attn_mask_max = torch.rand(L_max, L_max, device=query.device, dtype=query.dtype)/1

      attn_mask = torch.zeros(L, S, device=query.device, dtype=query.dtype)
#      attn_mask = torch.rand(L, S, device=query.device, dtype=query.dtype)

#      attn_mask = torch.zeros(L, S, device=query.device, dtype=query.dtype)
#      print('lower_indices: '+str(lower_indices))
#      attn_mask[lower_indices] = coherence_tril_tensor
#    print('attn_mask: '+str(attn_mask))
#    print('mask: '+str(mask))
#    attn_mask = attn_mask.masked_fill(mask==False, -float('inf'))
      attn_mask_max = attn_mask_max.masked_fill(mask_max==False, -65504.)
      if L==1:
        attn_mask[:,:] = attn_mask_max[S-1,:S]
      else:
        attn_mask[:,:] = attn_mask_max[:L,:S]
      attn_mask = attn_mask.reshape([1, 1, L, S])
#      if L == 1:
#      attn_mask = torch.zeros(1, 1, L, S, device=query.device, dtype=query.dtype)
#      attn_mask = torch.rand(1, 1, L, S, device=query.device, dtype=query.dtype)/100
#        attn_mask = attn_mask_max[:,:,S-1,:S]


#      attn_mask = torch.rand(L, S, device=query.device, dtype=query.dtype)/1
#      attn_mask = torch.zeros(L, S, device=query.device, dtype=query.dtype)
#      attn_mask = attn_mask.masked_fill(mask==False, -65504.)
#      attn_mask = attn_mask.reshape([1, 1, L, S])
#      if L == 1:
#      attn_mask = torch.zeros(1, 1, L, S, device=query.device, dtype=query.dtype)
#      attn_mask = torch.rand(1, 1, L, S, device=query.device, dtype=query.dtype)/100
#        attn_mask = torch.rand(1, 1, L, S, device=query.device, dtype=query.dtype)/1
        
    return attn_mask
    
