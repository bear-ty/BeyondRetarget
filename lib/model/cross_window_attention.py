#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-window attention for RGB2Robo temporal feature sequences. It supports
cached preceding/following windows and optional bidirectional context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class CrossWindowAttentionBlock(nn.Module):
    """Exchange temporal context between neighboring inference windows."""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        context_span: int = 8,
        bidirectional: bool = True,
        use_layer_norm: bool = True,
        max_position_length: Optional[int] = None,
    ):
        """Initialize the attention block and its context policy."""
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.context_span = context_span
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        max_len = max(int(max_position_length or 0), context_span * 2 + 20)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model)
            self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention projections."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
        
        nn.init.xavier_uniform_(self.w_o.weight, gain=0.1)
        nn.init.constant_(self.w_o.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        context_kv: Optional[Dict[str, torch.Tensor]] = None,
        enable_cross_window: bool = True,
        return_kv_cache: bool = False,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attend to the current window plus optional cached context."""
        if not enable_cross_window or context_kv is None:
            if return_kv_cache:
                return x, self._extract_kv_cache(x)
            return x
        
        batch_size, seq_len, d_model = x.shape
        
        residual = x
        if self.use_layer_norm:
            x = self.layer_norm1(x)
        
        q = self.w_q(x)  # [B, T, D]
        k_current = self.w_k(x)  # [B, T, D]
        v_current = self.w_v(x)  # [B, T, D]

        if isinstance(context_kv, dict):
            k_extended, v_extended = self._build_extended_kv(
                k_current, v_current, context_kv, seq_len
            )
        elif isinstance(context_kv, torch.Tensor):
            k_extended, v_extended = self._build_extended_kv_from_tensor(
                k_current, v_current, context_kv, seq_len
            )
        else:
            k_extended, v_extended = k_current, v_current
        
        extended_valid_mask = None
        if context_mask is not None:
            B = x.shape[0]
            ones_current = torch.ones(B, seq_len, dtype=torch.bool, device=x.device)

            if isinstance(context_kv, dict):
                prev_len = context_kv.get('prev_k', None).shape[1] if (isinstance(context_kv.get('prev_k', None), torch.Tensor)) else 0
                next_len = context_kv.get('next_k', None).shape[1] if (self.bidirectional and isinstance(context_kv.get('next_k', None), torch.Tensor)) else 0

                prev_part = context_mask[:, :self.context_span]
                prev_mask = prev_part[:, :prev_len] if prev_len > 0 else torch.zeros(B, 0, dtype=torch.bool, device=x.device)

                if self.bidirectional and next_len > 0:
                    next_part = context_mask[:, self.context_span:self.context_span + self.context_span]
                    next_mask = next_part[:, :next_len]
                else:
                    next_mask = torch.zeros(B, 0, dtype=torch.bool, device=x.device)

                extended_valid_mask = torch.cat([prev_mask, ones_current, next_mask], dim=1)

            elif isinstance(context_kv, torch.Tensor):
                L = context_kv.shape[1]
                if self.bidirectional:
                    prev_len = min(L, self.context_span)
                    next_len = min(max(L - prev_len, 0), self.context_span)
                else:
                    prev_len = min(L, self.context_span)
                    next_len = 0

                prev_part = context_mask[:, :self.context_span]
                prev_mask = prev_part[:, :prev_len] if prev_len > 0 else torch.zeros(B, 0, dtype=torch.bool, device=x.device)

                if self.bidirectional and next_len > 0:
                    next_part = context_mask[:, self.context_span:self.context_span + self.context_span]
                    next_mask = next_part[:, :next_len]
                else:
                    next_mask = torch.zeros(B, 0, dtype=torch.bool, device=x.device)

                extended_valid_mask = torch.cat([prev_mask, ones_current, next_mask], dim=1)

            else:
                extended_valid_mask = torch.ones(B, k_extended.shape[1], dtype=torch.bool, device=x.device)

        attn_output = self._multi_head_attention(q, k_extended, v_extended, mask=extended_valid_mask)
        
        x = residual + self.dropout(attn_output)
        
        residual = x
        if self.use_layer_norm:
            x = self.layer_norm2(x)
        
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        
        if return_kv_cache:
            kv_cache = self._extract_kv_cache_from_current(k_current, v_current)
            return x, kv_cache
        
        return x

    def _build_extended_kv(
        self,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        context_kv: Dict[str, torch.Tensor],
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build key/value tensors containing the optional context."""
        k_list = []
        v_list = []
        
        if 'prev_k' in context_kv and context_kv['prev_k'] is not None:
            prev_k = context_kv['prev_k']  # [B, context_span, D]
            prev_v = context_kv['prev_v']  # [B, context_span, D]
            
            prev_k_pos = self.pos_encoding(prev_k, offset=-self.context_span)
            k_list.append(prev_k_pos)
            v_list.append(prev_v)
        
        k_current_pos = self.pos_encoding(k_current, offset=0)
        k_list.append(k_current_pos)
        v_list.append(v_current)
        
        if (self.bidirectional and 'next_k' in context_kv and 
            context_kv['next_k'] is not None):
            next_k = context_kv['next_k']  # [B, context_span, D]
            next_v = context_kv['next_v']  # [B, context_span, D]
            
            next_k_pos = self.pos_encoding(next_k, offset=seq_len)
            k_list.append(next_k_pos)
            v_list.append(next_v)
        
        k_extended = torch.cat(k_list, dim=1)  # [B, extended_len, D]
        v_extended = torch.cat(v_list, dim=1)  # [B, extended_len, D]
        
        return k_extended, v_extended

    def _build_extended_kv_from_tensor(
        self,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        context_feat: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build extended key/value tensors from a compact context tensor."""
        k_list = []
        v_list = []

        B, L, D = context_feat.shape
        k_context = self.w_k(context_feat)  # [B, L, D]
        v_context = self.w_v(context_feat)  # [B, L, D]

        if self.bidirectional:
            prev_len = min(L, self.context_span)
            next_len = min(max(L - prev_len, 0), self.context_span)
        else:
            prev_len = min(L, self.context_span)
            next_len = 0

        if prev_len > 0:
            prev_k = k_context[:, :prev_len, :]
            prev_v = v_context[:, :prev_len, :]
            prev_k_pos = self.pos_encoding(prev_k, offset=-self.context_span)
            k_list.append(prev_k_pos)
            v_list.append(prev_v)

        k_current_pos = self.pos_encoding(k_current, offset=0)
        k_list.append(k_current_pos)
        v_list.append(v_current)

        if self.bidirectional and next_len > 0:
            next_k = k_context[:, prev_len:prev_len + next_len, :]
            next_v = v_context[:, prev_len:prev_len + next_len, :]
            next_k_pos = self.pos_encoding(next_k, offset=seq_len)
            k_list.append(next_k_pos)
            v_list.append(next_v)

        k_extended = torch.cat(k_list, dim=1)
        v_extended = torch.cat(v_list, dim=1)

        return k_extended, v_extended
    
    def _multi_head_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute masked multi-head attention."""
        batch_size, seq_len, _ = q.shape
        extended_len = k.shape[1]
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, extended_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, extended_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask_broadcast = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_broadcast, float('-inf'))
        
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)
    
    def _extract_kv_cache(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract a cache from input features."""
        k = self.w_k(x)
        v = self.w_v(x)
        return self._extract_kv_cache_from_current(k, v)
    
    def _extract_kv_cache_from_current(
        self,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract boundary frames from current keys and values."""
        seq_len = k.shape[1]
        span = min(self.context_span, seq_len)
        
        k_start = k[:, :span]
        v_start = v[:, :span]
        k_end = k[:, -span:]
        v_end = v[:, -span:]
        
        return {
            'boundary_start_k': k_start,
            'boundary_start_v': v_start,
            'boundary_end_k': k_end,
            'boundary_end_v': v_end
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with an optional offset."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding to a batch of temporal features."""
        seq_len = x.shape[1]
        max_len = self.pe.shape[1]
        idx = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.long)
        idx = idx % max_len
        pos_encoding = self.pe[:, idx]
        return x + pos_encoding
