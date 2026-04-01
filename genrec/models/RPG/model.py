# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class ConsensusCorrection(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        consensus_hidden_dim: int,
        beta: float,
        alpha_init: float,
        mask_self: bool,
        detach_confidence: bool,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.query = nn.Linear(hidden_dim, consensus_hidden_dim)
        self.key = nn.Linear(hidden_dim, consensus_hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.register_buffer('confidence_bias_scale', torch.tensor(beta, dtype=torch.float32))
        self.mask_self = mask_self
        self.detach_confidence = detach_confidence
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def _compute_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        log_vocab = torch.log(torch.tensor(probs.shape[-1], device=probs.device, dtype=probs.dtype))
        entropy = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=-1)
        confidence = 1.0 - entropy / log_vocab
        return confidence.detach() if self.detach_confidence else confidence

    def forward(self, logits: torch.Tensor, token_embs: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        summaries = torch.einsum('bmv,mvd->bmd', probs, token_embs)
        confidence = self._compute_confidence(probs)

        queries = self.query(summaries)
        keys = self.key(summaries)
        values = self.value(summaries)

        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attn_scores = attn_scores / (queries.shape[-1] ** 0.5)
        attn_scores = attn_scores + self.confidence_bias_scale * torch.log(confidence.clamp_min(self.eps)).unsqueeze(-2)

        if self.mask_self:
            diag_mask = torch.eye(attn_scores.shape[-1], device=attn_scores.device, dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        messages = torch.matmul(attn_weights, values)

        gate_inputs = torch.cat([summaries, messages, confidence.unsqueeze(-1)], dim=-1)
        gate = torch.sigmoid(self.gate(gate_inputs))
        delta_summary = gate * messages
        delta_logits = torch.einsum('bmd,mvd->bmv', delta_summary, token_embs)
        return logits + self.alpha * delta_logits


class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        self.use_consensus_correction = self.config.get('use_consensus_correction', False)
        self.consensus_correction = None
        if self.use_consensus_correction:
            self.consensus_correction = ConsensusCorrection(
                hidden_dim=self.config['n_embd'],
                consensus_hidden_dim=self.config.get('consensus_hidden_dim', 64),
                beta=self.config.get('consensus_beta', 0.2),
                alpha_init=self.config.get('consensus_alpha_init', 0.0),
                mask_self=self.config.get('consensus_mask_self', True),
                detach_confidence=self.config.get('consensus_detach_confidence', True),
            )

        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        # Hard-Family Holistic Item Scorer (HFRS)
        self.use_hfrs = self.config.get('use_hfrs', False)
        self.hfrs_stage = self.config.get('hfrs_stage', 'A')
        self.hfrs_res_dim = self.config.get('hfrs_res_dim', 16)
        self.hfrs_beta_scale = self.config.get('hfrs_beta_scale', 0.1)
        self.hfrs_rerank_topk = self.config.get('hfrs_rerank_topk', 16)
        self.hfrs_pool_topm = self.config.get('hfrs_pool_topm', 128)
        self.hfrs_min_pool_size = self.config.get('hfrs_min_pool_size', 4)
        self.hfrs_use_base_only = self.config.get('hfrs_use_base_only', True)
        self.hfrs_hard_negative_file = self.config.get('hfrs_hard_negative_file')

        self.register_buffer(
            'hfrs_hard_negative_table',
            torch.zeros((0, 0), dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            'hfrs_hard_negative_lengths',
            torch.zeros((0,), dtype=torch.long),
            persistent=False
        )

        if self.use_hfrs:
            self._init_hfrs_modules()
            self._load_hfrs_hard_negative_pool()
            self._configure_hfrs_stage()

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def _get_codebook_token_embs(self) -> torch.Tensor:
        token_embs = self.gpt2.wte.weight[1:-1]
        token_embs = F.normalize(token_embs, dim=-1)
        return token_embs.view(self.n_pred_head, self.config['codebook_size'], -1)

    def _init_hfrs_modules(self):
        query_input_dim = self.config['n_embd']
        self.hfrs_user_query = nn.Sequential(
            nn.Linear(query_input_dim, self.config['n_embd']),
            nn.SiLU(),
            nn.Linear(self.config['n_embd'], self.hfrs_res_dim),
        )
        self.hfrs_token_key = nn.Linear(self.config['n_embd'], self.hfrs_res_dim)
        self.hfrs_token_value = nn.Linear(self.config['n_embd'], self.hfrs_res_dim)
        self.hfrs_score_proj = nn.Linear(1, self.hfrs_res_dim)
        self.hfrs_attn_out = nn.Linear(self.hfrs_res_dim, 1)
        self.hfrs_position_emb = nn.Embedding(self.n_pred_head, self.hfrs_res_dim)
        self.hfrs_residual_head = nn.Sequential(
            nn.Linear(self.hfrs_res_dim * 2 + 4, self.config['n_embd']),
            nn.SiLU(),
            nn.Linear(self.config['n_embd'], 1),
        )

        beta_target = 0.01 / max(self.hfrs_beta_scale, 1e-8)
        beta_target = min(max(beta_target, -0.999), 0.999)
        self.hfrs_beta_raw = nn.Parameter(torch.tensor(math.atanh(beta_target), dtype=torch.float32))

    def _load_hfrs_hard_negative_pool(self):
        if not self.hfrs_hard_negative_file:
            return

        hard_negative_path = self.hfrs_hard_negative_file
        if not os.path.exists(hard_negative_path):
            return

        with open(hard_negative_path, 'r') as f:
            item_to_hard_negatives = json.load(f)

        table = torch.zeros((self.dataset.n_items, self.hfrs_pool_topm), dtype=torch.long)
        lengths = torch.zeros((self.dataset.n_items,), dtype=torch.long)
        for item_id_str, negative_ids in item_to_hard_negatives.items():
            item_id = int(item_id_str)
            filtered_negatives = []
            seen = set()
            for negative_id in negative_ids:
                negative_id = int(negative_id)
                if negative_id <= 0 or negative_id >= self.dataset.n_items or negative_id == item_id:
                    continue
                if negative_id in seen:
                    continue
                seen.add(negative_id)
                filtered_negatives.append(negative_id)
                if len(filtered_negatives) >= self.hfrs_pool_topm:
                    break
            if not filtered_negatives:
                continue
            lengths[item_id] = len(filtered_negatives)
            table[item_id, :len(filtered_negatives)] = torch.tensor(filtered_negatives, dtype=torch.long)

        self.hfrs_hard_negative_table = table.to(self.config['device'])
        self.hfrs_hard_negative_lengths = lengths.to(self.config['device'])

    def _configure_hfrs_stage(self):
        if not self.use_hfrs or self.hfrs_stage != 'A' or not self.hfrs_use_base_only:
            return

        for module in (self.gpt2, self.pred_heads):
            for parameter in module.parameters():
                parameter.requires_grad = False
        if self.consensus_correction is not None:
            for parameter in self.consensus_correction.parameters():
                parameter.requires_grad = False

    def _compute_codebook_logits(self, states: torch.Tensor) -> torch.Tensor:
        token_embs = self._get_codebook_token_embs()
        logits = torch.einsum('bmd,mvd->bmv', states, token_embs) / self.temperature
        if self.consensus_correction is not None:
            logits = self.consensus_correction(logits, token_embs)
        return logits

    def _get_last_step_states(self, final_states: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        return final_states.gather(
            dim=1,
            index=(seq_lens - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )[:, 0]

    def _get_target_item_ids(self, batch: dict) -> torch.Tensor:
        labels = batch['labels']
        if labels.dim() == 1:
            return labels
        if labels.shape[1] == 1:
            return labels[:, 0]
        return labels.gather(1, (batch['seq_lens'] - 1).view(-1, 1)).squeeze(1)

    @property
    def hfrs_beta_eff(self) -> torch.Tensor:
        return self.hfrs_beta_scale * torch.tanh(self.hfrs_beta_raw)

    def _compute_token_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        codebook_logits = self._compute_codebook_logits(states)
        return F.log_softmax(codebook_logits, dim=-1).reshape(codebook_logits.shape[0], -1)

    def _gather_candidate_tokens(self, item_ids: torch.Tensor) -> torch.Tensor:
        safe_item_ids = item_ids.clamp(min=1)
        return self.item_id2tokens[safe_item_ids]

    def _gather_item_token_scores(self, token_logits: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(0).expand(token_logits.shape[0], -1)
        candidate_tokens = self._gather_candidate_tokens(item_ids)
        gathered_scores = torch.gather(
            token_logits.unsqueeze(-2).expand(-1, candidate_tokens.shape[1], -1),
            dim=-1,
            index=(candidate_tokens - 1)
        )
        return gathered_scores

    def score_item_ids_base(self, token_logits: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return self._gather_item_token_scores(token_logits, item_ids).mean(dim=-1)

    def _get_hfrs_query(self, states: torch.Tensor) -> torch.Tensor:
        pooled_states = states.mean(dim=1)
        return self.hfrs_user_query(pooled_states)

    def _compute_hfrs_residual(self, states: torch.Tensor, token_logits: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(0).expand(token_logits.shape[0], -1)

        candidate_tokens = self._gather_candidate_tokens(item_ids)
        token_scores = self._gather_item_token_scores(token_logits, item_ids)
        token_embs = self.gpt2.wte(candidate_tokens)
        token_embs = F.normalize(token_embs, dim=-1)

        query = self._get_hfrs_query(states)
        token_keys = self.hfrs_token_key(token_embs)
        score_features = self.hfrs_score_proj(token_scores.unsqueeze(-1))
        position_features = self.hfrs_position_emb.weight.view(1, 1, self.n_pred_head, -1)

        attn_hidden = torch.tanh(
            query.unsqueeze(1).unsqueeze(1) +
            token_keys +
            score_features +
            position_features
        )
        attn_weights = torch.softmax(self.hfrs_attn_out(attn_hidden).squeeze(-1), dim=-1)

        token_values = self.hfrs_token_value(token_embs)
        holistic_item = (attn_weights.unsqueeze(-1) * token_values).sum(dim=-2)
        token_stats = torch.stack([
            token_scores.mean(dim=-1),
            token_scores.max(dim=-1).values,
            token_scores.std(dim=-1, correction=0),
        ], dim=-1)

        residual_inputs = torch.cat([
            query.unsqueeze(1).expand(-1, item_ids.shape[1], -1),
            holistic_item,
            token_stats
        ], dim=-1)
        residual = self.hfrs_residual_head(residual_inputs).squeeze(-1)
        return residual

    def score_item_ids_total(self, states: torch.Tensor, token_logits: torch.Tensor, item_ids: torch.Tensor):
        base_scores = self.score_item_ids_base(token_logits, item_ids)
        if not self.use_hfrs:
            return base_scores, torch.zeros_like(base_scores)
        residual_scores = self._compute_hfrs_residual(states, token_logits, item_ids)
        total_scores = base_scores + self.hfrs_beta_eff * residual_scores
        return total_scores, residual_scores

    def get_base_topk_candidates(self, token_logits: torch.Tensor, k: int):
        all_item_ids = torch.arange(1, self.dataset.n_items, device=token_logits.device, dtype=torch.long)
        all_item_ids = all_item_ids.unsqueeze(0).expand(token_logits.shape[0], -1)
        base_scores = self.score_item_ids_base(token_logits, all_item_ids)
        topk_scores, topk_indices = base_scores.topk(k, dim=-1)
        topk_item_ids = all_item_ids.gather(1, topk_indices)
        return topk_item_ids, topk_scores

    def _compute_hfrs_loss(self, states: torch.Tensor, token_logits: torch.Tensor, positive_item_ids: torch.Tensor):
        if self.hfrs_hard_negative_table.numel() == 0:
            raise FileNotFoundError(
                'HFRS training requires a valid hfrs_hard_negative_file with mined per-item hard negatives.'
            )

        hard_negative_lengths = self.hfrs_hard_negative_lengths[positive_item_ids]
        valid_mask = hard_negative_lengths >= self.hfrs_min_pool_size
        if not valid_mask.any():
            zero = token_logits.sum() * 0.0
            return zero, {
                'family_listwise_loss': 0.0,
                'beta_eff': self.hfrs_beta_eff.detach().item(),
                'residual_pos_mean': 0.0,
                'residual_neg_mean': 0.0,
                'family_positive_rank': 0.0,
                'family_training_examples': 0.0,
            }

        states = states[valid_mask]
        token_logits = token_logits[valid_mask]
        positive_item_ids = positive_item_ids[valid_mask]

        negative_ids = self.hfrs_hard_negative_table[positive_item_ids, :self.hfrs_rerank_topk]
        negative_mask = negative_ids > 0
        candidate_mask = torch.cat([
            torch.ones((positive_item_ids.shape[0], 1), dtype=torch.bool, device=positive_item_ids.device),
            negative_mask
        ], dim=1)

        candidate_item_ids = torch.cat([positive_item_ids.unsqueeze(1), negative_ids], dim=1)
        total_scores, residual_scores = self.score_item_ids_total(states, token_logits, candidate_item_ids)
        masked_scores = total_scores.masked_fill(~candidate_mask, -1e9)

        listwise_loss = -F.log_softmax(masked_scores, dim=-1)[:, 0].mean()
        residual_pos_mean = residual_scores[:, 0].mean()
        if negative_mask.any():
            residual_neg_mean = residual_scores[:, 1:][negative_mask].mean()
        else:
            residual_neg_mean = residual_scores[:, 0].new_zeros(())
        positive_rank = 1 + ((masked_scores[:, 1:] > masked_scores[:, :1]) & candidate_mask[:, 1:]).sum(dim=-1).float()

        return listwise_loss, {
            'family_listwise_loss': listwise_loss.detach().item(),
            'beta_eff': self.hfrs_beta_eff.detach().item(),
            'residual_pos_mean': residual_pos_mean.detach().item(),
            'residual_neg_mean': residual_neg_mean.detach().item(),
            'family_positive_rank': positive_rank.mean().detach().item(),
            'family_training_examples': float(positive_item_ids.shape[0]),
        }

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            if self.use_hfrs:
                selected_states = self._get_last_step_states(final_states, batch['seq_lens'])
                selected_states = F.normalize(selected_states, dim=-1)
                token_logits = self._compute_token_log_probs(selected_states)
                positive_item_ids = self._get_target_item_ids(batch)
                outputs.loss, outputs.log_dict = self._compute_hfrs_loss(
                    states=selected_states,
                    token_logits=token_logits,
                    positive_item_ids=positive_item_ids
                )
            else:
                label_mask = batch['labels'].view(-1) != -100
                selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
                selected_states = F.normalize(selected_states, dim=-1)
                token_logits = self._compute_codebook_logits(selected_states)
                token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
                losses = [
                    self.loss_fct(token_logits[:, i, :], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                    for i in range(self.n_pred_head)
                ]
                outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end]  # sub-block for items i

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = self.item_id2tokens[j_start:j_end]  # sub-block for items j

                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                avg_block = sum_block / n_digit

                # Write back into the final item_item_sim
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]

        # Initialize visited nodes tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Add initial nodes to visited set
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        for sid in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add neighbors to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)

                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1):
        outputs = self.forward(batch, return_loss=False)
        states = self._get_last_step_states(outputs.final_states, batch['seq_lens'])
        states = F.normalize(states, dim=-1)

        token_logits = self._compute_token_log_probs(states)

        if self.generate_w_decoding_graph:
            if self.use_hfrs:
                raise ValueError('HFRS only supports graph-off / exact evaluation. Set test_use_graph_decoding=False.')
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        else:
            all_item_ids = torch.arange(1, self.dataset.n_items, device=token_logits.device, dtype=torch.long)
            all_item_ids = all_item_ids.unsqueeze(0).expand(token_logits.shape[0], -1)
            base_scores = self.score_item_ids_base(token_logits, all_item_ids)

            if not self.use_hfrs:
                preds = base_scores.topk(n_return_sequences, dim=-1).indices + 1
                return preds.unsqueeze(-1)

            rerank_topk = min(self.hfrs_rerank_topk, base_scores.shape[1])
            candidate_scores, candidate_indices = base_scores.topk(rerank_topk, dim=-1)
            candidate_item_ids = all_item_ids.gather(1, candidate_indices)
            total_scores, _ = self.score_item_ids_total(states, token_logits, candidate_item_ids)
            topk_scores, topk_indices = total_scores.topk(min(n_return_sequences, rerank_topk), dim=-1)
            preds = candidate_item_ids.gather(1, topk_indices)
            return preds.unsqueeze(-1)
