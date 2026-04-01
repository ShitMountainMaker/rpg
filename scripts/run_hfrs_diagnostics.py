#!/usr/bin/env python3

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RPG')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_known_args()


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main():
    args, unparsed_args = parse_args()
    config_dict = parse_command_line_args(unparsed_args)
    config_dict['val_use_graph_decoding'] = False
    config_dict['test_use_graph_decoding'] = False

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=config_dict,
    )
    model = pipeline.model.to(pipeline.config['device'])
    model.eval()

    dataloader = DataLoader(
        pipeline.tokenized_datasets[args.split],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[args.split]
    )

    top16_hits = 0
    top32_hits = 0
    top64_hits = 0
    total_examples = 0
    margin_base = []
    margin_total = []
    margin_lift = []
    residual_pos = []
    residual_neg = []
    top_fp_suppressed = 0
    margin_improved = 0
    bucket_stats = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'HFRS diagnostics ({args.split})'):
            batch = {k: v.to(pipeline.config['device']) for k, v in batch.items()}
            outputs = model(batch, return_loss=False)
            states = model._get_last_step_states(outputs.final_states, batch['seq_lens'])
            states = F.normalize(states, dim=-1)
            token_logits = model._compute_token_log_probs(states)
            positive_item_ids = model._get_target_item_ids(batch)

            candidate_item_ids, candidate_scores = model.get_base_topk_candidates(token_logits, 64)
            positive_base_scores = model.score_item_ids_base(token_logits, positive_item_ids.unsqueeze(1)).squeeze(1)

            for row in range(candidate_item_ids.shape[0]):
                total_examples += 1
                positive_item_id = positive_item_ids[row].item()
                ranked_item_ids = candidate_item_ids[row].tolist()
                ranked_scores = candidate_scores[row].tolist()

                if positive_item_id in ranked_item_ids[:16]:
                    top16_hits += 1
                if positive_item_id in ranked_item_ids[:32]:
                    top32_hits += 1
                if positive_item_id in ranked_item_ids[:64]:
                    top64_hits += 1

                false_positive_item_id = None
                false_positive_base_score = None
                for candidate_item_id, candidate_score in zip(ranked_item_ids, ranked_scores):
                    if candidate_item_id != positive_item_id:
                        false_positive_item_id = candidate_item_id
                        false_positive_base_score = candidate_score
                        break
                if false_positive_item_id is None:
                    continue

                pair_item_ids = torch.tensor([[positive_item_id, false_positive_item_id]], device=states.device)
                total_scores, residual_scores = model.score_item_ids_total(
                    states[row:row + 1],
                    token_logits[row:row + 1],
                    pair_item_ids
                )

                base_margin = positive_base_scores[row].item() - false_positive_base_score
                total_margin = total_scores[0, 0].item() - total_scores[0, 1].item()
                lift = total_margin - base_margin
                margin_base.append(base_margin)
                margin_total.append(total_margin)
                margin_lift.append(lift)
                residual_pos.append(residual_scores[0, 0].item())
                residual_neg.append(residual_scores[0, 1].item())
                if total_margin > 0:
                    top_fp_suppressed += 1
                if lift > 0:
                    margin_improved += 1

                hamming_distance = int(
                    (model.item_id2tokens[positive_item_id] != model.item_id2tokens[false_positive_item_id]).sum().item()
                )
                bucket_stats[hamming_distance]['margin_base'].append(base_margin)
                bucket_stats[hamming_distance]['margin_total'].append(total_margin)
                bucket_stats[hamming_distance]['margin_lift'].append(lift)

    summary = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'use_hfrs': bool(model.use_hfrs),
        'beta_eff': float(model.hfrs_beta_eff.item()) if model.use_hfrs else 0.0,
        'num_examples': total_examples,
        'base_top16_hit_rate': top16_hits / max(total_examples, 1),
        'base_top32_hit_rate': top32_hits / max(total_examples, 1),
        'base_top64_hit_rate': top64_hits / max(total_examples, 1),
        'margin_base_mean': mean_or_zero(margin_base),
        'margin_total_mean': mean_or_zero(margin_total),
        'margin_lift_mean': mean_or_zero(margin_lift),
        'residual_pos_mean': mean_or_zero(residual_pos),
        'residual_neg_mean': mean_or_zero(residual_neg),
        'top_fp_suppressed_rate': top_fp_suppressed / max(total_examples, 1),
        'margin_improved_rate': margin_improved / max(total_examples, 1),
        'bucket_stats': {
            str(bucket): {
                'count': len(stats['margin_base']),
                'margin_base_mean': mean_or_zero(stats['margin_base']),
                'margin_total_mean': mean_or_zero(stats['margin_total']),
                'margin_lift_mean': mean_or_zero(stats['margin_lift']),
            }
            for bucket, stats in sorted(bucket_stats.items())
        }
    }

    if args.output is None:
        output_path = ROOT / 'analysis_results' / f'{Path(args.checkpoint).stem}_{args.split}_hfrs_diagnostics.json'
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
