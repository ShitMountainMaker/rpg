#!/usr/bin/env python3

import argparse
import json
import math
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
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--topm', type=int, default=128)
    parser.add_argument('--candidate_buffer', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_known_args()


def main():
    args, unparsed_args = parse_args()
    config_dict = parse_command_line_args(unparsed_args)
    config_dict['use_hfrs'] = False
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

    train_dataloader = DataLoader(
        pipeline.tokenized_datasets['train'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn['train']
    )

    item_confusions = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'margin_sum': 0.0}))
    top16_hits = 0
    top32_hits = 0
    top64_hits = 0
    total_examples = 0
    topk = min(args.topm + args.candidate_buffer, model.dataset.n_items - 1)

    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc='Mining HFRS hard negatives'):
            batch = {k: v.to(pipeline.config['device']) for k, v in batch.items()}
            outputs = model(batch, return_loss=False)
            states = model._get_last_step_states(outputs.final_states, batch['seq_lens'])
            states = F.normalize(states, dim=-1)
            token_logits = model._compute_token_log_probs(states)
            positive_item_ids = model._get_target_item_ids(batch)

            candidate_item_ids, candidate_scores = model.get_base_topk_candidates(token_logits, max(topk, 64))
            positive_scores = model.score_item_ids_base(token_logits, positive_item_ids.unsqueeze(1)).squeeze(1)

            for row in range(candidate_item_ids.shape[0]):
                positive_item_id = positive_item_ids[row].item()
                ranked_item_ids = candidate_item_ids[row].tolist()
                ranked_scores = candidate_scores[row].tolist()
                total_examples += 1

                if positive_item_id in ranked_item_ids[:16]:
                    top16_hits += 1
                if positive_item_id in ranked_item_ids[:32]:
                    top32_hits += 1
                if positive_item_id in ranked_item_ids[:64]:
                    top64_hits += 1

                seen = set()
                collected = 0
                for negative_item_id, negative_score in zip(ranked_item_ids, ranked_scores):
                    if negative_item_id == positive_item_id or negative_item_id in seen:
                        continue
                    seen.add(negative_item_id)
                    confusion_stats = item_confusions[positive_item_id][negative_item_id]
                    confusion_stats['count'] += 1
                    confusion_stats['margin_sum'] += positive_scores[row].item() - negative_score
                    collected += 1
                    if collected >= args.topm:
                        break

    item_to_hard_negatives = {}
    pool_lengths = []
    for item_id, negative_stats in item_confusions.items():
        sorted_negatives = sorted(
            negative_stats.items(),
            key=lambda kv: (-kv[1]['count'], kv[1]['margin_sum'] / max(kv[1]['count'], 1), kv[0])
        )
        negative_ids = [negative_id for negative_id, _ in sorted_negatives[:args.topm]]
        item_to_hard_negatives[str(item_id)] = negative_ids
        pool_lengths.append(len(negative_ids))

    if args.output is None:
        output_path = ROOT / 'cache' / args.dataset / config_dict.get('category', 'Sports_and_Outdoors') / 'processed' / (
            f'{Path(args.checkpoint).stem}_hfrs_top{args.topm}.json'
        )
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(item_to_hard_negatives, f)

    summary = {
        'checkpoint': args.checkpoint,
        'output': str(output_path),
        'num_items_with_pool': len(item_to_hard_negatives),
        'avg_pool_length': float(sum(pool_lengths) / max(len(pool_lengths), 1)),
        'min_pool_length': int(min(pool_lengths) if pool_lengths else 0),
        'max_pool_length': int(max(pool_lengths) if pool_lengths else 0),
        'train_examples': total_examples,
        'base_top16_hit_rate': top16_hits / max(total_examples, 1),
        'base_top32_hit_rate': top32_hits / max(total_examples, 1),
        'base_top64_hit_rate': top64_hits / max(total_examples, 1),
    }
    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
