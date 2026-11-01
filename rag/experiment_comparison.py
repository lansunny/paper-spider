"""Offline scoring and paired summaries for two chunking configurations."""
import json
from collections import defaultdict

from rag.benchmark import score_evidence

STAGES = ('union', 'rrf20', 'before_top5', 'after_top5', 'after_top10', 'after_top20', 'context_budget')


def evidence_object(hit, number):
    return {'id': f'S{number}', 'paper_id': hit['paper_id'], 'title': hit.get('title', ''),
            'pdf_page': hit['pdf_page'], 'text': hit['text']}


def pack_evidence(hits, char_budget):
    """Exact JSON-list character cap; greedy whole-chunk selection, no gold input.

    Includes citation IDs, titles, page numbers and list separators. Excludes the
    system prompt, question wrapper and answer. This is NOT a token budget.
    Neither neighbor expansion nor overlap merging is applied to either system.
    """
    if char_budget < 2:
        raise ValueError('Evidence budget must fit at least an empty JSON list')
    selected, sources, seen, skipped = [], [], set(), []
    for rank, hit in enumerate(hits, 1):
        identity = (hit['paper_id'], hit['document_sha256'], hit['pdf_page'], hit['text'])
        if identity in seen:
            continue
        source = evidence_object(hit, len(selected) + 1)
        size = len(json.dumps([*sources, source], ensure_ascii=False))
        if size > char_budget:
            skipped.append({'rank': rank, 'chunk_id': hit['chunk_id']})
            continue
        seen.add(identity)
        sources.append(source)
        selected.append({**hit, 'citation_id': source['id']})
    used = len(json.dumps(sources, ensure_ascii=False))
    return selected, {'limit_chars': char_budget, 'used_chars': used,
                      'fill_ratio': used / char_budget, 'selected_chunks': len(selected),
                      'body_chars': sum(len(h['text']) for h in selected), 'skipped': skipped}


def evaluate_stages(question, merged, ranked, pages, budget):
    candidates = merged[:20]
    if len(ranked) != len(candidates) or {h['chunk_id'] for h in ranked} != {h['chunk_id'] for h in candidates}:
        raise ValueError('Reranking must preserve the candidate set exactly')
    context, usage = pack_evidence(ranked, budget)
    lists = {'union': merged, 'rrf20': candidates, 'before_top5': candidates[:5],
             'after_top5': ranked[:5], 'after_top10': ranked[:10],
             'after_top20': ranked, 'context_budget': context}
    metrics = {stage: score_evidence(question, hits, pages, k=max(1, len(hits))) for stage, hits in lists.items()}
    unit_ids = [u['unit_id'] for u in question['evidence_units']]
    unit_map = {stage: {u['unit_id']: u['covered'] for u in m['units']} for stage, m in metrics.items()}
    diagnostics = []
    for unit in unit_ids:
        flags = {stage: unit_map[stage][unit] for stage in STAGES}
        if not flags['union']:
            reason = 'not_fully_recalled'
        elif not flags['rrf20']:
            reason = 'lost_at_fusion_top20'
        elif not flags['after_top5']:
            reason = 'outside_final_top5'
        else:
            reason = 'covered_at_top5'
        diagnostics.append({'unit_id': unit, 'top5_path': reason,
                            'budget_covered': flags['context_budget'],
                            'budget_omitted_available_evidence': flags['after_top20'] and not flags['context_budget'],
                            'stage_coverage': flags})
    return {'metrics': metrics, 'context_usage': usage, 'diagnostics': diagnostics,
            'rankings': lists}


def paired_summary(pairs):
    if not pairs:
        raise ValueError('No completed paired questions to summarize')
    ids = [p['question_id'] for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs in paired comparison')
    groups = defaultdict(list)
    details = []
    for pair in pairs:
        groups['all_answerable'].append(pair)
        groups['scope:' + pair['scope']].append(pair)
        groups['category:' + pair['category']].append(pair)
        changes = {}
        for stage in STAGES:
            old, new = (pair[s]['metrics'][stage] for s in ('baseline', 'structured'))
            delta = new['evidence_unit_recall'] - old['evidence_unit_recall']
            changes[stage] = {'baseline': old['evidence_unit_recall'], 'structured': new['evidence_unit_recall'],
                              'delta': delta, 'change': 'improved' if delta > 1e-12 else 'regressed' if delta < -1e-12 else 'unchanged',
                              'baseline_complete': old['all_required_evidence'],
                              'structured_complete': new['all_required_evidence']}
        details.append({'question_id': pair['question_id'], 'question': pair['question'],
                        'scope': pair['scope'], 'changes': changes})
    summaries = []
    for name, rows in sorted(groups.items()):
        stages = {}
        for stage in STAGES:
            stats = {}
            for system in ['baseline', 'structured']:
                stats[system] = {
                    'mean_evidence_unit_recall': sum(p[system]['metrics'][stage]['evidence_unit_recall'] for p in rows) / len(rows),
                    'all_required_evidence_rate': sum(p[system]['metrics'][stage]['all_required_evidence'] for p in rows) / len(rows)}
            stats['delta_recall'] = stats['structured']['mean_evidence_unit_recall'] - stats['baseline']['mean_evidence_unit_recall']
            stages[stage] = stats
        budget_usage = {system: {
            'mean_used_chars': sum(p[system]['context_usage']['used_chars'] for p in rows) / len(rows),
            'mean_selected_chunks': sum(p[system]['context_usage']['selected_chunks'] for p in rows) / len(rows),
            'mean_fill_ratio': sum(p[system]['context_usage']['fill_ratio'] for p in rows) / len(rows),
        } for system in ['baseline', 'structured']}
        summaries.append({'group': name, 'questions': len(rows), 'stages': stages, 'budget_usage': budget_usage})
    return {'groups': summaries, 'questions': details, 'changes': {
        stage: {label: [d['question_id'] for d in details if d['changes'][stage]['change'] == label]
                for label in ['improved', 'regressed', 'unchanged']} for stage in STAGES}}
