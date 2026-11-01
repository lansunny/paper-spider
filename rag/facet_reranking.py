"""Fuse query-specific ranks and select complementary facet representatives.

Top-three membership is a ranking proxy, never proof of factual support.
All policies reorder the exact same candidates, without reading gold evidence.
"""
import copy


def validate_rankings(rankings):
    if 'original' not in rankings:
        raise ValueError('Original ranking is required')
    expected = [hit['chunk_id'] for hit in rankings['original']]
    if len(expected) != len(set(expected)):
        raise ValueError('Duplicate original candidate IDs')
    originals = {h['chunk_id']: h for h in rankings['original']}
    for label, hits in rankings.items():
        ids = [h['chunk_id'] for h in hits]
        if len(ids) != len(expected) or set(ids) != set(expected):
            raise ValueError(f'Candidate set differs for {label}')
        if any(h['text'] != originals[h['chunk_id']]['text'] for h in hits):
            raise ValueError('Reranking changed candidate bodies')


def fuse_facet_rankings(rankings, *, k=60):
    """Original gets half the weight; facets share the other half equally."""
    validate_rankings(rankings)
    if k < 1:
        raise ValueError('RRF k must be positive')
    facets = [name for name in rankings if name != 'original']
    if not facets:
        return copy.deepcopy(rankings['original'])
    weights = {'original': .5, **{name: .5/len(facets) for name in facets}}
    positions = {label: {hit['chunk_id']: rank for rank, hit in enumerate(hits, 1)}
                 for label, hits in rankings.items()}
    output = []
    for hit in rankings['original']:
        ranks = {name: positions[name][hit['chunk_id']] for name in rankings}
        value = sum(weights[name]/(k+rank) for name, rank in ranks.items())
        output.append({**copy.deepcopy(hit), 'score': value,
                       'scores': {**hit.get('scores', {}), 'facet_rrf': value}, 'facet_ranks': ranks})
    return sorted(output, key=lambda h: (-h['score'], h['chunk_id']))


def facet_proxy(facets, selected, threshold=3):
    ids = {h['chunk_id'] for h in selected}
    return {name: bool(ids.intersection(h['chunk_id'] for h in hits[:threshold]))
            for name, hits in facets.items()}


def order_for_facet_coverage(rankings, fused, *, top_k=5, threshold=3):
    """Keep original top one, greedily represent unseen facet top-three lists.

Prefer a candidate representing more unseen facets, break ties by fused rank.
Once all facet proxies are represented, fill remaining slots in fused order.
"""
    validate_rankings(rankings)
    if top_k < 1 or threshold < 1:
        raise ValueError('Selection limits must be positive')
    original_ids = {h['chunk_id'] for h in rankings['original']}
    if len(fused) != len(original_ids) or {h['chunk_id'] for h in fused} != original_ids:
        raise ValueError('Fused ranking differs from the candidate set')
    facets = {name: hits for name, hits in rankings.items() if name != 'original'}
    if not facets or not fused:
        return copy.deepcopy(fused), []
    by_id = {h['chunk_id']: h for h in fused}
    shortlist = {name: {h['chunk_id'] for h in hits[:threshold]} for name, hits in facets.items()}
    selected = [by_id[rankings['original'][0]['chunk_id']]]
    steps = [{'chunk_id': selected[0]['chunk_id'], 'reason': 'retain_original_top1'}]
    fused_position = {h['chunk_id']: i for i, h in enumerate(fused)}
    while len(selected) < min(top_k, len(fused)):
        seen = {h['chunk_id'] for h in selected}
        remaining = {name for name, ids in shortlist.items() if not ids.intersection(seen)}
        if not remaining:
            break
        options = set().union(*(shortlist[name] for name in remaining)) - seen
        if not options:
            break
        chosen = min(options, key=lambda cid: (
            -sum(cid in shortlist[name] for name in remaining), fused_position[cid], cid))
        gained = sorted(name for name in remaining if chosen in shortlist[name])
        selected.append(by_id[chosen])
        steps.append({'chunk_id': chosen, 'reason': 'new_facet_rank_proxy', 'facets': gained})
    selected_ids = {h['chunk_id'] for h in selected}
    for hit in fused:
        if len(selected) >= min(top_k, len(fused)):
            break
        if hit['chunk_id'] not in selected_ids:
            selected.append(hit)
            selected_ids.add(hit['chunk_id'])
            steps.append({'chunk_id': hit['chunk_id'], 'reason': 'fill_by_fused_rank'})
    return [*selected, *[h for h in fused if h['chunk_id'] not in selected_ids]], steps
