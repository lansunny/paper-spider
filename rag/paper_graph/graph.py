"""Independent weighted kNN graphs and seeded Leiden communities."""
from collections import Counter

import numpy as np

from rag.ingestion import fingerprint


def similarity_graph(paper_ids, vectors, *, dimension, neighbors, threshold, resolution, seed):
    import igraph as ig
    import leidenalg

    values = np.asarray(vectors, dtype=np.float32)
    if (values.ndim != 2 or values.shape[0] != len(paper_ids) or len(set(paper_ids)) != len(paper_ids)
            or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=.01)):
        raise ValueError('Graph inputs must be unique paper IDs with finite, normalized vectors')
    if neighbors < 1 or not 0 <= threshold < 1 or resolution <= 0:
        raise ValueError('Invalid kNN or Leiden settings')
    similarities = np.clip(values @ values.T, -1, 1)
    selected, edges = set(), []
    for i, paper_id in enumerate(paper_ids):
        nearest = sorted((j for j in range(len(paper_ids)) if j != i),
                         key=lambda j: (-float(similarities[i, j]), paper_ids[j]))[:neighbors]
        for j in nearest:
            if similarities[i, j] > 0 and similarities[i, j] >= threshold:
                selected.add((min(i, j), max(i, j)))
    pairs = sorted(selected)
    graph = ig.Graph(n=len(paper_ids), edges=pairs, directed=False)
    weights = [float(similarities[i, j]) for i, j in pairs]
    graph.es['weight'] = weights
    degrees = graph.degree()
    membership = [None] * len(paper_ids)
    communities = []
    if pairs:
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
            weights=weights, resolution_parameter=resolution, n_iterations=-1, seed=seed)
        groups = [sorted((i for i in group if degrees[i] > 0), key=lambda i: paper_ids[i]) for group in partition]
        groups = [group for group in groups if len(group) > 1]
        groups.sort(key=lambda group: (-len(group), [paper_ids[i] for i in group]))
        strengths = graph.strength(weights='weight')
        for index, group in enumerate(groups, 1):
            members = [paper_ids[i] for i in group]
            identifier = dimension + '-' + fingerprint(members)[:12]
            for i in group:
                membership[i] = identifier
            representatives = [paper_ids[i] for i in sorted(group, key=lambda i: (-strengths[i], paper_ids[i]))[:3]]
            communities.append({'community_id': identifier, 'ordinal': index,
                'members': members, 'representative_paper_ids': representatives,
                'label_status': 'unnamed', 'label_basis': 'Leiden community; no inferred scholarly taxonomy'})
    for (i, j), weight in zip(pairs, weights, strict=True):
        edges.append({'source': paper_ids[i], 'target': paper_ids[j], 'dimension': dimension,
                      'weight': weight, 'relation': dimension + '_similarity'})
    nodes = [{'paper_id': p, 'community_id': membership[i], 'degree': degrees[i]} for i, p in enumerate(paper_ids)]
    return {'nodes': nodes, 'edges': edges, 'communities': communities,
        'statistics': {'nodes': len(nodes), 'edges': len(edges), 'communities': len(communities),
            'isolated': sum(d == 0 for d in degrees),
            'unassigned': sum(c is None for c in membership),
            'degree_distribution': dict(sorted(Counter(degrees).items())),
            'edge_cosine_min': min(weights) if weights else None,
            'edge_cosine_median': float(np.median(weights)) if weights else None,
            'edge_cosine_max': max(weights) if weights else None}}
