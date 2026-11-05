"""Independent community/description candidate branches; original text only."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from rag.formula_store import digest_file
from rag.ingestion import fingerprint, paper_catalog
from rag.paper_graph.retrieval import choose_dimension
from rag.paper_graph.sources import load_sources
from rag.query_planning_v2 import paper_facets
from rag.retrieval import reciprocal_rank_fusion
from rag.storage import atomic_json, read_json

VERSION = 'community-original-candidates-v1'


@dataclass(frozen=True)
class CommunityPolicy:
    dimension: str = 'auto'
    paper_limit: int = 8
    community_limit: int = 2
    claim_limit: int = 24
    anchor_limit: int = 40
    rerank_limit: int = 40
    context_chars: int = 16000

    def __post_init__(self):
        if self.dimension not in {'auto','topic','method','experiment','conclusion'}:
            raise ValueError('Unknown dimension')
        if any(v < 1 for k,v in asdict(self).items() if k != 'dimension'):
            raise ValueError('Retrieval limits must be positive')


def token_pieces(text, tokenizer, size=384, overlap=0):
    """Exact text slices with offsets, never synthetic decoding or silent truncation."""
    if not 0 <= overlap < size:
        raise ValueError('Invalid token overlap')
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)['offset_mapping']
    cursor = 0
    while cursor < len(offsets):
        end = min(cursor+size,len(offsets))
        a,b = offsets[cursor][0],offsets[end-1][1]
        raw = text[a:b]
        left = len(raw)-len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            yield a+left,a+right,text[a+left:a+right],end-cursor
        if end == len(offsets):
            break
        cursor = end-overlap


def load_claim_vectors(corpus, models, root, progress=None):
    """Small offline cache, separate from Qdrant. Both experimental branches share it."""
    encoder = corpus.manifest['encoder']
    cfg = models.settings
    if cfg.embedding_model != encoder['embedding_model'] or cfg.embedding_revision != encoder['embedding_revision']:
        raise ValueError('Graph and query encoder versions differ')
    claims = sorted(corpus.claims.values(),key=lambda c:c['claim_id'])
    identity = fingerprint({'version':VERSION,'graph':corpus.build_id,'encoder':encoder,
        'max_length':cfg.embedding_max_length,'piece_tokens':cfg.chunk_size,
        'pooling':'normalized_token_weighted_mean_of_exact_pieces',
        'claims':[(c['claim_id'],c['text']) for c in claims]})
    directory = Path(root)/'runtime/community_retrieval/cache'/identity
    manifest = read_json(directory/'manifest.json')
    if manifest:
        if digest_file(directory/'vectors.npz') != manifest['vectors_sha256']:
            raise ValueError('Claim vector cache checksum changed')
        with np.load(directory/'vectors.npz',allow_pickle=False) as f:
            ids,vectors = f['claim_ids'].tolist(),f['vectors'].copy()
        if ids != [c['claim_id'] for c in claims] or vectors.shape != (len(ids),cfg.vector_size):
            raise ValueError('Claim vector cache does not match graph claims')
        return ids,vectors,{**manifest,'cache_hit':True}
    started=perf_counter();pieces,owners,weights=[],[],[]
    for index,claim in enumerate(claims):
        for _,_,text,weight in token_pieces(claim['text'],models.tokenizer,min(cfg.chunk_size,cfg.embedding_max_length-8)):
            pieces.append(text);owners.append(index);weights.append(weight)
    if progress:progress(stage='claim_embeddings',claims=len(claims),pieces=len(pieces))
    values=models.encode(pieces)
    vectors=np.zeros((len(claims),cfg.vector_size),dtype=np.float32)
    for value,owner,weight in zip(values,owners,weights,strict=True):vectors[owner]+=value*weight
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(norms==0) or not np.isfinite(vectors).all():raise ValueError('Invalid claim embeddings')
    vectors/=norms;directory.mkdir(parents=True,exist_ok=True)
    temp=directory/'vectors.tmp.npz'
    np.savez_compressed(temp,claim_ids=np.asarray([c['claim_id'] for c in claims]),vectors=vectors)
    temp.replace(directory/'vectors.npz')
    manifest={'identity':identity,'graph_build_id':corpus.build_id,'claims':len(claims),'pieces':len(pieces),
        'encoder':encoder,'build_ms':round((perf_counter()-started)*1000,2),
        'vectors_sha256':digest_file(directory/'vectors.npz'),'generation_api_calls':0}
    atomic_json(directory/'manifest.json',manifest)
    return [c['claim_id'] for c in claims],vectors,{**manifest,'cache_hit':False}


class OriginalSources:
    """Versioned canonical pages for any indexed paper, independent of gold labels."""
    def __init__(self, root, manifest, corpus):
        self.root,self.manifest,self.corpus=Path(root),manifest,corpus
        self.documents={p['paper_id']:p for p in paper_catalog(self.root)}
        self.pages={}

    def page(self,pid,page_number):
        if pid in self.corpus.papers:
            return self.corpus.canonical_pages(pid)[page_number]
        if pid not in self.pages:
            doc=self.documents[pid]
            if doc['sha256'] != self.manifest['documents'].get(pid):raise ValueError('Indexed PDF version changed')
            blocks,_=load_sources(doc,self.root)
            grouped=defaultdict(list)
            for block in blocks:grouped[block['pdf_page']].append(block)
            pages={}
            for number,records in grouped.items():
                records.sort(key=lambda b:b['source_start']);text='\n\n'.join(b['text'] for b in records)
                digest=hashlib.sha256(text.encode()).hexdigest()
                if any(b['canonical_page_sha256']!=digest or text[b['source_start']:b['source_end']]!=b['text'] for b in records):
                    raise ValueError('Canonical page differs from its anchors')
                pages[number]={'text':text,'sha256':digest,'blocks':records}
            self.pages[pid]=pages
        return self.pages[pid][page_number]

    def canonical(self,hit):
        pid=hit['paper_id']
        if hit['document_sha256']!=self.manifest['documents'].get(pid):raise ValueError('Candidate PDF version mismatch')
        page=self.page(pid,hit['pdf_page']);text=hit['text'];spans=hit.get('source_spans') or []
        if not spans:
            trimmed=text.strip();start=page['text'].find(trimmed)
            if start<0 or page['text'].find(trimmed,start+1)>=0:raise ValueError('Candidate cannot be uniquely joined to original coordinates')
            text=trimmed
            spans=[{'block_number':-1,'source_start':start,'source_end':start+len(text),'text_start':0,'text_end':len(text)}]
        cursor=0
        for span in spans:
            a,b,l,r=(span[k] for k in ('source_start','source_end','text_start','text_end'))
            if not(0<=a<b<=len(page['text']) and cursor<=l<r<=len(text)) or text[cursor:l].strip() or text[l:r]!=page['text'][a:b]:
                raise ValueError('Candidate text/source span mismatch')
            cursor=r
        if text[cursor:].strip():raise ValueError('Unmapped candidate text')
        key=fingerprint([pid,hit['document_sha256'],hit['pdf_page'],[(s['source_start'],s['source_end']) for s in spans]])
        return {**hit,'chunk_id':'source-'+key[:32],'original_chunk_ids':hit.get('original_chunk_ids',[hit['chunk_id']]),
            'text':text,'source_spans':spans,'canonical_page_sha256':page['sha256'],
            'text_sha256':hashlib.sha256(text.encode()).hexdigest()}


def subtract_interval(start,end,covered):
    cursor=start;result=[]
    for a,b in sorted(covered):
        if b<=cursor:continue
        if a>=end:break
        if a>cursor:result.append((cursor,min(a,end)))
        cursor=max(cursor,b)
        if cursor>=end:break
    if cursor<end:result.append((cursor,end))
    return result


def source_key(hit):return hit['paper_id'],hit['document_sha256'],hit['pdf_page']


def merge_candidates(fulltext,anchors,*,rrf_k=60):
    """RRF before truncation. Exact/contained source duplicates do not get extra slots.

    Partial overlaps remain candidates. They are subtracted when packing context;
    no union of overlapping windows creates a longer passage for the cross-encoder.
    """
    paths=defaultdict(list);originals=defaultdict(set)
    for hits in (fulltext,anchors):
        for h in hits:
            for p in h.get('retrieval_paths',[]):
                if p not in paths[h['chunk_id']]:paths[h['chunk_id']].append(p)
            originals[h['chunk_id']].update(h.get('original_chunk_ids',[]))
    rankings={'fulltext':fulltext}
    if anchors:rankings['anchors']=anchors
    fused=reciprocal_rank_fusion(rankings,rrf_k)
    retained=[];covered=defaultdict(list);dropped=[]
    for hit in fused:
        key=source_key(hit);ranges=[(s['source_start'],s['source_end']) for s in hit['source_spans']]
        hit={**hit,'retrieval_paths':paths[hit['chunk_id']], 'original_chunk_ids':sorted(originals[hit['chunk_id']])}
        if all(not subtract_interval(a,b,covered[key]) for a,b in ranges):
            dropped.append(hit['chunk_id'])
            # Provenance is attached only to passages that actually contain its text.
            for prior in retained:
                if source_key(prior)==key and all(not subtract_interval(a,b,[(s['source_start'],s['source_end']) for s in prior['source_spans']]) for a,b in ranges):
                    prior['retrieval_paths'] += [p for p in hit['retrieval_paths'] if p not in prior['retrieval_paths']]
                    break
            continue
        retained.append(hit);covered[key].extend(ranges)
    return retained,{'exact_duplicate_count':len(fulltext)+len(anchors)-len(fused),
                     'contained_duplicate_count':len(dropped),'dropped_source_ids':dropped}


def novel_text(hit,covered):
    pieces=[];spans=[];cursor=0
    for span in hit['source_spans']:
        for a,b in subtract_interval(span['source_start'],span['source_end'],covered):
            text=hit['text'][span['text_start']+a-span['source_start']:span['text_start']+b-span['source_start']]
            if not text.strip():continue
            if pieces:cursor+=2
            spans.append({'block_number':span['block_number'],'source_start':a,'source_end':b,'text_start':cursor,'text_end':cursor+len(text)})
            pieces.append(text);cursor+=len(text)
    if not pieces:return None
    text='\n\n'.join(pieces)
    return {**hit,'text':text,'source_spans':spans,'parent_candidate_id':hit['chunk_id'],
        'chunk_id':'context-'+fingerprint([hit['chunk_id'],spans])[:32],
        'text_sha256':hashlib.sha256(text.encode()).hexdigest()}


def pack_context(ranked,budget):
    selected=[];serialized=[];covered=defaultdict(list);skipped=[];overlap_removed=0
    for hit in ranked:
        novel=novel_text(hit,covered[source_key(hit)])
        if novel is None:continue
        number=len(selected)+1
        obj={'id':'S'+str(number),'paper_id':novel['paper_id'],'title':novel['title'],
             'pdf_page':novel['pdf_page'],'text':novel['text']}
        if len(json.dumps([*serialized,obj],ensure_ascii=False))>budget:
            skipped.append(hit['chunk_id']);continue
        overlap_removed+=sum(s['source_end']-s['source_start'] for s in hit['source_spans'])-sum(s['source_end']-s['source_start'] for s in novel['source_spans'])
        selected.append({**novel,'citation_id':obj['id']});serialized.append(obj)
        covered[source_key(hit)].extend((s['source_start'],s['source_end']) for s in novel['source_spans'])
    return selected,{'limit_chars':budget,'used_chars':len(json.dumps(serialized,ensure_ascii=False)),
        'chunks':len(selected),'overlap_characters_removed':overlap_removed,'skipped_candidate_ids':skipped}


class CommunityOriginalRetriever:
    def __init__(self,corpus,models,claim_ids,claim_vectors,policy):
        self.corpus,self.models,self.claim_ids,self.claim_vectors,self.policy=corpus,models,claim_ids,claim_vectors,policy
        self.claim_index={c:i for i,c in enumerate(claim_ids)}

    def retrieve(self,question,variant,route,router,manifest,query_vector):
        if variant not in {'B','C'}:raise ValueError('Unknown anchor branch')
        started=perf_counter();cfg=self.policy;dimension,why=choose_dimension(question,cfg.dimension)
        trace={'variant':variant,'dimension':dimension,'dimension_reason':why,'status':'selected',
               'graph_build_id':self.corpus.build_id,'selected_paper_ids':[],'selected_claim_ids':[],
               'selected_community_ids':[],'generation_api_calls':0,'claim_rerank_pairs':0}
        if route['mode']=='single_paper' or len(route.get('paper_ids',[]))==1:
            return [],{**trace,'status':'protected_single_paper','elapsed_ms':0}
        facets=paper_facets(question,router) if router else []
        explicit={f.paper_id for f in facets}
        allowed=set(manifest['documents']) & set(self.corpus.papers)
        if explicit:allowed &= explicit
        elif route.get('paper_ids'):allowed &= set(route['paper_ids'])
        trace['scope_paper_ids']=sorted(allowed);trace['scope_reason']='explicit_multiple_papers' if explicit else 'graph_papers_in_retrieval_scope'
        ids,vectors=self.corpus.vectors[dimension]
        paper_scores={p:float(v) for p,v in zip(ids,vectors@query_vector,strict=True) if p in allowed}
        communities=[]
        if variant=='C':
            index={p:i for i,p in enumerate(ids)}
            for group in self.corpus.communities[dimension]:
                members=sorted(set(group['members'])&paper_scores.keys())
                if not members:continue
                centroid=vectors[[index[p] for p in members]].mean(axis=0)
                norm=float(np.linalg.norm(centroid))
                if not norm:continue
                communities.append({'community_id':group['community_id'],'score':float(centroid@query_vector/norm),'members':members})
            communities.sort(key=lambda g:(-g['score'],g['community_id']))
            communities=communities[:cfg.community_limit]
            eligible={p for g in communities for p in g['members']}
            paper_scores={p:s for p,s in paper_scores.items() if p in eligible}
        papers=sorted(paper_scores,key=lambda p:(-paper_scores[p],p))[:cfg.paper_limit]
        eligible_claims=[c for c in self.corpus.claims.values() if c['paper_id'] in papers and c['dimension']==dimension]
        values=self.claim_vectors@query_vector
        eligible_claims.sort(key=lambda c:(-float(values[self.claim_index[c['claim_id']]]),c['claim_id']))
        claims=eligible_claims[:cfg.claim_limit]
        candidates={}
        for claim in claims:
            cid=claim['claim_id'];pid=claim['paper_id'];community=self.corpus.memberships[dimension].get(pid) if variant=='C' else None
            for sid in claim['source_ids']:
                source=self.corpus.source(sid)
                if source['paper_id']!=pid or source['document_sha256']!=manifest['documents'][pid]:raise ValueError('Claim source escaped its paper/version scope')
                for a,b,text,_ in token_pieces(source['text'],self.models.tokenizer,self.models.settings.chunk_size,self.models.settings.chunk_overlap):
                    start,end=source['source_start']+a,source['source_start']+b
                    identifier='anchor-'+fingerprint([pid,source['document_sha256'],source['pdf_page'],start,end])[:32]
                    path={'branch':'dimension_claim' if variant=='B' else 'community_claim','dimension':dimension,
                        'community_id':community if variant=='C' else None,'paper_id':pid,'claim_id':cid,'source_id':sid,
                        'claim_similarity':float(values[self.claim_index[cid]])}
                    if identifier in candidates:
                        candidates[identifier]['retrieval_paths'].append(path);continue
                    candidates[identifier]={'chunk_id':identifier,'paper_id':pid,'document_sha256':source['document_sha256'],
                        'pdf_page':source['pdf_page'],'text':text,'title':source['title'],'bboxes':[source['bbox']],
                        'score':path['claim_similarity'],'scores':{'claim_dense':path['claim_similarity']},
                        'retrieval_paths':[path],'source_spans':[{'block_number':-1,'source_start':start,'source_end':end,'text_start':0,'text_end':len(text)}]}
        hits=sorted(candidates.values(),key=lambda h:(-h['score'],h['chunk_id']))[:cfg.anchor_limit]
        trace.update(selected_paper_ids=papers,selected_claim_ids=[c['claim_id'] for c in claims],
            selected_community_ids=[g['community_id'] for g in communities],community_scores=communities,
            paper_scores=paper_scores,raw_anchor_count=len(candidates),retained_anchor_count=len(hits),
            elapsed_ms=round((perf_counter()-started)*1000,2))
        return hits,trace
