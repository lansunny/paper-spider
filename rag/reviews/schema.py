"""Research cards and cited map/reduce responses, with explicit support scope."""
from typing import Literal
import re

from pydantic import Field

from rag.paper_graph.schema import StrictModel


class SupportedStatement(StrictModel):
    text: str = Field(min_length=1, max_length=650)
    support: Literal['multi_paper', 'paper_specific']
    claim_ids: list[str] = Field(min_length=1, max_length=12)


class CardRoute(StrictModel):
    method_community_id: str
    name: str = Field(min_length=1, max_length=90)
    overview: SupportedStatement
    differences: list[SupportedStatement] = Field(max_length=2)
    experiments: list[SupportedStatement] = Field(max_length=2)
    limitations: list[SupportedStatement] = Field(max_length=2)
    representative_paper_ids: list[str] = Field(min_length=1, max_length=3)


class CommunityCard(StrictModel):
    title: str = Field(min_length=1, max_length=90)
    scope: list[SupportedStatement] = Field(min_length=1, max_length=3)
    routes: list[CardRoute] = Field(min_length=1, max_length=8)


class CardScope(StrictModel):
    title: str = Field(min_length=1, max_length=90)
    scope: list[SupportedStatement] = Field(min_length=1, max_length=3)


class CitedText(StrictModel):
    text: str = Field(min_length=1, max_length=900)
    citations: list[str] = Field(min_length=1, max_length=12)


class RouteAnalysis(StrictModel):
    title: str = Field(min_length=1, max_length=90)
    relevant: bool
    insufficient_evidence: bool
    overview: list[CitedText] = Field(max_length=2)
    approaches: list[CitedText] = Field(max_length=4)
    differences: list[CitedText] = Field(max_length=3)
    experiments: list[CitedText] = Field(max_length=3)
    limitations: list[CitedText] = Field(max_length=3)
    missing_evidence: list[str] = Field(max_length=6)


class ReviewSynthesis(StrictModel):
    title: str = Field(min_length=1, max_length=110)
    overview: list[CitedText] = Field(max_length=3)
    comparisons: list[CitedText] = Field(max_length=5)
    limitations: list[CitedText] = Field(max_length=4)
    covered_route_ids: list[str] = Field(max_length=12)
    insufficient_evidence: bool


class ReviewRequest(StrictModel):
    question: str = Field(min_length=2, max_length=1200)
    scope: Literal['topic', 'library'] = 'topic'
    focus_route_ids: list[str] = Field(default_factory=list, max_length=4)
    generate: bool = True


class ReviewScopeSelection(StrictModel):
    selected_paper_ids: list[str] = Field(max_length=50)
    reason: str = Field(min_length=1, max_length=650)


def cited_sections(value):
    for field in ('overview', 'approaches', 'differences', 'experiments', 'comparisons', 'limitations'):
        for statement in getattr(value, field, []):
            yield statement


def validate_citations(value, evidence_ids, evidence=None):
    allowed = set(evidence_ids)
    for statement in cited_sections(value):
        if len(set(statement.citations)) != len(statement.citations) or not set(statement.citations) <= allowed:
            raise ValueError('Citations must be unique IDs from the provided original evidence')
        if '[' in statement.text and any(part.startswith('S') for part in statement.text.split('[')[1:]):
            raise ValueError('Put citation IDs in citations, not inside text')
        if evidence is not None:
            by_id = {s['id']:s for s in evidence}
            normalize = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
            original = normalize(' '.join(by_id[s].get('text','')+' '+by_id[s].get('title','') for s in statement.citations))
            # A narrow guard for model/version names; this does not establish semantic entailment.
            technical = re.findall(r'\b[A-Za-z][A-Za-z0-9]*(?:[-.\u2010-\u2015][A-Za-z0-9]+)+\b',statement.text)
            unsupported = [term for term in technical if any(c.isdigit() for c in term) and normalize(term) not in original]
            if unsupported:
                raise ValueError('Model/version names must occur in the cited original text; remove or recite: '+', '.join(unsupported))
    if not value.insufficient_evidence and not list(cited_sections(value)):
        raise ValueError('A substantive answer requires original-source citations')
    return value
