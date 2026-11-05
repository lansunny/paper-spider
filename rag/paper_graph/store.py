"""Read-only published graph bundles and atomic SQLite publication."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from rag.formula_store import digest_file
from rag.storage import atomic_json, read_json
from .schema import DIMENSIONS


def write_database(path, papers, profiles, sources, views):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.executescript('''
            CREATE TABLE papers(paper_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                document_sha256 TEXT NOT NULL, year INTEGER, profile_sha256 TEXT NOT NULL);
            CREATE TABLE sources(source_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers,
                anchor_json TEXT NOT NULL);
            CREATE TABLE profiles(paper_id TEXT REFERENCES papers, dimension TEXT, status TEXT,
                description TEXT, claims_json TEXT, reason TEXT, PRIMARY KEY(paper_id,dimension));
            CREATE TABLE nodes(dimension TEXT, paper_id TEXT REFERENCES papers, community_id TEXT,
                degree INTEGER, PRIMARY KEY(dimension,paper_id));
            CREATE TABLE edges(dimension TEXT, source TEXT REFERENCES papers, target TEXT REFERENCES papers,
                weight REAL NOT NULL CHECK(weight>0 AND weight<=1), PRIMARY KEY(dimension,source,target),
                CHECK(source<>target));
            CREATE INDEX edges_target ON edges(dimension,target);
            CREATE TABLE communities(dimension TEXT, community_id TEXT, data_json TEXT,
                PRIMARY KEY(dimension,community_id));
        ''')
        for paper in papers:
            identifier = paper['paper_id']
            profile = profiles[identifier]
            connection.execute('INSERT INTO papers VALUES(?,?,?,?,?)', (identifier, paper['title'],
                paper['sha256'], paper.get('publication_year') or paper.get('year'), profile['profile_sha256']))
            for source in sources[identifier]:
                connection.execute('INSERT INTO sources VALUES(?,?,?)',
                    (source['source_id'], identifier, json.dumps(source, ensure_ascii=False)))
            for dimension in DIMENSIONS:
                value = profile['dimensions'][dimension]
                connection.execute('INSERT INTO profiles VALUES(?,?,?,?,?,?)', (identifier, dimension,
                    value['status'], value['description'], json.dumps(value['claims'], ensure_ascii=False), value['reason']))
        for dimension, view in views.items():
            for node in view['nodes']:
                connection.execute('INSERT INTO nodes VALUES(?,?,?,?)',
                    (dimension, node['paper_id'], node['community_id'], node['degree']))
            for edge in view['edges']:
                connection.execute('INSERT INTO edges VALUES(?,?,?,?)',
                    (dimension, edge['source'], edge['target'], edge['weight']))
            for group in view['communities']:
                connection.execute('INSERT INTO communities VALUES(?,?,?)',
                    (dimension, group['community_id'], json.dumps(group, ensure_ascii=False)))
        if connection.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Graph contains dangling paper references')


class PaperGraphStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def bundle(self):
        pointer = read_json(self.root / 'runtime/paper_graph/CURRENT.json')
        if not pointer:
            return None
        directory = (self.root / pointer['directory']).resolve()
        if not directory.is_relative_to((self.root / 'runtime/paper_graph/builds').resolve()):
            raise ValueError('Graph bundle outside the artifact directory')
        if digest_file(directory / 'manifest.json') != pointer['manifest_sha256']:
            raise ValueError('Graph manifest integrity check failed')
        manifest = read_json(directory / 'manifest.json')
        if manifest['status'] != 'completed':
            raise ValueError('Only completed graph bundles are readable')
        return directory, manifest

    def summary(self):
        bundle = self.bundle()
        return bundle[1] if bundle else {'status': 'not_built', 'papers': 0, 'dimensions': {}}

    def _connection(self, bundle=None):
        bundle = bundle or self.bundle()
        if not bundle:
            raise FileNotFoundError('Paper similarity graph has not been built')
        path = bundle[0] / 'graph.sqlite3'
        if digest_file(path) != bundle[1]['artifacts']['graph.sqlite3']:
            raise ValueError('Graph database integrity check failed')
        connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def view(self, dimension):
        if dimension not in DIMENSIONS:
            raise ValueError('Unknown graph dimension')
        with closing(self._connection()) as connection:
            nodes = [dict(r) for r in connection.execute('''SELECT n.*,p.title,p.year,p.profile_sha256,
                v.description FROM nodes n JOIN papers p USING(paper_id)
                JOIN profiles v ON v.paper_id=n.paper_id AND v.dimension=n.dimension
                WHERE n.dimension=? ORDER BY n.paper_id''', (dimension,))]
            edges = [dict(r) for r in connection.execute('''SELECT e.*,a.profile_sha256 AS source_profile_sha256,
                b.profile_sha256 AS target_profile_sha256 FROM edges e
                JOIN papers a ON a.paper_id=e.source JOIN papers b ON b.paper_id=e.target
                WHERE e.dimension=? ORDER BY e.source,e.target''', (dimension,))]
            communities = [json.loads(r[0]) for r in connection.execute(
                'SELECT data_json FROM communities WHERE dimension=? ORDER BY community_id', (dimension,))]
            missing = [r[0] for r in connection.execute(
                "SELECT paper_id FROM profiles WHERE dimension=? AND status<>'available' ORDER BY paper_id", (dimension,))]
        return {'dimension': dimension, 'nodes': nodes, 'edges': edges, 'communities': communities,
                'missing_paper_ids': missing, 'edge_meaning': 'semantic_similarity_only'}

    def paper(self, paper_id):
        with closing(self._connection()) as connection:
            row = connection.execute('SELECT * FROM papers WHERE paper_id=?', (paper_id,)).fetchone()
            if not row:
                raise KeyError(paper_id)
            result = dict(row)
            result['dimensions'] = {}
            for row in connection.execute('SELECT * FROM profiles WHERE paper_id=?', (paper_id,)):
                value = dict(row)
                value['claims'] = json.loads(value.pop('claims_json'))
                for claim in value['claims']:
                    claim['sources'] = [self._source(connection, identifier) for identifier in claim['source_ids']]
                result['dimensions'][value.pop('dimension')] = value
        self._verify_sources(paper_id, [source for value in result['dimensions'].values()
            for claim in value['claims'] for source in claim['sources']])
        return result

    def _verify_sources(self, paper_id, sources):
        from rag.ingestion import paper_catalog
        from .sources import load_sources
        document = next((p for p in paper_catalog(self.root) if p['paper_id'] == paper_id), None)
        if document is None:
            raise ValueError('Original paper is no longer in the library')
        current, _ = load_sources(document, self.root)
        blocks = {(s['pdf_page'], s['block_number']): s for s in current}
        for source in sources:
            block = blocks.get((source['pdf_page'], source['block_number']))
            if (not block or source['document_sha256'] != block['document_sha256']
                    or source['canonical_page_sha256'] != block['canonical_page_sha256']
                    or source['source_start'] < block['source_start'] or source['source_end'] > block['source_end']
                    or block['text'][source['source_start'] - block['source_start']:
                                     source['source_end'] - block['source_start']] != source['text']):
                raise ValueError('Graph anchor no longer matches the current canonical source')
            source['source_validation'] = 'matched_current_pdf_and_canonical_text'

    @staticmethod
    def _source(connection, identifier):
        row = connection.execute('SELECT anchor_json FROM sources WHERE source_id=?', (identifier,)).fetchone()
        if not row:
            raise KeyError(identifier)
        source = json.loads(row[0])
        source['study_url'] = '/study?paper=' + quote(source['paper_id'], safe='') + '&page=' + str(source['pdf_page'])
        source['pdf_url'] = '/api/papers/' + quote(source['paper_id'], safe='') + '/pdf#page=' + str(source['pdf_page'])
        return source

    def source(self, identifier):
        with closing(self._connection()) as connection:
            source = self._source(connection, identifier)
        self._verify_sources(source['paper_id'], [source])
        return source

    def neighbors(self, paper_id, dimension, limit=10):
        if dimension not in DIMENSIONS or not 1 <= limit <= 100:
            raise ValueError('Invalid dimension or neighbor limit')
        with closing(self._connection()) as connection:
            if not connection.execute('SELECT 1 FROM papers WHERE paper_id=?', (paper_id,)).fetchone():
                raise KeyError(paper_id)
            rows = connection.execute('''SELECT p.paper_id,p.title,e.weight,n.community_id FROM edges e
                JOIN papers p ON p.paper_id=CASE WHEN e.source=? THEN e.target ELSE e.source END
                JOIN nodes n ON n.paper_id=p.paper_id AND n.dimension=e.dimension
                WHERE e.dimension=? AND (e.source=? OR e.target=?)
                ORDER BY e.weight DESC,p.paper_id LIMIT ?''', (paper_id, dimension, paper_id, paper_id, limit))
            return [dict(r) for r in rows]


def publish(root, directory, manifest):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if not directory.is_relative_to((root / 'runtime/paper_graph/builds').resolve()):
        raise ValueError('Graph publication directory outside project graph builds')
    atomic_json(directory / 'manifest.json', manifest)
    atomic_json(root / 'runtime/paper_graph/CURRENT.json', {
        'directory': directory.relative_to(root).as_posix(),
        'manifest_sha256': digest_file(directory / 'manifest.json')})
