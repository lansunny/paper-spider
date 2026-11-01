/* Paper spider research desk. Native controls and source-first reading, no client dependencies. */
'use strict';
const $ = id => document.getElementById(id);
const icons = '/static/assets/icons.svg#';
const state = {
  papers: [], selected: new Set(), health: null, result: null, view: 'research',
  sidebarLimit: 24, catalogLimit: 40, busy: false, activeCitation: null,
  reader: {kind: 'pdf', paper: null, page: 1, total: 1, formula: null, source: null, ready: false},
};
const formatCount = value => Number(value || 0).toLocaleString('zh-CN');
const modeNames = {hybrid: '混合检索', dense: '语义检索', bm25: '关键词检索'};
const matchNames = {formula_inside_chunk: '正文内公式', explicit_formula_reference: '正文引用',
  adjacent_explanation: '相邻解释关联', exact_equation_number: '编号精确匹配',
  literal_symbol_proxy_fallback: '符号文字匹配'};
const sidebarMedia = matchMedia('(max-width: 1099px)');
const readerMedia = matchMedia('(max-width: 767px)');
let readerRequest = 0, readerController = null, readerBlob = null, indexTimer, toastTimer;
let history = readStorage('paperrag.research-history', []);
if (!Array.isArray(history)) history = [];
history = history.filter(x => x && typeof x.question === 'string' && x.question.length <= 2000).slice(0, 8);
let preferences = readStorage('paperrag.retrieval-preferences', {});
if (!preferences || typeof preferences !== 'object') preferences = {};
preferences = {mode: ['hybrid', 'dense', 'bm25'].includes(preferences.mode) ? preferences.mode : 'hybrid',
  topK: [3, 5, 10].includes(preferences.topK) ? preferences.topK : 5,
  rerank: preferences.rerank !== false, generate: preferences.generate !== false};

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined && text !== null) el.textContent = String(text);
  if (className) el.className = className;
  return el;
}
function icon(name, className = '') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'icon ' + className); svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', icons + name); svg.append(use); return svg;
}
function button(text, className, action) {
  const el = node('button', text, className); el.type = 'button';
  if (action) el.addEventListener('click', action);
  return el;
}
function readStorage(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
function writeStorage(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Private windows can deny storage. */ } }
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function toast(message) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false; toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2600); }
async function copyText(text, success = '已复制') {
  try { await navigator.clipboard.writeText(text); toast(success); }
  catch { toast('无法自动复制，请选中文字后复制。'); }
}
function localResource(url) {
  try { const parsed = new URL(url, location.origin); return parsed.origin === location.origin && parsed.pathname.startsWith('/api/') ? parsed.href : null; }
  catch { return null; }
}
async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch (error) { if (error.name === 'AbortError') throw error; throw new Error('暂时无法连接本地服务，请检查服务后重试。'); }
  let body;
  try { body = await response.json(); }
  catch { throw new Error('服务返回了无法读取的内容，请稍后重试。'); }
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '请求未通过校验，请检查问题与论文范围。');
  return body;
}
function paperById(id) { return state.papers.find(p => p.paper_id === id); }
function paperPdf(paper) { return paper.pdf_url || '/api/papers/' + encodeURIComponent(paper.paper_id) + '/pdf'; }
function filteredPapers(term) {
  term = term.trim().toLocaleLowerCase();
  return state.papers.filter(p => (p.title + ' ' + (p.year || '') + ' ' + p.paper_id).toLocaleLowerCase().includes(term));
}
function selectionChanged(id, checked) {
  checked ? state.selected.add(id) : state.selected.delete(id);
  renderSidebar(); renderCatalog(); updateScope();
}
function updateScope() {
  const n = state.selected.size;
  $('scope').textContent = n ? '已选 ' + n + ' 篇 · 限定检索范围' : '全部已索引论文';
  $('clear-selection').hidden = !n;
  $('question-scope').textContent = n ? '在选定的 ' + n + ' 篇论文中检索' : '在全部已索引论文中检索';
}
function renderSidebar() {
  const focusedId = document.activeElement?.dataset?.paperSelect;
  const list = $('papers'); list.replaceChildren();
  const filtered = filteredPapers($('filter').value);
  for (const paper of filtered.slice(0, state.sidebarLimit)) {
    const row = node('article', null, 'paper-row');
    row.dataset.paperId = paper.paper_id;
    row.classList.toggle('is-selected', state.selected.has(paper.paper_id));
    row.classList.toggle('is-reading', state.reader.paper?.paper_id === paper.paper_id);
    const label = node('label', null, 'paper-select');
    const input = node('input'); input.type = 'checkbox'; input.checked = state.selected.has(paper.paper_id);
    input.disabled = !paper.indexed; input.dataset.paperSelect = paper.paper_id;
    input.setAttribute('aria-label', '限定检索范围：' + paper.title);
    if (!paper.indexed) input.title = '建立索引后可选择检索';
    input.addEventListener('change', () => selectionChanged(paper.paper_id, input.checked));
    const title = node('span', paper.title, 'paper-title'); title.title = paper.title;
    label.append(input, title);
    const meta = node('div', null, 'paper-row-meta');
    meta.append(node('span', (paper.year || '上传文献') + ' · ' + (paper.indexed ? '已入库' : '待入库')),
      button('阅读', '', () => { closeLibrary(); openPaper(paper); }));
    row.append(label, meta); list.append(row);
  }
  if (!filtered.length) list.append(node('p', state.papers.length ? '未找到匹配的论文。试试其他标题或年份。' : '论文库暂时为空，可先导入 PDF。', 'list-state muted'));
  $('load-more-papers').hidden = filtered.length <= state.sidebarLimit;
  $('load-more-papers').textContent = '显示更多 · ' + Math.min(state.sidebarLimit, filtered.length) + ' / ' + formatCount(filtered.length);
  $('nav-paper-count').textContent = formatCount(state.papers.length);
  if (focusedId) Array.from(list.querySelectorAll('input')).find(x => x.dataset.paperSelect === focusedId)?.focus({preventScroll: true});
  updateScope();
}
function renderCatalog() {
  let papers = filteredPapers($('library-search').value);
  if ($('library-sort').value === 'year') papers.sort((a, b) => Number(b.year || 0) - Number(a.year || 0) || a.title.localeCompare(b.title));
  if ($('library-sort').value === 'title') papers.sort((a, b) => a.title.localeCompare(b.title));
  $('library-summary').textContent = formatCount(papers.length) + ' 篇文献 · ' + formatCount(papers.filter(p => p.indexed).length) + ' 篇可检索';
  const list = $('catalog-list'); list.replaceChildren();
  for (const paper of papers.slice(0, state.catalogLimit)) {
    const row = node('article', null, 'catalog-row');
    const identity = node('div');
    identity.append(button(paper.title, 'catalog-title', () => openPaper(paper)), node('p', paper.paper_id, 'catalog-id'));
    const meta = node('div', paper.year || '—', 'catalog-meta'); meta.append(node('small', paper.indexed ? '已入库' : '待入库'));
    const choose = node('input', null, 'catalog-select'); choose.type = 'checkbox'; choose.checked = state.selected.has(paper.paper_id);
    choose.disabled = !paper.indexed; choose.setAttribute('aria-label', '选择论文：' + paper.title);
    choose.addEventListener('change', () => selectionChanged(paper.paper_id, choose.checked));
    row.append(identity, meta, choose); list.append(row);
  }
  if (!papers.length) list.append(node('p', state.papers.length ? '没有匹配的文献。' : '还没有文献，导入一篇 PDF 开始研究。', 'empty-result'));
  $('load-more-catalog').hidden = papers.length <= state.catalogLimit;
  $('load-more-catalog').textContent = '加载更多 · 已显示 ' + Math.min(papers.length, state.catalogLimit) + ' / ' + formatCount(papers.length);
}
function renderFeatured() {
  const candidates = [
    ['2024.eacl-demo.16', 'RAG 评估'],
    ['2020.emnlp-main.550', '检索训练'],
    ['2023.emnlp-main.495', '主动检索'],
  ];
  const entries = candidates.map(([id, label]) => ({paper: paperById(id), label})).filter(x => x.paper);
  for (const paper of state.papers) {
    if (entries.length >= 3) break;
    if (!entries.some(e => e.paper.paper_id === paper.paper_id)) entries.push({paper, label: '论文原文'});
  }
  const target = $('featured-papers'); target.replaceChildren();
  for (const {paper, label} of entries) {
    const article = node('article', null, 'featured-paper');
    const cover = button(null, 'featured-cover', () => openPaper(paper));
    cover.setAttribute('aria-label', '阅读 ' + paper.title);
    const preview = node('img'); preview.loading = 'lazy'; preview.decoding = 'async';
    preview.alt = paper.title + '，原文首页';
    preview.src = paperPdf(paper).replace(/\/pdf$/, '/pages/1');
    preview.addEventListener('error', () => { cover.hidden = true; }, {once: true});
    cover.append(preview);
    article.append(cover, node('span', label, 'featured-paper-label'), button(paper.title, '', () => openPaper(paper)),
      node('span', paper.year || '上传', 'featured-paper-meta'));
    target.append(article);
  }
  if (!entries.length) target.append(node('p', '导入论文后，可在这里打开原文。', 'muted'));
}
function renderHistory() {
  $('recent-section').hidden = !history.length;
  const target = $('recent-questions'); target.replaceChildren();
  for (const entry of history.slice(0, 3)) {
    const b = button(null, 'recent-question', () => { closeLibrary(); fillQuestion(entry.question); });
    b.title = entry.question; b.append(icon('history'), node('span', entry.question)); target.append(b);
  }
}
function rememberQuestion(question) {
  history = [{question}, ...history.filter(x => x.question !== question)].slice(0, 8);
  writeStorage('paperrag.research-history', history); renderHistory();
}
function setView(view) {
  state.view = view; $('research-view').hidden = view !== 'research'; $('library-view').hidden = view !== 'library';
  for (const id of ['research', 'library']) {
    const b = $('nav-' + id); b.classList.toggle('is-active', view === id);
    if (view === id) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  }
  $('workspace-location').textContent = view === 'research' ? '论文研究' : '全部文献';
  $('main-content').scrollTop = 0; closeLibrary();
  if (view === 'library') renderCatalog();
}
function fillQuestion(question) {
  setView('research'); $('question').value = question; updateQuestionLength(); $('question').focus();
  $('main-content').scrollTo({top: 0, behavior: 'auto'});
}
function updateQuestionLength() {
  const n = $('question').value.length; $('question-length').hidden = !n; $('question-length').textContent = n + ' / 2000';
}
function applyPreferenceControls() {
  $('retrieval-mode').value = preferences.mode; $('top-k').value = String(preferences.topK);
  $('rerank').checked = preferences.rerank;
  $('generate').checked = preferences.generate && !!state.health?.generation_configured;
  $('generate').disabled = !state.health?.generation_configured;
  $('generate-hint').textContent = state.health?.generation_configured ? '根据检索证据组织回答' : '回答模型未就绪，可先检索与阅读证据';
  $('mode-label').textContent = modeNames[preferences.mode];
  if (!state.busy) $('submit').querySelector('span').textContent = $('generate').checked ? '检索并回答' : '检索文献';
}
async function refresh() {
  $('refresh').disabled = true;
  const results = await Promise.allSettled([api('/api/health'), api('/api/papers')]);
  const [healthResult, paperResult] = results;
  if (healthResult.status === 'fulfilled') {
    state.health = healthResult.value;
    $('connection').className = 'connection is-connected';
    $('connection-text').textContent = formatCount(state.health.indexed_papers) + ' 篇已入库';
    $('pipeline-status').textContent = formatCount(state.health.indexed_chunks) + ' 个已入库片段。' +
      (state.health.generation_configured ? '回答模型已就绪。' : '当前可检索与阅读证据。');
  } else {
    $('connection').className = 'connection is-offline'; $('connection-text').textContent = '服务未连接';
    $('pipeline-status').textContent = '暂时无法读取索引状态，请检查本地服务。';
    showError(healthResult.reason.message);
  }
  if (paperResult.status === 'fulfilled') {
    state.papers = Array.isArray(paperResult.value.papers) ? paperResult.value.papers : [];
    for (const id of state.selected) if (!state.papers.some(p => p.paper_id === id && p.indexed)) state.selected.delete(id);
    renderSidebar(); renderCatalog(); renderFeatured();
    if (!state.reader.paper && state.papers.length) {
      const first = paperById('2024.eacl-demo.16') || state.papers[0]; openPaper(first, {reveal: false});
    }
  } else if (!state.papers.length) {
    $('papers').replaceChildren(node('p', '文献加载失败，可点击上方刷新重试。', 'muted list-state'));
    $('featured-papers').replaceChildren(node('p', '文献暂不可用。', 'muted'));
  }
  applyPreferenceControls(); $('refresh').disabled = false;
}
function evidenceId(hit, index) { return hit.citation_id || 'S' + (index + 1); }
function uniqueSources(evidence) {
  const map = new Map();
  evidence.forEach((h, i) => { if (!map.has(h.paper_id)) map.set(h.paper_id, {hit: h, citation: evidenceId(h, i)}); });
  return Array.from(map.values());
}
function appendCitedText(target, text) {
  // Build text nodes, never insert model output as HTML.
  const regex = /\[(S\d+)\]/g; let cursor = 0;
  for (const match of text.matchAll(regex)) {
    target.append(document.createTextNode(text.slice(cursor, match.index)));
    const hit = state.result.evidence.find((h, i) => evidenceId(h, i) === match[1]);
    if (hit) {
      const b = button('[' + match[1] + ']', 'citation-link', () => selectEvidence(hit, match[1]));
      b.setAttribute('aria-label', match[1] + '，查看 ' + hit.title + ' 第 ' + hit.pdf_page + ' 页');
      target.append(b);
    } else target.append(document.createTextNode(match[0]));
    cursor = match.index + match[0].length;
  }
  target.append(document.createTextNode(text.slice(cursor)));
}
function sourceSummary(evidence) {
  const section = node('section', null, 'source-summary');
  const head = node('div', null, 'section-heading');
  head.append(node('h3', '参考文献'), button('查看全部证据', 'text-button', () => setResultTab('evidence')));
  const list = node('div', null, 'source-summary-list');
  for (const {hit, citation} of uniqueSources(evidence)) {
    const row = button(null, 'source-summary-row', () => selectEvidence(hit, citation));
    row.append(node('span', citation, 'source-id'), node('span', hit.title), node('span', 'PDF ' + hit.pdf_page, 'source-page'));
    list.append(row);
  }
  section.append(head, list); return section;
}
function renderAnswer(data) {
  const target = $('panel-answer'); target.replaceChildren();
  if (data.answer) {
    const body = node('div', null, 'answer-body');
    for (const paragraph of data.answer.split(/\n\s*\n/).filter(Boolean)) {
      const p = node('p'); p.style.whiteSpace = 'pre-wrap'; appendCitedText(p, paragraph); body.append(p);
    }
    target.append(body);
    const note = node('div', null, 'answer-source-line');
    note.append(node('span', '依据 ' + uniqueSources(data.evidence).length + ' 篇论文 · ' + data.evidence.length + ' 条原文证据'));
    if (data.insufficient_evidence) note.append(node('span', '现有证据不足以完整回答'));
    target.append(note);
  } else {
    const copy = !data.evidence.length ? '未找到足够的论文证据。可以尝试改写问题，或清除论文范围后重试。'
      : data.generation_status === 'key_required' ? '已找到相关证据。回答模型尚未就绪，可先阅读证据并核对原文。'
      : data.generation_status === 'error' ? '证据检索已完成，回答生成暂未成功。可先查看证据，再重试。'
      : '已返回原文证据。本次未生成回答，可以直接核对来源。';
    const notice = node('div', null, 'answer-notice'); notice.append(node('p', copy));
    if (data.evidence.length) notice.append(button('阅读检索证据', 'secondary-button', () => setResultTab('evidence')));
    target.append(notice);
  }
  if (data.evidence.length) target.append(sourceSummary(data.evidence));
}
function renderEvidence(data) {
  const target = $('panel-evidence'); target.replaceChildren();
  if (!data.evidence.length) { target.append(node('p', '没有可供展示的证据。试试更具体的问题或更宽的文献范围。', 'empty-result')); return; }
  data.evidence.forEach((hit, index) => {
    const id = evidenceId(hit, index);
    const article = node('article', null, 'evidence-card'); article.dataset.citation = id;
    article.classList.toggle('is-selected', state.activeCitation === id);
    const head = node('div', null, 'evidence-card-heading');
    const cite = button('[' + id + ']', 'source-id', () => selectEvidence(hit, id));
    const scoreLabel = data.rerank ? '重排' : modeNames[data.retrieval_mode] || '相关性';
    const score = node('span', scoreLabel + ' ' + (Number.isFinite(Number(hit.score)) ? Number(hit.score).toFixed(4) : '—'), 'evidence-score');
    score.title = '用于候选排序，不是置信概率。';
    head.append(cite, score);
    const title = button(hit.title, 'evidence-title', () => selectEvidence(hit, id));
    const meta = node('p', 'PDF 第 ' + hit.pdf_page + ' 页 · ' + hit.paper_id, 'evidence-meta');
    const snippet = node('p', hit.text.length > 420 ? hit.text.slice(0, 420) + '…' : hit.text, 'evidence-snippet');
    article.append(head, title, meta, snippet);
    if (hit.text.length > 420 || hit.text.split(/\r?\n/).length > 6) {
      const detail = node('details'); detail.append(node('summary', '展开完整证据'), node('pre', hit.text)); article.append(detail);
    }
    const bottom = node('div', null, 'evidence-bottom');
    bottom.append(button('在阅读台核对原文 ↗', 'text-button', () => selectEvidence(hit, id)),
      node('span', hit.formula_ids?.length ? '关联 ' + hit.formula_ids.length + ' 个公式' : '原文证据'));
    article.append(bottom); target.append(article);
  });
}
function renderFormulas(data) {
  const target = $('panel-formulas'); target.replaceChildren();
  if (!data.formulas.length) {
    target.append(node('p', '本次没有联接到公式原图。可在问题中明确论文、公式编号或相关方法。', 'empty-result')); return;
  }
  target.append(node('p', '图片保留论文中的原始表达。文字代理仅辅助检索，阅读时以原图为准。', 'formula-intro'));
  for (const f of data.formulas) {
    const article = node('article', null, 'formula-card');
    const head = node('div', null, 'formula-card-head');
    head.append(node('h3', f.equation_number ? '公式 (' + f.equation_number + ')' : '未编号公式'), node('span', 'PDF 第 ' + f.pdf_page + ' 页'));
    const open = button(null, 'formula-preview', () => openFormula(f));
    open.setAttribute('aria-label', '查看 ' + f.title + ' 公式原图');
    const img = node('img'); const url = localResource(f.image_url);
    if (url) img.src = url;
    img.loading = 'lazy'; img.alt = f.title + ' 第 ' + f.pdf_page + ' 页公式原图';
    img.addEventListener('error', () => { const msg = node('span', '原图暂时无法读取，请打开论文核对。', 'empty-result'); open.replaceChildren(msg); }, {once: true});
    open.append(img);
    const methods = [...new Set((f.matches || []).map(m => matchNames[m.method] || m.method))].join(' · ');
    article.append(head, open, node('p', f.title, 'formula-source-title'), node('p', methods, 'formula-match'));
    const proxy = node('details', null, 'formula-proxy'); proxy.append(node('summary', '文字代理 · 未验证，仅辅助检索'), node('pre', f.text_proxy || '无可用文字代理'));
    article.append(proxy, button('核对公式来源 ↗', 'text-button', () => openFormula(f))); target.append(article);
  }
}
function renderRunDetails(data) {
  const target = $('run-details-content'); target.replaceChildren();
  const stages = node('div', null, 'run-stage-list');
  for (const stage of data.stages || []) {
    stages.append(node('span', stage.name + ' · ' + ({done: '完成', pending: '未执行', skipped: '已关闭'}[stage.status] || stage.status), stage.status === 'done' ? 'is-done' : ''));
  }
  target.append(stages);
  const dl = node('dl');
  const timings = Object.entries(data.timings || {}).map(([k, v]) => k + ': ' + v + ' ms').join(' / ');
  const entries = [
    ['检索范围', data.route?.mode === 'single_paper' ? '自动限定单篇论文' : data.route?.mode === 'manual' ? '手动选择论文范围' : '全部已索引论文'],
    ['检索方式', modeNames[data.retrieval_mode] || '—'],
    ['正文重排', data.rerank ? '已启用' : '未启用'],
    ['实际上下文', data.evidence.length + ' 条 · ' + formatCount(data.context_chars) + ' 字符'],
    ['论文子问题', data.query_plan?.facets?.length || 0],
    ['记录编号', data.trace_id || '—'], ['阶段耗时', timings || '—'],
  ];
  for (const [key, value] of entries) { const row = node('div'); row.append(node('dt', key), node('dd', value)); dl.append(row); }
  target.append(dl);
}
function renderResult(data) {
  state.result = {...data, evidence: Array.isArray(data.evidence) ? data.evidence : [], formulas: Array.isArray(data.formulas) ? data.formulas : []};
  const result = state.result; state.activeCitation = null;
  $('starter').hidden = true; $('result').hidden = false; $('result-title').textContent = '研究结果';
  $('evidence-count').textContent = result.evidence.length; $('formula-count').textContent = result.formulas.length;
  const seconds = Number(result.elapsed_ms);
  $('result-meta').textContent = uniqueSources(result.evidence).length + ' 篇来源' +
    (Number.isFinite(seconds) ? ' · ' + (seconds / 1000).toFixed(1) + ' 秒' : '');
  $('copy-answer').hidden = !result.answer;
  renderAnswer(result); renderEvidence(result); renderFormulas(result); renderRunDetails(result);
  setResultTab(result.answer ? 'answer' : 'evidence');
  if (result.evidence.length) selectEvidence(result.evidence[0], evidenceId(result.evidence[0], 0), {reveal: false});
}
function setResultTab(name, focus = false) {
  for (const value of ['answer', 'evidence', 'formulas']) {
    const selected = value === name, b = $('tab-' + value);
    b.setAttribute('aria-selected', String(selected)); b.tabIndex = selected ? 0 : -1; $('panel-' + value).hidden = !selected;
    if (selected && focus) b.focus();
  }
}
function selectEvidence(hit, id, options = {}) {
  state.activeCitation = id;
  document.querySelectorAll('.evidence-card').forEach(card => card.classList.toggle('is-selected', card.dataset.citation === id));
  const paper = paperById(hit.paper_id) || {paper_id: hit.paper_id, title: hit.title, pdf_url: '/api/papers/' + encodeURIComponent(hit.paper_id) + '/pdf'};
  openPaper(paper, {page: hit.pdf_page || 1, source: hit, citation: id, reveal: options.reveal !== false});
}
const readerHome = document.createComment('source-reader-home');
$('reader-pane').before(readerHome);
const mobileReaderDialog = node('dialog', null, 'reader-dialog');
mobileReaderDialog.setAttribute('aria-label', '原文阅读台'); document.body.append(mobileReaderDialog);
function showReader(reveal) {
  if (readerMedia.matches && reveal) {
    mobileReaderDialog.append($('reader-pane')); $('reader-pane').classList.add('is-open');
    if (!mobileReaderDialog.open) mobileReaderDialog.showModal();
  }
}
function restoreReader() {
  $('reader-pane').classList.remove('is-open'); readerHome.after($('reader-pane'));
}
mobileReaderDialog.addEventListener('close', restoreReader);
function openPaper(paper, {page = 1, source = null, citation = '', reveal = true} = {}) {
  const sameDocument = state.reader.kind === 'pdf' && state.reader.paper?.paper_id === paper.paper_id;
  state.reader = {kind: 'pdf', paper, page: Math.max(1, Number(page) || 1), total: sameDocument ? state.reader.total : 1,
    formula: null, source, citation, ready: false};
  updateReaderText(); showReader(reveal); loadReaderImage();
  document.querySelectorAll('.paper-row').forEach(row => row.classList.toggle('is-reading', row.dataset.paperId === paper.paper_id));
}
function openFormula(formula) {
  const paper = paperById(formula.paper_id) || {paper_id: formula.paper_id, title: formula.title};
  const ids = new Set((formula.matches || []).map(m => m.source_chunk_id));
  const source = state.result?.evidence.find(h => ids.has(h.chunk_id));
  state.reader = {kind: 'formula', paper, page: formula.pdf_page, total: 1, formula, source, citation: '', ready: false};
  updateReaderText(); showReader(true); loadReaderImage();
}
function updateReaderText() {
  const r = state.reader; if (!r.paper) return;
  $('reader-image-area').classList.toggle('is-formula', r.kind === 'formula');
  $('reader-kind').textContent = r.kind === 'formula' ? '原始公式 · ' + (r.formula.equation_number ? '式 (' + r.formula.equation_number + ')' : '未编号') : r.citation ? '[' + r.citation + '] 引用来源' : '原始文献';
  $('reader-title').textContent = r.paper.title;
  $('reader-subtitle').textContent = (r.paper.year ? r.paper.year + ' · ' : '') + 'PDF 第 ' + r.page + ' 页';
  $('reader-source').hidden = !r.source;
  $('reader-source-label').textContent = r.citation ? '[' + r.citation + '] 引用原文' : '关联正文';
  $('reader-source-text').textContent = r.source?.text || '';
  $('reader-formula-note').hidden = r.kind !== 'formula';
  const external = localResource(paperPdf(r.paper));
  $('reader-external').hidden = !external; $('pdf-external').hidden = !external;
  if (external) { $('reader-external').href = external + '#page=' + r.page; $('pdf-external').href = external + '#page=' + r.page; }
  $('pdf-title').textContent = r.paper.title;
  updateReaderNavigation();
}
function updateReaderNavigation() {
  const r = state.reader;
  const formulaIndex = r.kind === 'formula' ? state.result?.formulas.findIndex(f => f.element_id === r.formula.element_id && f.paper_id === r.formula.paper_id) ?? -1 : -1;
  const totalFormulas = state.result?.formulas.length || 0;
  $('reader-pagination').hidden = !r.ready;
  $('reader-page-status').textContent = r.kind === 'formula' ? '原公式图 · ' + (formulaIndex + 1) + ' / ' + totalFormulas : '第 ' + r.page + ' / ' + r.total + ' 页';
  $('reader-prev').disabled = !r.ready || (r.kind === 'formula' ? formulaIndex <= 0 : r.page <= 1);
  $('reader-next').disabled = !r.ready || (r.kind === 'formula' ? formulaIndex >= totalFormulas - 1 : r.page >= r.total);
  $('reader-prev').setAttribute('aria-label', r.kind === 'formula' ? '上一张公式' : '上一页');
  $('reader-next').setAttribute('aria-label', r.kind === 'formula' ? '下一张公式' : '下一页');
  $('expand-reader').disabled = !r.ready;
  $('pdf-prev').hidden = $('pdf-next').hidden = r.kind === 'formula';
  $('pdf-prev').disabled = !r.ready || r.page <= 1; $('pdf-next').disabled = !r.ready || r.page >= r.total;
  $('pdf-page-status').textContent = r.kind === 'formula' ? '原始公式 · PDF ' + r.page : '第 ' + r.page + ' / ' + r.total + ' 页';
}
async function loadReaderImage() {
  if (!state.reader.paper) return;
  const request = ++readerRequest;
  if (readerController) readerController.abort();
  readerController = new AbortController();
  const r = state.reader; r.ready = false; updateReaderNavigation();
  $('reader-image').hidden = true; $('reader-error').hidden = true; $('reader-placeholder').hidden = false;
  $('reader-placeholder').querySelector('p').textContent = r.kind === 'formula' ? '正在读取原始公式…' : '正在读取论文原文…';
  $('reader-image-area').classList.add('is-loading');
  const path = r.kind === 'formula' ? localResource(r.formula.image_url) : localResource(paperPdf(r.paper).replace(/\/pdf$/, '/pages/' + r.page));
  try {
    if (!path) throw new Error('无法定位此文献的原始文件。');
    const response = await fetch(path, {signal: readerController.signal});
    if (!response.ok) throw new Error('原文暂时无法读取，可打开完整 PDF 核对。');
    const blob = await response.blob();
    if (request !== readerRequest) return;
    if (r.kind === 'pdf') r.total = Math.max(r.page, Number(response.headers.get('X-Page-Count')) || r.page);
    const nextBlob = URL.createObjectURL(blob);
    if (readerBlob) URL.revokeObjectURL(readerBlob); readerBlob = nextBlob;
    $('reader-image').src = readerBlob; $('pdf-image').src = readerBlob;
    $('reader-image').alt = r.paper.title + ' · PDF 第 ' + r.page + ' 页' + (r.kind === 'formula' ? '公式原图' : '原文');
    await $('reader-image').decode();
    if (request !== readerRequest) return;
    $('reader-image').hidden = false; $('reader-placeholder').hidden = true; r.ready = true;
  } catch (error) {
    if (request !== readerRequest || error.name === 'AbortError') return;
    $('reader-placeholder').hidden = true; $('reader-error').hidden = false;
    $('reader-error').textContent = error.message || '原图加载失败，请重试。';
  } finally {
    if (request === readerRequest) { $('reader-image-area').classList.remove('is-loading'); updateReaderNavigation(); }
  }
}
function moveReader(direction) {
  const r = state.reader; if (!r.ready) return;
  if (r.kind === 'formula') {
    const list = state.result?.formulas || [], i = list.findIndex(f => f.element_id === r.formula.element_id && f.paper_id === r.formula.paper_id);
    if (list[i + direction]) openFormula(list[i + direction]);
  } else if (r.page + direction >= 1 && r.page + direction <= r.total) {
    r.page += direction; r.source = null; r.citation = ''; updateReaderText(); loadReaderImage();
  }
}
function openFullReader() {
  if (!state.reader.ready) return;
  $('pdf-image').src = readerBlob; updateReaderText(); $('pdf-dialog').showModal();
}
function openLibrary() { if (sidebarMedia.matches && !$('library-dialog').open) $('library-dialog').showModal(); }
function closeLibrary() { if ($('library-dialog').open) $('library-dialog').close(); }
function syncResponsiveShell() {
  if (sidebarMedia.matches) $('mobile-library-slot').append($('library-panel'));
  else { closeLibrary(); $('sidebar-slot').append($('library-panel')); }
  if (!readerMedia.matches && mobileReaderDialog.open) mobileReaderDialog.close();
}
async function submitQuestion(event) {
  event.preventDefault(); if (state.busy) return;
  const question = $('question').value.trim();
  if (!question) { showError('请输入研究问题。'); $('question').focus(); return; }
  const request = {question, paper_ids: Array.from(state.selected), top_k: Number($('top-k').value),
    mode: $('retrieval-mode').value, rerank: $('rerank').checked, generate_answer: $('generate').checked && !$('generate').disabled};
  showError(''); state.busy = true;
  $('submit').disabled = true; $('new-research').disabled = true; $('question').readOnly = true;
  $('submit').querySelector('span').textContent = '检索中'; $('loading-state').hidden = false; $('starter').hidden = true;
  $('result').setAttribute('aria-busy', 'true');
  if (state.result) $('result-title').textContent = '上一轮结果 · 正在更新';
  const began = performance.now();
  const timer = setInterval(() => { $('loading-elapsed').textContent = '已等待 ' + Math.floor((performance.now() - began) / 1000) + ' 秒 · 首次检索可能需要加载模型'; }, 1000);
  try {
    const result = await api('/api/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(request)});
    renderResult(result); rememberQuestion(question);
  } catch (error) {
    showError(error.message);
    if (!state.result) $('starter').hidden = false;
    else $('result-title').textContent = '上一轮研究结果';
  } finally {
    clearInterval(timer); state.busy = false;
    $('submit').disabled = false; $('new-research').disabled = false; $('question').readOnly = false;
    $('submit').querySelector('span').textContent = $('generate').checked ? '检索并回答' : '检索文献'; $('loading-state').hidden = true;
    $('result').setAttribute('aria-busy', 'false');
  }
}
function newResearch() {
  if (state.busy) return;
  state.result = null; state.activeCitation = null; setView('research');
  $('question').value = ''; updateQuestionLength(); showError('');
  $('result').hidden = true; $('starter').hidden = false; $('question').focus();
}
async function uploadFile() {
  const file = $('upload').files[0]; if (!file) return;
  const message = $('upload-message'); message.hidden = false;
  if (file.size > 30 * 1024 * 1024) { message.textContent = '文件超过 30 MB，请选择较小的 PDF。'; $('upload').value = ''; return; }
  if (!file.name.toLowerCase().endsWith('.pdf')) { message.textContent = '请选择 PDF 文件。'; $('upload').value = ''; return; }
  $('upload-trigger').disabled = true; $('library-import').disabled = true; $('upload').disabled = true;
  message.textContent = '正在导入并检查 PDF…';
  try {
    const form = new FormData(); form.append('file', file);
    const result = await api('/api/papers/upload', {method: 'POST', body: form});
    message.textContent = result.duplicate ? '这篇文献已在论文库中。' : '已导入。更新文献索引后即可检索。';
    toast(result.duplicate ? '文献已存在' : 'PDF 已导入'); await refresh();
  } catch (error) { message.textContent = error.message; }
  finally { $('upload-trigger').disabled = false; $('library-import').disabled = false; $('upload').disabled = false; $('upload').value = ''; }
}
let lastIndexStatus = null;
async function pollIndex() {
  clearTimeout(indexTimer);
  try {
    const job = await api('/api/index');
    const busy = ['queued', 'running'].includes(job.status);
    $('build-index').disabled = busy;
    $('index-progress').hidden = job.status === 'idle';
    const names = {queued: '排队中', running: '正在入库', completed: '索引已就绪', failed: '入库失败', interrupted: '上次入库已中断'};
    $('index-progress').textContent = (names[job.status] || job.status) +
      (job.total ? ' · ' + formatCount(job.completed) + ' / ' + formatCount(job.total) : '') +
      (job.message ? ' · ' + job.message : '');
    if (busy) indexTimer = setTimeout(pollIndex, 2000);
    else if (job.status === 'completed' && ['running', 'queued'].includes(lastIndexStatus)) await refresh();
    lastIndexStatus = job.status;
  } catch { $('build-index').disabled = false; $('index-progress').hidden = false; $('index-progress').textContent = '暂时无法读取入库进度，请稍后重试。'; }
}
$('question-form').addEventListener('submit', submitQuestion);
$('question').addEventListener('input', updateQuestionLength);
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('question-form').requestSubmit(); }
});
document.querySelectorAll('[data-query]').forEach(b => b.addEventListener('click', () => fillQuestion(b.dataset.query)));
$('nav-research').onclick = () => setView('research');
$('nav-library').onclick = $('browse-library').onclick = () => setView('library');
$('filter').addEventListener('input', () => { state.sidebarLimit = 24; renderSidebar(); });
$('library-search').addEventListener('input', () => { state.catalogLimit = 40; renderCatalog(); });
$('library-sort').addEventListener('change', renderCatalog);
$('load-more-papers').onclick = () => { state.sidebarLimit += 24; renderSidebar(); };
$('load-more-catalog').onclick = () => { state.catalogLimit += 40; renderCatalog(); };
$('clear-selection').onclick = () => { state.selected.clear(); renderSidebar(); renderCatalog(); updateScope(); };
$('refresh').onclick = refresh;
$('new-research').onclick = newResearch;
$('open-library').onclick = openLibrary; $('close-library').onclick = closeLibrary;
$('open-settings').onclick = () => { applyPreferenceControls(); $('settings-dialog').showModal(); };
$('close-settings').onclick = () => $('settings-dialog').close();
$('settings-dialog').addEventListener('close', applyPreferenceControls);
$('save-settings').onclick = () => {
  preferences = {mode: $('retrieval-mode').value, topK: Number($('top-k').value), rerank: $('rerank').checked,
    generate: $('generate').disabled ? preferences.generate : $('generate').checked};
  writeStorage('paperrag.retrieval-preferences', preferences); applyPreferenceControls(); $('settings-dialog').close();
};
$('copy-answer').onclick = () => { if (state.result?.answer) copyText(state.result.answer, '回答已复制，包含原文引用编号'); };
$('copy-source').onclick = () => { if (state.reader.source?.text) copyText(state.reader.source.text, '证据原文已复制'); };
$('reader-prev').onclick = $('pdf-prev').onclick = () => moveReader(-1);
$('reader-next').onclick = $('pdf-next').onclick = () => moveReader(1);
$('expand-reader').onclick = $('reader-image').onclick = openFullReader;
$('close-reader').onclick = () => mobileReaderDialog.close();
$('close-pdf').onclick = () => $('pdf-dialog').close();
$('keyboard-help').onclick = () => $('shortcuts-dialog').showModal();
$('close-shortcuts').onclick = () => $('shortcuts-dialog').close();
$('upload-trigger').onclick = $('library-import').onclick = () => $('upload').click();
$('upload').addEventListener('change', uploadFile);
$('build-index').onclick = async () => {
  $('build-index').disabled = true;
  try { await api('/api/index', {method: 'POST'}); lastIndexStatus = 'queued'; await pollIndex(); }
  catch (error) { $('build-index').disabled = false; $('index-progress').hidden = false; $('index-progress').textContent = error.message; }
};
document.querySelectorAll('[data-result-tab]').forEach(b => {
  b.addEventListener('click', () => setResultTab(b.dataset.resultTab));
  b.addEventListener('keydown', event => {
    const tabs = ['answer', 'evidence', 'formulas'], current = tabs.indexOf(b.dataset.resultTab);
    const index = event.key === 'ArrowRight' ? (current + 1) % 3 : event.key === 'ArrowLeft' ? (current + 2) % 3 : event.key === 'Home' ? 0 : event.key === 'End' ? 2 : -1;
    if (index >= 0) { event.preventDefault(); setResultTab(tabs[index], true); }
  });
});
document.addEventListener('keydown', event => {
  const input = event.target instanceof HTMLElement && (event.target.matches('input,textarea,select') || event.target.isContentEditable);
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    if (document.querySelector('dialog[open]:not(.library-dialog)')) return;
    event.preventDefault(); openLibrary(); $('filter').focus();
  } else if (!input && !document.querySelector('dialog[open]') && event.key === '/') {
    event.preventDefault(); setView('research'); $('question').focus();
  } else if (!input && !document.querySelector('dialog[open]') && event.key === '?') {
    event.preventDefault(); $('shortcuts-dialog').showModal();
  }
});
for (const dialog of document.querySelectorAll('dialog')) {
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
}
sidebarMedia.addEventListener('change', syncResponsiveShell); readerMedia.addEventListener('change', syncResponsiveShell);
window.addEventListener('pagehide', () => { clearTimeout(indexTimer); if (readerBlob) URL.revokeObjectURL(readerBlob); readerController?.abort(); });
syncResponsiveShell(); applyPreferenceControls(); renderHistory(); refresh(); pollIndex();

