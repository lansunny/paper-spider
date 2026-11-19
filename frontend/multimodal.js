'use strict';

function createMultimodal({$, api, node, button, localResource, paperById, openPaper, showError, updateComposer}) {
  const kinds = {body: ['正文', 'body'], figure: ['图片', 'figures'], table: ['表格', 'tables']};
  const scopes = Object.keys(kinds);
  const pick = $('multimodal-image');
  const profileSelect = $('multimodal-profile');
  const dialog = $('multimodal-object');
  let kind = 'all', filter = 'all', result = null, file = null, preview = null, profileRequest, detailRequest = 0;

  function chooseKind(next) {
    kind = next;
    $('multimodal-kinds').querySelectorAll('button').forEach(b => {
      b.setAttribute('aria-pressed', String(b.dataset.mmKind === next));
    });
    updateComposer();
  }

  function selectImage(next) {
    if (next && (next.size > 8 * 1024 * 1024 ||
        !(next.type ? /^image\/(png|jpeg|webp)$/.test(next.type) : /\.(png|jpe?g|webp)$/i.test(next.name)))) {
      showError('请选择不超过 8 MB 的 PNG、JPEG 或 WebP 图片。');
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    file = next;
    preview = file ? URL.createObjectURL(file) : null;
    const image = $('multimodal-preview-image');
    if (preview) image.src = preview;
    else image.removeAttribute('src');
    $('multimodal-preview-name').textContent = file?.name || '';
    showError('');
    updateComposer();
  }

  async function loadProfiles() {
    if (!profileRequest) {
      profileRequest = api('/api/docling/profiles').then(data => {
        const full = (data.profiles || []).filter(p => !p.pilot);
        profileSelect.replaceChildren(...full.map(p => {
          const option = node('option', p.papers + ' 篇 · ' + p.profile.slice(0, 8));
          option.value = p.profile;
          return option;
        }));
        if (full.some(p => p.profile === data.default_profile)) profileSelect.value = data.default_profile;
        $('multimodal-profile-field').hidden = full.length < 2;
        if (!full.length) profileRequest = null;
        return full;
      }).catch(error => { profileRequest = null; throw error; });
    }
    return profileRequest;
  }

  async function search(query, paperIds, topK) {
    const profiles = await loadProfiles();
    const profile = profiles.find(p => p.profile === profileSelect.value);
    if (!profile) throw new Error('多模态索引尚未就绪。');
    if (file) {
      if (!profile.clip_ready) throw new Error('图片索引尚未就绪，请先使用文字检索。');
      const form = new FormData();
      form.set('image', file);
      form.set('profile', profile.profile);
      form.set('top_k', topK);
      paperIds.forEach(id => form.append('paper_ids', id));
      return api('/api/docling/search/image', {method: 'POST', body: form});
    }
    return api('/api/docling/search', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, profile: profile.profile, paper_ids: paperIds, top_k: topK,
        kinds: kind === 'all' ? scopes : [kind]})});
  }

  function sourcePage(hit) { return hit.matched_page || hit.pages?.[0] || 1; }
  function sourceTitle(hit) { return paperById(hit.paper_id)?.title || hit.title || hit.paper_id; }
  function displayTitle(hit, type) {
    if (type === 'body') return hit.headings?.at(-1) || sourceTitle(hit);
    return hit.labels?.length ? (type === 'figure' ? '图 ' : '表 ') + hit.labels.join('、') : kinds[type][0];
  }
  function openSource(hit, reveal = true) {
    const paper = paperById(hit.paper_id) || {paper_id: hit.paper_id, title: sourceTitle(hit)};
    const text = hit.kind === 'paragraph' ? hit.text : hit.caption || hit.text;
    openPaper(paper, {page: sourcePage(hit), source: text ? {text} : null, reveal});
  }

  function cropImage(crop, alt) {
    const url = localResource(crop.image_url);
    if (!url) return null;
    const image = node('img');
    image.src = url; image.alt = alt; image.loading = 'lazy';
    return image;
  }

  function paragraphLinks(obj) {
    const holder = node('div', null, 'mm-context-links');
    const blocks = [...(obj.related_paragraphs || []), ...(obj.nearby_paragraph ? [obj.nearby_paragraph] : [])];
    if (!blocks.length) return holder;
    holder.append(node('h3', '关联正文'));
    for (const hit of new Map(blocks.map(p => [p.object_id, p])).values()) {
      const item = node('article');
      item.append(node('p', hit.text.slice(0, 360) + (hit.text.length > 360 ? '…' : '')),
        button('打开 PDF 第 ' + sourcePage(hit) + ' 页', 'text-button', () => {
          dialog.close();
          openSource(hit);
        }));
      holder.append(item);
    }
    return holder;
  }

  function tableGrid(data, matchedRows) {
    const grid = data.table;
    if (!data.html || !grid?.table_cells?.length || !grid.num_rows || !grid.num_cols || grid.num_rows * grid.num_cols > 20000) return null;
    const wrapper = node('div', null, 'mm-table-scroll');
    wrapper.tabIndex = 0; wrapper.setAttribute('role', 'region'); wrapper.setAttribute('aria-label', '解析后的表格');
    const table = node('table');
    const starts = new Map(), occupied = new Set();
    for (const cell of grid.table_cells) {
      const r = cell.start_row_offset_idx, c = cell.start_col_offset_idx;
      starts.set(r * grid.num_cols + c, cell);
      for (let rr = r; rr < cell.end_row_offset_idx; rr++) {
        for (let cc = c; cc < cell.end_col_offset_idx; cc++) occupied.add(rr * grid.num_cols + cc);
      }
    }
    const focus = new Set(matchedRows.map(row => row.row));
    for (let r = 0; r < grid.num_rows; r++) {
      const tr = node('tr', null, focus.has(r) ? 'mm-matched-row' : '');
      for (let c = 0; c < grid.num_cols; c++) {
        const index = r * grid.num_cols + c, cell = starts.get(index);
        if (!cell) {
          if (!occupied.has(index)) tr.append(node('td'));
          continue;
        }
        const td = node(cell.column_header || cell.row_header || cell.row_section ? 'th' : 'td', cell.text);
        td.rowSpan = cell.row_span; td.colSpan = cell.col_span;
        if (cell.column_header) td.scope = cell.col_span > 1 ? 'colgroup' : 'col';
        else if (cell.row_header) td.scope = 'row';
        tr.append(td);
      }
      table.append(tr);
    }
    wrapper.append(table);
    return wrapper;
  }

  async function showObject(hit, type, loaded = null) {
    const url = localResource(hit.object_url);
    if (!url) return;
    const request = ++detailRequest;
    $('multimodal-object-title').textContent = displayTitle(hit, type);
    const meta = $('multimodal-object-meta'), content = $('multimodal-object-content');
    const origin = button('打开 PDF 第 ' + sourcePage(hit) + ' 页 ↗', 'text-button', () => {
      dialog.close(); openSource(hit);
    });
    meta.replaceChildren(node('span', sourceTitle(hit)), origin);
    content.replaceChildren(node('p', '正在读取原文对象…', 'muted'));
    if (!dialog.open) dialog.showModal();
    try {
      const detail = loaded || await api(url);
      if (request !== detailRequest || !dialog.open) return;
      content.replaceChildren();
      if (detail.caption) content.append(node('p', detail.caption, 'mm-object-caption'));
      if (type === 'table') {
        const grid = tableGrid(detail, hit.matched_rows || []);
        if (grid) content.append(grid);
        const notes = (detail.footnotes || []).map(item => typeof item === 'string' ? item : item.text).filter(Boolean);
        if (notes.length) {
          const foot = node('div', null, 'mm-table-notes');
          notes.forEach(note => foot.append(node('p', note)));
          content.append(foot);
        }
      }
      for (const crop of detail.crops || []) {
        const image = cropImage(crop, displayTitle(hit, type) + ' · PDF 第 ' + crop.page + ' 页的原图');
        if (image) content.append(image);
      }
      const related = paragraphLinks({...detail, nearby_paragraph: hit.nearby_paragraph});
      if (related.childElementCount) content.append(related);
      if (!content.childElementCount) content.append(node('p', '请在 PDF 来源页核对内容。', 'muted'));
    } catch (error) {
      if (request === detailRequest && dialog.open) content.replaceChildren(node('p', error.message, 'error-message'));
    }
  }

  function linkedObjects(hit) {
    const details = node('details', null, 'mm-card-text');
    const list = node('div', null, 'mm-linked-objects');
    details.append(node('summary', '关联图表 ' + hit.linked_objects.length), list);
    let loaded = false, loading = false;
    details.addEventListener('toggle', async () => {
      if (!details.open || loading || loaded) return;
      loading = true;
      list.textContent = '正在读取关联图表…';
      const responses = await Promise.allSettled(hit.linked_objects.map(async link => {
        const path = localResource(link.object_url);
        if (!path) return null;
        return api(path);
      }));
      list.replaceChildren();
      for (const response of responses) {
        const obj = response.status === 'fulfilled' ? response.value : null;
        if (obj && (obj.kind === 'figure' || obj.kind === 'table')) {
          list.append(button(displayTitle(obj, obj.kind), 'text-button', () => showObject(obj, obj.kind, obj)));
        }
      }
      loaded = list.childElementCount > 0;
      if (!loaded) list.textContent = '关联图表暂时无法读取，请重新展开。';
      loading = false;
    });
    return details;
  }

  function card(hit, type) {
    const item = node('article', null, 'mm-card mm-card-' + type);
    if (type !== 'body' && hit.crops?.length) {
      const image = cropImage(hit.crops[0], displayTitle(hit, type) + ' · ' + sourceTitle(hit));
      if (image) {
        const previewButton = button(null, 'mm-card-preview', () => showObject(hit, type));
        previewButton.setAttribute('aria-label', '查看 ' + image.alt);
        previewButton.append(image); item.append(previewButton);
      }
    }
    const title = button(displayTitle(hit, type), 'mm-card-title',
      () => type === 'body' ? openSource(hit) : showObject(hit, type));
    item.append(title, node('p', sourceTitle(hit) + ' · PDF ' + sourcePage(hit), 'mm-card-source'));
    const text = type === 'body' ? hit.text : hit.caption;
    if (text) {
      const limit = type === 'body' ? 340 : 210;
      item.append(node('p', text.slice(0, limit) + (text.length > limit ? '…' : ''), 'mm-card-text'));
      if (type === 'body' && text.length > limit) {
        const more = node('details', null, 'mm-card-text');
        more.append(node('summary', '展开段落'), node('p', text)); item.append(more);
      }
    }
    if (type === 'table' && hit.matched_rows?.length) {
      const matched = node('ul', null, 'mm-match-rows');
      hit.matched_rows.slice(0, 2).forEach(row => matched.append(node('li', row.text)));
      item.append(matched);
    }
    if (type === 'body' && hit.linked_objects?.length) item.append(linkedObjects(hit));
    item.append(button('查看原文 ↗', 'text-button mm-source-button', () => openSource(hit)));
    return item;
  }

  function paint(focus = false) {
    const bar = $('multimodal-filters'), content = $('multimodal-results-content');
    bar.replaceChildren(); content.replaceChildren();
    const groups = scopes.map(type => [type, result[kinds[type][1]] || []]);
    const total = groups.reduce((n, [, items]) => n + items.length, 0);
    $('multimodal-result-meta').textContent = total + ' 条结果';
    if (!total) { content.append(node('p', '未找到匹配的内容。', 'mm-empty')); return; }
    const filters = [['all', '全部', total], ...groups.filter(([, items]) => items.length)
      .map(([type, items]) => [type, kinds[type][0], items.length])];
    for (const [value, label, count] of filters) {
      const b = button(label + ' ' + count, '', () => { filter = value; paint(true); });
      b.setAttribute('aria-pressed', String(filter === value)); bar.append(b);
      if (focus && filter === value) b.focus({preventScroll: true});
    }
    for (const [type, items] of groups) {
      if (!items.length || (filter !== 'all' && filter !== type)) continue;
      const section = node('section', null, 'mm-group');
      section.append(node('h3', kinds[type][0]));
      const cards = node('div', null, 'mm-card-list' + (type === 'figure' ? ' mm-figure-grid' : ''));
      items.forEach(hit => cards.append(card(hit, type))); section.append(cards); content.append(section);
    }
  }

  pick.onchange = () => { if (pick.files[0]) selectImage(pick.files[0]); pick.value = ''; };
  $('multimodal-image-pick').onclick = () => pick.click();
  $('multimodal-image-remove').onclick = () => { selectImage(null); $('question').focus(); };
  $('multimodal-kinds').querySelectorAll('button').forEach(b => b.onclick = () => chooseKind(b.dataset.mmKind));
  $('multimodal-object-close').onclick = () => dialog.close();
  dialog.addEventListener('close', () => { detailRequest++; $('multimodal-object-content').replaceChildren(); });
  window.addEventListener('pagehide', event => { if (!event.persisted && preview) URL.revokeObjectURL(preview); });
  return {
    chooseKind, search,
    get file() { return file; },
    get kind() { return kind; },
    get hasResult() { return !!result; },
    clearImage: () => selectImage(null),
    activate: () => loadProfiles().catch(() => {}),
    render(data) {
      result = data; filter = 'all'; paint();
      const first = data.body?.[0] || data.figures?.[0] || data.tables?.[0];
      if (first) openSource(first, false);
    },
    reset() {
      result = null; selectImage(null);
      $('multimodal-filters').replaceChildren();
      $('multimodal-results-content').replaceChildren();
    },
  };
}
