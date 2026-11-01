/* Pointer light and source-state affordances. No remote resources or animation loop at rest. */
(() => {
  'use strict';
  const root = document.documentElement;
  const button = document.getElementById('toggle-motion');
  const systemMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  let saved = null;
  try { saved = localStorage.getItem('paperrag.reduced-motion'); } catch {}
  let reduced = saved === null ? systemMotion.matches : saved === 'true';

  function applyMotion() {
    root.dataset.motion = reduced ? 'reduced' : 'full';
    button.setAttribute('aria-pressed', String(reduced));
    button.setAttribute('aria-label', reduced ? '开启界面动态效果' : '减少界面动态效果');
    document.getElementById('motion-label').textContent = reduced ? '动态关' : '动态开';
    if (reduced) document.querySelectorAll('.is-glass-active').forEach(el => el.classList.remove('is-glass-active'));
  }
  button.addEventListener('click', () => {
    reduced = !reduced;
    saved = String(reduced);
    try { localStorage.setItem('paperrag.reduced-motion', String(reduced)); } catch {}
    applyMotion();
  });
  systemMotion.addEventListener('change', () => {
    if (saved === null) { reduced = systemMotion.matches; applyMotion(); }
  });
  applyMotion();

  let lightFrame = 0;
  let active = null;
  let point = null;
  document.addEventListener('pointermove', event => {
    if (reduced || !finePointer.matches || event.pointerType === 'touch') return;
    const host = event.target instanceof Element ? event.target.closest('.lg') : null;
    if (host !== active) {
      active?.classList.remove('is-glass-active');
      active = host;
      active?.classList.add('is-glass-active');
    }
    if (!active) return;
    point = {x: event.clientX, y: event.clientY};
    if (lightFrame) return;
    lightFrame = requestAnimationFrame(() => {
      lightFrame = 0;
      if (!active || !point) return;
      const box = active.getBoundingClientRect();
      active.style.setProperty('--glass-x', `${point.x - box.left}px`);
      active.style.setProperty('--glass-y', `${point.y - box.top}px`);
      const stage = active.closest('.composer-stage');
      if (stage) {
        stage.style.setProperty('--source-x', `${((point.x - box.left) / box.width - .5) * 18}px`);
        stage.style.setProperty('--source-y', `${((point.y - box.top) / box.height - .5) * 12}px`);
      }
    });
  }, {passive: true});
  document.addEventListener('pointerleave', () => {
    active?.classList.remove('is-glass-active');
    active = null;
  });

  const tabs = document.querySelector('.result-tabs');
  const indicator = document.createElement('div');
  indicator.className = 'liquid-tab-indicator lg';
  indicator.setAttribute('aria-hidden', 'true');
  tabs.prepend(indicator);
  let tabFrame = 0;
  function updateTab() {
    cancelAnimationFrame(tabFrame);
    tabFrame = requestAnimationFrame(() => {
      const selected = tabs.querySelector('button[aria-selected=true]');
      if (!selected || !tabs.offsetWidth) return;
      indicator.style.width = `${selected.offsetWidth}px`;
      indicator.style.transform = `translateX(${selected.offsetLeft}px)`;
    });
  }
  new MutationObserver(updateTab).observe(tabs, {subtree: true, attributes: true, attributeFilter: ['aria-selected']});
  new ResizeObserver(updateTab).observe(tabs);
  document.fonts.ready.then(updateTab);
  window.LiquidGlass?.refresh();
  updateTab();

  const reader = document.getElementById('reader-image-area');
  const readerPane = document.getElementById('reader-pane');
  function updateReaderTone() {
    readerPane.classList.toggle('is-showing-formula', reader.classList.contains('is-formula'));
  }
  new MutationObserver(updateReaderTone).observe(reader, {attributes: true, attributeFilter: ['class']});
  updateReaderTone();
  window.addEventListener('pagehide', () => {
    cancelAnimationFrame(lightFrame);
    cancelAnimationFrame(tabFrame);
  });
})();
