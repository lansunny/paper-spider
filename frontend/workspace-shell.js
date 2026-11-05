/* Shared navigation and context tools for every research workspace. */
(() => {
  'use strict';
  const header=document.querySelector('[data-app-header]');
  if(!header)return;
  const page=header.dataset.appHeader;
  const menu=header.querySelector('#open-library');
  const searchStatus=header.querySelector('#mode-search-status');
  const tools=header.querySelector('.header-actions')||document.createElement('div');
  const copy=header.querySelector('#copy-map-link');
  if(copy)tools.append(copy);
  tools.className='ps-header-tools';
  tools.querySelector('.header-tools')?.classList.remove('lg');
  header.className='ps-header';
  const brand=document.createElement('div');brand.className='ps-brand-group';
  if(menu)brand.append(menu);
  const home=document.createElement('a');home.className='ps-brand';home.href='/?mode=study';
  home.innerHTML='<img src="/static/assets/favicon.svg" width="28" height="28" alt=""><span>论文蛛<span class="ps-brand-dot">.</span></span>';
  home.setAttribute('aria-label','论文蛛 Paper Spider 论文库');brand.append(home);
  const nav=document.createElement('nav');nav.className='ps-navigation';nav.setAttribute('aria-label','工作空间');
  const entries=[['study','论文精读','/?mode=study','book'],['search','多论文检索','/?mode=search','search'],['graph','论文地图','/graph','network'],['review','研究综述','/review','notes']];
  for(const [id,label,url,symbol] of entries){
    const a=document.createElement('a');a.href=url;a.dataset.workspace=id;
    if(page==='home'&&['study','search'].includes(id))a.id='mode-'+id;
    const sprite=symbol==='notes'?'study-icons.svg':'icons.svg';
    a.innerHTML='<svg class="icon" aria-hidden="true"><use href="/static/assets/'+sprite+'#'+symbol+'"/></svg><span>'+label+'</span>';
    if(id==='search'&&searchStatus)a.append(searchStatus);
    nav.append(a);
  }
  header.replaceChildren(brand,nav,tools);
  function setMode(mode){for(const a of nav.querySelectorAll('a')){if(a.dataset.workspace===mode)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');}}
  window.PaperSpiderUI={setMode};
  setMode(page==='home'?(new URLSearchParams(location.search).get('mode')==='search'?'search':'study'):page);
  document.body.dataset.screen=page;

  if(page==='review'){
    const aside=document.querySelector('.review-sidebar');
    if(aside){
      const placeholder=document.createComment('review sidebar');aside.before(placeholder);
      const dialog=document.createElement('dialog');dialog.className='ps-context-dialog';dialog.setAttribute('aria-label','综述记录');
      const close=document.createElement('button');close.type='button';close.className='ps-context-close';close.textContent='关闭';close.onclick=()=>dialog.close();dialog.append(close);document.body.append(dialog);
      const toggle=document.createElement('button');toggle.className='ps-history-toggle';toggle.type='button';toggle.textContent='历史记录';toggle.onclick=()=>{dialog.append(aside);dialog.showModal();};tools.append(toggle);
      dialog.addEventListener('close',()=>placeholder.after(aside));
      const media=matchMedia('(max-width:800px)');media.addEventListener('change',()=>{if(!media.matches&&dialog.open)dialog.close();});
    }
  }
})();
