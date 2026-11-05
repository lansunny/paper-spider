'use strict';
const $=id=>document.getElementById(id);
const state={cards:[],papers:new Map(),result:null,focus:[],focusName:'',run:null,busy:false,poll:0,activeCommunity:null};
const labels={planning:'选择相关主题与方法社区',evidence:'沿来源锚点收集原文',route_analysis:'分路线分析原文依据',synthesis:'汇总共性、差异与局限',cards:'整理带来源的研究卡片',starting:'准备研究范围',queued:'任务已排队'};
const fields={overview:'研究问题',approaches:'方法与代表工作',differences:'论文之间的差异',experiments:'实验依据',limitations:'成立条件与局限'};
function el(tag,text,cls){const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;}
function btn(text,cls,fn){const n=el('button',text,cls);n.type='button';n.onclick=fn;return n;}
function link(text,url,cls){const n=el('a',text,cls);n.href=url;return n;}
function count(n){return Number(n||0).toLocaleString('zh-CN');}
function notice(text=''){$('review-notice').hidden=!text;$('review-notice').textContent=text;}
async function api(url,options){const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:'请求未能完成，请稍后重试。');return d;}
function setBusy(busy){state.busy=busy;$('start-review').disabled=busy;$('prepare-cards').disabled=busy;$('start-review').textContent=busy?'正在整理…':'生成综述 ↗';}
function setFocus(ids=[],name=''){state.focus=ids;state.focusName=name;$('focus-route').hidden=!ids.length;$('focus-name').textContent='围绕路线：'+name;}
function routeFollow(route){setFocus([route.route_id],route.name);$('question').value='围绕“'+route.name+'”，进一步比较这些论文的方法、实验设置与适用局限。';$('question').focus();$('review-form').scrollIntoView({behavior:'smooth',block:'center'});}
async function showSource(value){
  const box=$('source-content');box.replaceChildren(el('p','正在读取原文…','muted'));if(!$('source-dialog').open)$('source-dialog').showModal();
  try{const source=typeof value==='string'?await api('/api/reviews/sources/'+encodeURIComponent(value)):value;
    box.replaceChildren(el('h2',source.title),el('p',(source.id?source.id+' · ':'')+'PDF 第 '+source.pdf_page+' 页','source-location'),el('blockquote',source.text,'source-quote'));
    const a=link('在论文精读中打开这一页 ↗',source.study_url,'source-open');a.target='_blank';a.rel='noopener';box.append(a);
  }catch(error){box.replaceChildren(el('p',error.message,'notice'));}
}
function cited(item){const p=el('p',item.text,'cited-paragraph');for(const id of item.citations){const source=state.result.evidence.find(e=>e.id===id);if(source){const b=btn(id,'citation',()=>showSource(source));b.title=source.title+' · PDF '+source.pdf_page;p.append(' ',b);}}return p;}
function section(title,items){const box=el('section',null,'analysis-group');box.append(el('h4',title));for(const item of items)box.append(cited(item));return box;}
function support(statement,box){box.append(el('span',(statement.support==='multi_paper'?'多篇支持':'单篇发现')+' · '+statement.supporting_paper_ids.length+' 篇来源','support-tag'));
  for(const sid of statement.source_ids.slice(0,4)){box.append(btn('原文 ↗','card-source',()=>showSource(sid)));}
}
function communityContent(card){
  const inside=el('div',null,'community-inner');for(const scope of card.scope){const p=el('div',null,'card-scope');p.append(el('p',scope.text));support(scope,p);inside.append(p);}
  for(const route of card.routes){const r=el('div',null,'card-route');r.append(el('h4',route.name),el('p',route.overview.text));support(route.overview,r);
    const more=el('details',null,'card-extra');more.append(el('summary','差异、实验与局限'));
    for(const [field,title] of [['differences','具体差异'],['experiments','实验依据'],['limitations','适用局限']])for(const statement of route[field]){const part=el('div',null,'card-scope');part.append(el('h5',title),el('p',statement.text));support(statement,part);more.append(part);}
    if(more.children.length>1)r.append(more);
    for(const pid of route.representative_paper_ids){const paper=state.papers.get(pid);r.append(link(paper?.title||pid,'/study?paper='+encodeURIComponent(pid),'card-paper'));}inside.append(r);}
  return inside;
}
function communityQuestion(card){setFocus();$('review-scope').value='topic';$('question').value='在我的论文库中，'+card.title+'有哪些研究路线？请比较代表工作、实验依据和局限。';$('question').focus();$('review-form').scrollIntoView({behavior:'smooth',block:'center'});}
function openCommunity(id){
  const card=state.cards.find(c=>c.topic_community_id===id),panel=$('community-detail');
  state.activeCommunity=card?id:null;
  const url=new URL(location.href);if(card)url.searchParams.set('community',id);else url.searchParams.delete('community');window.history.replaceState({},'',url);
  for(const b of document.querySelectorAll('.community-cover'))b.setAttribute('aria-expanded',String(b.dataset.communityId===state.activeCommunity));
  panel.hidden=!card;if(!card)return;
  CommunityVisuals.paint(panel,id);panel.replaceChildren();
  const heading=el('div',null,'community-deck-heading'),copy=el('div');
  copy.append(el('h3',card.title),el('p',card.paper_ids.length+' 篇论文 · '+card.routes.length+' 条研究路线'));
  const close=btn('×','deck-close',()=>{openCommunity(null);document.querySelector('.community-cover[data-community-id="'+id+'"]')?.focus({preventScroll:true});});close.setAttribute('aria-label','关闭群组');
  heading.append(CommunityVisuals.emblem(card.paper_ids,id),copy,close);panel.append(heading);
  const actions=el('div',null,'community-deck-actions');actions.append(btn('围绕此群组综述 ↗','',()=>communityQuestion(card)),link('在地图中查看 ↗','/graph?dimension=topic&group='+encodeURIComponent(id)));
  const content=el('div',null,'community-deck-content');content.append(communityContent(card));
  const members=el('details',null,'community-paper-members');members.append(el('summary','组内论文 · '+card.paper_ids.length+' 篇'));const list=el('div',null,'community-paper-grid');
  for(const pid of card.paper_ids){const paper=state.papers.get(pid);list.append(link(paper?.title||pid,'/study?paper='+encodeURIComponent(pid)));}
  members.append(list);content.append(members);panel.append(actions,content);
  panel.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'nearest'});
}
function renderCards(){
  const target=$('community-cards');target.replaceChildren();CommunityVisuals.register(state.cards.map(c=>c.topic_community_id));
  for(const card of state.cards){
    const id=card.topic_community_id,deck=CommunityVisuals.paint(el('article',null,'community-deck'),id);
    const cover=btn('','community-cover',()=>openCommunity(state.activeCommunity===id?null:id));cover.dataset.communityId=id;cover.setAttribute('aria-controls','community-detail');cover.setAttribute('aria-expanded',String(state.activeCommunity===id));
    cover.setAttribute('aria-label',card.title+'，'+card.paper_ids.length+'篇论文，'+card.routes.length+'条研究路线');
    const art=el('div',null,'community-cover-art');art.append(CommunityVisuals.emblem(card.paper_ids,id));
    const copy=el('div',null,'community-cover-copy');copy.append(el('h3',card.title),el('p',card.scope.map(x=>x.text).join(' '),'community-cover-description'));
    const bottom=el('div',null,'community-cover-bottom');bottom.append(el('span',card.routes.length+' 条路线'),el('span','打开 ↗'));copy.append(bottom);
    cover.append(art,copy);deck.append(cover);target.append(deck);
  }
}
async function loadCards(){const status=await api('/api/reviews/status');$('review-scope-meta').textContent=status.papers+' 篇论文';$('community-meta').textContent=status.ready?status.topic_cards+' 个主题':'研究卡片尚未准备';$('prepare-cards').hidden=status.ready;
  if(status.ready){const data=await api('/api/reviews/cards');state.cards=data.cards;state.papers=new Map(data.papers.map(p=>[p.paper_id,p]));renderCards();}
  else notice(status.message||'点击“整理研究卡片”，从已有论文图准备可追溯的研究线索。');
}
async function loadHistory(){try{const data=await api('/api/reviews/jobs');const target=$('recent-reviews');target.replaceChildren();const reviews=data.jobs.filter(j=>j.kind==='review'&&j.visibility!=='diagnostic');for(const job of reviews){const a=link(job.question,'/review?run='+job.run_id,'history-link'+(job.run_id===state.run?' is-current':''));a.append(el('small',(job.status==='completed'?'已完成':job.status==='failed'?'未完成':job.status==='interrupted'?'已中断':'处理中')+' · '+new Date(job.created_at).toLocaleDateString('zh-CN')));target.append(a);}if(!reviews.length)target.append(el('p','完成的研究综述将保存在这里。','muted'));}catch(error){$('recent-reviews').replaceChildren(el('p',error.message,'muted'));}}
function progress(job){$('job-progress').hidden=false;const p=job.progress||{},stage=p.stage||'starting';$('progress-title').textContent=labels[stage]||'正在处理';$('progress-count').textContent=p.total?Math.min(p.completed+1,p.total)+' / '+p.total:'';$('progress-detail').textContent=p.route_name||'';const stages=['planning','evidence','route_analysis','synthesis'];document.querySelectorAll('[data-stage]').forEach(n=>{const at=stages.indexOf(n.dataset.stage),current=stages.indexOf(stage);n.classList.toggle('active',at===current);n.classList.toggle('done',current>at);});}
function renderResult(result){state.result=result;const root=$('review-result');root.hidden=false;root.replaceChildren();$('review-main').classList.add('has-result');
  const head=el('header',null,'review-result-head');head.append(el('h2',result.synthesis.title));const meta=el('div',null,'result-meta');const supported=result.routes.filter(r=>r.analysis.relevant&&!r.analysis.insufficient_evidence);meta.append(el('span',result.generated?supported.length+' 条有证据路线':result.routes.filter(r=>r.evidence.length).length+' 条取证路线'),el('span',new Set(result.evidence.map(e=>e.paper_id)).size+' 篇取证论文'),el('span',result.evidence.length+' 段原文'),link('在地图中查看取证范围 ↗',result.map_url));head.append(meta);root.append(head);
  const overview=el('div',null,'review-overview');for(const item of result.synthesis.overview)overview.append(cited(item));if(!result.synthesis.overview.length)overview.append(el('p',result.generated?'当前证据不足以形成综述':'原文已整理','no-support'));root.append(overview);
  const layout=el('div',null,'result-layout'),nav=el('nav',null,'route-nav');nav.setAttribute('aria-label','本次研究路线');const routes=el('div',null,'route-results');result.routes.forEach((route,index)=>{
    const anchor='route-'+index,a=link('', '#'+anchor);a.append(el('small',String(index+1).padStart(2,'0')),el('span',route.name));nav.append(a);
    const article=el('article',null,'research-route');article.id=anchor;article.append(el('span','路线 '+String(index+1).padStart(2,'0')+' · '+route.paper_ids.length+' 篇来源','route-kicker'),el('h3',route.name));const papers=el('div',null,'paper-chips');route.papers.forEach((paper,i)=>{const p=link('','/study?paper='+encodeURIComponent(paper.paper_id),'paper-chip');p.append(el('span',String(i+1)),el('div',paper.title));papers.append(p);});article.append(papers);
    if(!route.analysis.relevant)article.append(el('p','与当前问题关联不足','no-support'));
    for(const [field,title] of Object.entries(fields))if(route.analysis[field]?.length)article.append(section(title,route.analysis[field]));
    if(route.analysis.missing_evidence.length){const missing=el('div',null,'missing-evidence');missing.append(el('strong','证据缺口'));for(const text of route.analysis.missing_evidence)missing.append(el('p',text));article.append(missing);}
    const detail=el('details',null,'run-cost');detail.append(el('summary','原文依据 · '+route.evidence.length+' 段 / '+route.paper_ids.length+' 篇'));for(const source of route.evidence){const p=el('p',null,'card-paper');p.append(btn(source.id+' · PDF '+source.pdf_page,'card-source',()=>showSource(source)),document.createTextNode(source.title));detail.append(p);}article.append(detail);
    const actions=el('div',null,'route-actions');actions.append(btn('围绕此路线追问 ↗','',()=>routeFollow(route)),link('在地图中定位',route.map_url));article.append(actions);routes.append(article);
  });layout.append(nav,routes);root.append(layout);
  if(result.synthesis.comparisons.length||result.synthesis.limitations.length){const compare=el('section',null,'review-comparison');compare.append(el('h3','共性、差异与边界'));if(result.synthesis.comparisons.length)compare.append(section('跨路线比较',result.synthesis.comparisons));if(result.synthesis.limitations.length)compare.append(section('当前证据的边界',result.synthesis.limitations));root.append(compare);}
  const cost=el('details',null,'run-cost');cost.append(el('summary','用量 · '+count(result.usage.total_tokens)+' token · '+(result.elapsed_ms/1000).toFixed(1)+' 秒'));const dl=el('dl');for(const [name,value] of [['输入 token',count(result.usage.prompt_tokens)],['输出 token',count(result.usage.completion_tokens)]])dl.append(el('dt',name),el('dd',value));cost.append(dl);root.append(cost);
}
async function watch(runId){clearTimeout(state.poll);state.run=runId;try{const job=await api('/api/reviews/jobs/'+encodeURIComponent(runId));if(state.run!==runId)return;
  if(job.question)$('question').value=job.question;
  if(job.status==='completed'){setBusy(false);$('job-progress').hidden=true;if(job.kind==='cards'){await loadCards();notice('研究卡片已准备好，可以开始整理综述。');}else{$('review-scope').value=job.result.scope;renderResult(job.result);}loadHistory();return;}
  if(['failed','interrupted'].includes(job.status)){setBusy(false);$('job-progress').hidden=true;notice(job.message||'任务未完成，请重试。');loadHistory();return;}
  setBusy(true);progress(job);state.poll=setTimeout(()=>watch(runId),1800);
}catch(error){setBusy(false);$('job-progress').hidden=true;notice(error.message);}}
async function start(cards=false){if(state.busy)return;notice();setBusy(true);progress({progress:{stage:'starting',completed:0,total:1}});try{const payload={question:$('question').value.trim(),scope:$('review-scope').value,focus_route_ids:state.focus,generate:!$('evidence-only').checked};const job=await api(cards?'/api/reviews/cards':'/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},...(cards?{}:{body:JSON.stringify(payload)})});const url=new URL(location.href);url.searchParams.set('run',job.run_id);window.history.replaceState({},'',url);progress(job);watch(job.run_id);loadHistory();}catch(error){setBusy(false);$('job-progress').hidden=true;notice(error.message);}}
$('review-form').onsubmit=e=>{e.preventDefault();start();};$('prepare-cards').onclick=()=>start(true);$('refresh-history').onclick=loadHistory;$('clear-focus').onclick=()=>setFocus();$('close-source').onclick=()=>$('source-dialog').close();$('source-dialog').onclick=e=>{if(e.target===$('source-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}};
async function init(){const run=new URLSearchParams(location.search).get('run');state.run=run;await Promise.allSettled([loadCards().catch(error=>notice(error.message)),loadHistory()]);if(run)await watch(run);const community=new URLSearchParams(location.search).get('community');if(community)openCommunity(community);document.body.dataset.ready='true';}
init();
