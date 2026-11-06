/* The same sourced reading assistant serves the study inspector and full workspace. */
(()=>{
  'use strict';
  const root=document.querySelector('[data-related-reading]');if(!root)return;
  const params=new URLSearchParams(location.search),paper=params.get('paper')||'',compact=root.dataset.layout==='compact';
  let run=params.get('reading_run')||(!compact?params.get('run'):'')||'',busy=false,result=null;
  const labels={queued:'等待开始',neighbors:'发现关联论文',followups:'整理值得追问的问题',claim_embeddings:'准备原文取证',evidence:'沿追问收集原文',advice:'整理对照阅读建议',grounding:'核对说明与阅读位置',completed:'已完成'};
  const directions={alternative:'相近问题 · 不同路线',transfer:'相近方法 · 其他问题',experiment:'实验对照线索',conclusion:'发现与局限',topic:'相近研究问题'};
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const link=(text,href,cls)=>{const a=el('a',text,cls);a.href=href;return a;};
  const api=async(url,options)=>{const r=await fetch(url,options);const body=await r.json();if(!r.ok)throw Error(typeof body.detail==='string'?body.detail:'暂时无法完成，请重试。');return body;};
  const head=el('header',undefined,'rr-heading');head.append(el(compact?'h2':'h1','关联阅读'));const seed=el('div','从当前论文出发，寻找值得对照阅读的工作。','rr-seed');head.append(seed);
  const form=el('form',undefined,'rr-form'),label=el('label','想进一步了解什么？');label.htmlFor='rr-goal';
  const goal=el('textarea');goal.id='rr-goal';goal.maxLength=1200;goal.required=true;goal.value='了解解决相近问题的其他方法，比较适用条件，并推荐下一篇值得读的论文和原文位置。';
  const actions=el('div',undefined,'rr-actions'),start=el('button','开始关联阅读','rr-start');start.type='submit';start.disabled=true;
  const evidenceLabel=el('label'),evidenceOnly=el('input');evidenceOnly.type='checkbox';evidenceLabel.append(evidenceOnly,document.createTextNode('仅整理原文'));
  actions.append(evidenceLabel,start);form.append(label,goal,actions);
  const status=el('div','正在读取论文范围…','rr-status');status.setAttribute('role','status');
  const open=link('展开阅读建议 ↗','/reading?paper='+encodeURIComponent(paper),'rr-open');if(!compact)open.hidden=true;
  const questions=el('details',undefined,'rr-questions');questions.hidden=true;questions.append(el('summary','本次比较问题'));
  const cards=el('div',undefined,'rr-cards'),history=el('nav',undefined,'rr-history');history.setAttribute('aria-label','关联阅读记录');
  root.append(head,form,status,open,questions,cards,history);
  const back=document.getElementById('rr-back');if(back&&paper){back.textContent='返回当前论文';back.href='/study?paper='+encodeURIComponent(paper);}
  function setStatus(text,error=false){status.textContent=text;status.classList.toggle('is-error',error);}
  function original(id,parent,quotes=[]){
    const hit=result?.evidence.find(e=>e.citation_id===id);if(!hit)return;
    const prior=parent.querySelector('[data-evidence="'+id+'"]');if(prior){prior.remove();return;}
    const box=el('div',undefined,'rr-original');box.dataset.evidence=id;const a=link(hit.title+' · PDF '+hit.pdf_page+' ↗',hit.study_url,'rr-source-link');a.target='_blank';a.rel='noopener';box.append(a);
    const chars=Array.from(hit.text),slice=(a,b)=>chars.slice(a,b).join('');
    const text=el('p'),ranges=quotes.filter(q=>q.citation_id===id&&Number.isInteger(q.text_start)&&Number.isInteger(q.text_end)&&q.text_start>=0&&q.text_end>q.text_start&&q.text_end<=chars.length&&slice(q.text_start,q.text_end)===q.quote).sort((a,b)=>a.text_start-b.text_start);
    let cursor=0;for(const q of ranges){if(q.text_start<cursor)continue;text.append(document.createTextNode(slice(cursor,q.text_start)),el('mark',slice(q.text_start,q.text_end)));cursor=q.text_end;}text.append(document.createTextNode(slice(cursor)));box.append(text);parent.append(box);
  }
  function cited(item,parent){const p=el('p',item.text);const type={direct:'原文位置',overview:'概述入口',source_excerpt:'原文片段'}[item.location_type];if(type)p.prepend(el('span',type,'rr-location'));for(const id of item.citations){const b=el('button',id,'rr-citation');b.type='button';b.title='查看引用原文';b.onclick=()=>original(id,parent,item.support_quotes||[]);p.append(b);}parent.append(p);}
  function render(data){
    result=data;goal.value=data.goal;evidenceOnly.checked=!data.generated;cards.replaceChildren();questions.replaceChildren(el('summary','本次比较问题'));
    if(data.questions.length){const list=el('ol');for(const q of data.questions)list.append(el('li',q.question));questions.append(list);questions.hidden=false;}
    else questions.hidden=true;
    const historical=data.generated&&!data.grounding_review&&data.version!=='graph-guided-related-reading-v2';
    for(const [index,r] of data.recommendations.entries()){
      const card=el('article',undefined,'rr-card'),heading=el('div',undefined,'rr-card-head'),title=el('h3');title.append(link(r.title,r.study_url));heading.append(el('span',String(index+1).padStart(2,'0'),'rr-number'),title);card.append(heading);
      const tags=el('div',undefined,'rr-directions');for(const d of r.directions.slice(0,2))tags.append(el('span',directions[d]||d,'rr-direction'));card.append(tags);
      if(historical)card.append(el('p','历史记录，说明尚未完成原文核对。可重新生成。','rr-unverified'));
      else if(r.status!=='supported')card.append(el('p',data.generated?'取证尚不足，保留为阅读线索':'原文已整理，尚未生成阅读解释','rr-unverified'));
      if(!historical)for(const [field,titleText] of [['why_read','为什么值得读'],['differences','与当前论文的差异'],['focus','重点看哪里']])if(r[field].length){card.append(el('h4',titleText));for(const item of r[field])cited(item,card);}
      for(const gap of r.evidence_gaps)card.append(el('p',gap,'rr-gap'));
      if(!data.generated||historical){for(const e of data.evidence.filter(e=>e.paper_id===r.paper_id).slice(0,2)){const p=el('p','PDF '+e.pdf_page+' · '+e.text.slice(0,100)+'…');const b=el('button',e.citation_id,'rr-citation');b.type='button';b.onclick=()=>original(e.citation_id,card);p.append(b);card.append(p);}}
      const read=link('打开论文精读 ↗',r.study_url,'rr-source-link');card.append(read);cards.append(card);
    }
    setStatus(data.message||data.recommendations.length+' 篇对照阅读 · '+data.evidence.length+' 段原文');
    if(run)open.href='/reading?paper='+encodeURIComponent(paper)+'&run='+encodeURIComponent(run);
  }
  async function loadHistory(){try{const data=await api('/api/related-reading/jobs?paper_id='+encodeURIComponent(paper));history.replaceChildren();for(const j of data.jobs.slice(0,4)){const a=link(new Date(j.created_at).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})+' · '+(j.status==='completed'?'阅读建议':j.status==='failed'?'未完成':'处理中'),'/reading?paper='+encodeURIComponent(paper)+'&run='+j.run_id);history.append(a);}}catch{}}
  async function poll(){
    try{const job=await api('/api/related-reading/jobs/'+encodeURIComponent(run));
      if(job.paper_id!==paper)throw Error('这份记录属于另一篇论文。');
      if(job.status==='completed'){busy=false;start.disabled=false;status.setAttribute('aria-busy','false');render(job.result);loadHistory();return;}
      if(['failed','interrupted'].includes(job.status))throw Error(job.message||'任务未完成，可重新提交。');
      const p=job.progress||{};setStatus((labels[p.stage]||'正在处理')+(p.stage==='evidence'?' '+(p.completed+1)+' / '+p.total:''));
      if(p.question)status.append(el('div',p.question));setTimeout(poll,1800);
    }catch(error){busy=false;start.disabled=false;status.setAttribute('aria-busy','false');setStatus(error.message,true);}
  }
  form.onsubmit=async event=>{event.preventDefault();if(busy||goal.value.trim().length<2)return;busy=true;start.disabled=true;status.setAttribute('aria-busy','true');setStatus('正在开始关联阅读…');
    try{const job=await api('/api/related-reading',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({paper_id:paper,goal:goal.value.trim(),generate:!evidenceOnly.checked,limit:5})});run=job.run_id;
      if(!compact){const url=new URL(location.href);url.searchParams.set('run',run);window.history.replaceState({},'',url);}poll();
    }catch(error){busy=false;start.disabled=false;status.setAttribute('aria-busy','false');setStatus(error.message,true);}
  };
  async function init(){if(!paper){setStatus('请先选择一篇论文。');return;}try{const options=await api('/api/related-reading/papers/'+encodeURIComponent(paper));if(!options.available){setStatus('这篇论文尚未纳入关联阅读图，可先使用多论文检索。');form.hidden=true;return;}seed.textContent=options.paper.title;start.disabled=false;setStatus('关联阅读覆盖 '+options.papers+' 篇论文');if(run){busy=true;start.disabled=true;status.setAttribute('aria-busy','true');poll();}loadHistory();}catch(error){setStatus(error.message,true);}}
  init();
})();
