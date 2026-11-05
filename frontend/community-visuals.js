/* Shared community identity. Ring dots represent actual member papers. */
(() => {
  'use strict';
  const ns='http://www.w3.org/2000/svg';
  const palette=[
    {ink:'#426e99',wash:'#edf3f9',line:'#c6d6e7'},
    {ink:'#527c6b',wash:'#edf5f0',line:'#cbded2'},
    {ink:'#976e43',wash:'#faf3e9',line:'#e7d5bc'},
    {ink:'#69758d',wash:'#eff1f7',line:'#d1d7e5'},
    {ink:'#826c63',wash:'#f6f1ed',line:'#dfd0c7'},
  ];
  const identities=new Map();
  function register(ids){[...new Set(ids)].sort().forEach((id,i)=>identities.set(id,palette[i%palette.length]));}
  function theme(id){return identities.get(id)||palette[0];}
  function svg(tag,attrs={}){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));return e;}
  function paint(element,id){const t=theme(id);element.style.setProperty('--group-ink',t.ink);element.style.setProperty('--group-wash',t.wash);element.style.setProperty('--group-line',t.line);return element;}
  function orbit(members,id,{compact=false}={}){
    const t=theme(id),g=svg('g',{class:'community-orbit'}),r=compact?68:72;
    g.append(svg('circle',{r:r+10,fill:t.wash,stroke:t.line,'stroke-width':1}),svg('circle',{r:r-6,fill:'#fff',stroke:t.line,'stroke-width':1}));
    const phase=compact?-.3:-.5;
    members.forEach((paper,i)=>{
      const angle=phase-Math.PI/2+i*Math.PI*2/Math.max(members.length,1),x=Math.cos(angle)*(r+10),y=Math.sin(angle)*(r+10);
      const p=svg('circle',{cx:x,cy:y,r:members.length>24?2.8:3.6,fill:t.ink,class:'community-member-dot','data-paper-id':typeof paper==='string'?paper:paper.paper_id});
      g.append(p);
    });
    const number=svg('text',{class:'community-orbit-count',y:0,'text-anchor':'middle',fill:t.ink});number.textContent=String(members.length);
    const label=svg('text',{class:'community-orbit-unit',y:23,'text-anchor':'middle',fill:t.ink});label.textContent='篇论文';g.append(number,label);
    return g;
  }
  function emblem(members,id){const root=svg('svg',{viewBox:'-100 -100 200 200',class:'community-emblem','aria-hidden':'true'});root.append(orbit(members,id));root.querySelector('.community-orbit-count').setAttribute('y','-4');root.querySelector('.community-orbit-unit').setAttribute('y','31');return root;}
  window.CommunityVisuals={register,theme,paint,orbit,emblem};
})();
