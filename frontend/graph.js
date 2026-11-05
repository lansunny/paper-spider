/* Paper-level similarity exploration. All relationships and descriptions come from the published graph. */
'use strict';
const $ = id => document.getElementById(id);
const DIMENSIONS = {
  topic: {label: '主题', description: '发现研究相近问题的论文，沿主题展开阅读。'},
  method: {label: '方法', description: '沿着工作流程与技术路线，寻找方法相近的论文。'},
  experiment: {label: '实验', description: '寻找数据集、指标与实验设置相关的论文。'},
  conclusion: {label: '结论', description: '比较主要发现、成立条件与局限之间的联系。'},
};
const FIELD_NAMES = {research_objects:'研究对象',problems:'核心问题',tasks:'研究任务',scenarios:'应用场景',
  core_idea:'核心思路',modules:'主要模块',training:'训练方式',inference:'推理流程',
  datasets:'数据集',baselines:'对比基线',metrics:'评估指标',settings:'实验设置',results:'实验结果',
  findings:'主要发现',conditions:'成立条件',limitations:'局限与适用范围'};
const mobile = matchMedia('(max-width:767px)');
const reducedMotion = matchMedia('(prefers-reduced-motion:reduce)');
const mapState = {summary:null,dimension:'topic',view:null,selected:null,compared:null,group:null,level:'groups',cardTitles:new Map(),
  search:'',threshold:0,views:new Map(),layouts:new Map(),papers:new Map(),loadRequest:0,detailRequest:0,
  nodes:[],edges:[],visibleNodes:[],visibleEdges:[],svg:null,world:null,zoom:null,nodeSelection:null,
  edgeSelection:null,hitSelection:null,groupSelection:null,hovered:null,viewport:{width:1000,height:740}};
let toastTimer, resizeTimer;
let reviewTrace=null;

async function loadReviewTrace(){
  const params=new URLSearchParams(location.search),id=params.get('review');
  document.getElementById('review-map-banner')?.remove();reviewTrace=null;
  if(!id)return;
  try{
    const job=await api('/api/reviews/jobs/'+encodeURIComponent(id));
    if(job.status!=='completed'||job.kind!=='review')throw Error('该综述尚未完成。');
    const result=job.result;
    if(result.graph_build_id!==mapState.summary.build_id)throw Error('此综述基于其他版本的论文图。');
    const route=params.get('route')?result.routes.find(r=>r.route_id===params.get('route')):null;
    const paperIds=route?route.paper_ids:result.map_trace.paper_ids;
    reviewTrace={papers:new Set(paperIds),run:id};
    const banner=element('div',null,'review-map-banner');banner.id='review-map-banner';
    banner.append(element('span',(route?route.name+' · ':'本次综述取证 · ')+paperIds.length+' 篇论文'),
      link('返回综述 ↗','/review?run='+encodeURIComponent(id)),button('清除高亮','quiet-button',()=>{
        reviewTrace=null;banner.remove();const url=new URL(location.href);url.searchParams.delete('review');url.searchParams.delete('route');history.replaceState({},'',url);updateFocus();renderPaperList();
      }));
    $('map-view').prepend(banner);
  }catch(error){toast(error.message);}
}

function element(tag,text,className){const e=document.createElement(tag);if(text!=null)e.textContent=String(text);if(className)e.className=className;return e;}
function icon(name){const e=document.createElementNS('http://www.w3.org/2000/svg','svg');e.setAttribute('class','icon');e.setAttribute('aria-hidden','true');const u=document.createElementNS(e.namespaceURI,'use');u.setAttribute('href','/static/assets/icons.svg#'+name);e.append(u);return e;}
function button(text,className,action){const e=element('button',text,className);e.type='button';if(action)e.onclick=action;return e;}
function link(text,url,className){const e=element('a',text,className);e.href=url;return e;}
function toast(text){clearTimeout(toastTimer);$('map-toast').textContent=text;$('map-toast').hidden=false;toastTimer=setTimeout(()=>$('map-toast').hidden=true,2600);}
function shortTitle(title,limit=30){const prefix=title.split(':')[0];if(prefix.length<=16&&prefix.length<title.length)return prefix;return title.length>limit?title.slice(0,limit).trim()+'…':title;}
function groupColor(id){return CommunityVisuals.theme(id).ink;}
function dot(color){const e=element('span',null,'group-dot');e.style.setProperty('--group-color',color);return e;}
function groupName(id){const group=mapState.view?.communities.find(c=>c.community_id===id);return mapState.cardTitles.get(id)||(group?DIMENSIONS[mapState.dimension].label+'群组 '+String(group.ordinal).padStart(2,'0'):'未分组');}
function paperUrl(id){return '/study?paper='+encodeURIComponent(id);}
function nodeById(id){return mapState.nodes.find(n=>n.paper_id===id);}
function endpointId(value){return typeof value==='string'?value:value.paper_id;}
function edgeBetween(a,b){return mapState.edges.find(e=>(endpointId(e.source)===a&&endpointId(e.target)===b)||(endpointId(e.source)===b&&endpointId(e.target)===a));}
function visibleNeighbors(id){return mapState.visibleEdges.filter(e=>endpointId(e.source)===id||endpointId(e.target)===id)
  .map(e=>({edge:e,node:nodeById(endpointId(e.source)===id?endpointId(e.target):endpointId(e.source))}))
  .sort((a,b)=>b.edge.weight-a.edge.weight||a.node.paper_id.localeCompare(b.node.paper_id));}

async function api(path){
  const response=await fetch(path,{cache:'no-store'});let value;
  try{value=await response.json();}catch{throw Error('论文图暂时无法读取，请稍后重试。');}
  if(!response.ok)throw Error(value.detail||'论文图暂时无法读取。');return value;
}
function getPaper(id){
  if(!mapState.papers.has(id))mapState.papers.set(id,api('/api/paper-graph/papers/'+encodeURIComponent(id)).catch(error=>{mapState.papers.delete(id);throw error;}));
  return mapState.papers.get(id);
}
function setMapStatus(message,retry=false){$('map-state-copy').textContent=message;$('map-state').hidden=false;$('retry-map').hidden=!retry;}
function readLocation(){const p=new URLSearchParams(location.search);return {dimension:Object.hasOwn(DIMENSIONS,p.get('dimension'))?p.get('dimension'):'topic',paper:p.get('paper'),compare:p.get('compare'),group:p.get('group'),level:p.get('level')==='papers'||p.has('paper')||p.has('group')||p.has('review')?'papers':'groups'};}
function saveLocation(push=false){
  const url=new URL(location.href);url.searchParams.set('dimension',mapState.dimension);
  if(mapState.selected)url.searchParams.set('paper',mapState.selected);else url.searchParams.delete('paper');
  if(mapState.compared)url.searchParams.set('compare',mapState.compared);else url.searchParams.delete('compare');
  if(mapState.group)url.searchParams.set('group',mapState.group);else url.searchParams.delete('group');
  if(mapState.level==='papers')url.searchParams.set('level','papers');else url.searchParams.delete('level');
  if(url.href!==location.href)window.history[push?'pushState':'replaceState']({},'',url);
}

async function loadDimension(dimension,{paper=mapState.selected,compare=null,group=null,level=mapState.level,record=true}={}){
  const request=++mapState.loadRequest;mapState.detailRequest++;mapState.hovered=null;$('map-tooltip').hidden=true;
  mapState.dimension=dimension;mapState.group=null;mapState.compared=null;
  for(const [name,info] of Object.entries(DIMENSIONS)){
    const selected=name===dimension,b=$('dimension-'+name);b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;
  }
  $('map-view').setAttribute('aria-labelledby','dimension-'+dimension);
  $('edge-threshold').disabled=true;setMapStatus('正在读取'+DIMENSIONS[dimension].label+'关系…');
  try{
    if(!mapState.views.has(dimension))mapState.views.set(dimension,await api('/api/paper-graph/views/'+dimension));
    if(request!==mapState.loadRequest)return;
    mapState.view=mapState.views.get(dimension);
    mapState.view.communities.sort((a,b)=>a.ordinal-b.ordinal);
    CommunityVisuals.register(mapState.view.communities.map(c=>c.community_id));
    mapState.level=level;mapState.group=mapState.view.communities.some(c=>c.community_id===group)?group:null;
    const min=mapState.view.edges.length?Math.floor(Math.min(...mapState.view.edges.map(e=>e.weight))*100)/100:0;
    $('edge-threshold').min=String(min);$('edge-threshold').max='1';$('edge-threshold').value=String(min);
    mapState.threshold=min;$('edge-threshold').disabled=!mapState.view.edges.length;
    mapState.selected=mapState.view.nodes.some(n=>n.paper_id===paper)?paper:null;
    createNetwork();updateFilters();browseTab(mapState.level==='groups'?'groups':'papers');
    if(compare&&mapState.selected&&edgeBetween(mapState.selected,compare))mapState.compared=compare;
    updateFocus();renderInspector();fitMap(false);$('map-state').hidden=mapState.nodes.length>0;
    if(!mapState.nodes.length)setMapStatus('这批论文还没有可用的'+DIMENSIONS[dimension].label+'表示。');
    if(record)saveLocation(true);
  }catch(error){if(request===mapState.loadRequest){setMapStatus(error.message,true);$('map-summary').textContent='读取失败';}}
}

function createNetwork(){
  const dimension=mapState.dimension,view=mapState.view;
  const nodes=view.nodes.map(n=>({...n,radius:5.5+Math.sqrt(n.degree)*.85}));
  const edges=view.edges.map(e=>({...e}));
  const cached=mapState.layouts.get(dimension);
  const groups=view.communities.length||1;
  const columns=Math.ceil(Math.sqrt(groups)),rows=Math.ceil(groups/columns);
  const centers=new Map(view.communities.map((g,i)=>{const row=Math.floor(i/columns),count=Math.min(columns,groups-row*columns),column=i%columns;
    return [g.community_id,{x:count===1?500:500+(column-(count-1)/2)*450/(columns-1||1),y:rows===1?365:230+row*300/(rows-1)}];}));
  const rng=d3.randomLcg(.314159+Object.keys(DIMENSIONS).indexOf(dimension)*.11);
  for(const node of nodes){const center=centers.get(node.community_id)||{x:500,y:365},saved=cached?.get(node.paper_id);node.x=saved?.x??center.x+(rng()-.5)*100;node.y=saved?.y??center.y+(rng()-.5)*100;}
  if(!cached){
    const simulation=d3.forceSimulation(nodes).randomSource(rng)
      .force('link',d3.forceLink(edges).id(n=>n.paper_id).distance(e=>48+(1-e.weight)*75).strength(.055))
      .force('charge',d3.forceManyBody().strength(-145))
      .force('collision',d3.forceCollide(n=>n.radius+13).iterations(2))
      .force('x',d3.forceX(n=>(centers.get(n.community_id)||{x:500}).x).strength(.17))
      .force('y',d3.forceY(n=>(centers.get(n.community_id)||{y:365}).y).strength(.17)).stop();
    simulation.tick(280);simulation.stop();
    mapState.layouts.set(dimension,new Map(nodes.map(n=>[n.paper_id,{x:n.x,y:n.y}])));
  }else{const byId=new Map(nodes.map(n=>[n.paper_id,n]));for(const edge of edges){edge.source=byId.get(edge.source);edge.target=byId.get(edge.target);}}
  mapState.nodes=nodes;mapState.edges=edges;
  const rectangle=$('map-canvas').getBoundingClientRect();
  mapState.viewport={width:1000,height:1000*rectangle.height/Math.max(1,rectangle.width)};
  const svg=d3.select('#network').attr('viewBox','0 0 1000 '+mapState.viewport.height);svg.selectAll('*').remove();
  const world=svg.append('g').attr('class','network-world');
  const paperWorld=world.append('g').attr('class','paper-network');mapState.paperLayer=paperWorld;
  mapState.svg=svg;mapState.world=world;
  mapState.zoom=d3.zoom().extent([[0,0],[1000,mapState.viewport.height]]).scaleExtent([.3,4.5]).clickDistance(5)
    .on('zoom',event=>{world.attr('transform',event.transform);updateNodeScale();$('map-tooltip').hidden=true;});
  svg.call(mapState.zoom).on('dblclick.zoom',null).on('click',event=>{if(event.target===$('network'))clearSelection();});
  mapState.groupSelection=paperWorld.append('g').attr('class','community-labels').selectAll('text').data(view.communities).join('text')
    .attr('class','community-label').attr('text-anchor','middle')
    .text(g=>DIMENSIONS[dimension].label+'组 '+String(g.ordinal).padStart(2,'0')+' · '+g.members.length+'篇');
  mapState.edgeSelection=paperWorld.append('g').selectAll('line').data(edges).join('line').attr('class','network-edge');
  mapState.hitSelection=paperWorld.append('g').selectAll('line').data(edges).join('line').attr('class','edge-hit')
    .attr('data-source',e=>e.source.paper_id).attr('data-target',e=>e.target.paper_id)
    .on('click',(event,e)=>{event.stopPropagation();comparePapers(e.source.paper_id,e.target.paper_id);})
    .on('pointerenter',(event,e)=>showTooltip(event,DIMENSIONS[dimension].label+'相似',e.weight.toFixed(3)+' · 点击对照两篇论文'))
    .on('pointerleave',()=>$('map-tooltip').hidden=true);
  const node=paperWorld.append('g').selectAll('g').data(nodes).join('g').attr('class','network-node')
    .attr('data-paper-id',n=>n.paper_id).attr('role','button').attr('tabindex',-1)
    .attr('aria-label',n=>n.title+'，'+n.degree+' 条'+DIMENSIONS[dimension].label+'相似联系')
    .on('click',(event,n)=>{if(event.defaultPrevented)return;event.stopPropagation();selectPaper(n.paper_id);})
    .on('keydown',(event,n)=>{if(['Enter',' '].includes(event.key)){event.preventDefault();selectPaper(n.paper_id);}})
    .on('pointerenter',(event,n)=>{mapState.hovered=n.paper_id;updateFocus();showTooltip(event,n.title,(n.year?n.year+' · ':'')+groupName(n.community_id)+' · '+visibleNeighbors(n.paper_id).length+' 条可见联系');})
    .on('pointerleave',()=>{mapState.hovered=null;updateFocus();$('map-tooltip').hidden=true;});
  node.append('circle').attr('class','node-ring').attr('r',n=>n.radius+4);
  node.append('circle').attr('class','node-core').attr('r',n=>n.radius).attr('fill',n=>groupColor(n.community_id));
  node.append('text').attr('class','node-label').attr('x',n=>n.radius+8).attr('dy','.35em').text(n=>shortTitle(n.title,32));
  node.call(d3.drag().clickDistance(5).on('start',()=>{$('map-tooltip').hidden=true;})
    .on('drag',(event,n)=>{n.x=event.x;n.y=event.y;mapState.layouts.get(dimension).set(n.paper_id,{x:n.x,y:n.y});positionNetwork();}));
  mapState.nodeSelection=node;positionNetwork();createCommunityOverview(centers);
}

function createCommunityOverview(centers){
  const layer=mapState.world.append('g').attr('class','community-overview');mapState.overviewLayer=layer;
  mapState.overviewGroups=mapState.view.communities.map(c=>({...c,...centers.get(c.community_id),radius:68+Math.sqrt(c.members.length)*7}));
  mapState.overviewLinks=layer.append('g').attr('class','community-overview-links');
  const groups=layer.append('g').selectAll('g').data(mapState.overviewGroups).join('g').attr('class','community-overview-group')
    .attr('data-community-id',g=>g.community_id).attr('transform',g=>'translate('+g.x+','+g.y+')').attr('role','button').attr('tabindex',0)
    .attr('aria-label',g=>groupName(g.community_id)+'，'+g.members.length+'篇论文，展开群组')
    .on('click',(event,g)=>{event.stopPropagation();openMapGroup(g.community_id);})
    .on('keydown',(event,g)=>{if(['Enter',' '].includes(event.key)){event.preventDefault();openMapGroup(g.community_id);}})
    .on('pointerenter',(event,g)=>showTooltip(event,groupName(g.community_id),g.members.length+' 篇论文'))
    .on('pointerleave',()=>$('map-tooltip').hidden=true);
  groups.each(function(g){
    const node=d3.select(this);this.append(CommunityVisuals.orbit(g.members,g.community_id));
    node.select('.community-orbit').attr('transform','scale('+(g.radius/82)+')');
    node.append('circle').attr('class','community-overview-focus').attr('r',g.radius+8).attr('stroke',groupColor(g.community_id));
    const name=groupName(g.community_id),parts=name.length>14?[name.slice(0,14),name.slice(14,28)+(name.length>28?'…':'')]:[name];
    const title=node.append('text').attr('class','community-overview-title').attr('y',g.radius+30);
    parts.forEach((text,i)=>title.append('tspan').attr('x',0).attr('dy',i?19:0).text(text));
  });
  mapState.overviewSelection=groups;
}

function updateCommunityOverview(){
  if(!mapState.overviewLayer)return;
  const grouped=mapState.level==='groups';mapState.paperLayer.style('display',grouped?'none':null);mapState.overviewLayer.style('display',grouped?null:'none');
  $('map-legend-kind').textContent=grouped?'群组':'论文';
  if(grouped)$('visible-count').textContent=mapState.overviewGroups.length+' 个群组 · '+mapState.nodes.length+' 篇论文';
  $('show-groups').setAttribute('aria-pressed',String(grouped));$('show-papers').setAttribute('aria-pressed',String(!grouped));
  const query=mapState.search.trim().toLowerCase();
  mapState.overviewSelection.attr('opacity',g=>!query||g.members.some(p=>{const n=nodeById(p);return [n?.title,p,n?.description].join(' ').toLowerCase().includes(query);})?1:.2);
  const totals=new Map(),byId=new Map(mapState.overviewGroups.map(g=>[g.community_id,g]));
  for(const edge of mapState.visibleEdges){const a=edge.source.community_id,b=edge.target.community_id;if(a===b)continue;const ids=[a,b].sort(),key=ids.join('|');if(!totals.has(key))totals.set(key,{source:byId.get(ids[0]),target:byId.get(ids[1]),count:0});totals.get(key).count++;}
  mapState.overviewLinks.selectAll('line').data([...totals.values()]).join('line')
    .attr('x1',e=>e.source.x).attr('y1',e=>e.source.y).attr('x2',e=>e.target.x).attr('y2',e=>e.target.y)
    .attr('stroke-width',e=>Math.min(4,1+Math.sqrt(e.count)*.22)).each(function(e){d3.select(this).selectAll('title').data([e]).join('title').text(d=>d.count+' 条跨组论文关联');});
}
function openMapGroup(id){
  mapState.hovered=null;$('map-tooltip').hidden=true;
  mapState.group=id;mapState.level='papers';mapState.selected=null;mapState.compared=null;
  updateFilters();browseTab('papers');renderInspector();fitMap();saveLocation();if(mobile.matches)$('map-sidebar').open=false;
}
function setMapLevel(level){
  mapState.hovered=null;$('map-tooltip').hidden=true;
  mapState.level=level;if(level==='groups'){mapState.group=null;mapState.selected=null;mapState.compared=null;}
  updateFilters();browseTab(level==='groups'?'groups':'papers');renderInspector();fitMap();saveLocation();
}

function positionNetwork(){
  for(const selection of [mapState.edgeSelection,mapState.hitSelection])selection
    .attr('x1',e=>e.source.x).attr('y1',e=>e.source.y).attr('x2',e=>e.target.x).attr('y2',e=>e.target.y);
  mapState.nodeSelection?.attr('transform',n=>'translate('+n.x+','+n.y+')');
  mapState.groupSelection?.attr('x',g=>d3.mean(mapState.nodes.filter(n=>n.community_id===g.community_id),n=>n.x))
    .attr('y',g=>d3.min(mapState.nodes.filter(n=>n.community_id===g.community_id),n=>n.y)-35);
}
function updateNodeScale(){
  if(!mapState.nodeSelection)return;
  const matrix=$('network').getScreenCTM();if(!matrix)return;
  const scale=Math.abs(matrix.a)*d3.zoomTransform($('network')).k;
  if(!scale)return;
  mapState.nodeSelection.select('.node-core').attr('r',n=>Math.max(n.radius,(mobile.matches?6.5:5.5)/scale));
  mapState.nodeSelection.select('.node-ring').attr('r',n=>Math.max(n.radius,(mobile.matches?6.5:5.5)/scale)+3/scale);
  mapState.nodeSelection.select('text').attr('x',n=>Math.max(n.radius,(mobile.matches?6.5:5.5)/scale)+6/scale)
    .style('font-size',11.5/scale+'px').style('stroke-width',3/scale+'px').text(n=>shortTitle(n.title,mobile.matches?18:32));
  mapState.groupSelection.style('font-size',11/scale+'px');
  mapState.overviewSelection?.each(function(g){
    const root=d3.select(this),factor=scale*g.radius/82;
    root.select('.community-orbit-count').style('font-size',Math.max(38,22/factor)+'px');
    root.select('.community-orbit-unit').style('font-size',Math.max(11,10.5/factor)+'px').attr('y',Math.max(23,20/factor));
    const name=groupName(g.community_id),limit=mobile.matches?14:24,parts=name.length>limit?[name.slice(0,limit),name.slice(limit,limit*2)+(name.length>limit*2?'…':'')]:[name];
    root.select('.community-overview-title').style('font-size',12/scale+'px').attr('y',g.radius+24/scale)
      .selectAll('tspan').data(parts).join('tspan').attr('x',0).attr('dy',(d,i)=>i?18/scale:0).text(d=>d);
  });
}
function showTooltip(event,title,subtitle){
  const target=$('map-tooltip');target.replaceChildren(element('div',title),element('small',subtitle));target.hidden=false;
  const r=$('map-view').getBoundingClientRect();target.style.left=Math.max(8,Math.min(event.clientX-r.left+14,r.width-target.offsetWidth-10))+'px';
  target.style.top=Math.max(8,Math.min(event.clientY-r.top+14,r.height-target.offsetHeight-10))+'px';
}
function fitMap(animate=true){
  if(mapState.level==='groups'&&mapState.overviewGroups?.length){
    const groups=mapState.overviewGroups,x0=d3.min(groups,g=>g.x-g.radius-55),x1=d3.max(groups,g=>g.x+g.radius+55),y0=d3.min(groups,g=>g.y-g.radius-35),y1=d3.max(groups,g=>g.y+g.radius+100);
    const {height}=mapState.viewport,unit=$('map-canvas').getBoundingClientRect().width/1000,bottom=(mobile.matches?105:36)/Math.max(unit,.1),usable=height-bottom,k=Math.min(2,900/(x1-x0),usable/(y1-y0));
    const transform=d3.zoomIdentity.translate(500-k*(x0+x1)/2,usable/2-k*(y0+y1)/2).scale(k);
    if(animate&&!reducedMotion.matches)mapState.svg.transition().duration(220).call(mapState.zoom.transform,transform);else mapState.svg.call(mapState.zoom.transform,transform);return;
  }
  const nodes=mapState.visibleNodes;if(!nodes.length||!mapState.svg)return;
  const x0=d3.min(nodes,n=>n.x-n.radius-40),x1=d3.max(nodes,n=>n.x+n.radius+120);
  const y0=d3.min(nodes,n=>n.y-n.radius-55),y1=d3.max(nodes,n=>n.y+n.radius+45);
  const height=mapState.viewport.height;
  const k=Math.min(2,.87/Math.max((x1-x0)/1000,(y1-y0)/height));
  const transform=d3.zoomIdentity.translate(500-k*(x0+x1)/2,height*.47-k*(y0+y1)/2).scale(k);
  if(animate&&!reducedMotion.matches)mapState.svg.transition().duration(220).call(mapState.zoom.transform,transform);
  else mapState.svg.call(mapState.zoom.transform,transform);
}
function updateFilters(){
  if(!mapState.view)return;
  const visible=mapState.nodes.filter(n=>!mapState.group||n.community_id===mapState.group),ids=new Set(visible.map(n=>n.paper_id));
  mapState.visibleNodes=visible;mapState.visibleEdges=mapState.edges.filter(e=>e.weight>=mapState.threshold&&ids.has(endpointId(e.source))&&ids.has(endpointId(e.target)));
  $('edge-threshold-value').textContent='≥'+mapState.threshold.toFixed(2);
  const label=DIMENSIONS[mapState.dimension].label;
  $('map-summary').textContent=mapState.summary.papers+' 篇论文 · '+mapState.view.communities.length+' 个'+label+'分组';
  $('visible-count').textContent=visible.length+' 篇论文 · '+mapState.visibleEdges.length+' 条可见联系';
  $('pilot-count').textContent=mapState.summary.papers+' 篇论文';$('mobile-paper-count').textContent=String(visible.length);
  $('group-count').textContent=String(mapState.view.communities.length);$('clear-group').hidden=!mapState.group;
  $('sidebar-scope').textContent=mapState.group?groupName(mapState.group):'全部论文';
  if(mapState.compared&&!mapState.visibleEdges.some(e=>new Set([endpointId(e.source),endpointId(e.target)]).has(mapState.selected)&&new Set([endpointId(e.source),endpointId(e.target)]).has(mapState.compared)))mapState.compared=null;
  renderPaperList();renderGroupList();updateFocus();
}
function updateFocus(){
  updateCommunityOverview();
  if(!mapState.nodeSelection)return;
  const selected=mapState.hovered||mapState.selected,neighbors=new Set(selected?visibleNeighbors(selected).map(n=>n.node.paper_id):[]);
  const visibleIds=new Set(mapState.visibleNodes.map(n=>n.paper_id)),visibleEdges=new Set(mapState.visibleEdges);
  const search=mapState.search.trim().toLowerCase();
  const matches=n=>!search||[n.title,n.paper_id,n.description].join(' ').toLowerCase().includes(search);
  const representatives=new Set(mapState.view.communities.flatMap(c=>c.representative_paper_ids.slice(0,1)));
  const closeLabels=new Set(selected?visibleNeighbors(selected).slice(0,mobile.matches?1:3).map(n=>n.node.paper_id):[]);
  mapState.nodeSelection.style('display',n=>visibleIds.has(n.paper_id)?null:'none')
    .classed('is-review-paper',n=>Boolean(reviewTrace?.papers.has(n.paper_id)))
    .classed('is-review-muted',n=>Boolean(reviewTrace&&!reviewTrace.papers.has(n.paper_id)))
    .classed('is-selected',n=>n.paper_id===mapState.selected||n.paper_id===mapState.compared)
    .classed('is-neighbor',n=>neighbors.has(n.paper_id))
    .classed('is-dim',n=>(Boolean(selected)&&n.paper_id!==selected&&!neighbors.has(n.paper_id)&&n.paper_id!==mapState.compared)||!matches(n))
    .attr('tabindex',n=>n.paper_id===mapState.selected?0:-1);
  mapState.nodeSelection.select('text').style('display',n=>
    reviewTrace?.papers.has(n.paper_id)||
    n.paper_id===mapState.selected||n.paper_id===mapState.hovered||n.paper_id===mapState.compared||
    (representatives.has(n.paper_id)&&(!mobile.matches||!selected))||closeLabels.has(n.paper_id)||(search&&matches(n))?null:'none');
  mapState.edgeSelection.style('display',e=>visibleEdges.has(e)?null:'none')
    .classed('is-dim',e=>Boolean(selected)&&endpointId(e.source)!==selected&&endpointId(e.target)!==selected)
    .classed('is-active',e=>Boolean(selected)&&(endpointId(e.source)===selected||endpointId(e.target)===selected))
    .classed('is-compared',e=>Boolean(mapState.compared)&&[endpointId(e.source),endpointId(e.target)].includes(mapState.selected)&&[endpointId(e.source),endpointId(e.target)].includes(mapState.compared));
  mapState.hitSelection.style('display',e=>visibleEdges.has(e)&&(!selected||endpointId(e.source)===selected||endpointId(e.target)===selected)?null:'none');
  mapState.groupSelection.style('display',g=>!mapState.group||g.community_id===mapState.group?null:'none').attr('opacity',selected ? .5 : 1);
  $('mobile-selection').hidden=!mapState.selected;
  if(mapState.selected)$('mobile-selection').textContent=shortTitle(nodeById(mapState.selected)?.title||'',16)+' · 详情 ↗';
}

function renderPaperList(){
  const target=$('paper-list');target.replaceChildren();const query=mapState.search.trim().toLowerCase();
  const papers=mapState.visibleNodes.filter(n=>!query||[n.title,n.paper_id,n.description].join(' ').toLowerCase().includes(query))
    .sort((a,b)=>b.degree-a.degree||a.title.localeCompare(b.title));
  $('paper-count').textContent=String(papers.length);
  for(const paper of papers){
    const row=button(null,'map-paper-row',()=>{selectPaper(paper.paper_id);if(mobile.matches)$('map-sidebar').open=false;});
    if(reviewTrace?.papers.has(paper.paper_id))row.classList.add('is-review-paper');
    row.dataset.paperId=paper.paper_id;row.setAttribute('aria-pressed',String(paper.paper_id===mapState.selected));
    const meta=element('span',null,'paper-row-meta');meta.append(dot(groupColor(paper.community_id)),element('span',paper.year||'年份未知'),element('span',paper.degree+' 条联系'));
    row.append(element('span',paper.title,'paper-row-title'),meta);target.append(row);
  }
  if(!papers.length)target.append(element('p',query?'没有找到匹配的论文，试试其他标题或关键词。':'当前分组没有可用论文。','map-list-empty'));
}
function renderGroupList(){
  const target=$('group-list');target.replaceChildren();
  for(const group of mapState.view.communities){
    const row=CommunityVisuals.paint(button(null,'map-group-row visual-group-row',()=>openMapGroup(group.community_id)),group.community_id);
    row.setAttribute('aria-pressed',String(mapState.group===group.community_id));
    const copy=element('span',null,'visual-group-copy');copy.append(element('strong',groupName(group.community_id)),element('small',group.members.length+' 篇论文'));
    row.append(CommunityVisuals.emblem(group.members,group.community_id),copy);target.append(row);
  }
  if(!mapState.view.communities.length)target.append(element('p','当前视角还没有形成分组。','map-list-empty'));
}
function selectPaper(id){
  if(!nodeById(id))return;
  const wasGroups=mapState.level==='groups';mapState.level='papers';
  if(mapState.group&&nodeById(id).community_id!==mapState.group){mapState.group=null;updateFilters();}
  mapState.selected=id;mapState.compared=null;updateFocus();renderPaperList();renderInspector();saveLocation();if(wasGroups){browseTab('papers');fitMap();}
}
function comparePapers(a,b){
  if(!edgeBetween(a,b))return;
  mapState.selected=a;mapState.compared=b;updateFocus();renderPaperList();renderInspector();saveLocation();
}
function clearSelection(){mapState.selected=null;mapState.compared=null;mapState.hovered=null;updateFocus();renderPaperList();renderInspector();saveLocation();}

function readLink(paper,label='进入精读'){
  const e=link(label,paperUrl(paper.paper_id),'read-paper');e.append(icon('up-right'));return e;
}
function relatedSearchLink(paper){
  const dimension=mapState.dimension,label=DIMENSIONS[dimension].label;
  const question='有哪些论文与《'+paper.title+'》在'+label+'上相关？请根据各自原文比较，并给出证据。';
  return link('图辅助检索 ↗','/?mode=search&graph='+encodeURIComponent(dimension)+'&question='+encodeURIComponent(question),'text-button');
}
function sourceDetails(dimension){
  const section=element('section',null,'inspector-section');section.append(element('h3','原文依据'));
  if(!dimension.claims.length){section.append(element('p','当前维度没有足够的原文依据。','dimension-copy'));return section;}
  const fields=new Map();for(const claim of dimension.claims){if(!fields.has(claim.field))fields.set(claim.field,[]);fields.get(claim.field).push(claim);}
  for(const [field,claims] of fields){
    const details=element('details',null,'claim-details'),summary=element('summary');summary.append(element('span',FIELD_NAMES[field]||field),icon('chevron'));
    details.append(summary);for(const claim of claims)details.append(element('p',claim.text,'claim-content'));
    const seen=new Set();for(const source of claims.flatMap(claim=>claim.sources||[])){
      if(seen.has(source.source_id))continue;seen.add(source.source_id);
      const card=element('div',null,'source-card'),heading=element('div',null,'source-heading');
      const jump=link('定位原文',source.study_url,'source-link');jump.append(icon('up-right'));
      heading.append(element('span','PDF 第 '+source.pdf_page+' 页'),jump);
      const quote=element('blockquote',source.text);card.append(heading,quote,
        element('p','原文范围 '+source.source_start+'–'+source.source_end+' · 已核对文档与文本','source-coordinates'));
      details.append(card);
    }
    section.append(details);
  }
  return section;
}
function overviewInspector(target){
  const selected=mapState.view?.communities.find(c=>c.community_id===mapState.group);
  if(selected){
    const badge=CommunityVisuals.paint(element('div',null,'selected-community-heading'),selected.community_id);badge.append(CommunityVisuals.emblem(selected.members,selected.community_id));target.append(badge,element('h2',groupName(selected.community_id),'inspector-title'),element('p',selected.members.length+' 篇论文','inspector-meta'));
    if(mapState.dimension==='topic')target.append(link('打开研究卡组 ↗','/review?community='+encodeURIComponent(selected.community_id),'text-button'));
    const members=element('section',null,'inspector-section');members.append(element('h3','组内论文'));
    for(const pid of selected.members){const paper=nodeById(pid);members.append(button(paper?.title||pid,'neighbor-row',()=>selectPaper(pid)));}target.append(members);return;
  }
  target.append(element('h2','研究群组','inspector-title'));
  const section=element('section',null,'inspector-section');
  for(const group of mapState.view?.communities||[]){const row=button(null,'neighbor-row',()=>openMapGroup(group.community_id));
    const copy=element('span',null,'overview-group-copy');const title=element('span',null,'overview-group-title');title.append(dot(groupColor(group.community_id)),element('span',groupName(group.community_id)));
    copy.append(title,element('span',nodeById(group.representative_paper_ids[0])?.title||'','overview-group-paper'));row.append(copy,element('span',group.members.length+' 篇','similarity-value'));section.append(row);}
  target.append(section);
}
async function renderInspector(){
  const request=++mapState.detailRequest,target=$('inspector-content');target.replaceChildren();
  if(!mapState.selected){overviewInspector(target);return;}
  const id=mapState.selected,compared=mapState.compared,dimension=mapState.dimension;
  const known=nodeById(id);target.append(element('div','正在读取论文与依据…','inspector-kicker'),element('h2',known.title,'inspector-title'));
  try{
    const papers=await Promise.all([getPaper(id),...(compared?[getPaper(compared)]:[])]);
    if(request!==mapState.detailRequest)return;target.replaceChildren();
    if(compared){renderComparison(target,papers[0],papers[1],dimension);return;}
    const paper=papers[0],value=paper.dimensions[dimension];
    const kicker=element('div',null,'inspector-kicker');kicker.append(element('span','已选论文'),element('span',groupName(known.community_id)));
    target.append(kicker,element('h2',paper.title,'inspector-title'),element('p',[paper.year,id].filter(Boolean).join(' · '),'inspector-meta'));
    const actions=element('div',null,'inspector-actions');actions.append(readLink(paper),relatedSearchLink(paper),button('浏览全图','text-button',clearSelection));target.append(actions);
    const section=element('section',null,'inspector-section');section.append(element('h3',DIMENSIONS[dimension].label+'概览'),
      element('p',value.status==='available'?value.description:value.reason,'dimension-copy'));target.append(section);
    const neighbors=visibleNeighbors(id),related=element('section',null,'inspector-section'),heading=element('div',null,'neighbor-heading');
    heading.append(element('h3',DIMENSIONS[dimension].label+'相近的论文'),element('span',neighbors.length+' 篇'));
    related.append(heading);
    for(const neighbor of neighbors.slice(0,6)){
      const row=button(null,'neighbor-row',()=>comparePapers(id,neighbor.node.paper_id));row.dataset.neighborId=neighbor.node.paper_id;
      row.setAttribute('aria-label','对照 '+neighbor.node.title+'，相似度 '+neighbor.edge.weight.toFixed(3));
      row.append(element('span',neighbor.node.title,'neighbor-title'),element('span',neighbor.edge.weight.toFixed(3),'similarity-value'));related.append(row);
    }
    if(!neighbors.length)related.append(element('p','当前筛选条件下没有可见联系，可以降低连线相似度或显示全部分组。','dimension-copy'));
    if(neighbors.length>6){const more=button('显示全部 '+neighbors.length+' 篇','text-button',()=>{for(const n of neighbors.slice(6)){const row=button(null,'neighbor-row',()=>comparePapers(id,n.node.paper_id));row.append(element('span',n.node.title,'neighbor-title'),element('span',n.edge.weight.toFixed(3),'similarity-value'));related.insertBefore(row,more);}more.remove();});related.append(more);}
    target.append(related,sourceDetails(value));
  }catch(error){if(request!==mapState.detailRequest)return;const box=element('div',error.message,'inspector-error');box.append(button('重新读取','secondary-button',()=>{mapState.papers.delete(id);if(compared)mapState.papers.delete(compared);renderInspector();}));target.append(box);}
}
function renderComparison(target,a,b,dimension){
  const edge=edgeBetween(a.paper_id,b.paper_id),back=button(null,'relation-back',()=>selectPaper(a.paper_id));back.append(icon('arrow'),element('span','返回论文详情'));
  const score=element('div',null,'relation-score');score.append(element('strong',edge.weight.toFixed(3)),element('span','余弦相似度'));
  const pair=element('div',null,'relation-pair');
  for(const paper of [a,b]){const name=button(null,'relation-pair-paper',()=>selectPaper(paper.paper_id));name.append(dot(groupColor(nodeById(paper.paper_id)?.community_id)),element('span',paper.title));pair.append(name);}
  target.append(back,element('h2',DIMENSIONS[dimension].label+'内容相近','inspector-title'),score,pair);
  for(const paper of [a,b]){
    const value=paper.dimensions[dimension],section=element('section',null,'relation-side');
    section.append(element('h3',paper.title),element('p',value.description||value.reason,'dimension-copy'));
    const actions=element('div',null,'inspector-actions');actions.append(readLink(paper),button('展开此论文','text-button',()=>selectPaper(paper.paper_id)));
    section.append(actions,sourceDetails(value));target.append(section);
  }
  if(dimension==='experiment')target.append(element('p','实验相关不等于结果可直接比较，仍需核对数据集、指标与设置。','relation-note'));
}

function browseTab(name){
  for(const tab of ['papers','groups']){const selected=tab===name;$('browse-'+tab).setAttribute('aria-selected',String(selected));$('browse-'+tab).tabIndex=selected?0:-1;$(tab==='papers'?'paper-list':'group-list').hidden=!selected;}
}
function resetMap(){mapState.group=null;mapState.level='groups';mapState.search='';$('map-search').value='';mapState.threshold=Number($('edge-threshold').min);$('edge-threshold').value=String(mapState.threshold);mapState.selected=null;mapState.compared=null;updateFilters();browseTab('groups');renderInspector();fitMap();saveLocation();}
function responsiveSidebar(){ $('map-sidebar').open=!mobile.matches; }
$('browse-papers').onclick=()=>browseTab('papers');$('browse-groups').onclick=()=>browseTab('groups');
document.querySelector('.sidebar-tabs').addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();const name=$('browse-papers').getAttribute('aria-selected')==='true'?'groups':'papers';browseTab(name);$('browse-'+name).focus();}});
for(const tab of document.querySelectorAll('[data-dimension]'))tab.onclick=()=>{if(mapState.dimension!==tab.dataset.dimension)loadDimension(tab.dataset.dimension);};
$('dimension-tabs').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const names=Object.keys(DIMENSIONS),i=names.indexOf(mapState.dimension),next=event.key==='Home'?0:event.key==='End'?3:(i+(event.key==='ArrowRight'?1:3))%4;$('dimension-'+names[next]).focus();loadDimension(names[next]);});
$('map-search').oninput=event=>{mapState.search=event.target.value;renderPaperList();updateFocus();};
$('map-search').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();$('paper-list').querySelector('.map-paper-row')?.click();}};
$('edge-threshold').oninput=event=>{mapState.threshold=Number(event.target.value);updateFilters();renderInspector();saveLocation();};
$('clear-group').onclick=()=>setMapLevel('groups');
$('show-groups').onclick=()=>setMapLevel('groups');$('show-papers').onclick=()=>setMapLevel('papers');
$('zoom-in').onclick=()=>{if(mapState.zoom)mapState.svg.call(mapState.zoom.scaleBy,1.3);};
$('zoom-out').onclick=()=>{if(mapState.zoom)mapState.svg.call(mapState.zoom.scaleBy,1/1.3);};
$('fit-map').onclick=()=>fitMap();$('clear-focus').onclick=clearSelection;$('reset-map').onclick=resetMap;
$('mobile-selection').onclick=()=>$('map-inspector').scrollIntoView({behavior:reducedMotion.matches?'auto':'smooth',block:'start'});
$('copy-map-link').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);toast('当前视图链接已复制');}catch{toast('暂时无法复制，请复制浏览器地址。');}};
$('retry-map').onclick=()=>{mapState.views.delete(mapState.dimension);init();};
document.addEventListener('keydown',event=>{if(event.key==='/'&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();if(mobile.matches)$('map-sidebar').open=true;$('map-search').focus();}if(event.key==='Escape'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName))clearSelection();});
mobile.addEventListener('change',responsiveSidebar);window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{
  if(!mapState.svg)return;const rectangle=$('map-canvas').getBoundingClientRect();
  mapState.viewport={width:1000,height:1000*rectangle.height/Math.max(1,rectangle.width)};
  mapState.svg.attr('viewBox','0 0 1000 '+mapState.viewport.height);mapState.zoom.extent([[0,0],[1000,mapState.viewport.height]]);
  fitMap(false);updateFocus();
},150);});
window.addEventListener('popstate',()=>{const value=readLocation();loadDimension(value.dimension,{...value,record:false});});
async function init(){
  setMapStatus('正在读取论文图…');
  try{
    if(typeof d3==='undefined')throw Error('地图组件未能加载，请刷新页面。');
    const [summary,cards]=await Promise.all([api('/api/paper-graph'),api('/api/reviews/cards').catch(()=>null)]);
    if(summary.status!=='completed'){setMapStatus('暂无论文地图',true);$('map-summary').textContent='暂无论文图';return;}
    if(mapState.summary?.build_id!==summary.build_id){mapState.views.clear();mapState.layouts.clear();mapState.papers.clear();}
    mapState.cardTitles=new Map((cards?.cards||[]).map(c=>[c.topic_community_id,c.title]));
    mapState.summary=summary;await loadReviewTrace();const value=readLocation();
    await loadDimension(value.dimension,{...value,record:false});saveLocation();
    document.body.dataset.ready='true';
  }catch(error){setMapStatus(error.message,true);$('map-summary').textContent='连接失败';}
}
responsiveSidebar();init();
