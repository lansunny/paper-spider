import {getDocument,GlobalWorkerOptions,TextLayer,AnnotationMode} from './assets/pdfjs/build/pdf.min.mjs';
import {clamp,kindNames,svg,relativePoint,normalizedRect,selectionGeometry,paintAnnotation,hitAnnotation,annotationPayload} from './study-annotations.js';

GlobalWorkerOptions.workerSrc='/static/assets/pdfjs/build/pdf.worker.min.mjs';
const $=id=>document.getElementById(id),params=new URLSearchParams(location.search);
const paperId=params.get('paper')||'',base='/api/study/papers/'+encodeURIComponent(paperId);
const state={metadata:null,pdf:null,page:1,zoom:'fit',scroll:0,scale:1,annotations:[],bookmarks:[],notesRevision:0,
  tool:'select',color:'#f6d96b',selected:null,undo:[],redo:[],selection:null,notesDirty:false,notesConflict:false,notesEpoch:0,
  renderVersion:0,renderTask:null,textTask:null,rendering:false,chatBusy:false,failed:new Map(),pending:0,
  searchVersion:0,searchText:new Map(),searchHits:[],draftAnnotation:null};
let persistence=Promise.resolve(),notesTimer,progressTimer,selectionTimer,searchTimer,resizeTimer,toastTimer,gesture=null,thumbnailObserver;
const operationVersions=new Map();
const leftMarker=document.createComment('study-navigation'),rightMarker=document.createComment('study-inspector');
$('study-sidebar').before(leftMarker);$('study-inspector').before(rightMarker);
const leftMedia=matchMedia('(max-width:799px)'),rightMedia=matchMedia('(max-width:1179px)');
const layout=document.querySelector('.study-layout');

function element(tag,text,className){const el=document.createElement(tag);if(text!==undefined&&text!==null)el.textContent=text;if(className)el.className=className;return el;}
function action(text,fn,className=''){const b=element('button',text,className);b.type='button';b.addEventListener('click',fn);return b;}
function toast(message){clearTimeout(toastTimer);$('study-toast').textContent=message;$('study-toast').hidden=false;toastTimer=setTimeout(()=>$('study-toast').hidden=true,3500);}
function draftKey(){return 'paper-spider.study.draft.'+paperId+'.'+state.metadata.document_sha256;}
function storeDraft(){try{localStorage.setItem(draftKey(),JSON.stringify({notes:$('paper-notes').value,notes_revision:state.notesRevision}));}catch{}}
function removeDraft(){try{localStorage.removeItem(draftKey());}catch{}}
async function request(url,options={}){
  const response=await fetch(url,{...options,headers:{...(options.body?{'Content-Type':'application/json'}:{}),...options.headers}});
  if(!response.ok){let detail;try{detail=(await response.json()).detail;}catch{}const message=typeof detail==='string'?detail:detail?.message||'请求未完成，请稍后重试。';const error=new Error(message);error.status=response.status;error.detail=detail;throw error;}
  return response.json();
}
function updateSaveStatus(){
  const failed=state.failed.size>0;
  $('save-status').textContent=failed?'有更改未保存':state.pending||state.notesDirty?'正在保存…':'已保存至本地';
  $('save-status').classList.toggle('is-error',failed);$('retry-save').hidden=!failed;
}
const retry=action('重试保存',()=>{for(const [key,task] of [...state.failed])if(key!=='notes'||!state.notesConflict)enqueue(key,task);},'text-button');retry.id='retry-save';retry.hidden=true;$('save-status').after(retry);
function enqueue(key,task){
  const version=(operationVersions.get(key)||0)+1;operationVersions.set(key,version);state.pending++;updateSaveStatus();
  const operation=persistence.then(task).then(value=>{if(operationVersions.get(key)===version)state.failed.delete(key);return value;})
    .catch(error=>{if(operationVersions.get(key)===version)state.failed.set(key,task);toast(error.message);throw error;})
    .finally(()=>{state.pending--;updateSaveStatus();});
  persistence=operation.catch(()=>{});return operation;
}
function saveState(patch,key='progress'){
  if(!state.metadata)return Promise.resolve();
  const notesEpoch=state.notesEpoch;
  return enqueue(key,async()=>{
    const payload={document_sha256:state.metadata.document_sha256,...patch};
    if(patch.notes!==undefined){
      if(notesEpoch!==state.notesEpoch)return;
      if(state.notesConflict)throw Error('请先处理笔记版本冲突，本地草稿已保留。');
      payload.notes_revision=state.notesRevision;
    }
    try{
      const saved=await request(base+'/state',{method:'PATCH',body:JSON.stringify(payload)});
      if(patch.notes!==undefined){state.notesRevision=saved.notes_revision;if($('paper-notes').value===patch.notes){state.notesDirty=false;removeDraft();$('notes-save-status').textContent='已保存';}else storeDraft();}
      return saved;
    }catch(error){
      if(key==='notes'){$('notes-save-status').textContent='草稿尚未保存';$('notes-save-status').classList.add('is-error');
        if(error.status===409&&error.detail?.notes_revision!==undefined){state.notesConflict=true;$('notes-conflict').hidden=false;}}
      throw error;
    }
  });
}
function flushNotes(){
  clearTimeout(notesTimer);if(!state.notesDirty||state.notesConflict)return Promise.resolve();
  $('notes-save-status').classList.remove('is-error');return saveState({notes:$('paper-notes').value},'notes');
}
function notesChanged(){
  state.notesDirty=true;storeDraft();$('notes-count').textContent=$('paper-notes').value.length.toLocaleString()+' 字';
  $('notes-save-status').textContent='正在保存…';updateSaveStatus();clearTimeout(notesTimer);notesTimer=setTimeout(()=>flushNotes().catch(()=>{}),650);
}
function appendNote(quote,text='',page=state.page){
  const excerpt=quote?'\n'+quote.split('\n').map(line=>'> '+line).join('\n')+'\n':'';
  $('paper-notes').value+=($('paper-notes').value?'\n\n':'')+'### PDF 第 '+page+' 页\n'+excerpt+(text?'\n'+text+'\n':'');
  notesChanged();setPanel('notes');openPanel('right');$('paper-notes').focus();$('paper-notes').scrollTop=$('paper-notes').scrollHeight;
}
function makeAnnotation(kind,fields={}){return {id:crypto.randomUUID(),kind,page:state.page,color:state.color,rects:[],points:[],width:Number($('pen-width').value),quote:'',text:'',...fields};}
function persistAnnotation(a){return enqueue('annotation:'+a.id,async()=>{
  if(a.deleted)return request(base+'/annotations/'+a.id+'?document_sha256='+state.metadata.document_sha256,{method:'DELETE'});
  return request(base+'/annotations/'+a.id,{method:'PUT',body:JSON.stringify({document_sha256:state.metadata.document_sha256,annotation:annotationPayload(a)})});
});}
function applyAnnotation(before,after,record=true){
  const id=(after||before).id;state.annotations=state.annotations.filter(a=>a.id!==id);if(after)state.annotations.push({...after});
  if(record){state.undo.push({before:before?{...before}:null,after:after?{...after}:null});state.redo=[];}
  state.selected=after?.id||null;renderAnnotations();renderAnnotationList();updateUndo();
  persistAnnotation(after||{...before,deleted:true}).catch(()=>{});
}
function updateUndo(){$('annotation-undo').disabled=!state.undo.length;$('annotation-redo').disabled=!state.redo.length;}
function undo(){const command=state.undo.pop();if(!command)return;applyAnnotation(command.after,command.before,false);state.redo.push(command);updateUndo();}
function redo(){const command=state.redo.pop();if(!command)return;applyAnnotation(command.before,command.after,false);state.undo.push(command);updateUndo();}
function setTool(tool){
  if(gesture){if($('gesture-layer').hasPointerCapture(gesture.pointer))$('gesture-layer').releasePointerCapture(gesture.pointer);gesture=null;$('gesture-layer').replaceChildren();}
  state.tool=tool;$('pdf-page').dataset.tool=tool;
  for(const b of document.querySelectorAll('[data-tool]')){const selected=b.dataset.tool===tool;b.classList.toggle('is-active',selected);b.setAttribute('aria-pressed',String(selected));}
  if(['ink','rectangle','note'].includes(tool)&&state.color==='#f6d96b')setColor('#345a89');
  if(tool==='highlight'&&state.color==='#345a89')setColor('#f6d96b');
  const hints={select:'',highlight:'拖选文字，即可高亮。',underline:'拖选文字，为原文加下划线。',ink:'在原文上自由书写。',rectangle:'拖动圈出图表、公式或重要区域。',note:'点击原文位置，添加一条批注。',eraser:'点击或划过，擦除整条标注。'};
  $('tool-hint').textContent=hints[tool];hideSelection();
}
function setColor(color){state.color=color;for(const b of document.querySelectorAll('[data-color]')){const yes=b.dataset.color===color;b.classList.toggle('is-active',yes);b.setAttribute('aria-pressed',String(yes));}}
function currentAnnotations(){return state.annotations.filter(a=>a.page===state.page);}
function renderAnnotations(){const layer=$('annotation-layer');layer.replaceChildren();for(const a of currentAnnotations())paintAnnotation(layer,a,state.scale,a.id===state.selected);}
function selectAnnotation(a){
  state.selected=a.id;navigate(a.page).then(()=>{
    const r=a.rects[0]||[a.points[0][0],a.points[0][1],0,0];
    $('document-viewport').scrollTop=Math.max(0,r[1]*$('pdf-page').offsetHeight-100);renderAnnotations();
    document.querySelectorAll('.annotation-card').forEach(el=>el.classList.toggle('is-selected',el.dataset.annotationId===a.id));
  });
}
function renderAnnotationList(){
  $('annotation-count').textContent=state.annotations.length;const list=$('annotation-list');list.replaceChildren();
  const filter=$('annotation-filter').value;
  const annotations=state.annotations.filter(a=>filter==='all'||(filter==='page'?a.page===state.page:a.kind===filter)).sort((a,b)=>a.page-b.page);
  if(!annotations.length){list.append(element('p',filter==='all'?'标记原文中的关键之处。高亮、画笔和批注都会保存在这里。':'当前筛选下没有标注。','empty-copy'));return;}
  for(const a of annotations){
    const card=element('article',null,'annotation-card');card.dataset.annotationId=a.id;card.style.setProperty('--mark-color',a.color);card.classList.toggle('is-selected',a.id===state.selected);
    const head=element('div',null,'annotation-card-head');head.append(element('span',null,'annotation-dot'),action('PDF '+a.page,()=>selectAnnotation(a)),element('span',kindNames[a.kind]));card.append(head);
    if(a.quote)card.append(element('blockquote',a.quote));if(a.text)card.append(element('p',a.text,'annotation-comment'));
    if(!a.quote&&!a.text)card.append(element('p',a.kind==='ink'?'原文上的手写标记':'已框选原文区域','annotation-comment'));
    const buttons=element('div',null,'annotation-card-actions');buttons.append(action('记入笔记',()=>appendNote(a.quote,a.text||'['+kindNames[a.kind]+']',a.page)),action('编辑批注',()=>editComment(a)),action('删除',()=>applyAnnotation(a,null)));
    card.append(buttons);list.append(card);
  }
}
function hideSelection(){$('selection-menu').hidden=true;}
function readSelection(){
  if(!state.pdf||!['select','highlight','underline'].includes(state.tool))return;
  const selected=selectionGeometry($('pdf-text'),$('pdf-page'));if(!selected){hideSelection();return;}
  state.selection={...selected,page:state.page};
  if(state.tool!=='select'){applyAnnotation(null,makeAnnotation(state.tool,selected));window.getSelection().removeAllRanges();state.selection=null;return;}
  const menu=$('selection-menu');menu.hidden=false;
  const bounds=selected.bounds,left=clamp(bounds.left+bounds.width/2-menu.offsetWidth/2,8,innerWidth-menu.offsetWidth-8);
  let top=bounds.top-menu.offsetHeight-9;if(top<65)top=bounds.bottom+9;
  menu.style.left=left+'px';menu.style.top=clamp(top,65,innerHeight-menu.offsetHeight-8)+'px';
}
function selectionAnnotation(kind){const s=state.selection;if(!s||s.page!==state.page)return;applyAnnotation(null,makeAnnotation(kind,{rects:s.rects,quote:s.quote}));hideSelection();window.getSelection().removeAllRanges();}
function editComment(a){state.draftAnnotation={...a};$('comment-quote').textContent=a.quote;$('comment-quote').hidden=!a.quote;$('comment-text').value=a.text;$('comment-title').textContent=a.kind==='note'?'原文批注':'为标注添加批注';$('comment-dialog').showModal();$('comment-text').focus();hideSelection();}
function noteFromSelection(){const s=state.selection;if(!s||s.page!==state.page)return;editComment(makeAnnotation('highlight',{rects:s.rects,quote:s.quote}));}
function pointerDown(event){
  if(event.button!==0||!state.pdf||state.rendering)return;
  const p=relativePoint(event,$('pdf-page'));
  if(state.tool==='note'){
    const w=18/$('pdf-page').offsetWidth,h=18/$('pdf-page').offsetHeight;
    editComment(makeAnnotation('note',{rects:[[clamp(p[0],0,1-w),clamp(p[1],0,1-h),clamp(p[0]+w,w,1),clamp(p[1]+h,h,1)]]}));return;
  }
  if(!['ink','rectangle','eraser'].includes(state.tool))return;
  event.preventDefault();$('gesture-layer').setPointerCapture(event.pointerId);
  gesture={pointer:event.pointerId,tool:state.tool,start:p,points:[p]};
  if(state.tool==='eraser')erase(p);
}
function erase(point){const hit=hitAnnotation(point,currentAnnotations(),$('pdf-page').getBoundingClientRect());if(hit)applyAnnotation(hit,null);}
function pointerMove(event){
  if(!gesture||gesture.pointer!==event.pointerId)return;
  const p=relativePoint(event,$('pdf-page'));if(gesture.tool==='eraser'){erase(p);return;}
  const previous=gesture.points.at(-1);
  if(Math.hypot(p[0]-previous[0],p[1]-previous[1])<.001)return;
  gesture.points.push(p);const layer=$('gesture-layer');layer.replaceChildren();
  if(gesture.tool==='ink')paintAnnotation(layer,makeAnnotation('ink',{points:gesture.points}),state.scale);
  else paintAnnotation(layer,makeAnnotation('rectangle',{rects:[normalizedRect(gesture.start,p)]}),state.scale);
}
function pointerEnd(event,cancel=false){
  if(!gesture||gesture.pointer!==event.pointerId)return;
  const g=gesture;gesture=null;$('gesture-layer').replaceChildren();
  if($('gesture-layer').hasPointerCapture(event.pointerId))$('gesture-layer').releasePointerCapture(event.pointerId);
  if(cancel||g.tool==='eraser')return;
  const end=relativePoint(event,$('pdf-page'));
  if(g.tool==='ink'){
    if(g.points.length<2)return;if(g.points.length>2499)g.points=g.points.filter((_,i)=>i%Math.ceil(g.points.length/2498)===0);g.points.push(end);
    applyAnnotation(null,makeAnnotation('ink',{points:g.points}));
  }else{const r=normalizedRect(g.start,end);if((r[2]-r[0])*$('pdf-page').offsetWidth>3&&(r[3]-r[1])*$('pdf-page').offsetHeight>3)applyAnnotation(null,makeAnnotation('rectangle',{rects:[r]}));}
}

function scrollFraction(){const el=$('document-viewport');return el.scrollHeight>el.clientHeight?clamp(el.scrollTop/(el.scrollHeight-el.clientHeight)):0;}
function saveProgress(){if(!state.pdf||state.rendering)return;saveState({page:state.page,zoom:state.zoom,scroll_position:scrollFraction()}).catch(()=>{});}
async function navigate(page,{restore=0,persist=true}={}){
  if(!state.pdf)return;if(gesture)setTool(state.tool);page=clamp(Math.trunc(Number(page))||1,1,state.pdf.numPages);state.page=page;hideSelection();state.selection=null;
  $('page-input').value=page;$('page-total').textContent='/ '+state.pdf.numPages;$('page-prev').disabled=page===1;$('page-next').disabled=page===state.pdf.numPages;
  $('page-description').textContent='PDF 第 '+page+' 页';$('document-progress').textContent=page+' / '+state.pdf.numPages+' · '+Math.round(page/state.pdf.numPages*100)+'%';
  document.querySelectorAll('[data-page]').forEach(el=>el.classList.toggle('is-current',Number(el.dataset.page)===page));
  $('bookmark-page').setAttribute('aria-pressed',String(state.bookmarks.includes(page)));$('bookmark-page').setAttribute('aria-label',state.bookmarks.includes(page)?'移除当前页书签':'为当前页添加书签');
  const version=++state.renderVersion;state.rendering=true;state.renderTask?.cancel();state.textTask?.cancel();
  try{
    const pdfPage=await state.pdf.getPage(page);if(version!==state.renderVersion)return;
    const viewportEl=$('document-viewport'),style=getComputedStyle(viewportEl),available=Math.max(120,viewportEl.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight));
    const unit=pdfPage.getViewport({scale:1}),scale=state.zoom==='fit'?available/unit.width:Number(state.zoom)/100;
    const viewport=pdfPage.getViewport({scale}),dpr=Math.min(devicePixelRatio||1,2),canvas=$('pdf-canvas'),sheet=$('pdf-page');
    state.scale=scale;sheet.hidden=false;$('document-message').hidden=true;sheet.style.width=viewport.width+'px';sheet.style.height=viewport.height+'px';sheet.style.setProperty('--scale-factor',scale);sheet.dataset.page=page;sheet.dataset.tool=state.tool;
    canvas.width=Math.ceil(viewport.width*dpr);canvas.height=Math.ceil(viewport.height*dpr);
    const text=$('pdf-text');text.replaceChildren();
    state.renderTask=pdfPage.render({canvasContext:canvas.getContext('2d'),viewport,transform:dpr===1?null:[dpr,0,0,dpr,0,0],annotationMode:AnnotationMode.ENABLE});
    await state.renderTask.promise;if(version!==state.renderVersion)return;
    const content=await pdfPage.getTextContent();if(version!==state.renderVersion)return;
    state.textTask=new TextLayer({textContentSource:content,container:text,viewport});await state.textTask.render();
    if(version!==state.renderVersion)return;
    renderAnnotations();renderAnnotationList();renderSearchHighlights();
    viewportEl.scrollTop=restore*Math.max(0,viewportEl.scrollHeight-viewportEl.clientHeight);viewportEl.scrollLeft=0;
    state.rendering=false;if(persist)saveProgress();
    $('page-description').textContent='PDF 第 '+page+' 页'+(content.items.some(x=>x.str?.trim())?'':' · 本页无文字层，可用画笔与区域批注');
  }catch(error){if(version!==state.renderVersion||error.name==='RenderingCancelledException'||error.name==='AbortException')return;state.rendering=false;showDocumentError('页面加载失败',error.message);}
}
function showDocumentError(title,copy){$('document-message').hidden=false;$('pdf-page').hidden=true;$('document-message').replaceChildren(element('strong',title),element('p',copy),action('重试加载',()=>state.pdf?navigate(state.page):location.reload(),'quiet-button'),Object.assign(element('a','返回文献库'),{href:'/'}));}
function renderOutline(){
  const panel=$('outline-panel');panel.replaceChildren();const outline=state.metadata.outline;
  panel.append(element('p',state.metadata.outline_source==='pdf'?'论文内置目录':'根据章节标题识别','outline-origin'));
  if(!outline.length){panel.append(element('p','这篇论文没有可识别的目录。可切换到页面缩略图，或使用全文查找。','empty-copy'));return;}
  for(const entry of outline){const b=action(null,()=>{navigate(entry.page);closePanel('left');},'outline-row');b.style.setProperty('--level',entry.level-1);b.dataset.page=entry.page;b.append(element('span',entry.title),element('span',entry.page,'outline-page'));panel.append(b);}
}
function renderBookmarks(){
  const panel=$('bookmarks-panel');panel.replaceChildren();$('bookmark-count').textContent=state.bookmarks.length;
  if(!state.bookmarks.length)panel.append(element('p','点击页码旁的书签图标，把重要页面留在这里。','empty-copy'));
  for(const page of state.bookmarks){const b=action(null,()=>{navigate(page);closePanel('left');},'bookmark-row');b.dataset.page=page;b.append(element('strong','PDF 第 '+page+' 页'),element('span','↗'));panel.append(b);}
}
function toggleBookmark(){
  if(state.bookmarks.includes(state.page))state.bookmarks=state.bookmarks.filter(p=>p!==state.page);else state.bookmarks.push(state.page);
  state.bookmarks.sort((a,b)=>a-b);renderBookmarks();$('bookmark-page').setAttribute('aria-pressed',String(state.bookmarks.includes(state.page)));
  saveState({bookmarks:[...state.bookmarks]},'bookmarks').catch(()=>{});
}
function renderThumbnails(){
  const container=$('pages-panel');if(container.childElementCount)return;
  let queue=Promise.resolve();thumbnailObserver=new IntersectionObserver(entries=>{
    for(const e of entries)if(e.isIntersecting){thumbnailObserver.unobserve(e.target);queue=queue.then(async()=>{
      try{const page=await state.pdf.getPage(Number(e.target.dataset.page)),unit=page.getViewport({scale:1}),viewport=page.getViewport({scale:148/unit.width}),canvas=e.target.querySelector('canvas');canvas.width=viewport.width;canvas.height=viewport.height;await page.render({canvasContext:canvas.getContext('2d'),viewport}).promise;}catch{}
    });}
  },{root:document.querySelector('.navigation-content'),rootMargin:'180px'});
  for(let page=1;page<=state.pdf.numPages;page++){const b=action(null,()=>{navigate(page);closePanel('left');},'thumbnail-button');b.dataset.page=page;b.setAttribute('aria-label','跳转 PDF 第 '+page+' 页');b.append(element('canvas'),element('span','PDF '+page));container.append(b);thumbnailObserver.observe(b);}
}
function setNavigation(name){
  for(const b of document.querySelectorAll('[data-nav]')){const yes=b.dataset.nav===name;b.setAttribute('aria-selected',String(yes));b.tabIndex=yes?0:-1;$(b.dataset.nav+'-panel').hidden=!yes;}
  if(name==='pages'&&state.pdf)renderThumbnails();
}
function setPanel(name){for(const b of document.querySelectorAll('[data-panel]')){const yes=b.dataset.panel===name;b.setAttribute('aria-selected',String(yes));b.tabIndex=yes?0:-1;$('panel-'+b.dataset.panel).hidden=!yes;}}
function openPanel(side){
  const narrow=side==='left'?leftMedia.matches:rightMedia.matches;
  if(narrow){const dialog=$(side+'-panel-dialog'),panel=$(side==='left'?'study-sidebar':'study-inspector');$(side+'-panel-slot').append(panel);if(!dialog.open)dialog.showModal();}
  else{layout.classList.remove(side+'-hidden');refreshExpanded();scheduleResize();}
}
function closePanel(side){const dialog=$(side+'-panel-dialog');if(dialog.open)dialog.close();}
function togglePanel(side){
  if((side==='left'?leftMedia:rightMedia).matches){if($(side+'-panel-dialog').open)closePanel(side);else openPanel(side);}
  else{layout.classList.toggle(side+'-hidden');refreshExpanded();scheduleResize();}
}
function syncTitle(){const title=$('document-short-title');if(leftMedia.matches)document.querySelector('.mobile-paper-heading').append(title);else document.querySelector('.document-title').append(title);}
function refreshExpanded(){$('toggle-outline').setAttribute('aria-expanded',String(!leftMedia.matches&&!layout.classList.contains('left-hidden')));$('toggle-notes').setAttribute('aria-expanded',String(!rightMedia.matches&&!layout.classList.contains('right-hidden')));}
function restorePanel(side){const panel=$(side==='left'?'study-sidebar':'study-inspector');(side==='left'?leftMarker:rightMarker).after(panel);refreshExpanded();scheduleResize();}
function scheduleResize(){clearTimeout(resizeTimer);const fraction=scrollFraction();resizeTimer=setTimeout(()=>{if(state.pdf)navigate(state.page,{restore:fraction,persist:false});},160);}

async function searchDocument(){
  const query=$('document-search').value.trim().toLowerCase(),version=++state.searchVersion;state.searchHits=[];$('search-results').replaceChildren();
  $('search-results').hidden=!query;$('search-status').hidden=!query;$('navigation-tabs').hidden=Boolean(query);document.querySelector('.navigation-content').hidden=Boolean(query);
  if(!query){renderAnnotations();return;}
  $('search-status').textContent='正在查找…';
  for(let page=1;page<=state.pdf.numPages;page++){
    if(version!==state.searchVersion)return;
    let text=state.searchText.get(page);
    if(text===undefined){const p=await state.pdf.getPage(page),content=await p.getTextContent();text=content.items.map(x=>x.str||'').join(' ').replace(/\s+/g,' ');state.searchText.set(page,text);}
    const index=text.toLowerCase().indexOf(query);if(index<0)continue;
    if(version!==state.searchVersion)return;
    state.searchHits.push(page);const b=action(null,()=>{navigate(page);closePanel('left');},'search-result');b.append(element('strong','PDF 第 '+page+' 页'),element('span','…'+text.slice(Math.max(0,index-40),index+query.length+85)+'…'));$('search-results').append(b);
  }
  if(version!==state.searchVersion)return;$('search-status').textContent=state.searchHits.length?'在 '+state.searchHits.length+' 页中找到匹配':'未找到匹配内容';renderAnnotations();renderSearchHighlights();
}
function renderSearchHighlights(){
  const query=$('document-search').value.trim().toLowerCase();if(!query)return;
  const bounds=$('pdf-page').getBoundingClientRect();
  for(const span of $('pdf-text').querySelectorAll('span')){
    if(span.children.length||span.firstChild?.nodeType!==Node.TEXT_NODE)continue;
    const value=span.textContent.toLowerCase();let start=value.indexOf(query),count=0;
    while(start>=0&&count++<30){const range=document.createRange();range.setStart(span.firstChild,start);range.setEnd(span.firstChild,start+query.length);
      for(const r of range.getClientRects())$('annotation-layer').append(svg('rect',{x:(r.left-bounds.left)/bounds.width*1000,y:(r.top-bounds.top)/bounds.height*1000,width:r.width/bounds.width*1000,height:r.height/bounds.height*1000,fill:'#74a9f5',opacity:.45}));start=value.indexOf(query,start+query.length);}
  }
}
function fillQuestion(value){setPanel('chat');openPanel('right');$('study-question').value=value.slice(0,2000);$('study-question').focus();}
async function askQuestion(event){
  event.preventDefault();const question=$('study-question').value.trim();if(!question||state.chatBusy)return;
  state.chatBusy=true;$('study-ask').disabled=true;$('study-ask').textContent='正在查证…';
  const messages=$('study-chat-messages');messages.querySelector('.chat-intro')?.remove();messages.append(element('div',question,'chat-message'));
  const answer=element('div','正在查找这篇论文中的相关证据…','chat-message is-assistant');messages.append(answer);messages.scrollTop=messages.scrollHeight;
  try{
    const data=await request('/api/ask',{method:'POST',body:JSON.stringify({question,paper_ids:[paperId],mode:'hybrid',top_k:5,rerank:true,generate_answer:true})});
    answer.replaceChildren();
    if(data.answer){
      let cursor=0;for(const match of data.answer.matchAll(/\[(S\d+)\]/g)){
        answer.append(document.createTextNode(data.answer.slice(cursor,match.index)));const evidence=data.evidence?.find(h=>h.citation_id===match[1]);
        if(evidence&&evidence.paper_id===paperId)answer.append(action(match[0],()=>{navigate(evidence.pdf_page);closePanel('right');},'chat-citation'));else answer.append(document.createTextNode(match[0]));cursor=match.index+match[0].length;
      }answer.append(document.createTextNode(data.answer.slice(cursor)));
      answer.append(action('保存到阅读笔记',()=>appendNote('',data.answer), 'text-button'));
    }else{answer.append(element('p',data.message||'未生成回答，可先查看检索证据。'));for(const hit of (data.evidence||[]).filter(h=>h.paper_id===paperId).slice(0,4))answer.append(action('PDF '+hit.pdf_page+' · '+hit.text.slice(0,100),()=>{navigate(hit.pdf_page);closePanel('right');},'question-suggestion'));}
    $('study-question').value='';
  }catch(error){answer.textContent=error.message;answer.classList.add('chat-error');}
  finally{state.chatBusy=false;$('study-ask').disabled=false;$('study-ask').textContent='提问 ↗';messages.scrollTop=messages.scrollHeight;}
}
async function exportFile(format){
  try{await flushNotes();await persistence;if(state.failed.size||state.notesDirty)throw Error('请先保存未完成的更改，再导出。');
    const response=await fetch(base+'/export?format='+format+'&document_sha256='+state.metadata.document_sha256);
    if(!response.ok)throw Error('导出未完成，请稍后重试。');const blob=await response.blob();download(blob,'paper-spider-'+paperId+'.'+format);$('export-dialog').close();toast('已导出阅读成果。');
  }catch(error){toast(error.message);}
}
function download(blob,name){const url=URL.createObjectURL(blob),a=Object.assign(element('a'),{href:url,download:name});a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function setupTabs(container){
  container.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;const buttons=[...container.querySelectorAll('[role=tab]')],index=buttons.indexOf(document.activeElement);if(index<0)return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;buttons[next].click();buttons[next].focus();});
}
for(const b of document.querySelectorAll('[data-tool]'))b.onclick=()=>setTool(b.dataset.tool);
for(const b of document.querySelectorAll('[data-color]'))b.onclick=()=>setColor(b.dataset.color);
for(const b of document.querySelectorAll('[data-nav]'))b.onclick=()=>setNavigation(b.dataset.nav);
for(const b of document.querySelectorAll('[data-panel]'))b.onclick=()=>setPanel(b.dataset.panel);
for(const b of document.querySelectorAll('[data-close-panel]'))b.onclick=()=>closePanel(b.dataset.closePanel);
for(const b of document.querySelectorAll('[data-close-dialog]'))b.onclick=()=>$(b.dataset.closeDialog).close();
for(const b of document.querySelectorAll('[data-export]'))b.onclick=()=>exportFile(b.dataset.export);
for(const b of document.querySelectorAll('[data-ask]'))b.onclick=()=>fillQuestion(b.dataset.ask);
setupTabs($('navigation-tabs'));setupTabs($('inspector-tabs'));
$('left-panel-dialog').addEventListener('close',()=>restorePanel('left'));$('right-panel-dialog').addEventListener('close',()=>restorePanel('right'));
for(const dialog of document.querySelectorAll('dialog'))dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
for(const media of [leftMedia,rightMedia])media.addEventListener('change',()=>{closePanel('left');closePanel('right');syncTitle();refreshExpanded();scheduleResize();});
$('toggle-outline').onclick=()=>togglePanel('left');$('toggle-notes').onclick=()=>togglePanel('right');
$('focus-mode').onclick=()=>{const focused=layout.classList.contains('left-hidden')&&layout.classList.contains('right-hidden');layout.classList.toggle('left-hidden',!focused);layout.classList.toggle('right-hidden',!focused);$('focus-mode').querySelector('span').textContent=focused?'专注阅读':'退出专注';refreshExpanded();scheduleResize();};
$('page-prev').onclick=()=>navigate(state.page-1);$('page-next').onclick=()=>navigate(state.page+1);
$('page-input').addEventListener('change',()=>navigate($('page-input').value));
$('zoom').onchange=()=>{const fraction=scrollFraction();state.zoom=$('zoom').value;navigate(state.page,{restore:fraction});};
$('bookmark-page').onclick=toggleBookmark;$('annotation-undo').onclick=undo;$('annotation-redo').onclick=redo;$('annotation-filter').onchange=renderAnnotationList;
$('paper-notes').addEventListener('input',notesChanged);$('paper-notes').addEventListener('blur',()=>flushNotes().catch(()=>{}));
$('insert-outline').onclick=()=>{$('paper-notes').value+=($('paper-notes').value?'\n\n':'')+'## 研究问题\n\n## 核心方法\n\n## 实验与结论\n\n## 我的理解与疑问\n';notesChanged();$('paper-notes').focus();};
$('reading-status').onchange=()=>saveState({reading_status:$('reading-status').value},'status').catch(()=>{});
$('download-draft').onclick=()=>download(new Blob([$('paper-notes').value],{type:'text/markdown;charset=utf-8'}),'paper-spider-local-draft.md');
$('reload-notes').onclick=async()=>{try{const metadata=await request(base);clearTimeout(notesTimer);state.notesEpoch++;$('paper-notes').value=metadata.state.notes;state.notesRevision=metadata.state.notes_revision;state.notesConflict=false;state.notesDirty=false;state.failed.delete('notes');$('notes-conflict').hidden=true;$('notes-count').textContent=metadata.state.notes.length+' 字';$('notes-save-status').textContent='已保存';$('notes-save-status').classList.remove('is-error');removeDraft();updateSaveStatus();}catch(error){toast(error.message);}};
$('document-search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>searchDocument().catch(error=>toast(error.message)),250);});
$('document-search').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();const hits=state.searchHits;if(hits.length){const i=hits.findIndex(p=>p>=state.page);navigate(hits[i>=0?i:0]);closePanel('left');}}});
$('document-viewport').addEventListener('scroll',()=>{hideSelection();if(state.rendering)return;clearTimeout(progressTimer);progressTimer=setTimeout(saveProgress,550);});
$('pdf-text').addEventListener('pointerup',()=>setTimeout(readSelection,30));
document.addEventListener('selectionchange',()=>{clearTimeout(selectionTimer);if(state.tool==='select')selectionTimer=setTimeout(readSelection,100);});
$('selection-menu').addEventListener('pointerdown',event=>event.preventDefault());
$('selection-highlight').onclick=()=>selectionAnnotation('highlight');$('selection-note').onclick=noteFromSelection;
$('selection-copy').onclick=async()=>{try{await navigator.clipboard.writeText(state.selection?.quote||'');toast('已复制原文。');}catch{toast('复制未完成，请使用 Ctrl / ⌘ C。');}hideSelection();};
$('selection-to-notes').onclick=()=>{if(state.selection)appendNote(state.selection.quote);hideSelection();};
$('selection-explain').onclick=()=>{if(state.selection)fillQuestion('请解释这篇论文中的以下原文，包括关键概念与方法含义，并引用论文证据：\n'+state.selection.quote.slice(0,1700));hideSelection();};
$('gesture-layer').addEventListener('pointerdown',pointerDown);$('gesture-layer').addEventListener('pointermove',pointerMove);$('gesture-layer').addEventListener('pointerup',event=>pointerEnd(event));$('gesture-layer').addEventListener('pointercancel',event=>pointerEnd(event,true));
$('comment-form').onsubmit=event=>{event.preventDefault();const a={...state.draftAnnotation,text:$('comment-text').value.trim()};if(!a.text)return;const before=state.annotations.find(x=>x.id===a.id)||null;applyAnnotation(before,a);$('comment-dialog').close();window.getSelection().removeAllRanges();setPanel('annotations');openPanel('right');};
$('study-chat-form').onsubmit=askQuestion;$('export-open').onclick=()=>$('export-dialog').showModal();$('keyboard-help').onclick=()=>$('shortcuts-dialog').showModal();
document.addEventListener('keydown',event=>{
  const editing=event.target.closest('input,textarea,[contenteditable=true]');
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='f'&&!document.querySelector('dialog[open]')){event.preventDefault();openPanel('left');$('document-search').focus();$('document-search').select();return;}
  if(editing||document.querySelector('dialog[open]'))return;
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redo():undo();return;}
  if(event.ctrlKey||event.metaKey||event.altKey)return;
  if(event.key==='ArrowRight'){event.preventDefault();navigate(state.page+1);}else if(event.key==='ArrowLeft'){event.preventDefault();navigate(state.page-1);}
  else if(event.key==='Escape'){setTool('select');window.getSelection().removeAllRanges();}
  else if(event.key==='?')$('shortcuts-dialog').showModal();
  else if(event.key==='Delete'&&state.selected){const a=state.annotations.find(a=>a.id===state.selected);if(a)applyAnnotation(a,null);}
  else{const tool={v:'select',h:'highlight',u:'underline',p:'ink',r:'rectangle',n:'note',e:'eraser'}[event.key.toLowerCase()];if(tool)setTool(tool);}
});
new ResizeObserver(()=>{if(state.pdf&&!state.rendering)scheduleResize();}).observe($('document-viewport'));
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden'){flushNotes().catch(()=>{});saveProgress();}});
window.addEventListener('pagehide',()=>{if(state.notesDirty)storeDraft();if(state.metadata)fetch(base+'/state',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_sha256:state.metadata.document_sha256,page:state.page,zoom:state.zoom,scroll_position:scrollFraction()}),keepalive:true}).catch(()=>{});});

async function initialize(){
  syncTitle();refreshExpanded();
  document.querySelector('.document-search kbd').textContent=/Mac|iPhone|iPad/.test(navigator.platform)?'⌘ F':'Ctrl F';
  if(!paperId){showDocumentError('选择一篇论文开始精读','从文献库点击“精读”，即可使用高亮、画笔、批注与阅读笔记。');$('save-status').textContent='尚未选择论文';return;}
  try{
    const metadata=await request(base);state.metadata=metadata;state.page=Number(params.get('page'))||metadata.state.page;state.zoom=metadata.state.zoom;state.annotations=metadata.state.annotations;state.bookmarks=metadata.state.bookmarks;state.notesRevision=metadata.state.notes_revision;
    document.title=metadata.paper.title+' · 论文蛛 Paper Spider';$('paper-title').textContent=metadata.paper.title;$('document-short-title').textContent=metadata.paper.title;$('document-short-title').title=metadata.paper.title;
    $('paper-byline').textContent=(metadata.paper.year||'论文')+' · '+metadata.pages+' 页'+(metadata.pdf_author?'\n'+metadata.pdf_author:'');
    $('paper-notes').value=metadata.state.notes;$('notes-count').textContent=metadata.state.notes.length.toLocaleString()+' 字';$('zoom').value=state.zoom;$('reading-status').value=metadata.state.reading_status;
    $('original-pdf').href='/api/papers/'+encodeURIComponent(paperId)+'/pdf';$('page-input').max=metadata.pages;
    for(const id of ['paper-notes','insert-outline','reading-status','document-search','zoom','bookmark-page','export-open','page-input'])$(id).disabled=false;
    renderOutline();renderBookmarks();renderAnnotationList();
    const loading=getDocument({url:$('original-pdf').href,cMapUrl:'/static/assets/pdfjs/cmaps/',cMapPacked:true,standardFontDataUrl:'/static/assets/pdfjs/standard_fonts/',wasmUrl:'/static/assets/pdfjs/wasm/',isEvalSupported:false});
    loading.onPassword=()=>{loading.destroy();showDocumentError('这份 PDF 需要密码','请使用可直接打开的 PDF 文件。');};state.pdf=await loading.promise;
    await navigate(state.page,{restore:params.has('page')?0:(metadata.state.scroll_position||0),persist:false});
    let draft;try{draft=JSON.parse(localStorage.getItem(draftKey())||'null');}catch{}
    if(draft&&draft.notes!==metadata.state.notes){const restore=action('恢复本地草稿',()=>{$('paper-notes').value=draft.notes;notesChanged();restore.remove();toast('已恢复本地草稿。');},'text-button');$('notes-conflict').before(restore);toast('找到未保存的本地草稿，可在笔记面板恢复。');}
    updateSaveStatus();document.body.dataset.ready='true';saveProgress();
  }catch(error){showDocumentError('未能打开论文',error.message);$('save-status').textContent='原文未就绪';}
}
initialize();
