/* Decorative desktop creature. Reads visible DOM and the existing loading flag only. */
(() => {
  'use strict';
  const overlay = document.getElementById('desktop-spider');
  const canvas = document.getElementById('desktop-spider-canvas');
  const loading = document.getElementById('loading-state');
  const ctx = canvas.getContext('2d');
  const figure = window.PaperSpiderFigure;
  if (!ctx || !figure || !loading) return;
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const library = document.getElementById('library-panel');
  const list = document.getElementById('papers');
  const composer = document.querySelector('.composer-stage');
  const header = document.querySelector('.app-header');
  let frame = 0, scheduled = 0, started = 0, lastPaint = 0, idleStarted = performance.now();
  let working = false, running = false, geometry = null, refreshed = 0;
  let bitmapW = 0, bitmapH = 0, bitmapRatio = 0;
  let scan = null, lastScrollAt = 0, manualUntil = 0;
  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
  const smooth = n => n * n * (3 - 2 * n);
  const reduce = () => motion.matches || document.documentElement.dataset.motion === 'reduced';

  function titleWords(title, clip) {
    const text = title.firstChild;
    if (!text || text.nodeType !== Node.TEXT_NODE) return [];
    const titleBox = title.getBoundingClientRect();
    if (titleBox.bottom < clip.top || titleBox.top > clip.bottom) return [];
    const candidates = [...text.textContent.matchAll(/[A-Za-z][A-Za-z0-9-]{3,}/g)];
    const words = [];
    for (const match of candidates.slice(0, 16)) {
      if (/^(with|from|based|using|and|the|for|that)$/i.test(match[0])) continue;
      const range = document.createRange();
      range.setStart(text, match.index); range.setEnd(text, match.index + match[0].length);
      const box = range.getBoundingClientRect();
      range.detach();
      if (box.width && box.top >= clip.top + 3 && box.bottom <= Math.min(clip.bottom - 3,titleBox.bottom + 1) && box.left >= clip.left && box.right <= clip.right + 1) {
        words.push({x: (box.left + box.right) / 2, y: (box.top + box.bottom) / 2,
          label: match[0].slice(0, 17)});
        if(words.length>=8)break;
      }
    }
    return words;
  }
  function measure() {
    const drawer = document.getElementById('library-dialog');
    const modal = [...document.querySelectorAll('dialog[open]')].some(el => el !== drawer);
    const parent = drawer.open ? drawer : document.body;
    if (overlay.parentElement !== parent) parent.append(overlay);
    if (modal || document.hidden) return null;
    const libraryBox = library.getBoundingClientRect();
    const listBox = list.getBoundingClientRect();
    const visible = libraryBox.width > 100 && listBox.height > 40 && libraryBox.left < innerWidth && libraryBox.right > 0;
    if (visible) {
      const clip = {left: listBox.left, right: listBox.right,
        top: Math.max(listBox.top, libraryBox.top, 12), bottom: Math.min(listBox.bottom, libraryBox.bottom, innerHeight - 12)};
      const compact = innerWidth < 768;
      const rail = clamp(clip.right + 3, 44, innerWidth - (compact?48:52));
      const anchor = {x: rail + 2, y: clamp(clip.top - 8, 8, innerHeight - 65)};
      const idle = {x: rail + 2, y: anchor.y + 34};
      const targets = [], words = [];
      for (const title of list.querySelectorAll('.paper-title')) {
        const pieces = titleWords(title, clip);
        if (!pieces.length) continue;
        const word=pieces[0];words.push(...pieces);
        targets.push({...word, bodyY: clamp(word.y + 13, clip.top + 30, clip.bottom - 28)});
      }
      return {kind: 'library', compact, rail, anchor, idle, targets, words:words.slice(0,100),
        scrollTop:list.scrollTop,clip,
        track:{top:clip.top+54,bottom:Math.max(clip.top+90,clip.bottom-54),right:rail-4,radius:13}};
    }
    const brand = document.querySelector('.brand').getBoundingClientRect();
    const tools = document.querySelector('.header-tools').getBoundingClientRect();
    const head = header.getBoundingClientRect();
    const questionBox = composer.getBoundingClientRect();
    const enoughGap = tools.left - brand.right > 36;
    const anchor = enoughGap ? {x: (brand.right + tools.left) / 2, y: head.top + 6}
      : {x: clamp(questionBox.right + 3, 28, innerWidth - 28), y: clamp(questionBox.top - 6, 10, innerHeight - 60)};
    const idle = {x: anchor.x, y: enoughGap ? head.bottom - 19 : anchor.y + 32};
    const rail = clamp(questionBox.right + 4, 44, innerWidth - 44);
    const query = document.getElementById('question').value;
    const words = query.match(/[A-Za-z][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,6}/g) || [];
    const targets = [...new Set(words)].slice(0, 12).map((label, index) => ({
      x: questionBox.left + 30 + (index%3) * Math.max(45,(questionBox.width-80)/3),
      y: clamp(questionBox.top + 48 + Math.floor(index/3) * 31, 25, Math.min(questionBox.bottom-38,innerHeight - 40)),
      bodyY: clamp(questionBox.top + 43 + (index%5) * 29, 35, Math.min(questionBox.bottom - 35, innerHeight - 40)),
      label: label.slice(0, 14),
    }));
    const trackTop=clamp(questionBox.top+40,50,innerHeight-130);
    return {kind: 'composer', rail, anchor, idle, targets,words:targets,
      track:{top:trackTop,bottom:Math.max(trackTop+65,Math.min(questionBox.bottom-42,innerHeight-70)),right:rail-2,radius:9}};
  }
  function beginScan() {
    scan={original:list.scrollTop,position:list.scrollTop,direction:1,lastWritten:list.scrollTop};
    lastScrollAt=0;manualUntil=0;list.classList.add('is-spider-scanning');
  }
  function restoreScan() {
    if(scan)list.scrollTop=clamp(scan.original,0,Math.max(0,list.scrollHeight-list.clientHeight));
    scan=null;lastScrollAt=0;list.classList.remove('is-spider-scanning');
  }
  function scrollLibrary(now) {
    const delta=lastScrollAt?Math.min(.08,Math.max(0,(now-lastScrollAt)/1000)):0;
    lastScrollAt=now;
    if(!scan||!geometry||geometry.kind!=='library'||reduce()||now<manualUntil)return;
    const max=Math.max(0,list.scrollHeight-list.clientHeight);
    if(max<1)return;
    const speed=Math.min(720,Math.max(500,list.clientHeight*1.2));
    let next=scan.position+scan.direction*speed*delta;
    if(next>max){next=Math.max(0,2*max-next);scan.direction=-1;}
    else if(next<0){next=Math.min(max,-next);scan.direction=1;}
    scan.position=clamp(next,0,max);
    scan.lastWritten=scan.position;
    list.scrollTop=scan.position;
  }
  function crawlAt(t) {
    const {top,bottom,right,radius}=geometry.track;
    const straight=Math.max(50,bottom-top),arc=Math.PI*radius;
    const speed=geometry.kind==='library'?155:100;
    let distance=(Math.max(0,t)*speed)%(straight*2+arc*2);
    if(distance<straight)return {x:right,y:top+distance,vx:0,vy:speed};
    distance-=straight;
    const center=right-radius;
    if(distance<arc){
      const angle=distance/radius;
      return {x:center+radius*Math.cos(angle),y:bottom+radius*Math.sin(angle),
        vx:-speed*Math.sin(angle),vy:speed*Math.cos(angle)};
    }
    distance-=arc;
    if(distance<straight)return {x:right-radius*2,y:bottom-distance,vx:0,vy:-speed};
    const angle=Math.PI+(distance-straight)/radius;
    return {x:center+radius*Math.cos(angle),y:top+radius*Math.sin(angle),
      vx:-speed*Math.sin(angle),vy:speed*Math.cos(angle)};
  }
  function prepare(bounds) {
    const left = Math.floor(clamp(bounds.left, 0, innerWidth));
    const top = Math.floor(clamp(bounds.top, 0, innerHeight));
    const width = Math.max(64, Math.ceil((bounds.right - left) / 64) * 64);
    const height = Math.max(64, Math.ceil((bounds.bottom - top) / 64) * 64);
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const w = Math.ceil(width * ratio), h = Math.ceil(height * ratio);
    if (bitmapW !== w || bitmapH !== h || bitmapRatio !== ratio) {
      bitmapW = w; bitmapH = h; bitmapRatio = ratio;
      canvas.width = w; canvas.height = h;
      canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    }
    canvas.style.transform = `translate3d(${left}px,${top}px,0)`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.translate(-left, -top);
  }
  function strand(a, b, bend, color, width = .7) {
    ctx.beginPath(); ctx.moveTo(a.x, a.y);
    ctx.bezierCurveTo(a.x + bend, a.y + (b.y-a.y)*.36, b.x - bend, a.y + (b.y-a.y)*.7, b.x, b.y);
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  }
  function onStrand(a,b,bend,t) {
    const m=1-t;
    return {
      x:m*m*m*a.x+3*m*m*t*(a.x+bend)+3*m*t*t*(b.x-bend)+t*t*t*b.x,
      y:m*m*m*a.y+3*m*m*t*(a.y+(b.y-a.y)*.36)+3*m*t*t*(a.y+(b.y-a.y)*.7)+t*t*t*b.y,
    };
  }
  function probesAt(t,pose) {
    const shift=geometry.kind==='library'?list.scrollTop-geometry.scrollTop:0;
    const visible=geometry.words.map(word=>({...word,y:word.y-shift})).filter(word=>
      geometry.kind!=='library'||(word.y>=geometry.clip.top+2&&word.y<=geometry.clip.bottom-2));
    const near=visible.filter(word=>Math.abs(word.y-pose.y)<225);
    const pool=near.length?near:visible;
    if(!pool.length)return [];
    const count=geometry.kind==='library'?(geometry.compact?15:22):10;
    return Array.from({length:count},(_,i)=>{
      const clock=t*.85+i*.173;
      const cycle=Math.floor(clock),phase=clock-cycle;
      const word=pool[(cycle*5+i*3)%pool.length];
      const reach=phase<.22?smooth(phase/.22):phase<.76?1:1-smooth((phase-.76)/.24);
      const root={x:pose.x-12+Math.sin(i*1.7)*7,y:pose.y+(i%7-3)*5.5};
      const tip={x:root.x+(word.x-root.x)*reach,y:root.y+(word.y-root.y)*reach};
      return {i,phase,reach,word,root,tip,bend:-28+(i%5)*11};
    });
  }
  function drawProbe(probe,t) {
    const {i,phase,reach,word,root,tip,bend}=probe;
    const rose=i%7===0;
    const rgb=rose?'207,67,145':'43,99,238';
    strand(root,tip,bend,`rgba(${rgb},${.10+reach*.29})`,.58+(i%3)*.14);
    if(i%2===0)strand({x:root.x+2,y:root.y+4},{x:tip.x+2,y:tip.y+2},bend+8,'rgba(76,122,240,.11)',.55);
    ctx.beginPath();ctx.arc(tip.x,tip.y,1.25+reach*.8,0,Math.PI*2);
    ctx.fillStyle=`rgba(${rgb},${.28+reach*.6})`;ctx.fill();
    for(let packet=0;packet<3;packet++){
      const transit=(t*1.65+packet/3+i*.27)%1;
      const dot=onStrand(tip,root,-bend,transit);
      const fade=Math.sin(transit*Math.PI)*reach;
      ctx.beginPath();ctx.arc(dot.x,dot.y,.75+(packet%2)*.55,0,Math.PI*2);
      ctx.fillStyle=`rgba(${rgb},${fade*.7})`;ctx.fill();
    }
    if(phase>.26&&phase<.92){
      const flight=(phase-.26)/.66;
      const wordPoint=onStrand({x:word.x,y:word.y},root,-bend,flight);
      const alpha=Math.sin(flight*Math.PI)*.86;
      ctx.font='500 10.5px "IBM Plex Sans","Microsoft YaHei",sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';
      const labelW=Math.min(ctx.measureText(word.label).width,94);
      ctx.fillStyle=`rgba(255,255,255,${alpha*.9})`;
      ctx.beginPath();ctx.roundRect(wordPoint.x-labelW/2-4,wordPoint.y-9,labelW+8,18,5);ctx.fill();
      ctx.fillStyle=`rgba(${rgb},${alpha})`;ctx.fillText(word.label,wordPoint.x,wordPoint.y,94);
    }
  }
  function paint(t = 0) {
    if (!geometry) return;
    const active = working && !reduce();
    const animate = !reduce();
    const scale = geometry.kind === 'library' ? (geometry.compact?(active?.62:.52):(active?.78:.64)) : (active?.56:.45);
    let pose = {...geometry.idle, vx: 0, vy: 0};
    if (active) {
      pose=crawlAt(t);
    } else if(animate) {
      pose.x+=Math.sin(t*.58)*3.8;
      pose.y+=Math.sin(t*.84)*1.9;
    }
    const head = {x: pose.x, y: pose.y - 18 * scale};
    const radius=scale*88+12;
    const probes=active?probesAt(t,pose):[];
    const points = [geometry.anchor, {x: pose.x - radius, y: pose.y - radius}, {x: pose.x + radius, y: pose.y + radius}];
    for(const probe of probes)points.push({x:probe.word.x-59,y:probe.word.y-24},{x:probe.word.x+59,y:probe.word.y+24});
    prepare({left: Math.min(...points.map(p=>p.x))-8, right: Math.max(...points.map(p=>p.x))+8,
      top: Math.min(...points.map(p=>p.y))-8, bottom: Math.max(...points.map(p=>p.y))+8});
    strand(geometry.anchor, head, animate ? Math.sin(t * .6) * (active?8:3) : 1, 'rgba(60,108,225,.43)', .7);
    probes.forEach(probe=>drawProbe(probe,t));
    if(active)for(let i=0;i<20;i++){
      const a=i/20*Math.PI*2+t*.32;
      const r=29+Math.sin(t*1.4+i)*9;
      ctx.beginPath();ctx.arc(pose.x+Math.cos(a)*r,pose.y+Math.sin(a)*r*.85,.8,0,Math.PI*2);
      ctx.fillStyle=`rgba(76,123,239,${.16+(i%3)*.065})`;ctx.fill();
    }
    ctx.save();ctx.translate(pose.x,pose.y);ctx.scale(scale,scale);
    figure.drawSpider(ctx,{x:0,y:0,time:animate?t:0,vx:pose.vx/scale,vy:pose.vy/scale,
      reducedMotion:!animate,idleMotion:!active,activeProbes:active?10:19,
      ...(!active?{heading:animate?Math.sin(t*.58)*.055:0}:{})});
    ctx.restore();
    overlay.dataset.state = active ? 'working' : 'resting';
    overlay.dataset.context = geometry.kind;
    // Coordinates describe this decorative figure, not retrieval measurements.
    overlay.dataset.x=pose.x.toFixed(1);overlay.dataset.y=pose.y.toFixed(1);
  }
  function stop() {
    running = false;cancelAnimationFrame(frame);frame = 0;
  }
  function tick(now) {
    if (!running) return;
    if (now-lastPaint >= 1000/(working?30:15)) {
      if(working)scrollLibrary(now);
      if (now-refreshed > (working?95:700)) { geometry = measure(); refreshed = now; }
      if (!geometry) { sync(); return; }
      paint(Math.max(0,(now-(working?started:idleStarted))/1000));lastPaint=now;
    }
    frame=requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(scheduled);scheduled=0;
    const next = !loading.hidden;
    if (next && !working) {started=performance.now();beginScan();}
    if (!next && working) {idleStarted=performance.now();restoreScan();}
    working=next;geometry=measure();refreshed=performance.now();
    overlay.hidden=!geometry;
    if (!geometry) {stop();lastScrollAt=0;return;}
    if (!reduce()) {
      if (!running) {running=true;frame=requestAnimationFrame(tick);}
      paint(Math.max(0,(performance.now()-(working?started:idleStarted))/1000));
    } else {stop();lastScrollAt=0;paint(0);}
  }
  function queue() {
    if (!scheduled) scheduled=requestAnimationFrame(sync);
  }
  new MutationObserver(queue).observe(loading,{attributes:true,attributeFilter:['hidden']});
  new MutationObserver(queue).observe(document.documentElement,{attributes:true,attributeFilter:['data-motion']});
  new MutationObserver(queue).observe(list,{childList:true,subtree:true});
  const resize = new ResizeObserver(queue);resize.observe(library);resize.observe(composer);resize.observe(header);
  for (const dialog of document.querySelectorAll('dialog'))new MutationObserver(queue).observe(dialog,{attributes:true,attributeFilter:['open']});
  list.addEventListener('scroll',()=>{
    if(scan&&Math.abs(list.scrollTop-scan.lastWritten)<2)return;
    if(scan){scan.position=list.scrollTop;scan.lastWritten=list.scrollTop;}
    queue();
  },{passive:true});
  function manualScroll() {
    if(!scan)return;
    manualUntil=performance.now()+1500;
    requestAnimationFrame(()=>{if(scan){scan.position=list.scrollTop;scan.lastWritten=list.scrollTop;}});
  }
  list.addEventListener('wheel',manualScroll,{passive:true});
  list.addEventListener('touchstart',manualScroll,{passive:true});
  list.addEventListener('pointerdown',manualScroll,{passive:true});
  list.addEventListener('keydown',manualScroll);
  document.getElementById('main-content').addEventListener('scroll',queue,{passive:true});
  window.addEventListener('resize',queue,{passive:true});
  document.addEventListener('visibilitychange',queue);motion.addEventListener('change',queue);
  document.fonts.ready.then(queue);
  window.addEventListener('pagehide',()=>{stop();restoreScan();cancelAnimationFrame(scheduled);});
  sync();
})();
