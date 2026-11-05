/* Geometry is normalized to the visible PDF page, independent of zoom and DPR. */
export const clamp = (v, low = 0, high = 1) => Math.max(low, Math.min(high, v));
export const kindNames = {highlight:'高亮', underline:'下划线', ink:'画笔', rectangle:'框选', note:'批注'};
const svgNS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs = {}) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, value);
  return element;
}

export function relativePoint(event, page) {
  const rect = page.getBoundingClientRect();
  return [clamp((event.clientX - rect.left) / rect.width), clamp((event.clientY - rect.top) / rect.height)];
}

export function normalizedRect(start, end) {
  return [Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];
}

export function selectionGeometry(textLayer, page) {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount) return null;
  const range = selection.getRangeAt(0);
  if (!textLayer.contains(range.startContainer) || !textLayer.contains(range.endContainer)) return null;
  const quote = selection.toString().replace(/\s+/g,' ').trim().slice(0,15000);
  if (!quote) return null;
  const bounds = page.getBoundingClientRect(), rects = [];
  for (const r of range.getClientRects()) {
    if (r.width < 1 || r.height < 2) continue;
    const x1=clamp((r.left-bounds.left)/bounds.width), y1=clamp((r.top-bounds.top)/bounds.height);
    const x2=clamp((r.right-bounds.left)/bounds.width), y2=clamp((r.bottom-bounds.top)/bounds.height);
    if (x2>x1 && y2>y1) rects.push([x1,y1,x2,y2]);
  }
  rects.sort((a,b)=>Math.abs(a[1]-b[1])>.003?a[1]-b[1]:a[0]-b[0]);
  const merged=[];
  for (const r of rects) {
    const previous=merged.at(-1);
    if (previous && Math.abs(previous[1]-r[1])<.004 && Math.abs(previous[3]-r[3])<.004 && r[0]<=previous[2]+.009) {
      previous[0]=Math.min(previous[0],r[0]); previous[2]=Math.max(previous[2],r[2]);
    } else merged.push([...r]);
  }
  if (!merged.length || merged.length>200) return null;
  return {quote,rects:merged,bounds:range.getBoundingClientRect()};
}

export function paintAnnotation(layer, annotation, scale = 1, selected = false) {
  const a=annotation, group=svg('g',{'data-annotation-id':a.id});
  if (a.kind==='ink') {
    group.append(svg('path',{d:a.points.map((p,i)=>(i?'L':'M')+(p[0]*1000)+','+(p[1]*1000)).join(' '),fill:'none',stroke:a.color,'stroke-width':a.width*scale,'stroke-linecap':'round','stroke-linejoin':'round','vector-effect':'non-scaling-stroke'}));
  } else for (const r of a.rects) {
    const attrs={x:r[0]*1000,y:r[1]*1000,width:(r[2]-r[0])*1000,height:(r[3]-r[1])*1000};
    if(a.kind==='highlight') group.append(svg('rect',{...attrs,fill:a.color,opacity:.42,rx:1}));
    else if(a.kind==='underline') group.append(svg('line',{x1:r[0]*1000,y1:r[3]*1000,x2:r[2]*1000,y2:r[3]*1000,stroke:a.color,'stroke-width':Math.max(1.2,a.width*scale),'vector-effect':'non-scaling-stroke'}));
    else if(a.kind==='rectangle') group.append(svg('rect',{...attrs,fill:'none',stroke:a.color,'stroke-width':a.width*scale,'vector-effect':'non-scaling-stroke'}));
    else {
      group.append(svg('rect',{...attrs,fill:a.color,rx:2}));
      const lineAttrs={stroke:'#fff','stroke-width':1,'vector-effect':'non-scaling-stroke'};
      group.append(svg('line',{x1:attrs.x+attrs.width*.25,y1:attrs.y+attrs.height*.35,x2:attrs.x+attrs.width*.75,y2:attrs.y+attrs.height*.35,...lineAttrs}));
      group.append(svg('line',{x1:attrs.x+attrs.width*.25,y1:attrs.y+attrs.height*.65,x2:attrs.x+attrs.width*.65,y2:attrs.y+attrs.height*.65,...lineAttrs}));
    }
    if(selected)group.append(svg('rect',{...attrs,fill:'none',stroke:'#0066cc','stroke-width':1,'stroke-dasharray':'3 3','vector-effect':'non-scaling-stroke'}));
  }
  layer.append(group);return group;
}

export function hitAnnotation(point, annotations, size) {
  const p=[point[0]*size.width,point[1]*size.height];
  for(const a of [...annotations].reverse()) {
    if(a.kind!=='ink') {
      if(a.rects.some(r=>point[0]>=r[0]-.006&&point[0]<=r[2]+.006&&point[1]>=r[1]-.004&&point[1]<=r[3]+.004))return a;
    } else for(let i=1;i<a.points.length;i++) {
      const start=[a.points[i-1][0]*size.width,a.points[i-1][1]*size.height],end=[a.points[i][0]*size.width,a.points[i][1]*size.height];
      const dx=end[0]-start[0],dy=end[1]-start[1],length=dx*dx+dy*dy;
      const t=length?clamp(((p[0]-start[0])*dx+(p[1]-start[1])*dy)/length):0;
      if(Math.hypot(p[0]-start[0]-t*dx,p[1]-start[1]-t*dy)<Math.max(7,a.width+3))return a;
    }
  }
  return null;
}

export function annotationPayload(a) {
  return Object.fromEntries(['id','kind','page','color','rects','points','width','quote','text'].map(k=>[k,a[k]]));
}
