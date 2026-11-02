// Pure reconciliation helpers are tested without a browser.
function reconcilePlayback(oldRows, rows, currentId, currentUrl, offset, waiting) {
  const row = rows.find(r => r.id === currentId && r.start !== null && r.url);
  if (!row) return {action: 'invalidate', id: null, offset: 0};
  if (row.url !== currentUrl) return {action: 'replace', id: row.id, offset: Math.min(offset, Math.max(0, row.duration - .01))};
  const i = rows.indexOf(row), next = rows[i + 1];
  if (waiting && next && next.start !== null && next.url) return {action: 'advance', id: next.id, offset: 0};
  return {action: 'keep', id: row.id, offset};
}
function sectionAtTime(rows, time) {
  return rows.find(r => r.start !== null && time >= r.start && time < r.end) || rows.filter(r => r.start !== null).at(-1);
}
function restoredAnchor(rows, anchor) {
  const row = anchor && rows.find(r => r.id === anchor.id && r.start !== null && r.url);
  return row ? {id: row.id, offset: Math.min(anchor.offset, Math.max(0, row.duration-.01))} : null;
}
// UI: built-in browser audio plus section markers; no remote scripts or waveform slicing.
const host = element.querySelector('.lt-preview');
const audio = host.querySelector('audio');
const q = name => host.querySelector(`[data-role="${name}"]`);
let data = {rows:[], prefix_count:0, total:0}, selectedId = null, selectedRevision = null;
let pendingAnchor = null;
let currentId = null, currentUrl = null, intent = false, waiting = false, dirty = false, conflict = false;
const fmt = t => `${Math.floor(t/60)}:${(t%60).toFixed(2).padStart(5,'0')}`;
const text = key => (data.labels || {})[key] || key;
function notice(key) { q('notice').textContent = text(key); }
function seek(row, offset=0, play=intent) {
  if (!row || row.start === null || !row.url) return;
  pendingAnchor=null;q('notice').textContent='';currentId=row.id; waiting=false;
  const apply = () => { audio.currentTime=Math.min(offset, Math.max(0,row.duration-.001)); if(play) audio.play().catch(()=>{intent=false;notice('pressPlay');}); };
  if(currentUrl !== row.url) { audio.pause();currentUrl=row.url;audio.src=row.url;audio.onloadedmetadata=apply;audio.load(); }
  else apply();
  intent=play; clock();
}
function clock() {
  const row=data.rows.find(r=>r.id===currentId);const value=row && row.start!==null ? row.start+audio.currentTime : 0;
  q('seek').max=data.total||0;q('seek').value=Math.min(value,data.total);
  q('clock').textContent=`${fmt(value)} / ${fmt(data.total)}`;
  q('play').textContent=text(intent?'pause':'play');
}
function editor(reset=false) {
  const row=data.rows.find(r=>r.id===selectedId);
  if(!row) {q('selected').textContent=text('choose');q('editor').value='';q('editor').disabled=true;for(const n of ['save','regenerate','split','fromSelection','reload'])q(n).disabled=true;return;}
  q('reload').disabled=false;
  q('selected').textContent=`${text('section')} ${row.number} · ${row.start===null?text('notInPrefix'):`${fmt(row.start)} – ${fmt(row.end)}`} · ${text(row.status)}`;
  if(reset || !dirty) {q('editor').value=row.text;selectedRevision=row.revision;dirty=false;conflict=false;}
  else if(selectedRevision!==row.revision) {conflict=true;notice('conflict');}
  q('editor').disabled=!!data.busy;
  for(const name of ['save','regenerate','split'])q(name).disabled=!!data.busy || conflict;
  q('fromSelection').disabled=row.start===null;
}
function select(id, jump=true) {
  selectedId=id;editor(true);paintSelection();
  const row=data.rows.find(r=>r.id===id);if(jump && row && row.start!==null)seek(row,0,intent);
}
function paintSelection() {
  host.querySelectorAll('[data-section]').forEach(el=>{const chosen=el.dataset.section===selectedId;el.classList.toggle('selected',chosen);el.setAttribute('aria-pressed',String(chosen));});
}
function markers() {
  q('markers').replaceChildren();q('sections').replaceChildren();
  for(const row of data.rows) {
    const button=document.createElement('button');button.type='button';button.dataset.section=row.id;
    button.textContent=`${row.number} · ${row.start===null?text('notInPrefix'):`${fmt(row.start)} – ${fmt(row.end)}`} · ${text(row.status)} — ${row.text}`;
    button.title=row.error||row.text;button.addEventListener('click',()=>select(row.id));q('sections').append(button);
    if(row.start!==null) {
      const marker=document.createElement('button');marker.type='button';marker.dataset.section=row.id;marker.style.flex=`${row.duration} 1 0px`;
      marker.textContent=String(row.number);marker.title=`${text('section')} ${row.number}: ${fmt(row.start)} – ${fmt(row.end)}`;marker.setAttribute('aria-label',marker.title);marker.addEventListener('click',()=>select(row.id));q('markers').append(marker);
    }
  }
  paintSelection();
}
function update() {
  let next;try {next=JSON.parse(props.value||'{}');} catch {return;}
  if(!next.rows) return;
  const changedProject=data.project_id!==next.project_id;
  const oldRows=data.rows;const decision=reconcilePlayback(oldRows,next.rows,currentId,currentUrl,audio.currentTime,waiting);
  data=next;
  host.querySelectorAll('[data-label]').forEach(el=>el.textContent=text(el.dataset.label));
  if(changedProject){audio.pause();audio.removeAttribute('src');currentId=null;currentUrl=null;pendingAnchor=null;intent=false;waiting=false;selectedId=null;dirty=false;}
  else if(currentId) {
    if(decision.action==='invalidate') {pendingAnchor={id:currentId,offset:audio.currentTime};audio.pause();audio.removeAttribute('src');currentId=null;currentUrl=null;intent=false;waiting=false;notice('invalidated');}
    else if(decision.action==='replace') {intent=false;seek(data.rows.find(r=>r.id===decision.id),decision.offset,false);notice('replaced');}
    else if(decision.action==='advance') seek(data.rows.find(r=>r.id===decision.id),0,intent);
    // keep: do not assign src, call load, seek, or pause when only the tail grew.
  }
  if(!changedProject && !currentId && pendingAnchor) {const anchor=restoredAnchor(data.rows,pendingAnchor);if(anchor){seek(data.rows.find(r=>r.id===anchor.id),anchor.offset,false);notice('replaced');}}
  if(!data.rows.some(r=>r.id===selectedId)){selectedId=data.rows[0]?.id||null;dirty=false;}
  q('summary').textContent=`${data.prefix_count}/${data.rows.length} ${text('prefix')} · ${fmt(data.total)}`;
  q('gap').textContent=!data.rows.length?text('noProject'):(data.prefix_count<data.rows.length?text('gap'):text('complete'));
  q('play').disabled=!data.prefix_count;markers();editor();clock();
}
q('play').addEventListener('click',()=>{
  if(intent){intent=false;waiting=false;audio.pause();clock();return;}
  intent=true;
  const row=data.rows.find(r=>r.id===currentId && r.start!==null);
  if(!row || audio.ended)seek(data.rows.find(r=>r.start!==null),0,true);else audio.play().catch(()=>{intent=false;notice('pressPlay');});
});
q('seek').addEventListener('input',()=>{const t=Number(q('seek').value);const r=sectionAtTime(data.rows,t);if(r)seek(r,t-r.start,intent);});
q('fromSelection').addEventListener('click',()=>{intent=true;seek(data.rows.find(r=>r.id===selectedId),0,true);});
q('reload').addEventListener('click',()=>editor(true));
q('editor').addEventListener('input',()=>{dirty=true;});
for(const action of ['save','regenerate','split'])q(action).addEventListener('click',()=>{
  if(data.busy||conflict||!selectedId)return;
  trigger({save:'section_save',regenerate:'section_regenerate',split:'section_split'}[action],{project_id:data.project_id,section_id:selectedId,revision:selectedRevision,text:q('editor').value});
  dirty=false;
});
audio.addEventListener('timeupdate',clock);
audio.addEventListener('pause',clock);audio.addEventListener('play',clock);
audio.addEventListener('ended',()=>{
  const i=data.rows.findIndex(r=>r.id===currentId),next=data.rows[i+1];
  if(intent&&next&&next.start!==null&&next.url)seek(next,0,true);
  else {waiting=intent && data.prefix_count<data.rows.length;if(waiting)notice('waiting');else intent=false;clock();}
});
audio.addEventListener('error',()=>{intent=false;notice('audioError');});
watch('value',update);update();
