"use strict";
const DATA = JSON.parse(document.getElementById("review-data").textContent);
const el = id => document.getElementById(id);
const FAMILY_NAMES = {cnn:"Temporal CNN",pose_set:"Pose set",feature_set:"Feature set",lstm:"LSTM",bilstm:"BiLSTM"};
const STORAGE_KEY = "gesture-review-" + DATA.bundle_id;
let storageWorks = true, annotations = {}, foregroundSeconds = 0;
try { const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); annotations=saved.annotations||{}; foregroundSeconds=saved.foreground_seconds||0; }
catch (error) { storageWorks=false; }
let filtered=[], current=null, playing=true, playbackTime=0, lastTick=null, revealed=false;
let timerTick=performance.now();

function textNode(tag, text, className) { const node=document.createElement(tag);node.textContent=text;if(className)node.className=className;return node; }
function addOption(select,value,text){const option=document.createElement("option");option.value=value;option.textContent=text;select.appendChild(option);}
function selectedPredictions(clip){return clip.predictions.filter(p=>(el("family").value==="all"||p.family===el("family").value)&&(el("seed").value==="all"||p.seed===Number(el("seed").value)));}
function isFailure(clip){return selectedPredictions(clip).some(p=>p.wrong);}
function persist(){try{localStorage.setItem(STORAGE_KEY,JSON.stringify({annotations,foreground_seconds:foregroundSeconds}));}catch(error){storageWorks=false;}}
function completedCount(){return Object.values(annotations).filter(n=>n.saved&&n.assessment).length;}
function reviewStatus(message){el("review-status").textContent=(message?message+" ":"")+completedCount()+" observations saved."+(storageWorks?" Export a copy when finished.":" Browser storage unavailable: export before closing.");}

function frameAt(clip,time){let lo=0,hi=clip.times.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(clip.times[mid]<=time)lo=mid;else hi=mid-1;}return lo;}
function mapper(clip,width,height,mode){
 if(mode==="camera"){
  const w=Math.min(width-40,(height-50)*clip.aspect),h=w/clip.aspect,ox=(width-w)/2,oy=(height-h)/2;
  return {point:p=>[ox+(1-p[0])*w,oy+p[1]*h],rect:[ox,oy,w,h]};
 }
 if(!clip.centerBox){
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  clip.centered_xyz.forEach((points,i)=>{if(clip.visible[i])points.forEach(p=>{xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);ymin=Math.min(ymin,p[1]);ymax=Math.max(ymax,p[1]);});});
  clip.centerBox=Number.isFinite(xmin)?{cx:(xmin+xmax)/2,cy:(ymin+ymax)/2,w:Math.max(xmax-xmin,.5),h:Math.max(ymax-ymin,.5)}:{cx:0,cy:0,w:1,h:1};
 }
 const box=clip.centerBox,s=Math.min((width-110)/box.w,(height-110)/box.h);
 return {point:p=>[width/2-(p[0]-box.cx)*s,height/2+(p[1]-box.cy)*s],rect:null};
}
function drawHand(canvas,clip,time,mode){
 const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
 ctx.clearRect(0,0,w,h);ctx.fillStyle="#101f2a";ctx.fillRect(0,0,w,h);
 if(!clip){ctx.fillStyle="#c3d6df";ctx.font="17px system-ui";ctx.textAlign="center";ctx.fillText("No reference for this clip",w/2,h/2);return;}
 const frame=frameAt(clip,time),map=mapper(clip,w,h,mode);
 if(map.rect){ctx.strokeStyle="#29404e";ctx.lineWidth=1;ctx.strokeRect(...map.rect);}
 ctx.font="15px system-ui";ctx.textAlign="left";ctx.fillStyle="#adcad8";
 ctx.fillText(mode==="camera"?"Mirrored camera position · wrist trail":"Centered shape · wrist movement removed from this view",18,25);
 if(!clip.visible[frame]){ctx.textAlign="center";ctx.font="23px system-ui";ctx.fillStyle="#d7e3e8";ctx.fillText("No tracked hand in this frame",w/2,h/2);return;}
 if(mode==="camera"){
  ctx.beginPath();let started=false;
  for(let i=0;i<=frame;i++){if(!clip.visible[i]){started=false;continue;}const p=map.point(clip.xyz[i][0]);if(!started){ctx.moveTo(...p);started=true;}else ctx.lineTo(...p);}
  ctx.strokeStyle="#81b5cf80";ctx.lineWidth=3;ctx.stroke();
 }
 const points=(mode==="camera"?clip.xyz:clip.centered_xyz)[frame].map(map.point);
 const fingers=[[0,1,2,3,4],[0,5,6,7,8],[5,9,10,11,12],[9,13,14,15,16],[13,17,18,19,20]],colors=["#f4b65c","#88d395","#6ec9e5","#b7a3ef","#ec99ad"];
 ctx.strokeStyle="#9ab7c6";ctx.lineWidth=4;ctx.beginPath();[0,5,9,13,17,0].forEach((j,i)=>i?ctx.lineTo(...points[j]):ctx.moveTo(...points[j]));ctx.stroke();
 fingers.forEach((finger,i)=>{ctx.strokeStyle=colors[i];ctx.lineWidth=4;ctx.beginPath();finger.forEach((j,k)=>k?ctx.lineTo(...points[j]):ctx.moveTo(...points[j]));ctx.stroke();finger.slice(1).forEach(j=>{ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(points[j][0],points[j][1],j===finger[4]?7:4,0,2*Math.PI);ctx.fill();});});
 ctx.fillStyle="#ffffff";ctx.beginPath();ctx.arc(points[0][0],points[0][1],7,0,2*Math.PI);ctx.fill();
 ctx.font="13px system-ui";ctx.textAlign="left";ctx.fillStyle="#bdd0da";ctx.fillText("Thumb / index / middle / ring / little finger · wrist in white",18,h-15);
}

function renderSignals(){
 const s=current.signals,box=el("signals");box.replaceChildren();
 const measures=[["Tracked frames",(100*s.tracking_fraction).toFixed(0)+"%"],["Recording length",s.duration_seconds.toFixed(2)+" seconds"],["Wrist path",s.motion_or_shape_defined?s.wrist_path_palm_units.toFixed(3)+" palm units":"Not defined: no hand"],["Shape variation",s.motion_or_shape_defined?s.pose_variation_rms.toFixed(3):"Not defined: no hand"]];
 measures.forEach(([label,value])=>{const cell=textNode("div",label);cell.appendChild(textNode("strong",value));box.appendChild(cell);});
 el("session").textContent="Validation session: "+current.session+". "+s.raw_frames+" original frames; largest frame gap "+s.max_gap_seconds.toFixed(3)+" seconds.";
}
function renderPredictions(){
 const grid=el("prediction-grid");grid.replaceChildren();grid.hidden=!revealed;
 if(!revealed)return;
 DATA.families.forEach(family=>{const values=current.predictions.filter(p=>p.family===family),card=textNode("div","","family");
  card.appendChild(textNode("h3",FAMILY_NAMES[family]));card.appendChild(textNode("p",values.filter(p=>p.wrong).length+" / "+values.length+" models missed this clip"));
  values.forEach(p=>{const row=textNode("div","","seed-row"+(p.wrong?" incorrect":""));row.appendChild(textNode("span","Seed "+p.seed+" · "+p.predicted));row.appendChild(textNode("span",p.score.toFixed(3)));card.appendChild(row);});grid.appendChild(card);
 });
}
function renderReference(){
 const kind=el("reference").value,info=current.neighbours[kind],ref=info?DATA.references[info.id]:null;
 el("reference-caption").textContent=ref?ref.label+" · training example · "+ref.filename:"No matching reference available. Pose comparison is undefined when no hand was tracked.";
 el("reference-distance").textContent=ref?"Pose RMS distance "+info.pose_profile_rms.toFixed(3)+"; wrist RMS distance "+info.wrist_profile_rms.toFixed(3)+". Chosen by pose distance; proximity does not establish intent or label correctness.":"";
 return ref;
}
function renderFrame(){if(!current)return;const i=frameAt(current,playbackTime),mode=el("view").value;el("scrub").value=i;el("frame-time").textContent="Frame "+(i+1)+" / "+current.times.length+" · "+current.times[i].toFixed(2)+" / "+current.times.at(-1).toFixed(2)+" s";
 drawHand(el("main-canvas"),current,playbackTime,mode);const reference=renderReference();const proportion=playbackTime/Math.max(current.times.at(-1),.001);drawHand(el("reference-canvas"),reference,reference?proportion*reference.times.at(-1):0,mode);
 el("view-explanation").textContent=mode==="camera"?"Coordinates and tracking gaps are replayed at their recorded times. The training example is aligned by relative clip time. Wrist trail shows the path so far.":"This view centers and scales each visible hand frame using the existing palm normalization. It helps compare shape but hides global wrist movement. Original model inputs are unchanged.";
}
function saveDraft(saved=false){if(!current)return;const old=annotations[current.id]||{};annotations[current.id]={...old,clip_id:current.id,clip_sha256:current.sha256,original_label:current.label,assessment:el("assessment").value,suggested_label:el("suggested-label").value,notes:el("notes").value,saved,viewed_predictions:old.viewed_predictions||revealed,first_opened_utc:old.first_opened_utc||new Date().toISOString(),last_edited_utc:new Date().toISOString()};persist();reviewStatus(saved?"Observation saved.":"Draft stored.");}
function selectClip(id){current=DATA.clips.find(c=>c.id===id);if(!current)return;playbackTime=0;lastTick=null;revealed=false;playing=true;el("play").textContent="Pause";el("reveal").textContent="Show model guesses";el("reference").value="same_label";el("reference").options[1].disabled=true;
 const note=annotations[id]||{};annotations[id]={...note,first_opened_utc:note.first_opened_utc||new Date().toISOString()};persist();
 el("assessment").value=note.assessment||"";el("suggested-label").value=note.suggested_label||"";el("notes").value=note.notes||"";
 el("clip-label").textContent="Original label: "+current.label;el("clip-filename").textContent=current.filename;el("clip-hint").textContent="Recording prompt: "+current.hint;el("scrub").max=current.times.length-1;
 const ps=selectedPredictions(current),wrong=ps.filter(p=>p.wrong).length;el("failure-badge").textContent=wrong+" / "+ps.length+" selected models wrong";el("failure-badge").className="badge "+(wrong?"bad":"good");
 const index=filtered.findIndex(c=>c.id===id);el("clip-position").textContent="Clip "+(index+1)+" of "+filtered.length+" in this filter";el("previous").disabled=index<=0;el("next").disabled=index>=filtered.length-1;
 for(const button of el("clip-list").children)button.classList.toggle("selected",button.dataset.clipId===id);
 renderSignals();renderPredictions();renderFrame();reviewStatus();
}
function renderList(){
 const previous=current?.id,category=el("category").value,scope=el("scope").value;
 filtered=DATA.clips.filter(c=>(category==="all"||c.label===category)&&(scope==="all"||isFailure(c))&&(scope!=="unreviewed"||!annotations[c.id]?.saved));
 const list=el("clip-list");list.replaceChildren();el("list-summary").textContent=filtered.length+" unique clips · failures depend on selected models";
 filtered.forEach(clip=>{const ps=selectedPredictions(clip),wrong=ps.filter(p=>p.wrong).length,button=textNode("button","","clip-button");button.dataset.clipId=clip.id;button.appendChild(textNode("strong",clip.label));button.appendChild(textNode("small",clip.filename));button.appendChild(textNode("span",wrong+"/"+ps.length+" wrong"+(annotations[clip.id]?.saved?" · reviewed":""),"badge "+(wrong?"bad":"good")));button.addEventListener("click",()=>selectClip(clip.id));list.appendChild(button);});
 el("content").hidden=!filtered.length;el("empty").hidden=filtered.length>0;
 if(filtered.length)selectClip(filtered.some(c=>c.id===previous)?previous:filtered[0].id);else current=null;
}
function move(direction){if(!current)return;const index=filtered.findIndex(c=>c.id===current.id),next=filtered[index+direction];if(next)selectClip(next.id);}
function togglePlay(){playing=!playing;lastTick=null;el("play").textContent=playing?"Pause":"Play";}
function exportObject(){return {version:1,bundle_id:DATA.bundle_id,study_manifest_sha256:DATA.study_manifest_sha256,exported_utc:new Date().toISOString(),foreground_review_seconds:foregroundSeconds,time_scope:"Approximate foreground viewer time, including thought/pauses; not pure labeling time.",reviews:Object.values(annotations).filter(n=>n.saved&&n.assessment).map(({saved,...note})=>note)};}
function exportNotes(){const data=exportObject();if(!data.reviews.length){reviewStatus("Save at least one observation before exporting.");return;}const blob=new Blob([JSON.stringify(data,null,2)+"\n"],{type:"application/json"});const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download="gesture_review_notes.json";document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);reviewStatus("Export downloaded. Keep it and tell the assistant when you finish.");}

addOption(el("category"),"all","All original categories");DATA.labels.forEach(label=>addOption(el("category"),label,label+" · "+DATA.summary[label].unique_any_model_failures+"/"+DATA.summary[label].validation_clips+" fail"));el("category").value="STOP";
addOption(el("family"),"all","All five families");DATA.families.forEach(f=>addOption(el("family"),f,FAMILY_NAMES[f]));
addOption(el("seed"),"all","All seeds");DATA.seeds.forEach(s=>addOption(el("seed"),String(s),String(s)));
addOption(el("suggested-label"),"","No proposed change");DATA.labels.forEach(label=>addOption(el("suggested-label"),label,label));
["category","family","seed","scope"].forEach(id=>el(id).addEventListener("change",renderList));
el("previous").addEventListener("click",()=>move(-1));el("next").addEventListener("click",()=>move(1));el("play").addEventListener("click",togglePlay);el("restart").addEventListener("click",()=>{playbackTime=0;lastTick=null;renderFrame();});
el("scrub").addEventListener("input",()=>{playing=false;el("play").textContent="Play";playbackTime=current.times[Number(el("scrub").value)];lastTick=null;renderFrame();});
el("view").addEventListener("change",renderFrame);el("reference").addEventListener("change",renderFrame);
["assessment","suggested-label","notes"].forEach(id=>el(id).addEventListener(id==="notes"?"input":"change",()=>saveDraft(false)));
el("save-observation").addEventListener("click",()=>{if(!el("assessment").value){reviewStatus("Choose an assessment, then save.");return;}saveDraft(true);if(el("scope").value==="unreviewed")renderList();});
el("export-notes").addEventListener("click",exportNotes);
el("reveal").addEventListener("click",()=>{revealed=!revealed;el("reveal").textContent=revealed?"Hide model guesses":"Show model guesses";el("reference").options[1].disabled=!revealed;if(!revealed)el("reference").value="same_label";if(revealed){annotations[current.id]={...annotations[current.id],viewed_predictions:true};persist();}renderPredictions();renderFrame();});
document.addEventListener("keydown",event=>{if(["INPUT","TEXTAREA","SELECT","BUTTON"].includes(event.target.tagName))return;if(event.code==="Space"){event.preventDefault();togglePlay();}if(event.key==="ArrowLeft"){event.preventDefault();move(-1);}if(event.key==="ArrowRight"){event.preventDefault();move(1);}});
setInterval(()=>{const now=performance.now();if(!document.hidden)foregroundSeconds+=(now-timerTick)/1000;timerTick=now;persist();},1000);
document.addEventListener("visibilitychange",()=>{timerTick=performance.now();lastTick=null;});
function animate(now){if(current&&!document.hidden){if(playing&&lastTick!==null)playbackTime=(playbackTime+(now-lastTick)/1000*Number(el("speed").value))%Math.max(current.times.at(-1),.001);lastTick=now;renderFrame();}else lastTick=null;requestAnimationFrame(animate);}
renderList();requestAnimationFrame(animate);
// Test hooks expose pure calculations/current UI state; no raw-label mutation.
window.gestureReview={frameAt,mapper,isFailure,selectedPredictions,exportObject,getCurrent:()=>current};
