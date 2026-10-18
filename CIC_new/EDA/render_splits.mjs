// Workbook companion for 04_audit_splits.py. Uses only bundled artifact-tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const input = path.resolve(process.argv[2]);
const out = path.dirname(input);
const reports = path.resolve(process.argv[3]);
const d = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const names = ['Comparison', 'Composition', 'Attack coverage', 'Activity', 'Episodes', 'Daily', 'Isolation', 'Method'];
const sheets = Object.fromEntries(names.map(name => [name, wb.worksheets.add(name)]));
const navy = '#17324D';
const col = i => { let n=i+1,s=''; while(n){s=String.fromCharCode(65+(n-1)%26)+s;n=Math.floor((n-1)/26);}return s; };
function table(name, title, headers, rows, opts={}) {
  const s=sheets[name], end=Math.max(5,rows.length+4), last=col(headers.length-1);
  s.showGridLines=false;
  const all=s.getRange(`A1:${last}${end}`);
  all.format.font={name:'Arial',size:10,color:'#233344'};
  all.format.columnWidth=18;
  all.format.rowHeight=24;
  all.format.verticalAlignment='center';
  s.getRange('A2').values=[[title]];
  s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:navy};
  s.getRange(`A4:${last}4`).values=[headers];
  s.getRange(`A4:${last}4`).format={fill:navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    wrapText:true,rowHeight:56,horizontalAlignment:'center',verticalAlignment:'center',
    borders:{insideVertical:{style:'thin',color:'#FFFFFF'}}};
  if(rows.length){
    s.getRange(`A5:${last}${end}`).values=rows;
    s.getRange(`A5:${last}${end}`).setNumberFormat('#,##0');
    s.tables.add(`A4:${last}${end}`,true,`SplitTable${names.indexOf(name)}`).showFilterButton=true;
  }
  for(const [i,width] of Object.entries(opts.widths||{}))s.getRange(`${col(+i)}4:${col(+i)}${end}`).format.columnWidth=width;
  for(const i of opts.rates||[])s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('0.0%');
  for(const i of opts.dates||[])s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
  for(const i of opts.decimals||[])s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('#,##0.0');
  for(const i of opts.booleans||[])s.getRange(`${col(i)}5:${col(i)}${end}`).format.horizontalAlignment='center';
  if(rows.length>15){s.freezePanes.freezeRows(4);s.freezePanes.freezeColumns(opts.freeze||1);}
  return s;
}

const comp=table('Composition','Real sequence composition after candidate exclusions',
  ['Candidate','Split','Sequences','Positive sequences','Positive share','Flows','Attack flows','Sources','UTC dates','Original share'],
  d.composition.map(r=>[r.candidate,r.split,r.sequences,r.positive_sequences,null,r.flows,r.attack_flows,r.sources,r.days,null]),
  {widths:{0:25,1:16},rates:[4,9],freeze:2});
d.composition.forEach((r,i)=>{
  const row=i+5;
  comp.getRange(`E${row}`).formulas=[[`=IF(C${row}=0,"n.a.",D${row}/C${row})`]];
  comp.getRange(`J${row}`).formulas=[[`=C${row}/${d.metadata.source_sequences}`]];
});
const coverage=table('Attack coverage','Attack types retained in each candidate; episodes use a 300s gap',
  ['Candidate','Attack type','Train windows','Validation windows','Test windows','Excluded windows',
   'Test unseen in train','Test only','Validation unseen','All three sets','Train flows','Validation flows','Test flows',
   'Train sources','Validation sources','Test sources','Train episodes','Validation episodes','Test episodes'],
  d.coverage.map(r=>[r.candidate,r.attack_type,...['train','validation','test','excluded'].map(s=>r[s+'_windows']),
    +r.test_unseen_in_train,+r.test_only,+r.validation_unseen_in_train,+r.present_in_all_three,
    ...['train','validation','test'].map(s=>r[s+'_flows']),...['train','validation','test'].map(s=>r[s+'_sources']),
    ...['train','validation','test'].map(s=>r[s+'_episodes'])]),
  {widths:{0:25,1:34},booleans:[6,7,8,9],freeze:2});
for(const c of ['G','H','I'])coverage.getRange(`${c}5:${c}${d.coverage.length+4}`).conditionalFormats.add('cellIs',
  {operator:'equal',formula:1,format:{fill:'#FFF0CC',font:{color:'#865400',bold:true}}});

table('Activity','Observed attack activity; a gap-based episode is not a confirmed campaign',
  ['Attack type','Gap threshold (s)','Candidate episodes','UTC dates','Sources','100s windows','Attack flows','First start (UTC)','Last start (UTC)','Largest episode flow share'],
  d.activity.map(r=>[r.attack_type,r.gap_seconds,r.candidate_episodes,r.days,r.sources,r.windows,r.attack_flows,
    new Date(r.first_utc),new Date(r.last_utc),r.largest_episode_flow_share]),
  {widths:{0:34,7:23,8:23,9:25},dates:[7,8],rates:[9]});
table('Episodes','Candidate attack episodes at a 300s gap; all sources of one type combined',
  ['Episode ID','Attack type','First start (UTC)','Last start (UTC)','Start span (s)','Attack flows','100s windows','Sources','Active 10s bins'],
  d.episodes.map(r=>[r.episode_id,r.attack_type,new Date(r.first_utc),new Date(r.last_utc),r.span_seconds,
    r.attack_flows,r.windows,r.sources,r.bins]),{widths:{1:34,2:23,3:23},dates:[2,3],decimals:[4],freeze:2});
table('Daily','Candidate assignments and exclusions by UTC date',
  ['Candidate','Split','UTC date','Sequences','Positive sequences','Flows','Attack flows'],
  d.daily.map(r=>[r.candidate,r.split,new Date(`${r.day_utc}T00:00:00Z`),r.sequences,r.positive_sequences,r.flows,r.attack_flows]),
  {widths:{0:25,1:16,2:23},dates:[2],freeze:2});
const iso=table('Isolation','Measured separation diagnostics after exclusions; no zero proves campaign independence',
  ['Candidate','Shared global 100s','Adjacent source pairs crossing sets','Interval overlap windows','Shared train/test sources',
   'Cross episodes: 60s','Cross episodes: 300s','Cross episodes: 900s','Train/test episodes: 300s','Retained sequences','Excluded sequences',
   'Embargo exclusions','Interval exclusions','Fold transitions'],
  d.isolation.map(r=>[r.candidate,r.shared_global_100s,r.adjacent_source_pairs_cross_split,r.interval_envelope_overlap_sequences,
    r.source_overlap_train_test,r.cross_episodes_60s,r.cross_episodes_300s,r.cross_episodes_900s,r.train_test_episodes_300s,
    r.retained_sequences,r.excluded_sequences,r.embargo_removed,r.interval_removed,r.boundaries??'n.a.']),
  {widths:{0:25,2:25},freeze:1});
for(const c of ['B','C','D','F','G','H','I'])iso.getRange(`${c}5:${c}${d.isolation.length+4}`).conditionalFormats.add('cellIs',
  {operator:'greaterThan',formula:0,format:{fill:'#FFF0CC',font:{color:'#865400'}}});

const compare=table('Comparison','CIC split candidates: coverage and separation',
  ['Candidate','Train share','Validation share','Test share','Excluded share','Train types','Validation types','Test types',
   'Validation types unseen','Test types unseen','Types in all three','Cross episodes: 300s'],
  d.candidates.map(p=>[p.candidate,...Array(11).fill(null)]),
  {widths:{0:25,8:23,9:23},rates:[1,2,3,4]});
compare.tabColor=navy;
d.candidates.forEach((p,i)=>{
  const row=i+5, maxComp=d.composition.length+4, maxCover=d.coverage.length+4;
  for(const [j,split] of ['train','validation','test','excluded'].entries())
    compare.getRange(`${col(j+1)}${row}`).formulas=[[
      `=SUMIFS('Composition'!$C$5:$C$${maxComp},'Composition'!$A$5:$A$${maxComp},$A${row},'Composition'!$B$5:$B$${maxComp},"${split}")/${d.metadata.source_sequences}`]];
  for(const [j,c] of ['C','D','E'].entries())compare.getRange(`${col(j+5)}${row}`).formulas=[[
    `=COUNTIFS('Attack coverage'!$A$5:$A$${maxCover},$A${row},'Attack coverage'!$${c}$5:$${c}$${maxCover},">0")`]];
  for(const [dest,c] of [['I','I'],['J','G'],['K','J']])compare.getRange(`${dest}${row}`).formulas=[[
    `=COUNTIFS('Attack coverage'!$A$5:$A$${maxCover},$A${row},'Attack coverage'!$${c}$5:$${c}$${maxCover},1)`]];
  compare.getRange(`L${row}`).formulas=[[`='Isolation'!G${row}`]];
});
// Comparison reads completed scientific diagnostics; Isolation is source results,
// not a terminal workbook check/gate. No internal QA formulas feed outputs.
compare.getRange('A16').values=[['Interpretation']];
compare.getRange('A16').format.font={name:'Arial',size:11,bold:true,color:navy};
const notes=[
  ['random_100s','Current implementation baseline; neighboring activity can enter multiple sets.'],
  ['chronological','Measures later-time generalization; attack types can be absent from train.'],
  ['day groups','Keep whole dates together; only a small number of independent date groups exist.'],
  ['global blocks','Keep all sources at one time together; one attack episode can span several blocks.'],
  ['guarded_30m','Activity separation is stronger; this seed leaves only two attack types in validation and two in test.'],
  ['Purging',`Whole-window exclusions use actual flow ends and a ${d.metadata.embargo_seconds}s guard on each side of fold transitions.`],
  ['Scope','Descriptive candidates only. Builder unchanged; exact full-row duplicate audit not performed.'],
];
compare.getRange('A18:B24').values=notes;
compare.getRange('A18:A24').format.font={name:'Arial',size:10,bold:true,color:navy};
compare.getRange('B18:B24').format.wrapText=false;

const methodRows=[
  ['Dataset',d.metadata.dataset],['Raw source','Local datasets/cic_2018_traffic.db, network_flows'],
  ['Catalog source','EDA 02 windows_100s.csv.gz; full catalog, no sampling'],
  ['Raw flows reconciled',d.metadata.raw_flows_reconciled],['Attack flows reconciled',d.metadata.raw_attack_flows_reconciled],
  ['Original sequences',d.metadata.source_sequences],['Seed',d.metadata.seed],['Target proportions','60/20/20 of original sequence counts; groups are indivisible'],
  ['Episode rule','For one attack type, combine all sources; start a new episode after a gap > threshold or at UTC midnight.'],
  ['Gap sensitivity','60, 300 and 900 seconds; activity bins are 10s with exact first/last attack starts.'],
  ['Episode meaning','Candidate contiguous activity only. Similarity, independence and campaign identity are unverified.'],
  ['Group randomization','Sorted groups, NumPy default_rng(seed) permutation, cuts nearest cumulative 60/80% sequence mass. No label-based search.'],
  ['Chronological weighting','Cut by cumulative observed sequence count, not elapsed calendar time; all sources at one time stay together.'],
  ['Guarded 30min groups','Union every observed global 30min block between the first and last attack starts of each 300s-gap episode.'],
  ['Boundary guard',`${d.metadata.embargo_seconds}s before and after each change in assigned global fold; exclude complete intersecting windows.`],
  ['Long flows','Also exclude a window if [first flow start, maximum actual flow end] overlaps another fold\'s assigned 100s window. Conservative envelope.'],
  ['Interval diagnostic','Overlapping envelopes are temporal dependence risks, not proof of duplicate packets. Inclusive flow ends are respected.'],
  ['Episode coverage','Episode counts and attack-window counts may repeat across types or thresholds; do not sum them as unique sequences.'],
  ['Source overlap','Descriptive; same source across sets is allowed in a same-environment evaluation. It is not itself a failed leakage check.'],
  ['Exact duplicate content','Not audited. Different raw row IDs can contain duplicate content; no zero-leakage claim is made.'],
  ['Fitting and synthesis','None. No vocabulary/scaler/model fit; no superposition; validation and test are unmodified real observations except exclusions.'],
  ['Final split','Not selected or frozen by this EDA. Candidate manifests stay in ignored outputs.'],
  ['Runtime (s)',d.metadata.analysis_seconds],['Script SHA256',d.metadata.sha256],
  ...d.candidates.map(p=>[p.candidate,p.method])
];
const method=table('Method','Definitions, sources and reproducibility',['Item','Definition or value'],methodRows,{widths:{0:29,1:112}});
method.getRange(`B5:B${methodRows.length+4}`).format.wrapText=true;
method.getRange(`A5:B${methodRows.length+4}`).format.rowHeight=42;
method.tabColor='#B1BAC4';

wb.recalculate();
await fs.writeFile(path.join(out,'workbook_inspection.ndjson'),(await wb.inspect({kind:'table',range:'Comparison!A4:L12',
  include:'values,formulas',tableMaxRows:10,tableMaxCols:12,maxChars:4500})).ndjson);
await fs.writeFile(path.join(out,'workbook_errors.ndjson'),(await wb.inspect({kind:'match',
  searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30}})).ndjson);
await fs.mkdir(path.join(out,'previews'),{recursive:true});
const selected=process.env.CIC_EDA_PREVIEW_SHEETS?.split(',')||names;
const ranges={'Comparison':'A1:L24','Composition':'A1:J13','Attack coverage':'A1:S12','Activity':'A1:J13',
  'Episodes':'A1:I13','Daily':'A1:G13','Isolation':'A1:N12','Method':'A1:B18'};
for(const name of selected){
  if(!names.includes(name))throw new Error(`Unknown sheet: ${name}`);
  const blob=await wb.render({sheetName:name,range:ranges[name],scale:1,format:'png'});
  await fs.writeFile(path.join(out,'previews',`${name.replaceAll(' ','_')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(reports,'04_cic_split_audit.xlsx'));
// artifact-tool's optional export inspection sidecar belongs with local previews.
const sidecar=path.join(reports,'04_cic_split_audit.xlsx.inspect.ndjson');
try{await fs.rename(sidecar,path.join(out,'workbook_export.inspect.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
console.log(`[excel] Exported ${names.length} sheets to ${reports}`);
