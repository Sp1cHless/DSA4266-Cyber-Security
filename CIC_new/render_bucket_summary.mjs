// English, compact, Git-visible report; authoring uses bundled artifact-tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const input=path.resolve(process.argv[2]), out=path.dirname(input), preview=path.resolve(process.argv[3]);
const d=JSON.parse(await fs.readFile(input,'utf8'));
const wb=Workbook.create();
const names=['Summary','Split','Composition','Attack coverage','Features','Checks','Method'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const col=i=>{let n=i+1,s='';while(n){s=String.fromCharCode(65+(n-1)%26)+s;n=Math.floor((n-1)/26);}return s;};
function table(name,title,headers,rows,{widths={},rates=[],decimals=[],wrap=[],height=25,freeze=1}={}){
  const s=sheets[name],last=col(headers.length-1),end=rows.length+4;
  s.showGridLines=false;
  const r=s.getRange(`A1:${last}${Math.max(5,end)}`);
  r.format.font={name:'Arial',size:10,color:'#233344'};
  r.format.columnWidth=20;r.format.rowHeight=height;r.format.verticalAlignment='center';
  s.getRange(`A1:${last}3`).format.rowHeight=24;
  s.getRange('A2').values=[[title]];
  s.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#17324D'};
  s.getRange(`A4:${last}4`).values=[headers];
  s.getRange(`A4:${last}4`).format={fill:'#17324D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:44,horizontalAlignment:'center',verticalAlignment:'center'};
  if(rows.length){s.getRange(`A5:${last}${end}`).values=rows;s.getRange(`A5:${last}${end}`).setNumberFormat('#,##0');s.tables.add(`A4:${last}${end}`,true,`BucketReport${names.indexOf(name)}`).showFilterButton=true;}
  for(const [i,w] of Object.entries(widths))s.getRange(`${col(+i)}4:${col(+i)}${end}`).format.columnWidth=w;
  for(const i of rates)s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('0.00%');
  for(const i of decimals)s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('0.000000');
  for(const i of wrap)s.getRange(`${col(i)}5:${col(i)}${end}`).format.wrapText=true;
  if(rows.length>15){s.freezePanes.freezeRows(4);s.freezePanes.freezeColumns(freeze);}
  return s;
}
table('Composition','Real and synthetic data by split',
 ['Split','Sample type','Sequences','Positive sequences','Bucket positions','Flow presentations','Attack flow presentations','Active buckets','Mean occupancy','Mixed buckets'],
 d.composition.map(r=>[r.split,r.sample_type,r.sequences,r.positive_sequences,r.bucket_positions,r.flow_presentations,r.attack_flow_presentations,r.active_buckets,r.mean_occupancy,r.mixed_buckets]),
 {widths:{0:17,1:29,5:24,6:28},rates:[8]});
const summary=table('Summary','CIC dataset build: global time-block split',
 ['Split','Sequences','Positive sequences','Positive share','Real sequences','Synthetic sequences','Bucket positions','Active buckets','Mean occupancy'],
 ['train','validation','test'].map(s=>[s,null,null,null,null,null,null,null,null]),
 {widths:{0:17,2:23,5:24},rates:[3,8]});
summary.tabColor='#17324D';
const end=d.composition.length+4;
for(let i=5;i<=7;i++){
  const match=`'Composition'!$A$5:$A$${end},A${i}`;
  summary.getRange(`B${i}`).formulas=[[`=SUMIFS('Composition'!C5:C${end},${match})`]];
  summary.getRange(`C${i}`).formulas=[[`=SUMIFS('Composition'!D5:D${end},${match})`]];
  summary.getRange(`D${i}`).formulas=[[`=IF(B${i}=0,"n.a.",C${i}/B${i})`]];
  summary.getRange(`E${i}`).formulas=[[`=SUMIFS('Composition'!C5:C${end},${match},'Composition'!B5:B${end},"real_benign")+SUMIFS('Composition'!C5:C${end},${match},'Composition'!B5:B${end},"real_malicious")`]];
  summary.getRange(`F${i}`).formulas=[[`=B${i}-E${i}`]];
  summary.getRange(`G${i}`).formulas=[[`=SUMIFS('Composition'!E5:E${end},${match})`]];
  summary.getRange(`H${i}`).formulas=[[`=SUMIFS('Composition'!H5:H${end},${match})`]];
  summary.getRange(`I${i}`).formulas=[[`=IF(G${i}=0,"n.a.",H${i}/G${i})`]];
}
const metrics=[['Original training N',d.plan.original_train_N],['Original training positives P',d.plan.original_train_P],
 ['Synthetic B+A',d.plan.synthetic_positive],['Synthetic B+B',d.plan.synthetic_negative],
 ['Feature columns (without mask)',d.schema.feature_count],['L7 slots including UNK',d.schema.l7_ratio_count],
 ['Buckets per sequence',10],['Scaler-fit active buckets',d.scaler_active_buckets],['Unique source flows',d.controls.flows],['Unique source attack flows',d.controls.attack_flows],
 ['Excluded original sequences',d.split_notes.excluded_sequences],['Excluded original flows',d.split_notes.excluded_flows]];
summary.getRange('A10:I22').format.font={name:'Arial',size:10,color:'#233344'};
summary.getRange('A10:I22').format.rowHeight=25;
summary.getRange('A10:I22').format.verticalAlignment='center';
summary.getRange('A10:B10').values=[['Build metric','Value']];
summary.getRange('A10:B10').format={fill:'#17324D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:36,verticalAlignment:'center',horizontalAlignment:'center'};
summary.getRange(`A11:B${10+metrics.length}`).values=metrics;
summary.getRange(`A10:A${10+metrics.length}`).format.columnWidth=42;
summary.getRange(`B11:B${10+metrics.length}`).setNumberFormat('#,##0');
summary.getRange('D11:D15').values=[['Global 15min blocks; 100s guards and actual flow-end exclusions. See Split for original proportions.'],
 ['Keep all retained original training windows; append P B+A and P B+B.'],
 ['Validation/test are original-only. Batch balance is not imposed here.'],
 ['Flow presentations include donor reuse; they are not unique source flows.'],
 ['Features are direct numeric statistics and L7 ratios. No embedding or L2 is required.']];
summary.getRange('D11:D15').format.wrapText=false;
const split=table('Split','Original data after boundary exclusions; before augmentation',
 ['Split','Sequences','Positive sequences','Retained share','Original share','Flows','Attack flows','Sources'],
 d.split_notes.original_composition.map(r=>[r.split,r.sequences,r.positive_sequences,null,null,r.flows,r.attack_flows,r.sources]),
 {widths:{0:35,2:23},rates:[3,4]});
d.split_notes.original_composition.forEach((r,i)=>{
  const row=i+5;
  split.getRange(`D${row}`).formulas=[[`=IF(A${row}="excluded","n.a.",B${row}/SUM(B5:B7))`]];
  split.getRange(`E${row}`).formulas=[[`=B${row}/${d.controls.original_sequences}`]];
});
split.getRange('D5:D8').format.horizontalAlignment='right';
split.getRange('A11').values=[['Attack coverage after exclusions']];
split.getRange('A11').format.font={name:'Arial',size:12,bold:true,color:'#17324D'};
split.getRange('A13:F13').values=[['Attack type','Train windows','Validation windows','Test windows','Validation unseen','Test unseen']];
split.getRange('A13:F13').format={fill:'#17324D',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:44,horizontalAlignment:'center',verticalAlignment:'center'};
const lastType=13+d.split_notes.coverage.length;
split.getRange(`A14:F${lastType}`).values=d.split_notes.coverage.map(r=>[r.attack_type,r.train_sequences,r.validation_sequences,r.test_sequences,
 r.validation_unseen_in_train?'yes':'no',r.test_unseen_in_train?'yes':'no']);
split.getRange(`A14:F${lastType}`).format.font={name:'Arial',size:10,color:'#233344'};
split.getRange(`A14:F${lastType}`).format.rowHeight=25;
split.getRange(`B14:D${lastType}`).setNumberFormat('#,##0');
split.getRange(`E14:F${lastType}`).format.horizontalAlignment='center';
split.tables.add(`A13:F${lastType}`,true,'SplitAttackCoverage').showFilterButton=true;
split.getRange(`C14:D${lastType}`).conditionalFormats.add('cellIs',{operator:'equal',formula:0,
 format:{fill:'#FFF0CC',font:{color:'#865400'}}});
table('Attack coverage','Attack membership; multi-type sequences may appear more than once',
 ['Attack type','Split','Real sequences','Real sources','Real attack flows','Synthetic selected','Synthetic containing'],
 d.attacks.map(r=>[r.attack_type,r.split,r.real_sequences,r.real_sources,r.real_attack_flows,r.synthetic_selected,r.synthetic_containing]),
 {widths:{0:35,1:17,5:24,6:26},freeze:2});
table('Features','Feature schema and original-train scaling',
 ['Index','Feature','Family','Definition','Log1p','Train mean after log','Train scale','Train constant'],
 d.features.map(r=>[r.index,r.feature,r.family,r.definition,r.log1p?'yes':'no',r.train_mean,r.train_scale,r.train_constant?'yes':'no']),
 {widths:{0:10,1:43,2:24,3:100,4:12,5:26,6:26,7:20},decimals:[5,6],wrap:[3],height:43,freeze:2});
for(const c of ['E','H'])sheets['Features'].getRange(`${c}5:${c}${d.features.length+4}`).format.horizontalAlignment='center';
const cell=v=>typeof v==='object'&&v!==null?JSON.stringify(v):v;
table('Checks','Final data reconciliation', ['Check','Expected','Observed','Passed'],
 d.checks.map(r=>[r.check,cell(r.expected),cell(r.observed),r.passed?'yes':'no']),
 {widths:{0:55,1:80,2:80,3:14},wrap:[1,2],height:49});
sheets['Checks'].getRange(`D5:D${d.checks.length+4}`).format.horizontalAlignment='center';
const m=d.metadata,c=m.config;
const methodRows=[['Dataset','NF-CICIDS2018-v3, local SQLite network_flows'],['Generated (UTC+08)',m.generated_at],
 ['Source',c.source.path],['Source size / mtime',`${c.source.bytes} bytes; mtime_ns: ${c.source.mtime_ns}`],
 ['Dataset output',m.output_directory],['Reports',m.report_directory],['Split',JSON.stringify(c.split_ratios)],
 ['Seed',c.seed],['Window / bucket','100s / 10s. Assign by flow start; preserve internal empty buckets.'],
 ['Real split boundary','All sources in one global 15min block have one assignment. Exclude boundary windows with 100s guards plus actual flow-end checks.'],
 ['Coverage','Protect all attack types in training after exclusions; greedily preserve feasible validation/test types. No model-score or seed search.'],
 ['Proportions','Target 60/20/20 of retained original sequences, before augmentation. Group reservations and purge can cause deviations.'],
 ['Evaluation','Same environment, held-out time blocks. Source IPs and attack campaigns may recur across sets.'],
 ['Categorical data','Merge flow records first. Recompute histograms, ratios and diversity. Train-only L7 vocabulary plus UNK.'],
 ['Normalization','log1p on nonnegative magnitudes, then original-train active-bucket mean/std. Restore empty buckets to zero.'],
 ['Shifts','Shift both endpoints of the second donor by one common offset; all starts remain in [0,100000). Long flow ends may extend beyond it.'],
 ['Sampling','Equal attack-type/source/sequence donor selection; benign backgrounds by active-bin band/source/sequence.'],
 ['Metadata exclusions','IP literals, labels, attack names, source IDs and synthetic flags are not model inputs.'],
 ['Missing validation types',d.split_notes.coverage.filter(r=>!r.validation_sequences).map(r=>r.attack_type).join(', ')||'none'],
 ['Missing test types',d.split_notes.coverage.filter(r=>!r.test_sequences).map(r=>r.attack_type).join(', ')||'none'],
 ['Duplicate rows','Identical complete rows share the source/start grouping key and cannot cross sets. Within-window duplicates are retained.'],
 ['Code SHA256',JSON.stringify(c.code_sha256)],
 ['Runtime',`Python ${m.python}; NumPy ${m.numpy}`],['Further definitions','See metrics_explained.md alongside this workbook.']];
const method=table('Method','Definitions and reproducibility', ['Item','Definition or value'],methodRows,
 {widths:{0:31,1:135},wrap:[1],height:50});
method.getRange('B6').values=[[new Date(m.generated_at.slice(0,19)+'Z')]];
method.getRange('B6').setNumberFormat('yyyy-mm-dd hh:mm:ss');
wb.recalculate();
await fs.mkdir(preview,{recursive:true});
const inspect=await wb.inspect({kind:'table',range:'Summary!A4:I7',include:'values,formulas',tableMaxRows:4,tableMaxCols:9,maxChars:4000});
await fs.writeFile(path.join(preview,'inspection.ndjson'),inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Final formula error scan'});
await fs.writeFile(path.join(preview,'errors.ndjson'),errors.ndjson);
const ranges={Summary:'A1:I23',Split:`A1:H${Math.max(18,lastType)}`,Composition:'A1:J13','Attack coverage':'A1:G13',Features:'A1:H11',Checks:'A1:D16',Method:'A1:B17'};
const previews=process.env.CIC_BUCKET_PREVIEW_SHEETS?process.env.CIC_BUCKET_PREVIEW_SHEETS.split(','):names;
for(const name of previews){
  if(!names.includes(name))throw new Error(`Unknown preview sheet ${name}`);
  const blob=await wb.render({sheetName:name,range:ranges[name],scale:1,format:'png'});
  await fs.writeFile(path.join(preview,name.replaceAll(' ','_')+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
const temporary=path.join(out,'bucket_summary.xlsx.partial');
await xlsx.save(temporary);
await fs.rename(temporary,path.join(out,'bucket_summary.xlsx'));
try{
  await fs.rename(temporary+'.inspect.ndjson',path.join(preview,'export_inspection.ndjson'));
}catch(error){if(error.code!=='ENOENT')throw error;}
console.log(`[report] Exported ${names.length} sheets to ${out}`);
