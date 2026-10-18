// Workbook companion for 03_audit_features.py; bundled artifact-tool only.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const input = path.resolve(process.argv[2]), out = path.dirname(input);
const d = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const names = ['Overview', 'Profiles', 'Categories', 'Legacy 88', 'Aggregation', 'Redundancy', 'Diagnostics', 'Method'];
const sheets = Object.fromEntries(names.map(n => [n, wb.worksheets.add(n)]));
const navy = '#17324D';
const col = i => { let x = i + 1, s = ''; while (x) { s = String.fromCharCode(65 + (x - 1) % 26) + s; x = Math.floor((x - 1) / 26); } return s; };
function table(name, title, headers, rows, {widths = {}, rates = [], decimals = [], wrap = [], height = 24, freeze = 1} = {}) {
  const s = sheets[name], end = rows.length + 4, last = col(headers.length-1);
  s.showGridLines = false;
  const area = s.getRange(`A1:${last}${Math.max(5, end)}`);
  area.format.font = {name:'Arial', size:10, color:'#233344'};
  area.format.rowHeight = height;
  area.format.columnWidth = 18;
  area.format.verticalAlignment = 'center';
  s.getRange(`A1:${last}3`).format.rowHeight = 24;
  s.getRange('A2').values = [[title]];
  s.getRange('A2').format.font = {name:'Arial',size:14,bold:true,color:navy};
  s.getRange(`A4:${last}4`).values = [headers];
  s.getRange(`A4:${last}4`).format = {fill:navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:44,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
  if (rows.length) {
    s.getRange(`A5:${last}${end}`).values = rows;
    s.getRange(`A5:${last}${end}`).setNumberFormat('#,##0');
    s.tables.add(`A4:${last}${end}`,true,`FeatureAudit${names.indexOf(name)}`).showFilterButton = true;
  }
  for (const [i,width] of Object.entries(widths)) s.getRange(`${col(+i)}4:${col(+i)}${end}`).format.columnWidth = width;
  for (const i of rates) s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('0.00%');
  for (const i of decimals) s.getRange(`${col(i)}5:${col(i)}${end}`).setNumberFormat('#,##0.000');
  for (const i of wrap) s.getRange(`${col(i)}5:${col(i)}${end}`).format.wrapText = true;
  if (rows.length > 18) {s.freezePanes.freezeRows(4); s.freezePanes.freezeColumns(freeze);}
  return s;
}
const c = d.controls, m = d.metadata;
const overviewRows = [
  ['Raw flows', c.flows],['Benign flows', c.benign],['Attack flows', c.attack],['Raw fields',c.raw_fields],
  ['Numeric raw fields',c.numeric_fields],['Legacy model features',c.legacy_features],
  ['Observed L7 categories',c.l7_categories],['Flows outside legacy L7 vocabulary',c.l7_outside_legacy],
  ['Outside legacy L7 / all flows',null],['Outside L7 / benign flows',null],['Outside L7 / attack flows',null],
  ['Legacy L7 categories with no flow support',c.unsupported_legacy_l7],
  ['Full-table constant numeric fields',d.profiles.filter(r=>r.population==='all'&&r.constant).length],
  ['Bernoulli sample rows',m.sample_actual],['Sample benign rows',m.sample_benign],['Sample attack rows',m.sample_attack],
];
const ov = table('Overview','CIC feature audit', ['Metric','Value'],overviewRows,{widths:{0:49,1:23}});
ov.tabColor = navy;
ov.getRange('B13').formulas = [['=B12/B5']];
ov.getRange('B13:B15').setNumberFormat('0.00%');
ov.getRange('D5').values = [['Feature decisions']];
ov.getRange('D5').format.font = {name:'Arial',size:11,bold:true,color:navy};
const notes = [
  `${c.l7_categories} L7 codes; ${(100*c.l7_outside_legacy/c.flows).toFixed(2)}% of flows fall outside the old vocabulary. Rebuild the training vocabulary.`,
  `${d.legacy.filter(r=>r.supporting_flows===0).length} legacy features have no supporting flow and are identically zero under the existing definitions.`,
  'FTP return code is constant zero. Its 8 derived inputs can be removed from the CIC schema.',
  `${(100*d.diagnostics.find(r=>r.check==='duration_differs_from_endpoint_span').affected/c.flows).toFixed(2)}% of durations differ from end minus start. Preserve both values and verify semantics.`,
  'Some exported packet IAT averages lie outside their extrema. Inspect semantics before adding this group.',
  'Keep additional raw candidates. This audit does not rank predictive value or measure bucket-feature variance.',
  'Fit vocabulary, scaling and feature selection after splitting original donors. This EDA fits none.',
];
ov.getRange('D7:D13').values = notes.map(x=>[x]);
ov.getRange('D7:D13').format.columnWidth = 103;
ov.getRange('D7:D13').format.wrapText = true;
ov.getRange('A7:D13').format.rowHeight = 35;

const qs = new Map(d.quantiles.map(q=>[`${q.field}/${q.population}`,q]));
table('Profiles','Full-table quality and sampled raw-value quantiles',
  ['Field','Population','Rows','Finite','SQL nulls','Nonfinite nonnull','Zero values','Zero / finite','Negative','Minimum','Maximum','Mean','Population std','Sample finite','Sample P50','Sample P90','Sample P99','Constant'],
  d.profiles.map(r=>{const q=qs.get(`${r.field}/${r.population}`);return [r.field,r.population,r.rows,r.finite,r.nulls,r.nonfinite,r.zeros,r.zero_fraction,r.negative,r.minimum,r.maximum,r.mean,r.std_population,q?.sample_finite??null,q?.p50??null,q?.p90??null,q?.p99??null,r.constant?'yes':'no'];}),
  {widths:{0:37,1:15,5:20,9:24,10:24,11:28,12:24,14:28,15:28,16:28},rates:[7],decimals:[11,12,14,15,16],freeze:2});
sheets['Profiles'].getRange(`R5:R${d.profiles.length+4}`).format.horizontalAlignment = 'center';

// All small-cardinality categories; only ports are shortened to top 50 in Excel.
// Complete port frequencies remain in categories.csv and results.json.
let categories = d.categories.filter(r=>!r.field.startsWith('l4_'));
for (const f of ['l4_src_port','l4_dst_port']) categories.push(...d.categories.filter(r=>r.field===f).sort((a,b)=>b.flows-a.flows||Number(a.value)-Number(b.value)).slice(0,50));
categories.sort((a,b)=>a.field.localeCompare(b.field)||b.flows-a.flows||Number(a.value)-Number(b.value));
const categoryRows = categories.map(r=>[r.field,String(r.value),r.flows,r.benign,r.attack,c.flows,r.flow_fraction,r.attack_fraction,
  r.in_legacy_l7===null?'n.a.':r.in_legacy_l7?'yes':'no',r.field.startsWith('l4_')?'top 50 ports; full CSV':'all observed values']);
categoryRows.push(...d.flags.map(r=>[r.field,r.flag,r.support_flows,r.benign,r.attack,r.tcp_flows,r.tcp_fraction,
  r.support_flows?r.attack/r.support_flows:null,'n.a.','TCP-eligible flag occurrences']));
table('Categories','Category frequencies and TCP bit support',
  ['Field','Value / bit','Flows','Benign','Attack','Share denominator','Flow share','Attack / value flows','In legacy L7','Coverage'],
  categoryRows,{widths:{0:30,1:18,7:23,9:40},rates:[6,7],freeze:2});
const categoryEnd = categoryRows.length + 4;
sheets['Categories'].getRange(`B5:B${categoryEnd}`).setNumberFormat('@');
sheets['Categories'].getRange(`I5:I${categoryEnd}`).format.horizontalAlignment = 'center';
ov.getRange('B14').formulas = [[`=SUMIFS('Categories'!D5:D${categoryEnd},'Categories'!A5:A${categoryEnd},"l7_proto",'Categories'!I5:I${categoryEnd},"no")/B6`]];
ov.getRange('B15').formulas = [[`=SUMIFS('Categories'!E5:E${categoryEnd},'Categories'!A5:A${categoryEnd},"l7_proto",'Categories'!I5:I${categoryEnd},"no")/B7`]];

table('Legacy 88','Existing feature definitions and CIC support',
  ['Feature','Family','Raw dependencies','Merge rule','Supporting flows','Recommendation','Interpretation'],
  d.legacy.map(r=>[r.feature,r.family,r.raw_dependencies,r.merge_rule,r.supporting_flows,r.recommendation,r.note]),
  {widths:{0:34,1:24,2:50,3:72,4:20,5:32,6:86},wrap:[2,3,5,6],height:48,freeze:2});
table('Aggregation','Raw-field candidate registry for event superposition',
  ['Raw field','SQLite type','Family','Unit','Legacy dependency','Proposed role','Constant','Aggregation proposal','Caveat'],
  d.registry.map(r=>[r.field,r.sqlite_type,r.family,r.unit,r.used_by_legacy?'yes':'no',r.model_role,r.constant_full_table?'yes':'no',r.aggregation,r.note]),
  {widths:{0:37,1:14,2:23,3:28,4:18,5:28,6:14,7:90,8:94},wrap:[3,5,7,8],height:58,freeze:1});
const redundancy = table('Redundancy','Sample raw-scalar correlations; no automatic feature removal',
  ['Left field','Right field','Paired sample','Pearson log1p','Absolute correlation','Equal raw values'],
  d.correlations.map(r=>[r.left,r.right,r.paired_sample,r.pearson_log1p,r.abs_correlation,r.equal_value_fraction]),
  {widths:{0:39,1:39,2:20,3:23,4:24,5:23},decimals:[3,4],rates:[5],freeze:2});
redundancy.getRange(`D5:E${d.correlations.length+4}`).setNumberFormat('0.000000');
const diag = table('Diagnostics','Temporal, range and exporter consistency diagnostics',
  ['Check','Affected rows','Eligible rows','Affected share','Interpretation'],
  d.diagnostics.map(r=>[r.check,r.affected,r.eligible,null,r.note]),
  {widths:{0:51,1:20,2:20,3:20,4:75},rates:[3],height:30});
for(let i=0;i<d.diagnostics.length;i++)diag.getRange(`D${i+5}`).formulas=[[`=IF(C${i+5}=0,"n.a.",B${i+5}/C${i+5})`]];
const textRows = d.text_profiles.map(r=>[r.field,`${r.rows} rows; ${r.nulls} SQL nulls; ${r.empty_strings} empty strings; ${r.distinct_nonnull} distinct nonnull values`]);
const methodRows = [
  ['Dataset','NF-CICIDS2018-v3; raw SQLite network_flows; no raw writes.'],
  ['Raw source',m.raw_database],['Schema reference',m.legacy_schema],
  ['Dataset documentation',m.sources[0]],['Exporter field reference',m.sources[1]],
  ['Generated (UTC+08)',m.generated_at],['Raw size / mtime',`${m.raw_bytes} bytes; mtime_ns: ${m.raw_mtime_ns}`],
  ['Analysis SHA256',m.script_sha256],['Legacy schema SHA256',m.legacy_sha256],
  ['Runtime',`${m.python} Python; numpy ${m.numpy}; SQLite ${m.sqlite}; ${m.analysis_seconds}s`],
  ['Full scan','All flows. Numeric null/nonfinite/zero/negative counts, extrema and population moments. Floating moments may have rounding error.'],
  ['Sampling',`${m.sampling}; seed ${m.sample_seed}; target ${m.sample_target}; p=${m.sample_probability}; actual ${m.sample_actual}`],
  ['Quantiles','Sample empirical quantiles, linear interpolation. Not exact full-data quantiles; sample size can be small for rare values.'],
  ['Correlations','Pearson correlation of log1p nonnegative finite raw values. Pairwise filtering. Constant pairs omitted. No IP, labels, dates, category codes or bitmask integers.'],
  ['Interpretation','Correlation is pooled across labels and sources. It can reflect composition, not causality, model importance or post-bucket redundancy.'],
  ['Populations','all = full table; benign / attack = raw label 0 / 1; eligible = field-specific protocol/presence/packet-count condition. Eligible overlaps label groups.'],
  ['TCP eligibility','protocol=6; used for all three flags and TCP window fields.'],
  ['ICMP eligibility','protocol=1. Type code 0 can be a valid ICMP value.'],
  ['DNS eligibility','Legacy heuristic: nonzero query ID OR query type OR answer TTL. Not exporter-verified DNS ground truth.'],
  ['FTP eligibility','Legacy heuristic: nonzero return code. Not all FTP traffic necessarily has an extracted return code.'],
  ['IAT eligibility','in_pkts>=2 for src_to_dst; out_pkts>=2 for dst_to_src. A diagnostic condition, not proof of available packet timestamps. Units not independently verified.'],
  ['Zeros','Zero is reported as stored. It can mean true zero, protocol not applicable, no exporter extraction, or sentinel. No general zero-to-missing conversion.'],
  ['Categories','Exact frequencies over all flows; TCP bit rows use TCP-flow denominator. Excel ports show top 50; full categories.csv / results.json contain every value.'],
  ['Legacy support','Fixed L7 slots, TCP bits and FTP code/presence features have supporting-flow counts. Blank means not applicable. Zero support implies zero-valued legacy slots; this is not a complete bucket-variance audit.'],
  ['Identity and target','Keep raw source/destination/timestamps/labels for reconstruction and split provenance. Literal IDs, absolute time, labels and attack names are not proposed model inputs.'],
  ['Aggregation','Candidate rules only. Keep counts and frequency maps, then recompute ratios/distinct/entropy after event merge. An average of per-flow summaries is not a pooled packet statistic.'],
  ['Scope','Source-IP grouping, 10s buckets and 100s sequences remain the chosen design. This audit examines raw features, not a new sequence-length study or model ranking.'],
  ['Next use','Split original donors first; fit vocabulary/scaler/selector on training only. Preserve separate validity masks and activity mask.'],
  ...textRows,
];
const method = table('Method','Definitions, source references and reproducibility', ['Item','Definition or value'],methodRows,
  {widths:{0:32,1:130},wrap:[1],height:49});
// Excel dates have no timezone; preserve the Beijing wall clock explicitly.
method.getRange('B10').values = [[new Date(m.generated_at.slice(0,19)+'Z')]];
method.getRange('B10').setNumberFormat('yyyy-mm-dd hh:mm:ss');
wb.recalculate();
const inspection=await wb.inspect({kind:'table',range:'Overview!A4:B20',include:'values,formulas',tableMaxRows:18,tableMaxCols:2,maxChars:4000});
await fs.writeFile(path.join(out,'workbook_inspection.ndjson'),inspection.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Final formula error scan'});
await fs.writeFile(path.join(out,'workbook_errors.ndjson'),errors.ndjson);
await fs.mkdir(path.join(out,'previews'),{recursive:true});
const previewRanges={Overview:'A1:D20',Profiles:'A1:R10',Categories:'A1:J13','Legacy 88':'A1:G10',Aggregation:'A1:I10',Redundancy:'A1:F13',Diagnostics:'A1:E13',Method:'A1:B13'};
const selected=process.env.CIC_EDA_PREVIEW_SHEETS?process.env.CIC_EDA_PREVIEW_SHEETS.split(','):names;
for(const name of selected){
  if(!names.includes(name))throw new Error(`Unknown preview sheet ${name}`);
  const blob=await wb.render({sheetName:name,range:previewRanges[name],scale:1,format:'png'});
  await fs.writeFile(path.join(out,'previews',`${name.replaceAll(' ','_')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(out,'03_cic_feature_audit.xlsx'));
console.log(`[excel] ${names.length} sheets exported`);
