// Companion renderer, invoked by 01_compare_perspectives.py.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const input = path.resolve(process.argv[2]);
const out = path.dirname(input);
const inputText = await fs.readFile(input, 'utf8');
const data = JSON.parse(inputText);
// Nanosecond timestamps exceed JS's exact integer range; preserve the JSON digits.
const sourceMtimeNs = inputText.match(/"database_mtime_ns":\s*(\d+)/)?.[1] ?? '';
const wb = Workbook.create();
const colors = { navy: '#17324D', teal: '#147D92', muted: '#EAF1F5', text: '#233344' };
const names = ['Overview', 'Entity classes', 'Bucket density', 'Daily', 'Attack types', 'Top entities', 'Method'];
const sheets = Object.fromEntries(names.map(name => [name, wb.worksheets.add(name)]));
const col = index => {
  let n = index + 1, result = '';
  while (n) { result = String.fromCharCode(65 + (n - 1) % 26) + result; n = Math.floor((n - 1) / 26); }
  return result;
};

function table(name, title, headers, rows, options = {}) {
  const sheet = sheets[name];
  sheet.showGridLines = false;
  const last = col(headers.length - 1), end = Math.max(5, rows.length + 4);
  const range = sheet.getRange(`A1:${last}${end}`);
  range.format.font = { name: 'Arial', size: 10, color: colors.text };
  range.format.rowHeight = 23;
  range.format.columnWidth = 19;
  range.format.verticalAlignment = 'center';
  sheet.getRange('A2').values = [[title]];
  sheet.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: colors.navy };
  sheet.getRange(`A4:${last}4`).values = [headers];
  sheet.getRange(`A4:${last}4`).format = {
    fill: colors.navy, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    wrapText: true, rowHeight: 42, horizontalAlignment: 'center', verticalAlignment: 'center',
  };
  if (rows.length) {
    sheet.getRange(`A5:${last}${rows.length + 4}`).values = rows;
    sheet.getRange(`A5:${last}${rows.length + 4}`).setNumberFormat('#,##0');
    const t = sheet.tables.add(`A4:${last}${rows.length + 4}`, true, `Data${names.indexOf(name)}`);
    t.showFilterButton = true;
  }
  for (const [index, width] of Object.entries(options.widths || {})) {
    sheet.getRange(`${col(Number(index))}4:${col(Number(index))}${end}`).format.columnWidth = width;
  }
  for (const index of options.rates || []) sheet.getRange(`${col(index)}5:${col(index)}${end}`).setNumberFormat('0.00%');
  for (const index of options.decimals || []) sheet.getRange(`${col(index)}5:${col(index)}${end}`).setNumberFormat('#,##0.00');
  for (const index of options.dates || []) sheet.getRange(`${col(index)}5:${col(index)}${end}`).setNumberFormat('yyyy-mm-dd');
  if (rows.length > 15) { sheet.freezePanes.freezeRows(4); sheet.freezePanes.freezeColumns(options.freezeColumns || 1); }
  return sheet;
}

const summaries = data.views.map(v => v.summary);
const overviewMetrics = [
  ['Observed entities', 'entities'], ['Benign-only entities', 'benign_only_entities'],
  ['Attack-only entities', 'attack_only_entities'], ['Hybrid entities (all dates)', 'hybrid_entities'],
  ['Flow memberships', 'flows'], ['Attack flow memberships', 'attack_flows'],
  ['Active buckets', 'active_buckets'], ['Benign-only buckets', 'benign_only_buckets'],
  ['Attack-only buckets', 'attack_only_buckets'], ['Hybrid buckets', 'hybrid_buckets'],
  ['One-flow buckets', 'one_flow_buckets'], ['Hybrid / active buckets', null],
  ['One-flow / active buckets', null], ['Median flows / active bucket', 'median'],
  ['P90 flows / active bucket', 'p90'], ['Largest entity flow share', 'top1_flow_share'],
  ['Top 10 entities flow share', 'top10_flow_share'], ['Largest entity attack share', 'top1_attack_share'],
  ['Top 10 entities attack share', 'top10_attack_share'],
  ['Hybrid / attack-bearing buckets', null],
];
const overviewRows = overviewMetrics.map(([label, key]) => [label, ...data.views.map(v => {
  if (key === 'median') return v.density[0].p50;
  if (key === 'p90') return v.density[0].p90;
  return key ? v.summary[key] ?? (key.endsWith('_share') ? null : 0) : null;
})]);
const overview = table('Overview', 'CIC observation perspectives', ['Metric', ...summaries.map(x => x.perspective)], overviewRows,
  { widths: { 0: 37, 1: 20, 2: 20, 3: 20, 4: 20 } });
overview.tabColor = colors.navy;
overview.getRange('A25:H26').format.columnWidth = 19;
overview.getRange('A4:A24').format.columnWidth = 37;
overview.getRange('A25').values = [['Endpoint totals count a flow once at each distinct endpoint. Network counts each raw flow once.']];
overview.getRange('A26').values = [['Hybrid entity spans all dates; hybrid bucket requires both labels in the same time bin.']];
overview.getRange('A25:A26').format.font = { name: 'Arial', size: 10, italic: true, color: '#526476' };
for (let j = 1; j <= summaries.length; j++) {
  const c = col(j);
  overview.getRange(`${c}16`).formulas = [[`=${c}14/${c}11`]];
  overview.getRange(`${c}17`).formulas = [[`=${c}15/${c}11`]];
  overview.getRange(`${c}24`).formulas = [[`=IF(SUM(${c}13:${c}14)=0,"n.a.",${c}14/SUM(${c}13:${c}14))`]];
}
overview.getRange('B16:E17').setNumberFormat('0.00%');
overview.getRange('B18:E19').setNumberFormat('#,##0.00');
overview.getRange('B20:E23').setNumberFormat('0.00%');
overview.getRange('B24:E24').setNumberFormat('0.00%');
overview.getRange('A30:B33').values = [['Perspective', 'Hybrid bucket share'], ...summaries.slice(0, 3).map(s => [s.perspective, null])];
overview.getRange('B31').formulas = [['=B16']];
overview.getRange('B32').formulas = [['=C16']];
overview.getRange('B33').formulas = [['=D16']];
overview.getRange('B31:B33').setNumberFormat('0.00%');
const chart = overview.charts.add('bar', overview.getRange('A30:B33'));
chart.title = `Benign and attack coexistence (${data.metadata.bucket_seconds}s buckets)`;
chart.titleTextStyle.typeface = 'Arial';
chart.titleTextStyle.fontSize = 13;
chart.hasLegend = false;
chart.series.items[0].fill = colors.teal;
chart.yAxis = { numberFormatCode: '0.00%', numberFormatSourceLinked: false, textStyle: { typeface: 'Arial', fontSize: 11 } };
chart.xAxis = { axisType: 'textAxis', textStyle: { typeface: 'Arial', fontSize: 11 } };
chart.setPosition('D29', 'I44');

table('Entity classes', 'Entity composition across all observed dates',
  ['Perspective', 'Activity class', 'Entities', 'Flow memberships', 'Attack flows', 'Active buckets', 'Hybrid buckets'],
  data.views.flatMap(v => v.entity_classes).map(r => [r.perspective, r.activity_class, r.entities || 0, r.flows || 0,
    r.attack_flows || 0, r.active_buckets || 0, r.hybrid_buckets || 0]), { widths: { 1: 24 } });

table('Bucket density', `Flow starts per nonempty ${data.metadata.bucket_seconds}s bucket`,
  ['Perspective', 'Bucket class', 'Buckets', 'Mean', 'P25', 'Median', 'P75', 'P90', 'P99', 'Maximum'],
  data.views.flatMap(v => v.density).map(r => [r.perspective, r.bucket_class, r.n, r.mean, r.p25, r.p50, r.p75, r.p90, r.p99, r.max]),
  { widths: { 1: 24 }, decimals: [3, 4, 5, 6, 7, 8] });

table('Daily', 'Observed activity by UTC calendar day',
  ['Perspective', 'UTC date', 'Entities', 'Flow memberships', 'Attack flows', 'Active buckets', 'Benign buckets', 'Attack-only buckets', 'Hybrid buckets'],
  data.views.flatMap(v => v.daily).map(r => [r.perspective, new Date(`${r.day_utc}T00:00:00Z`), r.entities, r.flows,
    r.attack_flows, r.active_buckets, r.benign_only_buckets || 0, r.attack_only_buckets || 0, r.hybrid_buckets || 0]), { dates: [1] });

table('Attack types', 'Raw attack labels and endpoint coverage',
  ['Attack type', 'Raw flows', 'Source entities', 'Destination entities', 'First flow start (UTC)', 'Last flow start (UTC)'],
  data.attacks.map(r => [r.attack_type, r.flows, r.source_entities, r.destination_entities,
    new Date(r.first_start_utc), new Date(r.last_start_utc)]),
  { widths: { 0: 32, 4: 33, 5: 33 }, dates: [4, 5] });
if (data.attacks.length) sheets['Attack types'].getRange(`E5:F${data.attacks.length + 4}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');

table('Top entities', `Top ${data.metadata.top_n} per perspective and rank basis; complete entities are in CSV`,
  ['Perspective', 'Rank basis', 'Rank', 'Entity', 'Activity class', 'Flows', 'Attack flows', 'Attack share',
    'Active days', 'Active buckets', 'Hybrid buckets', 'Hybrid bucket share', 'Max active run (buckets)',
    'Max within-day unobserved gap (s)', 'Outgoing flows', 'Incoming flows'],
  data.views.flatMap(v => v.top_entities).map(r => [r.perspective, r.rank_basis, r.rank, r.entity, r.activity_class,
    r.flows, r.attack_flows, r.attack_fraction, r.active_days_utc, r.active_buckets, r.hybrid_buckets,
    r.hybrid_bucket_fraction, r.longest_active_run_buckets, r.max_unobserved_gap_within_day_s,
    r.outgoing_flows, r.incoming_flows]), { widths: { 1: 22, 3: 22, 4: 24, 12: 27, 13: 31 }, rates: [7, 11], freezeColumns: 4 });

const methodRows = [
  ['Dataset', data.metadata.dataset], ['Generated (UTC)', new Date(data.metadata.generated_at)],
  ['Source SQLite', data.metadata.database], ['Raw table', 'network_flows; full scan, no sampling or label-based row exclusion'],
  ['Source size (bytes)', data.metadata.database_bytes], ['Source mtime fingerprint', `mtime_ns: ${sourceMtimeNs}`],
  ['Script SHA256', data.metadata.script_sha256], ['Bucket seconds', data.metadata.bucket_seconds],
  ['Sequence length', 'Not selected. This EDA compares observation perspectives before window-length selection.'],
  ['Source', 'Entity = ipv4_src_addr. Each raw flow belongs to one source entity.'],
  ['Destination', 'Entity = ipv4_dst_addr. Each raw flow belongs to one destination entity.'],
  ['Endpoint', 'Entity = either endpoint IP. Count once per distinct endpoint; source=destination counts once.'],
  ['Address identity', 'Entities are literal recorded IPs. Special addresses such as 0.0.0.0 are retained and are not assumed to identify individual devices.'],
  ['Network reference', 'All flows together, one entity ALL. This includes the whole observed dataset, not a declared production sensor.'],
  ['Flow memberships', 'Endpoint totals are 2 x raw rows minus self-flows. They are not additional or independent raw flows.'],
  ['Activity classes', 'benign_only: attack=0; attack_only: attack=total; hybrid: both labels occur.'],
  ['Entity vs bucket hybrid', 'Entity hybrid can span different days. Bucket hybrid means both labels have flow starts in the same aligned bin.'],
  ['Bucket time', 'floor(flow_start_milliseconds / bucket_ms) * bucket_ms. Whole-flow statistics are allocated by start time.'],
  ['No-start bins', 'Not enumerated. No flow start does not prove inactivity: an older flow may still be active or capture may be absent.'],
  ['Active run', 'Consecutive observed bins for the entity; reset at every gap and UTC date boundary. No missing bins are filled.'],
  ['Within-day gap', 'Unobserved intervals between active bins on the same UTC date. May include capture gaps; not asserted to be host idle time.'],
  ['Density quantiles', 'Exact linear interpolation over every active bucket, computed from integer-frequency histograms.'],
  ['Top entities', `Excel shows top ${data.metadata.top_n} by flows and by attack flows for each view. Complete entity details are separate CSV files.`],
  ['Reusable cache', 'perspective_counts.sqlite stores additive per-entity/per-bucket counts for subsequent sequence-length EDA.'],
  ['Source fingerprint', 'Size, mtime and raw totals are recorded; raw database contents are not cryptographically hashed.'],
  ['Raw flows', data.raw.flows], ['Raw attack flows', data.raw.attack_flows], ['Self-flows', data.raw.self_flows],
  ['Flow end before start', data.raw.end_before_start], ['Duration != end minus start', data.raw.duration_mismatch_rows],
  ['Flows lasting >10s', data.raw.flows_longer_than_10s], ['Analysis runtime (s)', data.metadata.analysis_seconds],
  ['Python / SQLite', `${data.metadata.python} / ${data.metadata.sqlite}`],
];
const method = table('Method', 'Definitions and reproducibility', ['Item', 'Definition or value'], methodRows, { widths: { 0: 34, 1: 114 } });
method.getRange(`B5:B${methodRows.length + 4}`).format.wrapText = true;
method.getRange(`A5:B${methodRows.length + 4}`).format.rowHeight = 42;
method.getRange('B6').setNumberFormat('yyyy-mm-dd hh:mm:ss');

wb.recalculate();
const inspection = await wb.inspect({ kind: 'table', range: 'Overview!A4:E24', include: 'values,formulas', tableMaxRows: 21, tableMaxCols: 5, maxChars: 5000 });
await fs.writeFile(path.join(out, 'workbook_inspection.ndjson'), inspection.ndjson, 'utf8');
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 50 }, summary: 'Final formula error scan' });
await fs.writeFile(path.join(out, 'workbook_errors.ndjson'), errors.ndjson, 'utf8');
await fs.mkdir(path.join(out, 'previews'), { recursive: true });
const previewNames = process.env.CIC_EDA_PREVIEW_SHEETS ? process.env.CIC_EDA_PREVIEW_SHEETS.split(',') : names;
for (const name of previewNames) {
  if (!names.includes(name)) throw new Error(`Unknown preview sheet: ${name}`);
  const range = name === 'Overview' ? 'A1:I44' : name === 'Method' ? 'A1:B16' : name === 'Top entities' ? 'A1:H13' : 'A1:F13';
  const preview = await wb.render({ sheetName: name, range, scale: 1, format: 'png' });
  await fs.writeFile(path.join(out, 'previews', `${name.replaceAll(' ', '_')}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(out, '01_cic_perspectives.xlsx'));
console.log(`[excel] Exported ${names.length} sheets and rendered previews`);
