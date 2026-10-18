// Workbook companion for 02_compare_sequence_lengths.py.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const input = path.resolve(process.argv[2]);
const out = path.dirname(input);
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const names = ['Overview', 'Classes', 'Distributions', 'Occupancy', 'Attack types', 'Daily', 'Coverage', 'Method'];
const sheets = Object.fromEntries(names.map(name => [name, wb.worksheets.add(name)]));
const navy = '#17324D', teal = '#147D92', orange = '#C97725';
const col = index => {
  let n = index + 1, result = '';
  while (n) { result = String.fromCharCode(65 + (n - 1) % 26) + result; n = Math.floor((n - 1) / 26); }
  return result;
};

function table(name, title, headers, rows, options = {}) {
  const sheet = sheets[name];
  const last = col(headers.length - 1), end = Math.max(5, rows.length + 4);
  sheet.showGridLines = false;
  const range = sheet.getRange(`A1:${last}${end}`);
  range.format.font = { name: 'Arial', size: 10, color: '#233344' };
  range.format.rowHeight = 23;
  range.format.columnWidth = 19;
  range.format.verticalAlignment = 'center';
  sheet.getRange('A2').values = [[title]];
  sheet.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: navy };
  sheet.getRange(`A4:${last}4`).values = [headers];
  sheet.getRange(`A4:${last}4`).format = {
    fill: navy, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    wrapText: true, rowHeight: 45, horizontalAlignment: 'center', verticalAlignment: 'center',
  };
  if (rows.length) {
    sheet.getRange(`A5:${last}${end}`).values = rows;
    sheet.getRange(`A5:${last}${end}`).setNumberFormat('#,##0');
    const t = sheet.tables.add(`A4:${last}${end}`, true, `Sequences${names.indexOf(name)}`);
    t.showFilterButton = true;
  }
  for (const [index, width] of Object.entries(options.widths || {})) sheet.getRange(`${col(Number(index))}4:${col(Number(index))}${end}`).format.columnWidth = width;
  for (const index of options.rates || []) sheet.getRange(`${col(index)}5:${col(index)}${end}`).setNumberFormat('0.00%');
  for (const index of options.decimals || []) sheet.getRange(`${col(index)}5:${col(index)}${end}`).setNumberFormat('#,##0.00');
  for (const index of options.dates || []) sheet.getRange(`${col(index)}5:${col(index)}${end}`).setNumberFormat('yyyy-mm-dd');
  if (rows.length > 15) { sheet.freezePanes.freezeRows(4); sheet.freezePanes.freezeColumns(options.freezeColumns || 1); }
  return sheet;
}

const metric = (length, name, stat) => length.distributions.find(r => r.activity_class === 'all' && r.metric === name)?.[stat] ?? null;
const overviewItems = [
  ['Buckets / window', r => r.summary.buckets_per_window],
  ['Stride (s)', r => r.summary.stride_s],
  ['Nonempty windows', r => r.summary.windows],
  ['Source IPs represented', r => r.summary.source_hosts],
  ['Benign-only windows', r => r.summary.benign_only],
  ['Attack-only windows', r => r.summary.attack_only],
  ['Hybrid windows', r => r.summary.hybrid],
  ['Attack-bearing window share', () => null],
  ['Hybrid / attack-bearing windows', () => null],
  ['Hybrid with a mixed bucket', r => r.summary.same_bucket_hybrid_windows],
  ['Hybrid across separate buckets only', r => r.summary.cross_bucket_only_hybrid_windows],
  ['Mean bucket occupancy', r => metric(r, 'occupancy', 'mean')],
  ['Median bucket occupancy', r => metric(r, 'occupancy', 'p50')],
  ['P90 bucket occupancy', r => metric(r, 'occupancy', 'p90')],
  ['One-active-bucket window share', () => null],
  ['Fully active window share', () => null],
  ['Median flows / window', r => metric(r, 'flows', 'p50')],
  ['P90 flows / window', r => metric(r, 'flows', 'p90')],
  ['Median longest empty run (s)', r => metric(r, 'longest_empty_run_s', 'p50')],
  ['Median flow-start span (s)', r => metric(r, 'flow_start_span_s', 'p50')],
  ['Raw flow coverage', r => r.summary.flow_coverage],
  ['Window share: global no-start bins', () => null],
  ['Top 10 source window share', r => r.summary.top10_window_share],
  ['Top 10 source attack-window share', r => r.summary.top10_attack_window_share],
];
const overview = table('Overview', 'CIC source-IP sequence lengths', ['Metric', ...data.lengths.map(r => `${r.summary.length_s}s`)],
  overviewItems.map(([label, getter]) => [label, ...data.lengths.map(getter)]), { widths: { 0: 42 } });
overview.tabColor = navy;
for (let j = 1; j <= data.lengths.length; j++) {
  const c = col(j), s = data.lengths[j - 1].summary;
  overview.getRange(`${c}12`).formulas = [[`=SUM(${c}10:${c}11)/${c}7`]];
  overview.getRange(`${c}13`).formulas = [[`=IF(SUM(${c}10:${c}11)=0,"n.a.",${c}11/SUM(${c}10:${c}11))`]];
  // Counts are preserved on Coverage; the ratios reference that table.
  const coverageRow = j + 4;
  overview.getRange(`${c}19`).formulas = [[`='Coverage'!D${coverageRow}/${c}7`]];
  overview.getRange(`${c}20`).formulas = [[`='Coverage'!E${coverageRow}/${c}7`]];
  overview.getRange(`${c}26`).formulas = [[`='Coverage'!F${coverageRow}/${c}7`]];
}
const lastCol = col(data.lengths.length);
for (const row of [12, 13, 16, 17, 18, 19, 20, 25, 26, 27, 28]) overview.getRange(`B${row}:${lastCol}${row}`).setNumberFormat('0.00%');
overview.getRange(`B25:${lastCol}25`).setNumberFormat('0.0000%');
overview.getRange(`B21:${lastCol}24`).setNumberFormat('#,##0.00');
overview.getRange('A30').values = [['Nonoverlapping windows; empty bins retained. Day-end partial windows are excluded and reconciled on Coverage.']];
overview.getRange('A30').format.font = { name: 'Arial', size: 10, italic: true, color: '#526476' };
overview.getRange(`A33:C${data.lengths.length + 33}`).values = [['Length', 'Mean occupancy', 'Median occupancy'],
  ...data.lengths.map(r => [`${r.summary.length_s}s`, null, null])];
for (let j = 1; j <= data.lengths.length; j++) {
  const row = j + 33, c = col(j);
  overview.getRange(`B${row}`).formulas = [[`=${c}16`]];
  overview.getRange(`C${row}`).formulas = [[`=${c}17`]];
}
overview.getRange(`B34:C${data.lengths.length + 33}`).setNumberFormat('0.00%');
const chart = overview.charts.add('line', overview.getRange(`A33:C${data.lengths.length + 33}`));
chart.title = 'Active bucket share by window length';
chart.titleTextStyle.typeface = 'Arial';
chart.titleTextStyle.fontSize = 13;
chart.legend = { position: 'top', textStyle: { typeface: 'Arial', fontSize: 11 } };
chart.yAxis = { numberFormatCode: '0.0%', numberFormatSourceLinked: false, textStyle: { typeface: 'Arial', fontSize: 11 } };
chart.xAxis = { axisType: 'textAxis', textStyle: { typeface: 'Arial', fontSize: 11 } };
chart.series.items[0].line = { fill: teal, style: 'solid', width: 2 };
chart.series.items[1].line = { fill: orange, style: 'solid', width: 2 };
chart.setPosition('E33', 'K49');

table('Classes', 'Natural class composition of nonempty source windows',
  ['Length (s)', 'Activity class', 'Windows', 'Source IPs', 'Flow starts', 'Attack flows', 'Active buckets',
    'Mixed-bucket hybrid windows', 'Separate-bucket hybrid windows', 'Global gap windows'],
  data.lengths.flatMap(r => r.classes).map(r => [r.length_s, r.activity_class, r.windows, r.source_hosts,
    r.flows, r.attack_flows, r.active_buckets, r.same_bucket_hybrid_windows, r.cross_bucket_only_hybrid_windows, r.global_gap_windows]),
  { widths: { 1: 24, 7: 26, 8: 28 } });

const distributions = data.lengths.flatMap(r => r.distributions);
const distributionSheet = table('Distributions', 'Exact empirical quantiles by length and class',
  ['Length (s)', 'Activity class', 'Metric', 'Unit', 'Windows', 'Mean', 'P25', 'Median', 'P75', 'P90', 'P99', 'Maximum'],
  distributions.map(r => [r.length_s, r.activity_class, r.metric, r.unit, r.n, r.mean, r.p25, r.p50, r.p75, r.p90, r.p99, r.max]),
  { widths: { 1: 24, 2: 30, 3: 14 }, decimals: [5, 6, 7, 8, 9, 10, 11], freezeColumns: 3 });
for (let i = 0; i < distributions.length; i++) if (distributions[i].unit === 'fraction') distributionSheet.getRange(`F${i + 5}:L${i + 5}`).setNumberFormat('0.00%');

table('Occupancy', 'Observed active-bucket counts; wholly empty windows are not enumerated',
  ['Length (s)', 'Activity class', 'Active buckets', 'Total buckets', 'Active share', 'Windows', 'Share within class'],
  data.lengths.flatMap(r => r.occupancy).map(r => [r.length_s, r.activity_class, r.active_buckets,
    r.total_buckets, r.occupancy, r.windows, r.class_fraction]), { widths: { 1: 24 }, rates: [4, 6] });

table('Attack types', 'Attack coverage and window composition; multi-type windows may repeat',
  ['Length (s)', 'Attack type', 'Windows', 'Source IPs', 'Attack-only windows', 'Hybrid windows', 'Attack flows retained', 'Raw attack flows', 'Flow coverage'],
  data.lengths.flatMap(r => r.attacks).map(r => [r.length_s, r.attack_type, r.windows, r.source_hosts,
    r.attack_only_windows, r.hybrid_windows, r.attack_flows, r.raw_attack_flows, r.flow_coverage]),
  { widths: { 1: 32 }, rates: [8] });

table('Daily', 'Window composition and density by UTC date',
  ['Length (s)', 'UTC date', 'Windows', 'Source IPs', 'Benign-only', 'Attack-only', 'Hybrid', 'Flow starts', 'Attack flows', 'Mean occupancy', 'Global gap windows'],
  data.lengths.flatMap(r => r.daily).map(r => [r.length_s, new Date(`${r.day_utc}T00:00:00Z`), r.windows,
    r.source_hosts, r.benign_only, r.attack_only, r.hybrid, r.flows, r.attack_flows, r.mean_occupancy, r.global_gap_windows]),
  { dates: [1], rates: [9] });

const coverage = table('Coverage', 'Unique flow-start coverage and boundary exclusions',
  ['Length (s)', 'Total buckets', 'Windows', 'One-active-bucket windows', 'Fully active windows', 'Global gap windows',
    'Flows retained', 'Raw flows', 'Tail flows excluded', 'Attack flows retained', 'Raw attack flows', 'Tail attacks excluded',
    'Flow coverage', 'UTC day tail (s)', 'Active buckets retained', 'Tail buckets excluded'],
  data.lengths.map(r => { const s = r.summary; return [s.length_s, s.buckets_per_window, s.windows,
    s.single_active_bucket_windows || 0, s.fully_active_windows || 0, s.global_gap_windows || 0,
    s.flows, data.controls.flows, s.excluded_tail_flows, s.attack_flows, data.controls.attack_flows,
    s.excluded_tail_attack_flows, s.flow_coverage, s.tail_seconds_per_utc_day, s.active_buckets, s.excluded_tail_buckets]; }),
  { widths: { 3: 27, 4: 23, 5: 23 }, rates: [12] });
coverage.getRange(`M5:M${data.lengths.length + 4}`).setNumberFormat('0.0000%');

const methodRows = [
  ['Dataset', data.metadata.dataset], ['Generated (UTC)', new Date(data.metadata.generated_at)],
  ['Grouping', 'source IP. All recorded addresses are retained, including special addresses such as 0.0.0.0.'],
  ['Bucket', '10 seconds, allocated by flow start. No-start bins do not prove no active connections.'],
  ['Window lengths (s)', data.metadata.lengths_seconds.join(', ')],
  ['Stride', 'Equal to window length. No overlap within one length; different lengths reuse the same original data.'],
  ['Alignment', 'Reset at UTC midnight each date. Window start = midnight + floor(seconds since midnight / length) * length.'],
  ['Day-end tail', 'Exclude incomplete windows crossing midnight; Coverage reports every excluded flow, attack and active bucket.'],
  ['Capture boundaries', 'No capture-session ground truth is assumed. Within-day windows can span no-start network intervals; these are flagged, not filled or silently dropped.'],
  ['Global gap window', 'At least one 10s position has no flow start anywhere in the dataset. This is a diagnostic, not a reliable sensor-availability indicator.'],
  ['Wholly empty windows', 'Not enumerated. All catalog rows have at least one observed source flow start.'],
  ['Internal empty bins', 'Preserved in position. Occupancy = observed source buckets / total window buckets.'],
  ['Classes', 'benign_only: no attack flows; attack_only: no benign flows; hybrid: both labels occur somewhere in the window.'],
  ['Two hybrid types', 'Mixed-bucket: at least one bucket contains both labels. Separate-bucket only: both labels occur in the window but never share a bucket.'],
  ['Temporal overlap', 'Sharing a bucket does not establish actual interval overlap. Separate-bucket flow starts may still belong to overlapping long flows.'],
  ['Longest empty run', 'Includes leading, trailing and internal empty positions. Duration = consecutive empty bucket count x 10 seconds.'],
  ['Flow-start span', 'Last observed flow start minus first observed flow start. It is not connection duration.'],
  ['Quantiles', 'Exact linear-interpolated empirical quantiles. Missing class: count 0, quantiles unavailable.'],
  ['Pooling', 'Overview distributions pool all classes. Use Classes and Distributions to distinguish density differences from changing label composition.'],
  ['Source counts', 'Distinct sources are counted within each length/class/day. A source can recur in multiple classes or dates, so these source counts are not additive.'],
  ['Attack names', 'Read from raw label=1 flows, then checked against cache attack counts. A multi-type window appears in multiple attack rows.'],
  ['Coverage', 'Each retained raw flow is counted once per length. Different lengths are alternatives; their flow totals must not be added together.'],
  ['Catalogs', 'windows_<length>s.csv.gz contains every retained window, its source, absolute boundaries, counts, activity mask and attack names.'],
  ['Activity mask', 'Hex integer; bit 0 is the first bucket. Empty positions are zero. The hex string avoids Excel integer precision limits.'],
  ['Source cache', data.metadata.cache], ['Raw SQLite', data.metadata.raw_database],
  ['Raw mtime fingerprint', `mtime_ns: ${data.metadata.raw_database_mtime_ns}`],
  ['Cache mtime fingerprint', `mtime_ns: ${data.metadata.cache_mtime_ns}`],
  ['Analysis script SHA256', data.metadata.script_sha256], ['EDA 01 analysis SHA256', data.metadata.eda01_script_sha256],
  ['Runtime (s)', data.metadata.analysis_seconds], ['Python / SQLite', `${data.metadata.python} / ${data.metadata.sqlite}`],
  ['Training', 'None. No augmentation, balancing, model selection, threshold calibration or train/test split.'],
];
const method = table('Method', 'Definitions and reproducibility', ['Item', 'Definition or value'], methodRows,
  { widths: { 0: 35, 1: 120 } });
method.getRange(`B5:B${methodRows.length + 4}`).format.wrapText = true;
method.getRange(`A5:B${methodRows.length + 4}`).format.rowHeight = 44;
method.getRange('B6').setNumberFormat('yyyy-mm-dd hh:mm:ss');

wb.recalculate();
const checks = await wb.inspect({ kind: 'table', range: `Overview!A4:${lastCol}28`, include: 'values,formulas',
  tableMaxRows: 25, tableMaxCols: data.lengths.length + 1, maxChars: 6000 });
await fs.writeFile(path.join(out, 'workbook_inspection.ndjson'), checks.ndjson, 'utf8');
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 50 }, summary: 'Final formula error scan' });
await fs.writeFile(path.join(out, 'workbook_errors.ndjson'), errors.ndjson, 'utf8');
await fs.mkdir(path.join(out, 'previews'), { recursive: true });
const previewNames = process.env.CIC_EDA_PREVIEW_SHEETS ? process.env.CIC_EDA_PREVIEW_SHEETS.split(',') : names;
const previewRanges = { Classes: 'A1:J13', Occupancy: 'A1:G13', 'Attack types': 'A1:I13', Daily: 'A1:K13' };
for (const name of previewNames) {
  if (!names.includes(name)) throw new Error(`Unknown preview sheet: ${name}`);
  const range = name === 'Overview' ? `A1:${col(Math.max(10, data.lengths.length))}49` : name === 'Method' ? 'A1:B16'
    : name === 'Distributions' ? 'A1:L13' : name === 'Coverage' ? 'A1:P10' : previewRanges[name];
  const blob = await wb.render({ sheetName: name, range, scale: 1, format: 'png' });
  await fs.writeFile(path.join(out, 'previews', `${name.replaceAll(' ', '_')}.png`), new Uint8Array(await blob.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(out, '02_cic_sequence_lengths.xlsx'));
console.log(`[excel] Exported ${names.length} sheets`);
