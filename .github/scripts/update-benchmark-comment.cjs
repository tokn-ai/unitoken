const fs = require('node:fs');
const path = require('node:path');

const MARKER = '<!-- unitoken-benchmark-report -->';
const MAX_REPORT_BYTES = 2 * 1024 * 1024;
const TRAINER_LABELS = new Map([
  ['smoke_en_byte_v300', 'English byte, vocab 300'],
  ['smoke_en_byte_v1000', 'English byte, vocab 1k'],
  ['smoke_zh_unicode_v300', 'Chinese Unicode, vocab 300'],
  ['smoke_zh_unicode_v1000', 'Chinese Unicode, vocab 1k'],
  ['smoke_zh_unicode_bbpe_r90_v1000', 'Chinese Unicode BBPE, vocab 1k'],
]);

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

function average(values) {
  const valid = values.map(finiteNumber).filter((value) => value !== null);
  if (valid.length === 0) {
    return null;
  }
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function loadReport(resultsDir, relativePath, contract, errors, optional) {
  const reportPath = path.join(resultsDir, relativePath);
  try {
    const stat = fs.lstatSync(reportPath);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > MAX_REPORT_BYTES) {
      throw new Error('invalid report file');
    }
    const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
    if (
      !isRecord(report)
      || report.schema_version !== 1
      || report.contract !== contract
      || !isRecord(report.gates)
      || typeof report.gates.passed !== 'boolean'
      || !Array.isArray(report.samples)
    ) {
      throw new Error('invalid report contract');
    }
    return { status: 'present', report };
  } catch (error) {
    if (optional && error?.code === 'ENOENT') {
      return { status: 'absent', report: null };
    }
    errors.push(relativePath);
    return { status: 'invalid', report: null };
  }
}

function readReport(resultsDir, relativePath, contract, errors) {
  return loadReport(resultsDir, relativePath, contract, errors, false).report;
}

function readOptionalReport(resultsDir, relativePath, contract, errors) {
  return loadReport(resultsDir, relativePath, contract, errors, true);
}

function readMetadata(resultsDir) {
  const metadataPath = path.join(resultsDir, 'metadata.json');
  const stat = fs.lstatSync(metadataPath);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 4096) {
    throw new Error('invalid benchmark metadata file');
  }
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  if (
    !isRecord(metadata)
    || metadata.schema_version !== 1
    || !Number.isSafeInteger(metadata.pull_request_number)
    || metadata.pull_request_number <= 0
    || !/^[0-9a-f]{40}$/.test(metadata.base_sha)
    || !/^[0-9a-f]{40}$/.test(metadata.head_sha)
  ) {
    throw new Error('invalid benchmark metadata contract');
  }
  return metadata;
}

function variantLabel(variant) {
  if (variant?.occurrence_mode === 'exact') {
    return 'exact';
  }
  if (
    variant?.occurrence_mode === 'bounded'
    && Number.isSafeInteger(variant.hot_pair_window_size)
    && variant.hot_pair_window_size > 0
  ) {
    return `k${variant.hot_pair_window_size}`;
  }
  return null;
}

function stableValue(value) {
  if (Array.isArray(value)) {
    return value.map(stableValue);
  }
  if (!isRecord(value)) {
    return value;
  }
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, stableValue(value[key])]),
  );
}

function trainerWorkload(sample) {
  const requestCase = isRecord(sample?.request?.case) ? sample.request.case : {};
  const expectedInputSha256 = requestCase.expected_input_sha256;
  const normalizedCase = {
    ...requestCase,
    bbpe_fallback: requestCase.bbpe_fallback ?? false,
    primary_vocab_ratio: requestCase.primary_vocab_ratio ?? 0.9,
  };

  // Checkout-local paths differ between the base and candidate worktrees. The
  // expected and measured input fingerprints identify the corpus instead.
  delete normalizedCase.words_path;
  // Golden inputs and outputs are assertions about the work, not its settings.
  delete normalizedCase.expected_input_sha256;
  delete normalizedCase.expected_model_sha256;

  return JSON.stringify(stableValue({
    case: normalizedCase,
    variant: sample?.request?.variant,
    input_sha256: sample?.measurement?.input?.sha256
      ?? expectedInputSha256
      ?? null,
  }));
}

function trainerRows(report) {
  const rows = new Map();
  for (const sample of report?.samples ?? []) {
    const caseName = sample?.request?.case?.name;
    const variant = variantLabel(sample?.request?.variant);
    if (typeof caseName !== 'string' || caseName.length === 0 || variant === null) {
      continue;
    }
    const key = JSON.stringify([caseName, variant]);
    let row = rows.get(key);
    if (!row) {
      row = {
        caseName,
        variant,
        times: [],
        rssValues: [],
        workloads: new Set(),
        failed: report.gates.passed === false,
      };
      rows.set(key, row);
    }
    row.failed ||= sample?.status === 'failed' || sample?.error != null;
    row.times.push(sample?.measurement?.timing?.core_training_ns);
    row.rssValues.push(
      sample?.measurement?.memory?.process_peak_rss_through_training_bytes,
    );
    row.workloads.add(trainerWorkload(sample));
  }
  for (const row of rows.values()) {
    row.time = average(row.times);
    row.rss = average(row.rssValues);
  }
  return rows;
}

function unionTrainerRows(
  baseline,
  candidate,
  baselineAvailable,
  candidateAvailable,
) {
  const rows = new Map();
  for (const [key, row] of baseline) {
    rows.set(key, {
      caseName: row.caseName,
      variant: row.variant,
      baseline: row,
      candidate: null,
      baselineAvailable,
      candidateAvailable,
    });
  }
  for (const [key, row] of candidate) {
    const existing = rows.get(key);
    if (existing) {
      existing.candidate = row;
    } else {
      rows.set(key, {
        caseName: row.caseName,
        variant: row.variant,
        baseline: null,
        candidate: row,
        baselineAvailable,
        candidateAvailable,
      });
    }
  }
  return [...rows.values()];
}

function sameWorkloads(baseline, candidate) {
  if (baseline.workloads.size !== candidate.workloads.size) {
    return false;
  }
  return [...baseline.workloads].every((workload) => candidate.workloads.has(workload));
}

function pretokenizerValues(report) {
  const values = new Map();
  const samples = report?.samples ?? [];
  values.set('bigram', average(
    samples.map((sample) => sample?.measurement?.timing?.bigram_pass_ns),
  ));
  values.set('word', average(
    samples.map((sample) => sample?.measurement?.timing?.word_pass_ns),
  ));
  values.set('total', average(
    samples.map((sample) => sample?.measurement?.timing?.core_pretokenizer_ns),
  ));
  values.set('rss', average(
    samples.map(
      (sample) => sample?.measurement?.memory?.process_peak_rss_through_core_bytes,
    ),
  ));
  return values;
}

function codecValues(report) {
  const values = new Map();
  const samples = report?.samples ?? [];
  values.set('encode', average(
    samples.map((sample) => sample?.encode?.timing?.encode_ns),
  ));
  values.set('decode', average(
    samples.map((sample) => sample?.decode?.timing?.decode_ns),
  ));
  values.set('encode_rss', average(
    samples.map(
      (sample) => sample?.encode?.memory?.process_peak_rss_through_phase_bytes,
    ),
  ));
  values.set('decode_rss', average(
    samples.map(
      (sample) => sample?.decode?.memory?.process_peak_rss_through_phase_bytes,
    ),
  ));
  return values;
}

function codecWorkload(report) {
  const config = { ...(report?.config ?? {}) };
  for (const key of Object.keys(config)) {
    if (key === 'name' || key.endsWith('_path') || key.startsWith('expected_')) {
      delete config[key];
    }
  }
  const measurements = [...new Set(
    (report?.samples ?? []).map((sample) => JSON.stringify(stableValue({
      input: sample?.encode?.input ?? null,
      model: sample?.encode?.model ?? null,
    }))),
  )].sort();
  return JSON.stringify(stableValue({ config, measurements }));
}

function formatMilliseconds(value) {
  return value === null ? 'n/a' : `${(value / 1_000_000).toFixed(2)} ms`;
}

function formatMebibytes(value) {
  return value === null ? 'n/a' : `${(value / 1024 / 1024).toFixed(1)} MiB`;
}

function formatDelta(baseline, candidate) {
  if (baseline === null || candidate === null || baseline === 0) {
    return 'n/a';
  }
  const delta = ((candidate - baseline) / baseline) * 100;
  return `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`;
}

function tableRow(label, baseline, candidate, formatter) {
  return `| ${label} | ${formatter(baseline)} | ${formatter(candidate)} | ${formatDelta(baseline, candidate)} |`;
}

function escapeTableCell(value) {
  return value.replaceAll('|', '\\|').replace(/[\r\n]+/g, ' ');
}

function trainerCellValue(row, reportAvailable, metric, formatter) {
  if (!reportAvailable) {
    return 'unavailable';
  }
  if (!row) {
    return 'missing';
  }
  return row.failed ? 'failed' : formatter(row[metric]);
}

function trainerTableRow(row, metric, formatter) {
  const baseline = row.baseline;
  const candidate = row.candidate;
  const baselineValue = trainerCellValue(
    baseline,
    row.baselineAvailable,
    metric,
    formatter,
  );
  const candidateValue = trainerCellValue(
    candidate,
    row.candidateAvailable,
    metric,
    formatter,
  );
  let delta = 'n/a';
  if (baseline && candidate && !baseline.failed && !candidate.failed) {
    delta = sameWorkloads(baseline, candidate)
      ? formatDelta(baseline[metric], candidate[metric])
      : 'changed';
  }
  const caseLabel = TRAINER_LABELS.get(row.caseName) ?? row.caseName;
  const label = escapeTableCell(`Trainer — ${caseLabel} (${row.variant})`);
  return `| ${label} | ${baselineValue} | ${candidateValue} | ${delta} |`;
}

function optionalReportCell(state, values, metric, formatter) {
  if (state.status === 'invalid') {
    return 'unavailable';
  }
  if (state.status === 'absent') {
    return 'missing';
  }
  if (state.report.gates.passed !== true) {
    return 'failed';
  }
  return formatter(values.get(metric) ?? null);
}

function optionalCodecTableRow(
  label,
  baseline,
  candidate,
  baselineValues,
  candidateValues,
  metric,
  formatter,
) {
  const baselineValue = baselineValues.get(metric) ?? null;
  const candidateValue = candidateValues.get(metric) ?? null;
  let delta = 'n/a';
  if (
    baseline.status === 'present'
    && baseline.report.gates.passed === true
    && candidate.status === 'present'
    && candidate.report.gates.passed === true
  ) {
    delta = codecWorkload(baseline.report) === codecWorkload(candidate.report)
      ? formatDelta(baselineValue, candidateValue)
      : 'changed';
  }
  return `| ${label} | ${optionalReportCell(baseline, baselineValues, metric, formatter)} | ${optionalReportCell(candidate, candidateValues, metric, formatter)} | ${delta} |`;
}

function reportSet(resultsDir, side, errors) {
  const prefix = `${side}/`;
  return {
    trainer: readReport(
      resultsDir,
      `${prefix}trainer.json`,
      'unitoken_trainer_regression_v1',
      errors,
    ),
    pretokenizer: readReport(
      resultsDir,
      `${prefix}pretokenizer.json`,
      'unitoken_pretokenizer_regression_v1',
      errors,
    ),
    byteCodec: readReport(
      resultsDir,
      `${prefix}codec-byte.json`,
      'unitoken_codec_regression_v1',
      errors,
    ),
    unicodeCodec: readReport(
      resultsDir,
      `${prefix}codec-unicode.json`,
      'unitoken_codec_regression_v1',
      errors,
    ),
    bbpeUnicodeCodec: readOptionalReport(
      resultsDir,
      `${prefix}codec-unicode-bbpe.json`,
      'unitoken_codec_regression_v1',
      errors,
    ),
  };
}

function buildComment({ resultsDir, conclusion, baseSha, headSha, runUrl }) {
  const errors = [];
  const baseline = reportSet(resultsDir, 'baseline', errors);
  const candidate = reportSet(resultsDir, 'candidate', errors);
  const trainerRowsToRender = unionTrainerRows(
    trainerRows(baseline.trainer),
    trainerRows(candidate.trainer),
    baseline.trainer !== null,
    candidate.trainer !== null,
  );
  const reports = [
    baseline.trainer,
    baseline.pretokenizer,
    baseline.byteCodec,
    baseline.unicodeCodec,
    baseline.bbpeUnicodeCodec.report,
    candidate.trainer,
    candidate.pretokenizer,
    candidate.byteCodec,
    candidate.unicodeCodec,
    candidate.bbpeUnicodeCodec.report,
  ].filter((report) => report !== null);
  const bbpeCodecRegression = baseline.bbpeUnicodeCodec.status === 'present'
    && candidate.bbpeUnicodeCodec.status === 'absent';
  const passed = conclusion === 'success'
    && errors.length === 0
    && !bbpeCodecRegression
    && reports.every((report) => report?.gates?.passed === true)
    && trainerRowsToRender.every(
      (row) => !row.baseline?.failed && !row.candidate?.failed,
    );
  const basePretokenizer = pretokenizerValues(baseline.pretokenizer);
  const headPretokenizer = pretokenizerValues(candidate.pretokenizer);
  const baseByteCodec = codecValues(baseline.byteCodec);
  const headByteCodec = codecValues(candidate.byteCodec);
  const baseUnicodeCodec = codecValues(baseline.unicodeCodec);
  const headUnicodeCodec = codecValues(candidate.unicodeCodec);
  const baseBbpeUnicodeCodec = codecValues(baseline.bbpeUnicodeCodec.report);
  const headBbpeUnicodeCodec = codecValues(candidate.bbpeUnicodeCodec.report);
  const renderBbpeUnicodeCodec = baseline.bbpeUnicodeCodec.status !== 'absent'
    || candidate.bbpeUnicodeCodec.status !== 'absent';
  const lines = [
    MARKER,
    '## Benchmark report',
    '',
    passed
      ? '✅ All base and PR correctness gates passed.'
      : '❌ The benchmark run or at least one correctness gate failed.',
    '',
    `Compared \`${baseSha.slice(0, 7)}\` → \`${headSha.slice(0, 7)}\` sequentially on the same runner. Timing deltas are informational.`,
    '',
    'Trainer and optional-codec cells marked `missing` are absent cases or reports; `unavailable` means a required report is missing or a report is invalid; `failed` means that revision failed its report or case; `changed` means the workloads are not comparable.',
    '',
    '| Benchmark | Base | PR | Δ |',
    '| --- | ---: | ---: | ---: |',
  ];

  for (const row of trainerRowsToRender) {
    lines.push(trainerTableRow(row, 'time', formatMilliseconds));
  }
  for (const [label, key] of [
    ['Pretokenizer — bigram pass', 'bigram'],
    ['Pretokenizer — word pass', 'word'],
    ['Pretokenizer — total', 'total'],
  ]) {
    lines.push(tableRow(
      label,
      basePretokenizer.get(key) ?? null,
      headPretokenizer.get(key) ?? null,
      formatMilliseconds,
    ));
  }
  for (const [label, values, key] of [
    ['Codec — byte encode', [baseByteCodec, headByteCodec], 'encode'],
    ['Codec — byte decode', [baseByteCodec, headByteCodec], 'decode'],
    ['Codec — Unicode encode', [baseUnicodeCodec, headUnicodeCodec], 'encode'],
    ['Codec — Unicode decode', [baseUnicodeCodec, headUnicodeCodec], 'decode'],
  ]) {
    lines.push(tableRow(
      label,
      values[0].get(key) ?? null,
      values[1].get(key) ?? null,
      formatMilliseconds,
    ));
  }
  if (renderBbpeUnicodeCodec) {
    for (const [label, key] of [
      ['Codec — Unicode BBPE encode, vocab 1k', 'encode'],
      ['Codec — Unicode BBPE decode, vocab 1k', 'decode'],
    ]) {
      lines.push(optionalCodecTableRow(
        label,
        baseline.bbpeUnicodeCodec,
        candidate.bbpeUnicodeCodec,
        baseBbpeUnicodeCodec,
        headBbpeUnicodeCodec,
        key,
        formatMilliseconds,
      ));
    }
  }

  lines.push('', '<details>', '<summary>Peak RSS</summary>', '');
  lines.push('| Benchmark | Base | PR | Δ |');
  lines.push('| --- | ---: | ---: | ---: |');
  for (const row of trainerRowsToRender) {
    lines.push(trainerTableRow(row, 'rss', formatMebibytes));
  }
  lines.push(tableRow(
    'Pretokenizer',
    basePretokenizer.get('rss') ?? null,
    headPretokenizer.get('rss') ?? null,
    formatMebibytes,
  ));
  for (const [label, values, key] of [
    ['Codec — byte encode', [baseByteCodec, headByteCodec], 'encode_rss'],
    ['Codec — byte decode', [baseByteCodec, headByteCodec], 'decode_rss'],
    ['Codec — Unicode encode', [baseUnicodeCodec, headUnicodeCodec], 'encode_rss'],
    ['Codec — Unicode decode', [baseUnicodeCodec, headUnicodeCodec], 'decode_rss'],
  ]) {
    lines.push(tableRow(
      label,
      values[0].get(key) ?? null,
      values[1].get(key) ?? null,
      formatMebibytes,
    ));
  }
  if (renderBbpeUnicodeCodec) {
    for (const [label, key] of [
      ['Codec — Unicode BBPE encode, vocab 1k', 'encode_rss'],
      ['Codec — Unicode BBPE decode, vocab 1k', 'decode_rss'],
    ]) {
      lines.push(optionalCodecTableRow(
        label,
        baseline.bbpeUnicodeCodec,
        candidate.bbpeUnicodeCodec,
        baseBbpeUnicodeCodec,
        headBbpeUnicodeCodec,
        key,
        formatMebibytes,
      ));
    }
  }
  lines.push('', '</details>', '');
  if (bbpeCodecRegression) {
    lines.push(
      'Optional benchmark regression: `candidate/codec-unicode-bbpe.json` is absent while the base report is present.',
      '',
    );
  }
  if (errors.length > 0) {
    lines.push(`Missing or invalid reports: ${errors.map((name) => `\`${name}\``).join(', ')}.`, '');
  }
  lines.push(`[Open benchmark run](${runUrl})`);
  return lines.join('\n');
}

async function updateBenchmarkComment({ github, context, core }) {
  const workflowRun = context.payload.workflow_run;
  const metadata = readMetadata(process.env.BENCHMARK_RESULTS_DIR);
  const pullRequests = workflowRun?.pull_requests ?? [];
  if (
    pullRequests.length > 1
    || (pullRequests.length === 1 && pullRequests[0].number !== metadata.pull_request_number)
  ) {
    core.info('Benchmark metadata does not match the triggering pull request; skipping comment.');
    return;
  }

  const pullNumber = metadata.pull_request_number;
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const { data: pullRequest } = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: pullNumber,
  });
  const headRepository = workflowRun?.head_repository?.full_name;
  if (
    pullRequest.head.sha !== metadata.head_sha
    || pullRequest.head.ref !== workflowRun?.head_branch
    || pullRequest.head.repo?.full_name !== headRepository
  ) {
    core.info('The benchmark origin or PR head no longer matches; skipping stale results.');
    return;
  }

  const runId = process.env.WORKFLOW_RUN_ID;
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${runId}`;
  const body = buildComment({
    resultsDir: process.env.BENCHMARK_RESULTS_DIR,
    conclusion: process.env.WORKFLOW_CONCLUSION,
    baseSha: metadata.base_sha,
    headSha: metadata.head_sha,
    runUrl,
  });
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: pullNumber,
    per_page: 100,
  });
  const existing = comments.find(
    (comment) => comment.user?.type === 'Bot' && comment.body?.includes(MARKER),
  );
  if (existing) {
    await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: existing.id,
      body,
    });
  } else {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: pullNumber,
      body,
    });
  }
}

module.exports = updateBenchmarkComment;
module.exports.buildComment = buildComment;
