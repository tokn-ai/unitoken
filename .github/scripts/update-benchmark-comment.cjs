const fs = require('node:fs');
const path = require('node:path');

const MARKER = '<!-- unitoken-benchmark-report -->';
const MAX_REPORT_BYTES = 2 * 1024 * 1024;

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

function readReport(resultsDir, relativePath, contract, errors) {
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
      || report.gates.passed !== true
      || !Array.isArray(report.samples)
    ) {
      throw new Error('invalid report contract');
    }
    return report;
  } catch (_error) {
    errors.push(relativePath);
    return null;
  }
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

function trainerValues(report) {
  const values = new Map();
  if (!report) {
    return values;
  }
  const cases = [
    'smoke_en_byte_v300',
    'smoke_en_byte_v1000',
    'smoke_zh_unicode_v300',
    'smoke_zh_unicode_v1000',
  ];
  for (const caseName of cases) {
    for (const variant of ['exact', 'k4096']) {
      const matching = report.samples.filter((sample) => {
        const request = sample?.request;
        const occurrence = request?.variant?.occurrence_mode;
        const label = occurrence === 'exact'
          ? 'exact'
          : occurrence === 'bounded' && request?.variant?.hot_pair_window_size === 4096
            ? 'k4096'
            : null;
        return request?.case?.name === caseName && label === variant;
      });
      values.set(`${caseName}:${variant}:time`, average(
        matching.map((sample) => sample?.measurement?.timing?.core_training_ns),
      ));
      values.set(`${caseName}:${variant}:rss`, average(
        matching.map(
          (sample) => sample?.measurement?.memory?.process_peak_rss_through_training_bytes,
        ),
      ));
    }
  }
  return values;
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
  };
}

function buildComment({ resultsDir, conclusion, baseSha, headSha, runUrl }) {
  const errors = [];
  const baseline = reportSet(resultsDir, 'baseline', errors);
  const candidate = reportSet(resultsDir, 'candidate', errors);
  const passed = conclusion === 'success' && errors.length === 0;
  const baseTrainer = trainerValues(baseline.trainer);
  const headTrainer = trainerValues(candidate.trainer);
  const basePretokenizer = pretokenizerValues(baseline.pretokenizer);
  const headPretokenizer = pretokenizerValues(candidate.pretokenizer);
  const baseByteCodec = codecValues(baseline.byteCodec);
  const headByteCodec = codecValues(candidate.byteCodec);
  const baseUnicodeCodec = codecValues(baseline.unicodeCodec);
  const headUnicodeCodec = codecValues(candidate.unicodeCodec);
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
    '| Benchmark | Base | PR | Δ |',
    '| --- | ---: | ---: | ---: |',
  ];

  const trainerCases = [
    ['English byte, vocab 300', 'smoke_en_byte_v300'],
    ['English byte, vocab 1k', 'smoke_en_byte_v1000'],
    ['Chinese Unicode, vocab 300', 'smoke_zh_unicode_v300'],
    ['Chinese Unicode, vocab 1k', 'smoke_zh_unicode_v1000'],
  ];
  for (const [label, caseName] of trainerCases) {
    for (const variant of ['exact', 'k4096']) {
      const key = `${caseName}:${variant}:time`;
      lines.push(tableRow(
        `Trainer — ${label} (${variant})`,
        baseTrainer.get(key) ?? null,
        headTrainer.get(key) ?? null,
        formatMilliseconds,
      ));
    }
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

  lines.push('', '<details>', '<summary>Peak RSS</summary>', '');
  lines.push('| Benchmark | Base | PR | Δ |');
  lines.push('| --- | ---: | ---: | ---: |');
  for (const [label, caseName] of trainerCases) {
    for (const variant of ['exact', 'k4096']) {
      const key = `${caseName}:${variant}:rss`;
      lines.push(tableRow(
        `Trainer — ${label} (${variant})`,
        baseTrainer.get(key) ?? null,
        headTrainer.get(key) ?? null,
        formatMebibytes,
      ));
    }
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
  lines.push('', '</details>', '');
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
