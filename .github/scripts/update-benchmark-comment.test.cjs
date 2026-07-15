const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const updateBenchmarkComment = require('./update-benchmark-comment.cjs');
const { buildComment } = updateBenchmarkComment;

function writeReport(root, side, name, contract, samples, gatesPassed = true) {
  const directory = path.join(root, side);
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, name), JSON.stringify({
    schema_version: 1,
    contract,
    gates: { passed: gatesPassed },
    samples,
  }));
}

function trainerSample(caseName, occurrenceMode, time, rss, caseOverrides = {}) {
  return {
    request: {
      case: { name: caseName, ...caseOverrides },
      variant: {
        occurrence_mode: occurrenceMode,
        hot_pair_window_size: occurrenceMode === 'bounded' ? 4096 : null,
      },
    },
    measurement: {
      timing: { core_training_ns: time },
      memory: { process_peak_rss_through_training_bytes: rss },
    },
    status: 'completed',
    error: null,
  };
}

function populateReports(root, side, multiplier, options = {}) {
  const {
    extraTrainerSamples = [],
    trainerCaseOverrides = {},
    trainerGatesPassed = true,
  } = options;
  const trainerSamples = [];
  for (const caseName of [
    'smoke_en_byte_v300',
    'smoke_en_byte_v1000',
    'smoke_zh_unicode_v300',
    'smoke_zh_unicode_v1000',
  ]) {
    trainerSamples.push(trainerSample(
      caseName,
      'exact',
      1_000_000 * multiplier,
      1_048_576,
      trainerCaseOverrides,
    ));
    trainerSamples.push(trainerSample(
      caseName,
      'bounded',
      2_000_000 * multiplier,
      2_097_152,
      trainerCaseOverrides,
    ));
  }
  trainerSamples.push(...extraTrainerSamples);
  writeReport(
    root,
    side,
    'trainer.json',
    'unitoken_trainer_regression_v1',
    trainerSamples,
    trainerGatesPassed,
  );
  writeReport(
    root,
    side,
    'pretokenizer.json',
    'unitoken_pretokenizer_regression_v1',
    [{
      measurement: {
        timing: {
          bigram_pass_ns: 1_000_000 * multiplier,
          word_pass_ns: 2_000_000 * multiplier,
          core_pretokenizer_ns: 3_000_000 * multiplier,
        },
        memory: { process_peak_rss_through_core_bytes: 3_145_728 },
      },
    }],
  );
  for (const name of ['codec-byte.json', 'codec-unicode.json']) {
    writeReport(root, side, name, 'unitoken_codec_regression_v1', [{
      encode: {
        timing: { encode_ns: 4_000_000 * multiplier },
        memory: { process_peak_rss_through_phase_bytes: 4_194_304 },
      },
      decode: {
        timing: { decode_ns: 5_000_000 * multiplier },
        memory: { process_peak_rss_through_phase_bytes: 5_242_880 },
      },
    }]);
  }
}

function writeMetadata(root) {
  fs.writeFileSync(path.join(root, 'metadata.json'), JSON.stringify({
    schema_version: 1,
    pull_request_number: 7,
    base_sha: '0123456789abcdef0123456789abcdef01234567',
    head_sha: 'fedcba9876543210fedcba9876543210fedcba98',
  }));
}

test('buildComment renders a comparable trainer delta with legacy BBPE defaults', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1);
    populateReports(root, 'candidate', 1.1, {
      trainerCaseOverrides: {
        bbpe_fallback: false,
        primary_vocab_ratio: 0.9,
      },
    });
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'success',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/1',
    });
    assert.match(comment, /<!-- unitoken-benchmark-report -->/);
    assert.match(comment, /All base and PR correctness gates passed/);
    assert.match(comment, /Trainer — English byte, vocab 300 \(exact\)/);
    assert.match(
      comment,
      /Trainer — English byte, vocab 300 \(exact\) \| 1\.00 ms \| 1\.10 ms \| \+10\.0%/,
    );
    assert.match(comment, /Open benchmark run/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildComment renders missing for a candidate-only trainer row', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1);
    populateReports(root, 'candidate', 1, {
      extraTrainerSamples: [trainerSample(
        'smoke_zh_unicode_bbpe_r90_v1000',
        'exact',
        3_000_000,
        undefined,
        {
          bbpe_fallback: true,
          primary_vocab_ratio: 0.9,
          target_vocab_size: 1000,
          unit: 'unicode',
        },
      )],
    });
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'success',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/1',
    });
    assert.match(
      comment,
      /Trainer — Chinese Unicode BBPE, vocab 1k \(exact\) \| missing \| 3\.00 ms \| n\/a/,
    );
    assert.match(
      comment,
      /Trainer — Chinese Unicode BBPE, vocab 1k \(exact\) \| missing \| n\/a \| n\/a/,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildComment marks trainer deltas changed when workloads differ', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1, {
      trainerCaseOverrides: { target_vocab_size: 1000 },
    });
    populateReports(root, 'candidate', 1.1, {
      trainerCaseOverrides: { target_vocab_size: 2000 },
    });
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'success',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/1',
    });
    assert.match(
      comment,
      /Trainer — English byte, vocab 300 \(exact\) \| 1\.00 ms \| 1\.10 ms \| changed/,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildComment renders failed instead of missing for a false trainer gate', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1);
    populateReports(root, 'candidate', 1.1, { trainerGatesPassed: false });
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'success',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/1',
    });
    assert.match(comment, /benchmark run or at least one correctness gate failed/);
    assert.match(
      comment,
      /Trainer — English byte, vocab 300 \(exact\) \| 1\.00 ms \| failed \| n\/a/,
    );
    assert.doesNotMatch(comment, /Missing or invalid reports/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildComment renders failed instead of missing for a failed trainer sample', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1);
    const failedSample = trainerSample(
      'smoke_en_byte_v300',
      'exact',
      undefined,
      undefined,
    );
    failedSample.status = 'failed';
    failedSample.measurement = null;
    failedSample.error = { phase: 'training', message: 'boom' };
    populateReports(root, 'candidate', 1.1, {
      extraTrainerSamples: [failedSample],
    });
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'success',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/1',
    });
    assert.match(comment, /benchmark run or at least one correctness gate failed/);
    assert.match(
      comment,
      /Trainer — English byte, vocab 300 \(exact\) \| 1\.00 ms \| failed \| n\/a/,
    );
    assert.doesNotMatch(
      comment,
      /Trainer — English byte, vocab 300 \(exact\) \| 1\.00 ms \| missing/,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildComment renders unavailable for a missing trainer report', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1);
    populateReports(root, 'candidate', 1.1);
    fs.rmSync(path.join(root, 'candidate', 'trainer.json'));
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'failure',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/2',
    });
    assert.match(
      comment,
      /Trainer — English byte, vocab 300 \(exact\) \| 1\.00 ms \| unavailable \| n\/a/,
    );
    const trainerLines = comment
      .split('\n')
      .filter((line) => line.startsWith('| Trainer'));
    assert.ok(trainerLines.length > 0);
    assert.ok(trainerLines.every((line) => !line.includes('| missing |')));
    assert.match(comment, /Missing or invalid reports: `candidate\/trainer\.json`/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildComment rejects a missing report', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  try {
    populateReports(root, 'baseline', 1);
    populateReports(root, 'candidate', 1);
    fs.rmSync(path.join(root, 'candidate', 'codec-unicode.json'));
    const comment = buildComment({
      resultsDir: root,
      conclusion: 'failure',
      baseSha: '0123456789abcdef',
      headSha: 'fedcba9876543210',
      runUrl: 'https://example.test/actions/runs/2',
    });
    assert.match(comment, /benchmark run or at least one correctness gate failed/);
    assert.match(comment, /candidate\/codec-unicode\.json/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('updateBenchmarkComment binds artifact metadata to the current PR head', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'unitoken-benchmark-comment-'));
  const previousEnvironment = {
    BENCHMARK_RESULTS_DIR: process.env.BENCHMARK_RESULTS_DIR,
    WORKFLOW_CONCLUSION: process.env.WORKFLOW_CONCLUSION,
    WORKFLOW_RUN_ID: process.env.WORKFLOW_RUN_ID,
  };
  try {
    populateReports(root, 'baseline', 1);
    populateReports(root, 'candidate', 1);
    writeMetadata(root);
    process.env.BENCHMARK_RESULTS_DIR = root;
    process.env.WORKFLOW_CONCLUSION = 'success';
    process.env.WORKFLOW_RUN_ID = '123';
    let created = null;
    const github = {
      rest: {
        pulls: {
          get: async () => ({
            data: {
              head: {
                sha: 'fedcba9876543210fedcba9876543210fedcba98',
                ref: 'benchmark-branch',
                repo: { full_name: 'contributor/unitoken' },
              },
            },
          }),
        },
        issues: {
          listComments: async () => ({ data: [] }),
          createComment: async (request) => {
            created = request;
          },
          updateComment: async () => {
            throw new Error('unexpected update');
          },
        },
      },
      paginate: async () => [],
    };
    await updateBenchmarkComment({
      github,
      context: {
        payload: {
          workflow_run: {
            head_branch: 'benchmark-branch',
            head_repository: { full_name: 'contributor/unitoken' },
            pull_requests: [],
          },
        },
        repo: { owner: 'tokn-ai', repo: 'unitoken' },
        serverUrl: 'https://github.com',
      },
      core: { info: () => {} },
    });
    assert.equal(created.issue_number, 7);
    assert.match(created.body, /<!-- unitoken-benchmark-report -->/);
    assert.match(created.body, /actions\/runs\/123/);
  } finally {
    for (const [name, value] of Object.entries(previousEnvironment)) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
    fs.rmSync(root, { recursive: true, force: true });
  }
});
