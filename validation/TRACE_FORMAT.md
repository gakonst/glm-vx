# Trace comparison format, version 1

`trace_gate.py` compares offline, teacher-forced traces within a separately reviewed
contract. It does not execute either model, attest a producer's provenance, establish
oracle independence, or prove correctness outside the enumerated fixtures. A passing
synthetic fixture is **not** full-model parity or release eligibility. The caller
must additionally enforce the release corpus and serving/codec/hardware gates in
`docs/CORRECTNESS-AND-PERFORMANCE.md`.

Run from the repository root (NumPy is the only non-stdlib dependency):

```sh
.venv/bin/python -m validation.trace_gate \
  --contract /trusted/contract.json \
  --reference-manifest /trusted/reference.json \
  --reference-trace /trusted/reference.npz \
  --candidate-manifest /candidate/trace.json \
  --candidate-trace /candidate/trace.npz \
  --receipt /results/receipt.json
```

Exit 0 means all requirements passed; exit 1 means comparison/input failure. Argument
or receipt-write errors also exit nonzero. Every readable input's exact bytes are
SHA-256 hashed in the receipt, including failed runs. Missing files have null hashes.
Receipts include scope, model/corpus/build pins, compared tensor count, numerical
summaries and the first failure. They contain no raw activation arrays. Diagnostic
order is input integrity/coverage first, then the trusted case/tensor list order.
Arrange tensors in execution order to identify the earliest compared divergence.

## Trust boundary

Freeze the contract, reference NPZ and gate implementation outside optimizer write
access. The calling evaluation system must pin or approve the **contract file hash**;
an arbitrary candidate-supplied contract has no authority. Reference NPZ bytes are
pinned inside the contract. Both producer manifests must name the exact contract
hash. Candidate manifests cannot override budgets, required keys or tie rules:
unknown fields are rejected. Pin the candidate build for each evaluated artifact;
budget/corpus changes require separate review. Manifests are claims that a trusted
runner must populate from measured build/model artifacts; this offline gate cannot
prove the candidate binary actually produced the arrays. It never imports code from
trace files, and NumPy reads use `allow_pickle=False`.

## Contract JSON

Objects have exactly the fields below; unknown/missing fields and duplicate JSON
keys are errors. Digests are lowercase 64-character SHA-256 strings.

| Field | Value |
| --- | --- |
| `schema_version` | Integer `1` |
| `contract_id` | Nonempty versioned identifier |
| `scope` | Nonempty description of evaluated domain and limitations |
| `pins` | Shared model/corpus/precision pins, below |
| `producers` | Exactly `reference` and `candidate` build objects, below |
| `reference_trace_sha256` | Hash of the exact trusted NPZ bytes |
| `budgets` | Operation name to budget object; exactly the floating operations used |
| `cases` | Nonempty ordered list of case objects |

`pins` contains `config_sha256`, `weights_sha256` (nonempty shard name → digest
map), `tokenizer_sha256`, `template_sha256`, `corpus_sha256`, `corpus_version`,
`reference_revision` (full immutable 40/64-character lowercase commit hash),
`numerical_mode`, and `tie_breaking` (currently only `stable-lowest-index`). Each
build object contains `binary_sha256`, `compiler` (version/revision string),
`libraries_sha256` (name → digest map, including PTX if used), `flags` (explicit
string list), and `hardware` (nonempty identity string). These are reproduced in
receipts and compared exactly, without coercing JSON booleans to numbers.

Each case contains exactly:

- `id`: unique case name.
- `token_ids`: nonempty prescribed token sequence, including decode inputs. Both
  producers must feed this sequence regardless of predicted tokens.
- `positions`: nonempty sorted unique zero-based token sequence offsets to capture.
- `execution`: `prefill`, `decode` or `chunked`.
- `chunks`: positive chunk lengths summing to token count; prefill has one chunk,
  decode has all one-token chunks. Each supported split is a distinct case.
- `cache_owner`: request identity; both manifests must match exactly.
- `layers`: nonempty unique layer IDs; declare every layer in the reviewed domain.
- `vocab_size`: integer >= 2; all prescribed IDs must be in range.
- `tensors`: nonempty ordered tensor specifications.

Each tensor specifies exactly `key`, `operation`, `kind`, `shape`, `dtype`,
`position`, `layer`, `infinity`. Keys are globally unique NPZ member names (without
`.npy`). Shapes are nonempty lists of positive dimensions; empty captures cannot
pass. Dtypes are `float16`, `float32`, `float64`, `int32`, `int64`, `uint32`, `uint64`,
or `bool`. The shape and dtype must match exactly on both sides. Position must be
selected by the case; layer is one of the case's IDs or null for global tensors.

Kinds:

- `float`: an intermediate with a frozen operation budget.
- `logits`: a finite one-dimensional **complete vocabulary** vector. Exactly one
  is required for every selected position, with null layer and `[vocab_size]` shape.
- `indices`: integral discrete selections, compared exactly in order.
- `exact`: exact logical cache values, positions, ownership encoding, reuse flags
  or other discrete state. Floating cache values receive no numerical tolerance.

Use `indices` for routing and DSA selection, and `exact` for logical caches. Capture
canonical logical arrays rather than implementation-specific compressed layouts.
Every declared layer/position must have a `float` intermediate; additional operation
coverage is enumerated explicitly by the trusted tensor list. The contract author
must include every required embedding, norm, attention, indexer, router, MLP,
residual, cache and logit stage applicable to that domain; this generic checker
cannot infer model architecture from a configuration hash. Exact tensor and case
coverage means extra, missing, duplicate, skipped or incomplete results fail.

`infinity` is `forbid` or `matching_negative`. The latter permits negative infinity
only at identical reference/candidate coordinates, suitable for masks; those
coordinates are excluded from numerical metrics. All-masked intermediates are
allowed only under that explicit policy. Positive infinity and NaN always fail,
even if both producers match. Logits always use `forbid`.

## Frozen operation budgets

Every `float`/`logits` operation names one budget object with exactly these fields:

```json
{
  "atol": 0.000001,
  "rtol": 0.000001,
  "normalized_l2": 0.000001,
  "normalization_floor": 0.000001,
  "softmax_tv": 0.000001,
  "top_k": 5,
  "top_k_margin_atol": 0.000001
}
```

These numbers only illustrate syntax; they are **not calibrated accuracy budgets**.
Freeze separately calibrated budgets for each operation and numerical mode before
optimization. All budgets must be finite and nonnegative; normalization floor is
strictly positive. `softmax_tv` <= 1; `top_k` is a positive integer smaller than the
vocabulary. Logit-specific fields are required but unused for intermediates.

Both conditions must hold for finite intermediate elements:

- Elementwise `abs(candidate-reference) <= atol + rtol*abs(reference)`.
- `RMS(candidate-reference) / max(RMS(reference), normalization_floor) <= normalized_l2`.

For logits, also require softmax total variation <= `softmax_tv`, identical ordered
top-k token IDs, and absolute change of the kth-versus-(k+1)th logit margin <=
`top_k_margin_atol`. Sorting uses descending score then ascending token ID. Discrete
mismatches cannot be waived through generous numerical budgets. Near ties that
choose different experts/positions fail until a separate reviewed contract and
reference resolve them.

## Producer manifest JSON and NPZ

Each producer manifest has exactly:

```text
schema_version: 1
role: "reference" or "candidate"
contract_sha256: hash of exact contract JSON bytes
pins: exact copy of trusted pins
producer: exact corresponding trusted build object
status: "complete"
skipped: []
cases: exact ordered trusted cases, with only the tensors field removed
trace_sha256: hash of this producer's NPZ bytes
```

NPZ contains exactly one numeric/boolean NPY array per trusted key, with no metadata
or extra ZIP members. Duplicate members, object arrays, missing arrays, dtype
changes and shape changes fail. Per-case tensor lists and budgets appear only in
the trusted contract. The implementation loads archives into memory; generate
bounded trace shards and evaluate each required shard under the enclosing release
runner instead of treating missing large cases as optional.

The executable fixture in `tests/test_trace_gate.py` constructs both manifests,
a contract and NPZ arrays. Its mutations exercise gate behavior only and are not
claimed as measured model parity evidence.
