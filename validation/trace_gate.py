"""Fail-closed, offline comparison of explicitly scoped teacher-forced traces.

The contract and reference archive must come from a trusted, independently reviewed
source outside candidate control. This gate verifies evidence, not the truth of a
producer's claimed provenance, and finite fixtures do not prove global correctness.
See validation/TRACE_FORMAT.md for the version 1 interchange contract.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

VERSION = 1


class GateError(ValueError):
    def __init__(self, code, message, **context):
        super().__init__(message)
        self.detail = {"code": code, "message": message, **context}


def require(condition, code, message, **context):
    if not condition:
        raise GateError(code, message, **context)


def fields(value, expected, where):
    require(isinstance(value, dict) and set(value) == set(expected),
            "schema", f"{where}: expected exactly {sorted(expected)}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(value, where):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), "schema", f"{where}: invalid SHA-256")


def same_json(left, right):
    """Unlike Python equality, JSON booleans must not alias integer metadata."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def nonempty(value, where):
    require(isinstance(value, str) and bool(value.strip()), "schema", f"{where}: empty string")


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def unique_ints(value, where, minimum=0):
    require(isinstance(value, list) and value and all(integer(x, minimum) for x in value)
            and len(value) == len(set(value)), "schema", f"{where}: expected nonempty unique integers")


def number(value, where, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "schema", f"{where}: invalid budget")


def parse_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "schema", f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise GateError("schema", f"non-finite JSON number: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def validate_contract(contract):
    fields(contract, {"schema_version", "contract_id", "scope", "pins", "producers",
                      "reference_trace_sha256", "budgets", "cases"}, "contract")
    require(type(contract["schema_version"]) is int and contract["schema_version"] == VERSION,
            "schema", "unsupported contract version")
    nonempty(contract["contract_id"], "contract_id")
    nonempty(contract["scope"], "scope")
    digest(contract["reference_trace_sha256"], "reference_trace_sha256")
    pins = contract["pins"]
    fields(pins, {"config_sha256", "weights_sha256", "tokenizer_sha256", "template_sha256",
                  "corpus_sha256", "corpus_version", "reference_revision", "numerical_mode",
                  "tie_breaking"}, "pins")
    for key in ("config_sha256", "tokenizer_sha256", "template_sha256", "corpus_sha256"):
        digest(pins[key], key)
    require(isinstance(pins["weights_sha256"], dict) and pins["weights_sha256"],
            "schema", "weights_sha256 must pin every checkpoint shard")
    for name, value in pins["weights_sha256"].items():
        nonempty(name, "weight name")
        digest(value, name)
    for key in ("corpus_version", "reference_revision", "numerical_mode", "tie_breaking"):
        nonempty(pins[key], key)
    revision = pins["reference_revision"]
    require(len(revision) in (40, 64) and all(c in "0123456789abcdef" for c in revision),
            "schema", "reference_revision must be an immutable commit hash")
    require(pins["tie_breaking"] == "stable-lowest-index", "schema", "unsupported tie-breaking rule")
    fields(contract["producers"], {"reference", "candidate"}, "producers")
    for role, producer in contract["producers"].items():
        fields(producer, {"binary_sha256", "compiler", "libraries_sha256", "flags", "hardware"}, role)
        digest(producer["binary_sha256"], role + " binary")
        nonempty(producer["compiler"], "compiler")
        nonempty(producer["hardware"], "hardware")
        require(isinstance(producer["flags"], list) and all(isinstance(x, str) for x in producer["flags"]),
                "schema", "flags must be an explicit list")
        require(isinstance(producer["libraries_sha256"], dict), "schema", "libraries must be a map")
        for name, value in producer["libraries_sha256"].items():
            nonempty(name, "library name")
            digest(value, name)
    budgets = contract["budgets"]
    require(isinstance(budgets, dict) and budgets, "schema", "missing operation budgets")
    for operation, budget in budgets.items():
        nonempty(operation, "operation")
        fields(budget, {"atol", "rtol", "normalized_l2", "normalization_floor",
                        "softmax_tv", "top_k", "top_k_margin_atol"}, operation)
        for key in ("atol", "rtol", "normalized_l2", "softmax_tv", "top_k_margin_atol"):
            number(budget[key], key)
        number(budget["normalization_floor"], "normalization_floor", positive=True)
        require(integer(budget["top_k"], 1), "schema", "top_k must be positive")
        require(budget["softmax_tv"] <= 1, "schema", "softmax_tv must be <= 1")
    cases = contract["cases"]
    require(isinstance(cases, list) and cases, "schema", "empty required case corpus")
    ids, keys, used_operations = set(), set(), set()
    for case in cases:
        fields(case, {"id", "token_ids", "positions", "execution", "chunks", "cache_owner",
                      "layers", "vocab_size", "tensors"}, "case")
        nonempty(case["id"], "case id")
        require(case["id"] not in ids, "schema", "duplicate case id")
        ids.add(case["id"])
        tokens = case["token_ids"]
        require(isinstance(tokens, list) and tokens and all(integer(x) for x in tokens),
                "schema", "token_ids must prescribe every teacher-forced token")
        unique_ints(case["positions"], "positions")
        require(case["positions"] == sorted(case["positions"]) and max(case["positions"]) < len(tokens),
                "schema", "positions must be ordered indices into token_ids")
        require(case["execution"] in ("prefill", "decode", "chunked"), "schema", "unsupported execution")
        chunks = case["chunks"]
        require(isinstance(chunks, list) and chunks and all(integer(x, 1) for x in chunks)
                and sum(chunks) == len(tokens), "schema", "chunks must account for all input tokens")
        require(case["execution"] != "prefill" or chunks == [len(tokens)], "schema", "invalid prefill chunks")
        require(case["execution"] != "decode" or chunks == [1] * len(tokens), "schema", "invalid decode chunks")
        nonempty(case["cache_owner"], "cache_owner")
        unique_ints(case["layers"], "layers")
        require(integer(case["vocab_size"], 2) and max(tokens) < case["vocab_size"], "schema", "invalid vocabulary")
        specs = case["tensors"]
        require(isinstance(specs, list) and specs, "schema", "empty tensor coverage")
        logits, layer_positions = set(), set()
        for spec in specs:
            fields(spec, {"key", "operation", "kind", "shape", "dtype", "position", "layer", "infinity"}, "tensor")
            key = spec["key"]
            nonempty(key, "tensor key")
            require(key not in keys, "schema", "duplicate tensor key")
            keys.add(key)
            nonempty(spec["operation"], "operation")
            require(spec["kind"] in ("float", "logits", "indices", "exact"), "schema", "unsupported tensor kind")
            require(isinstance(spec["shape"], list) and spec["shape"] and
                    all(integer(x, 1) for x in spec["shape"]), "schema", "empty or invalid tensor shape")
            require(spec["dtype"] in ("float16", "float32", "float64", "int32", "int64", "uint32", "uint64", "bool"),
                    "schema", "unsupported tensor dtype")
            require(type(spec["position"]) is int and spec["position"] in case["positions"], "schema", "invalid tensor position")
            require(spec["layer"] is None or (type(spec["layer"]) is int and spec["layer"] in case["layers"]),
                    "schema", "invalid tensor layer")
            require(spec["infinity"] in ("forbid", "matching_negative"), "schema", "invalid infinity policy")
            if spec["kind"] in ("float", "logits"):
                require(spec["dtype"].startswith("float") and spec["operation"] in budgets,
                        "schema", "floating operation has no frozen budget")
                used_operations.add(spec["operation"])
            if spec["kind"] == "indices":
                require(spec["dtype"] in ("int32", "int64", "uint32", "uint64"), "schema", "indices must be integral")
            if spec["kind"] == "logits":
                require(spec["shape"] == [case["vocab_size"]] and spec["layer"] is None
                        and spec["infinity"] == "forbid", "schema", "logits must contain the full finite vocabulary")
                require(spec["position"] not in logits, "schema", "duplicate logits position")
                logits.add(spec["position"])
                require(budgets[spec["operation"]]["top_k"] < case["vocab_size"], "schema", "top_k must leave a boundary")
            elif spec["layer"] is not None and spec["kind"] == "float":
                layer_positions.add((spec["layer"], spec["position"]))
        require(logits == set(case["positions"]), "coverage", "full logits required at every selected position", case=case["id"])
        require(layer_positions == {(l, p) for l in case["layers"] for p in case["positions"]},
                "coverage", "intermediate float trace required at every layer/position", case=case["id"])
    require(used_operations == set(budgets), "schema", "budgets must exactly cover floating operations")


def case_metadata(case):
    return {key: value for key, value in case.items() if key != "tensors"}


def validate_manifest(manifest, contract, contract_hash, role, archive_hash):
    fields(manifest, {"schema_version", "role", "contract_sha256", "pins", "producer",
                      "status", "skipped", "cases", "trace_sha256"}, role + " manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == VERSION,
            "schema", "unsupported manifest version", role=role)
    require(manifest["role"] == role, "identity", "wrong producer role", role=role)
    require(manifest["contract_sha256"] == contract_hash, "identity", "stale or different contract hash", role=role)
    require(same_json(manifest["pins"], contract["pins"]), "identity", "model/corpus/numerical pins differ", role=role)
    require(same_json(manifest["producer"], contract["producers"][role]), "identity", "binary/build/hardware pins differ", role=role)
    require(manifest["status"] == "complete" and manifest["skipped"] == [], "incomplete", "run incomplete or skipped", role=role)
    require(same_json(manifest["cases"], [case_metadata(c) for c in contract["cases"]]),
            "coverage", "case coverage, teacher forcing, chunks or cache ownership differ", role=role)
    require(manifest["trace_sha256"] == archive_hash, "identity", "trace archive hash differs", role=role)


def archive(data, expected_keys, role):
    # Reject duplicate ZIP entries and non-NPY payloads before NumPy can hide them.
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        names = zipped.namelist()
        require(len(names) == len(set(names)), "coverage", "duplicate archive members", role=role)
        require(set(names) == {key + ".npy" for key in expected_keys}, "coverage",
                "archive tensor keys differ from required coverage", role=role,
                missing=sorted(set(expected_keys) - {x[:-4] for x in names if x.endswith('.npy')}),
                unexpected=sorted(set(names) - {key + '.npy' for key in expected_keys}))
    return np.load(io.BytesIO(data), allow_pickle=False)


def compare_tensor(reference, candidate, spec, budget, context):
    for role, value in (("reference", reference), ("candidate", candidate)):
        require(list(value.shape) == spec["shape"], "shape", "tensor shape differs", role=role, **context)
        require(value.dtype == np.dtype(spec["dtype"]), "dtype", "tensor dtype differs", role=role, **context)
        require(not np.isnan(value).any(), "nonfinite", "NaN in tensor", role=role, **context)
        if spec["infinity"] == "forbid":
            require(np.isfinite(value).all(), "nonfinite", "unexpected infinity", role=role, **context)
        else:
            require(not np.isposinf(value).any(), "nonfinite", "unexpected positive infinity", role=role, **context)
    require(np.array_equal(np.isneginf(reference), np.isneginf(candidate)),
            "nonfinite", "masked infinity locations differ", **context)
    if spec["kind"] in ("indices", "exact"):
        different = reference != candidate
        require(not different.any(), "exact", "exact tensor differs", **context,
                index=list(map(int, np.argwhere(different)[0])) if different.any() else None)
        return {"exact": True}
    finite = np.isfinite(reference)
    r, c = reference[finite].astype(np.float64), candidate[finite].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = np.abs(c - r)
        bound = budget["atol"] + budget["rtol"] * np.abs(r)
        # Scale before the RMS to avoid overflow on large finite inputs.
        scale = max(float(np.max(np.abs(r), initial=0)), budget["normalization_floor"])
        denom = max(float(np.sqrt(np.mean((r / scale) ** 2))) if r.size else 0,
                    budget["normalization_floor"] / scale)
        normalized = float(np.sqrt(np.mean((delta / scale) ** 2)) / denom) if r.size else 0.0
    metrics = {"max_abs": float(np.max(delta, initial=0)), "normalized_l2": normalized}
    require(all(math.isfinite(x) for x in metrics.values()) and np.isfinite(bound).all(),
            "numeric", "non-finite comparison metric or bound", **context)
    mismatch = delta > bound
    if mismatch.any():
        flat_index = int(np.flatnonzero(finite)[int(np.flatnonzero(mismatch)[0])])
        raise GateError("numeric", "absolute/relative budget exceeded", **context,
                        index=list(map(int, np.unravel_index(flat_index, reference.shape))), metrics=metrics)
    require(normalized <= budget["normalized_l2"], "numeric", "normalized error budget exceeded", **context, metrics=metrics)
    if spec["kind"] == "logits":
        def softmax(x):
            exp = np.exp(x - np.max(x))
            return exp / np.sum(exp)
        tv = float(np.sum(np.abs(softmax(r) - softmax(c))) / 2)
        k = budget["top_k"]
        # Stable descending order makes equal-score ties use the lowest token ID.
        ri, ci = np.argsort(-r, kind="stable"), np.argsort(-c, kind="stable")
        margin_r = float(r[ri[k - 1]] - r[ri[k]])
        margin_c = float(c[ci[k - 1]] - c[ci[k]])
        metrics.update(softmax_tv=tv, reference_top_k_margin=margin_r, candidate_top_k_margin=margin_c)
        require(all(math.isfinite(x) for x in metrics.values()), "numeric", "non-finite logit metric", **context)
        require(np.array_equal(ri[:k], ci[:k]), "logits", "top-k token order differs", **context, metrics=metrics)
        require(tv <= budget["softmax_tv"] and abs(margin_r - margin_c) <= budget["top_k_margin_atol"],
                "logits", "distribution or top-k margin budget exceeded", **context, metrics=metrics)
    return metrics


def run_gate(contract_path, reference_manifest_path, reference_trace_path,
             candidate_manifest_path, candidate_trace_path):
    """Return a JSON-safe receipt; all input/format/comparison failures fail closed."""
    receipt = {"schema_version": VERSION, "status": "fail", "eligible": False,
               "scope": None, "files": {}, "compared_tensors": 0, "comparisons": [],
               "first_failure": None,
               "limitation": "Finite trace evidence only; no proof of global correctness or producer attestation."}
    inputs = {"contract": contract_path, "reference_manifest": reference_manifest_path,
              "reference_trace": reference_trace_path, "candidate_manifest": candidate_manifest_path,
              "candidate_trace": candidate_trace_path}
    payloads, read_errors = {}, []
    for name, path in inputs.items():
        try:
            data = Path(path).read_bytes()
            payloads[name] = data
            receipt["files"][name] = {"path": str(path), "sha256": sha256(data), "bytes": len(data)}
        except (OSError, ValueError) as exc:
            receipt["files"][name] = {"path": str(path), "sha256": None, "error": str(exc)}
            read_errors.append(name)
    try:
        require(not read_errors, "missing_input", "unreadable required inputs", inputs=read_errors)
        contract = parse_json(payloads["contract"])
        validate_contract(contract)
        receipt.update(contract_id=contract["contract_id"], scope=contract["scope"],
                       pins=contract["pins"], producers=contract["producers"])
        require(sha256(payloads["reference_trace"]) == contract["reference_trace_sha256"],
                "identity", "reference trace differs from trusted contract")
        for role in ("reference", "candidate"):
            validate_manifest(parse_json(payloads[role + "_manifest"]), contract,
                              sha256(payloads["contract"]), role, sha256(payloads[role + "_trace"]))
        keys = [s["key"] for case in contract["cases"] for s in case["tensors"]]
        with archive(payloads["reference_trace"], keys, "reference") as reference, \
                archive(payloads["candidate_trace"], keys, "candidate") as candidate:
            for case in contract["cases"]:
                for spec in case["tensors"]:
                    context = {"case": case["id"], "tensor": spec["key"], "operation": spec["operation"],
                               "position": spec["position"], "layer": spec["layer"]}
                    try:
                        metrics = compare_tensor(reference[spec["key"]], candidate[spec["key"]], spec,
                                                 contract["budgets"].get(spec["operation"]), context)
                    except (ValueError, TypeError) as exc:
                        if isinstance(exc, GateError):
                            raise
                        raise GateError("tensor_load", str(exc), **context) from exc
                    receipt["comparisons"].append({**context, **metrics})
                    receipt["compared_tensors"] += 1
        receipt.update(status="pass", eligible=True)
    except GateError as exc:
        receipt["first_failure"] = exc.detail
    except Exception as exc:
        # Unrecognized/corrupt encodings, ZIP/NPY errors and malformed contracts
        # must never become eligibility; preserve a bounded diagnostic.
        receipt["first_failure"] = {"code": "invalid_input", "message": f"{type(exc).__name__}: {exc}"[:1000]}
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "reference-manifest", "reference-trace", "candidate-manifest", "candidate-trace", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = [args.contract, args.reference_manifest, args.reference_trace, args.candidate_manifest, args.candidate_trace]
    if args.receipt.resolve() in {path.resolve() for path in inputs}:
        parser.error("receipt must not overwrite an input")
    receipt = run_gate(*inputs)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "eligible": receipt["eligible"], "first_failure": receipt["first_failure"]}))
    return 0 if receipt["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
