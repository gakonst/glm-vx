"""Mutation tests for the gate, independent of any model implementation."""
import copy
import io
import json
import zipfile

import numpy as np
import pytest

from validation.trace_gate import case_metadata, main, run_gate, sha256


def npz_bytes(arrays):
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


@pytest.fixture
def fixture(tmp_path):
    arrays = {
        "p0/l0/attention": np.array([[1., 2.], [3., 4.]], dtype="float32"),
        "p0/l0/scores": np.array([0., -np.inf], dtype="float32"),
        "p0/l0/selected": np.array([0, 1], dtype="int64"),
        "p0/l0/cache_positions": np.array([0, 1], dtype="int64"),
        "p0/l0/cache_values": np.array([[1., 2.], [3., 4.]], dtype="float32"),
        "p0/logits": np.array([1., 2., 3., 4.], dtype="float32"),
    }
    specs = []
    for key, array in arrays.items():
        operation = key.split("/")[-1]
        kind = {"selected": "indices", "cache_positions": "exact", "cache_values": "exact",
                "logits": "logits"}.get(operation, "float")
        specs.append(dict(key=key, operation=operation, kind=kind, shape=list(array.shape),
                          dtype=str(array.dtype), position=0, layer=None if kind == "logits" else 0,
                          infinity="matching_negative" if operation == "scores" else "forbid"))
    h = "a" * 64
    producer = dict(binary_sha256=h, compiler="vxc test revision", libraries_sha256={"lib.so": h},
                    flags=["-O0"], hardware="synthetic CPU test")
    budget = dict(atol=1e-4, rtol=1e-5, normalized_l2=1e-4, normalization_floor=1e-8,
                  softmax_tv=1e-4, top_k=2, top_k_margin_atol=1e-4)
    contract = dict(schema_version=1, contract_id="mutation-fixture-v1", scope="synthetic gate tests only",
                    pins=dict(config_sha256=h, weights_sha256={"shard-1": h}, tokenizer_sha256=h,
                              template_sha256=h, corpus_sha256=h, corpus_version="fixture-v1",
                              reference_revision="b" * 40, numerical_mode="strict-f32", tie_breaking="stable-lowest-index"),
                    producers={"reference": producer, "candidate": copy.deepcopy(producer)},
                    reference_trace_sha256=sha256(npz_bytes(arrays)),
                    budgets={op: copy.deepcopy(budget) for op in ("attention", "scores", "logits")},
                    cases=[dict(id="case-1", token_ids=[1, 2], positions=[0], execution="decode", chunks=[1, 1],
                                cache_owner="request-1", layers=[0], vocab_size=4, tensors=specs)])
    paths = [tmp_path / name for name in ("contract.json", "reference.json", "reference.npz", "candidate.json", "candidate.npz")]

    def write(candidate=None, mutate_contract=None, mutate_manifest=None, reference=None):
        trusted = copy.deepcopy(contract)
        ref = npz_bytes(arrays if reference is None else reference)
        cand = npz_bytes(arrays if candidate is None else candidate)
        trusted["reference_trace_sha256"] = sha256(ref)
        if mutate_contract:
            mutate_contract(trusted)
        contract_bytes = json.dumps(trusted).encode()
        manifests = {}
        for role, data in (("reference", ref), ("candidate", cand)):
            manifest = dict(schema_version=1, role=role, contract_sha256=sha256(contract_bytes),
                            pins=copy.deepcopy(trusted["pins"]), producer=copy.deepcopy(trusted["producers"][role]),
                            status="complete", skipped=[], cases=[case_metadata(c) for c in trusted["cases"]],
                            trace_sha256=sha256(data))
            if mutate_manifest and role == "candidate":
                mutate_manifest(manifest)
            manifests[role] = json.dumps(manifest).encode()
        for path, data in zip(paths, (contract_bytes, manifests["reference"], ref, manifests["candidate"], cand)):
            path.write_bytes(data)
        return run_gate(*paths)
    return arrays, paths, write


def assert_fail(receipt, code, tensor=None):
    assert receipt["status"] == "fail"
    assert receipt["eligible"] is False
    assert receipt["first_failure"]["code"] == code, receipt
    if tensor:
        assert receipt["first_failure"]["tensor"] == tensor


def test_complete_trace_passes_and_hashes_exact_inputs(fixture):
    arrays, paths, write = fixture
    receipt = write()
    assert receipt["status"] == "pass"
    assert receipt["compared_tensors"] == len(arrays)
    for name, path in zip(receipt["files"], paths):
        assert receipt["files"][name]["sha256"] == sha256(path.read_bytes())
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("tensor,mutation,code", [
    ("p0/l0/attention", lambda a: a.T, "numeric"),
    ("p0/l0/attention", lambda a: a + .1, "numeric"),
    ("p0/l0/attention", lambda a: a.astype("float64"), "dtype"),
    ("p0/l0/attention", lambda a: a.flatten(), "shape"),
    ("p0/l0/attention", lambda a: np.full_like(a, np.nan), "nonfinite"),
    ("p0/l0/attention", lambda a: np.full_like(a, np.inf), "nonfinite"),
    ("p0/l0/selected", lambda a: a[::-1], "exact"),
    ("p0/l0/cache_positions", lambda a: a + 1, "exact"),
    ("p0/l0/cache_values", lambda a: a + 1e-6, "exact"),
    ("p0/logits", lambda a: a + np.array([.2, 0., 0., 0.], dtype="float32"), "numeric"),
    ("p0/l0/scores", lambda a: a[::-1], "nonfinite"),
    ("p0/l0/scores", lambda a: np.array([0., np.inf], dtype="float32"), "nonfinite"),
    ("p0/l0/scores", lambda a: np.array([0., 0.], dtype="float32"), "nonfinite"),
])
def test_array_mutations(fixture, tensor, mutation, code):
    arrays, _, write = fixture
    candidate = copy.deepcopy(arrays)
    candidate[tensor] = mutation(candidate[tensor])
    receipt = write(candidate)
    assert_fail(receipt, code, tensor)
    assert receipt["first_failure"]["case"] == "case-1"
    assert receipt["first_failure"]["position"] == 0


@pytest.mark.parametrize("mutation,code", [
    (lambda m: m["pins"].update(config_sha256="c" * 64), "identity"),
    (lambda m: m["pins"]["weights_sha256"].update({"shard-1": "c" * 64}), "identity"),
    (lambda m: m["producer"].update(binary_sha256="c" * 64), "identity"),
    (lambda m: m.update(contract_sha256="c" * 64), "identity"),
    (lambda m: m.update(trace_sha256="c" * 64), "identity"),
    (lambda m: m.update(status="incomplete"), "incomplete"),
    (lambda m: m.update(skipped=["long-context"]), "incomplete"),
    (lambda m: m.update(cases=[]), "coverage"),
    (lambda m: m["cases"][0].update(token_ids=[2, 1]), "coverage"),
    (lambda m: m["cases"][0].update(positions=[1]), "coverage"),
    (lambda m: m["cases"][0].update(cache_owner="other-request"), "coverage"),
    (lambda m: m["cases"][0].update(chunks=[2]), "coverage"),
    (lambda m: m.update(budgets={"atol": 1e10}), "schema"),
    (lambda m: m.update(schema_version=2), "schema"),
])
def test_manifest_mutations(fixture, mutation, code):
    assert_fail(fixture[2](mutate_manifest=mutation), code)


def test_missing_and_extra_tensor_fail(fixture):
    arrays, _, write = fixture
    for name in arrays:
        candidate = copy.deepcopy(arrays)
        del candidate[name]
        assert_fail(write(candidate), "coverage")
    assert_fail(write({**arrays, "unexpected": np.ones(1)}), "coverage")


def test_object_arrays_are_not_loaded_with_pickle(fixture):
    arrays, _, write = fixture
    candidate = copy.deepcopy(arrays)
    candidate["p0/l0/attention"] = np.array([[object(), object()], [object(), object()]])
    assert_fail(write(candidate), "tensor_load", "p0/l0/attention")


def test_first_failure_follows_trusted_stage_order(fixture):
    arrays, _, write = fixture
    candidate = {k: v.copy() for k, v in arrays.items()}
    candidate["p0/l0/attention"] += 1
    candidate["p0/logits"] += 1
    receipt = write(candidate)
    assert_fail(receipt, "numeric", "p0/l0/attention")
    assert receipt["compared_tensors"] == 0


def test_small_errors_within_frozen_budgets_pass(fixture):
    arrays, _, write = fixture
    candidate = copy.deepcopy(arrays)
    candidate["p0/l0/attention"] += 1e-6
    assert write(candidate)["eligible"]


@pytest.mark.parametrize("metric", ["normalized_l2", "softmax_tv", "top_k_margin_atol"])
def test_independent_numerical_budgets(fixture, metric):
    arrays, _, write = fixture
    candidate = copy.deepcopy(arrays)
    candidate["p0/logits"][0 if metric == "softmax_tv" else 2] += .01
    def loosen_pointwise(c):
        b = c["budgets"]["logits"]
        b.update(atol=1, rtol=1, normalized_l2=1, softmax_tv=1, top_k_margin_atol=1)
        b[metric] = 0
    assert_fail(write(candidate, mutate_contract=loosen_pointwise),
                "numeric" if metric == "normalized_l2" else "logits", "p0/logits")


def test_top_k_order_cannot_be_waived_by_numeric_budget(fixture):
    arrays, _, write = fixture
    candidate = copy.deepcopy(arrays)
    candidate["p0/logits"] = candidate["p0/logits"][::-1]
    def loose(c):
        c["budgets"]["logits"].update(atol=100, rtol=100, normalized_l2=100,
                                       softmax_tv=1, top_k_margin_atol=100)
    assert_fail(write(candidate, mutate_contract=loose), "logits", "p0/logits")


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(cases=[]),
    lambda c: c["cases"][0].update(positions=[]),
    lambda c: c["cases"][0].update(layers=[0, 1]),
    lambda c: c["cases"][0]["tensors"].pop(),
    lambda c: c["cases"][0]["tensors"][-1].update(shape=[2]),
    lambda c: c["cases"][0]["tensors"].append(c["cases"][0]["tensors"][0]),
    lambda c: c["budgets"]["attention"].update(atol=float("nan")),
    lambda c: c["budgets"]["attention"].update(normalization_floor=0),
    lambda c: c["budgets"]["attention"].update(rtol=-1),
    lambda c: c["budgets"]["attention"].update(top_k=True),
    lambda c: c["budgets"].pop("attention"),
    lambda c: c["pins"].update(config_sha256="stale"),
])
def test_invalid_or_vacuous_contract_fails(fixture, mutation):
    receipt = fixture[2](mutate_contract=mutation)
    assert receipt["status"] == "fail"
    assert not receipt["eligible"]


def test_reference_nan_rejected_even_if_both_match(fixture):
    arrays, _, write = fixture
    reference = copy.deepcopy(arrays)
    reference["p0/l0/attention"][0, 0] = np.nan
    assert_fail(write(reference, reference=reference), "nonfinite", "p0/l0/attention")


def test_reference_archive_is_frozen(fixture):
    _, paths, write = fixture
    write()
    paths[2].write_bytes(paths[2].read_bytes() + b"tampered")
    assert_fail(run_gate(*paths), "identity")


def test_missing_input_and_malformed_json_fail_with_receipt(fixture):
    _, paths, write = fixture
    write()
    paths[4].unlink()
    assert_fail(run_gate(*paths), "missing_input")
    write()
    paths[0].write_text('{"schema_version": 1, "schema_version": 1}')
    assert_fail(run_gate(*paths), "schema")


def test_duplicate_archive_members_fail(fixture):
    _, paths, write = fixture
    write()
    with zipfile.ZipFile(paths[4], "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("p0/logits.npy", b"bad")
    manifest = json.loads(paths[3].read_text())
    manifest["trace_sha256"] = sha256(paths[4].read_bytes())
    paths[3].write_text(json.dumps(manifest))
    assert_fail(run_gate(*paths), "coverage")


def test_cli_exit_codes_and_receipt(fixture, tmp_path):
    _, paths, write = fixture
    output = tmp_path / "receipt.json"
    args = []
    for flag, path in zip(("contract", "reference-manifest", "reference-trace", "candidate-manifest", "candidate-trace"), paths):
        args.extend(["--" + flag, str(path)])
    args.extend(["--receipt", str(output)])
    write()
    assert main(args) == 0
    assert json.loads(output.read_text())["eligible"] is True
    paths[4].unlink()
    assert main(args) == 1
    assert json.loads(output.read_text())["eligible"] is False
    with pytest.raises(SystemExit):
        main(args[:-1] + [str(paths[0])])


@pytest.mark.parametrize("mutation", [
    lambda m: m["cases"][0].update(positions=[False]),
    lambda m: m["cases"][0].update(token_ids=[True, 2]),
])
def test_boolean_does_not_alias_integer_metadata(fixture, mutation):
    assert_fail(fixture[2](mutate_manifest=mutation), "coverage")


@pytest.mark.parametrize("mutation", [
    lambda c: c["pins"].update(reference_revision="main"),
    lambda c: c["pins"].update(tie_breaking="any tied index accepted"),
])
def test_reference_revision_and_ties_must_be_frozen(fixture, mutation):
    assert_fail(fixture[2](mutate_contract=mutation), "schema")


def test_zero_reference_uses_absolute_floor(fixture):
    arrays, _, write = fixture
    reference = copy.deepcopy(arrays)
    reference["p0/l0/attention"][:] = 0
    candidate = copy.deepcopy(reference)
    candidate["p0/l0/attention"][0, 0] = 1e-6
    # Pointwise tolerance permits this, but the frozen normalized budget does not.
    assert_fail(write(candidate, reference=reference), "numeric", "p0/l0/attention")


def test_all_masked_tensor_is_explicitly_permitted(fixture):
    arrays, _, write = fixture
    reference = copy.deepcopy(arrays)
    reference["p0/l0/scores"][:] = -np.inf
    assert write(reference, reference=reference)["eligible"]
