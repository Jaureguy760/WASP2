from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import betabinom, chi2, false_discovery_control
from typer.testing import CliRunner

from analysis import __main__ as analysis_cli
from analysis import run_analysis
from analysis.run_analysis_cohort_shared import run_cohort_shared_snv_analysis


def _counts() -> pd.DataFrame:
    rows = []
    counts = {
        "d1": [(18, 2), (10, 10), (14, 6), (7, 13)],
        "d2": [(16, 4), (9, 11), (13, 7), (8, 12)],
        "d3": [(19, 1), (11, 9), (15, 5), (6, 14)],
        "d4": [(17, 3), (10, 10), (12, 8), (9, 11)],
    }
    for sample, sample_counts in counts.items():
        for index, (ref_count, alt_count) in enumerate(sample_counts, start=1):
            pos = index * 100
            rows.append(
                {
                    "sample": sample,
                    "snv_id": f"chr1:{pos - 1}:A>G",
                    "chrom": "chr1",
                    "pos": pos,
                    "ref": "A",
                    "alt": "G",
                    "ref_count": ref_count,
                    "alt_count": alt_count,
                }
            )
    return pd.DataFrame(rows)


def _bb_nll(prob: float, rho: float, ref_count: np.ndarray, total: np.ndarray) -> float:
    rho = float(np.clip(rho, 1e-10, 1 - 1e-10))
    alpha = prob * (1 - rho) / rho
    beta = (1 - prob) * (1 - rho) / rho
    return float(-betabinom.logpmf(ref_count, total, alpha, beta).sum())


def _python_per_donor_oracle(frame: pd.DataFrame, pseudocount: int) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["ref_pc"] = prepared["ref_count"] + pseudocount
    prepared["alt_pc"] = prepared["alt_count"] + pseudocount
    prepared["N_pc"] = prepared["ref_pc"] + prepared["alt_pc"]
    donor_rho = {}
    for sample, donor in prepared.groupby("sample", sort=True):
        donor_ref = donor["ref_pc"].to_numpy()
        donor_total = donor["N_pc"].to_numpy()
        fit = minimize_scalar(
            lambda rho, ref=donor_ref, total=donor_total: _bb_nll(0.5, rho, ref, total),
            bounds=(0, 1),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 500},
        )
        assert fit.success
        donor_rho[sample] = float(fit.x)

    rows = []
    for snv_id, snv in prepared.groupby("snv_id", sort=True):
        snv_rows = tuple(snv.itertuples())

        def objective(prob: float, observations=snv_rows) -> float:
            return sum(
                _bb_nll(
                    prob,
                    donor_rho[row.sample],
                    np.array([row.ref_pc]),
                    np.array([row.N_pc]),
                )
                for row in observations
            )

        fit = minimize_scalar(
            objective,
            bounds=(0, 1),
            method="bounded",
            options={"xatol": 1e-8},
        )
        assert fit.success
        null_ll = -objective(0.5)
        alt_ll = -float(fit.fun)
        lrt = max(0.0, 2 * (alt_ll - null_ll))
        rows.append(
            {
                "snv_id": snv_id,
                "null_ll": null_ll,
                "alt_ll": alt_ll,
                "mu": float(fit.x),
                "lrt": lrt,
                "pval": float(chi2.sf(lrt, 1)),
            }
        )
    result = pd.DataFrame(rows).sort_values("snv_id").reset_index(drop=True)
    result["fdr_pval"] = false_discovery_control(result["pval"], method="bh")
    return result


def test_rust_per_donor_matches_independent_scipy_oracle(tmp_path: Path) -> None:
    wasp2_rust = pytest.importorskip("wasp2_rust")
    assert hasattr(wasp2_rust, "analyze_cohort_snvs"), "rebuild the Rust extension"
    counts = _counts()
    count_path = tmp_path / "counts.tsv"
    counts.to_csv(count_path, sep="\t", index=False)

    run = wasp2_rust.analyze_cohort_snvs(
        str(count_path),
        min_count=10,
        pseudocount=1,
        method="per-donor",
        min_donor_observations=4,
        min_informative_donors=4,
    )
    rust = pd.DataFrame(run["results"]).sort_values("snv_id").reset_index(drop=True)
    oracle = _python_per_donor_oracle(counts, pseudocount=1)

    assert len(run["donor_dispersion"]) == 4
    assert rust["donor_count"].eq(4).all()
    for column in ["null_ll", "alt_ll", "mu", "lrt", "pval", "fdr_pval"]:
        np.testing.assert_allclose(rust[column], oracle[column], rtol=2e-5, atol=2e-5)


def test_rust_cohort_rejects_duplicate_observation(tmp_path: Path) -> None:
    wasp2_rust = pytest.importorskip("wasp2_rust")
    counts = pd.concat([_counts(), _counts().iloc[[0]]], ignore_index=True)
    count_path = tmp_path / "duplicates.tsv"
    counts.to_csv(count_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="duplicate donor/SNV"):
        wasp2_rust.analyze_cohort_snvs(
            str(count_path),
            min_donor_observations=4,
            min_informative_donors=4,
        )


def test_legacy_binding_rejects_ambiguous_per_donor_method(tmp_path: Path) -> None:
    wasp2_rust = pytest.importorskip("wasp2_rust")
    count_path = tmp_path / "counts.tsv"
    _counts().to_csv(count_path, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="cohort-shared SNV API"):
        wasp2_rust.analyze_imbalance(str(count_path), method="per_donor")


def test_cli_routes_cohort_snvs_and_defaults_to_per_donor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = {}

    def fake_run(count_file: str | Path, out_file: str | Path, **kwargs: object):
        observed.update(kwargs)
        return {"results": Path(out_file)}

    monkeypatch.setattr(run_analysis, "run_cohort_shared_snv_analysis", fake_run)
    output = tmp_path / "results.tsv"
    run_analysis.run_ai_analysis(
        "counts.tsv",
        out_file=str(output),
        scope="cohort-shared",
        unit="snv",
    )
    assert observed["model"] == "per-donor"
    assert observed["min_informative_donors"] == 3


def test_cli_rejects_phased_cohort_snv_before_io() -> None:
    with pytest.raises(ValueError, match="unphased"):
        run_analysis.run_ai_analysis(
            "missing.tsv",
            scope="cohort-shared",
            unit="snv",
            phased=True,
        )


def test_console_interface_parses_cohort_options(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(analysis_cli, "run_ai_analysis", fake_run)
    try:
        result = CliRunner().invoke(
            analysis_cli.app,
            [
                "find-imbalance",
                "counts.tsv",
                "--scope",
                "cohort-shared",
                "--unit",
                "snv",
                "--model",
                "per-donor",
                "--min-informative-donors",
                "4",
            ],
        )
    except RuntimeError as error:
        if "Type not yet supported" in str(error):
            pytest.skip("installed Typer predates the project's declared minimum version")
        raise
    assert result.exit_code == 0, result.output
    assert observed["scope"] == "cohort-shared"
    assert observed["unit"] == "snv"
    assert observed["min_informative_donors"] == 4


def test_wrapper_refuses_existing_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    count_path = tmp_path / "counts.tsv"
    _counts().to_csv(count_path, sep="\t", index=False)
    output = tmp_path / "results.tsv"
    output.write_text("existing\n")
    monkeypatch.setattr(
        "analysis.run_analysis_cohort_shared.rust_analyze_cohort_snvs",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_cohort_shared_snv_analysis(count_path, output, min_donor_observations=4)


def test_wrapper_writes_candidate_artifact_set(tmp_path: Path) -> None:
    count_path = tmp_path / "counts.tsv"
    _counts().to_csv(count_path, sep="\t", index=False)
    output = tmp_path / "results.tsv"
    paths = run_cohort_shared_snv_analysis(
        count_path,
        output,
        min_donor_observations=4,
        min_informative_donors=4,
    )
    assert set(paths) == {"results", "dispersion", "qc", "provenance"}
    assert all(path.is_file() for path in paths.values())
    results = pd.read_csv(paths["results"], sep="\t")
    assert results["donor_count"].eq(4).all()
    assert {"significant_q05", "significant_q10"}.issubset(results)


def test_rayon_thread_count_is_byte_deterministic(tmp_path: Path) -> None:
    count_path = tmp_path / "counts.tsv"
    _counts().to_csv(count_path, sep="\t", index=False)
    script = """
import json
import sys
from pathlib import Path
import wasp2_rust

run = wasp2_rust.analyze_cohort_snvs(
    sys.argv[1], min_count=10, pseudocount=1, method="per-donor",
    min_donor_observations=4, min_informative_donors=4,
)
Path(sys.argv[2]).write_text(json.dumps(run, sort_keys=True, separators=(",", ":")))
"""
    outputs = []
    for threads in ["1", "8"]:
        output = tmp_path / f"threads_{threads}.json"
        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = threads
        subprocess.run(
            [sys.executable, "-c", script, str(count_path), str(output)],
            check=True,
            env=environment,
        )
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_oracle_fixture_has_finite_likelihoods() -> None:
    oracle = _python_per_donor_oracle(_counts(), pseudocount=1)
    numeric = oracle.drop(columns="snv_id").to_numpy()
    assert np.isfinite(numeric).all()
