"""Cohort-shared ATAC SNV analysis with fail-closed provenance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import pandas as pd

try:
    from wasp2_rust import analyze_cohort_snvs as rust_analyze_cohort_snvs
except ImportError:
    rust_analyze_cohort_snvs = None

UTC = timezone.utc
CohortSnvModel = Literal["single", "linear", "per-donor"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_record(path: Path) -> dict[str, str | int]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _package_version() -> str:
    try:
        return version("wasp2")
    except PackageNotFoundError:
        return "unknown"


def _artifact_paths(result_path: Path) -> dict[str, Path]:
    if result_path.suffix != ".tsv":
        raise ValueError("Cohort SNV output must use a .tsv filename")
    stem = result_path.stem
    return {
        "results": result_path,
        "dispersion": result_path.with_name(f"{stem}.dispersion.tsv"),
        "qc": result_path.with_name(f"{stem}.qc.tsv"),
        "provenance": result_path.with_name(f"{stem}.provenance.json"),
    }


def _validate_canonical_lock(input_path: Path, input_sha256: str) -> dict[str, Any] | None:
    if input_path.parent.name != "inputs":
        return None
    lock_path = input_path.parent.parent / "LOCK.json"
    if not lock_path.is_file():
        return None
    lock = json.loads(lock_path.read_text())
    relative = f"inputs/{input_path.name}"
    if lock.get("status") != "SEALED_INPUT":
        raise ValueError("Canonical input lock is not sealed")
    if lock.get("files", {}).get(relative) != input_sha256:
        raise ValueError("Count file SHA-256 does not match the canonical input lock")
    return {
        "lock_id": lock.get("lock_id"),
        "status": lock["status"],
        "manifest": _file_record(lock_path),
    }


def _dispersion_frame(run: dict[str, Any], model: CohortSnvModel) -> pd.DataFrame:
    if model == "per-donor":
        frame = pd.DataFrame(run["donor_dispersion"])
        frame.insert(0, "scope", "donor")
        frame["linear_d1"] = pd.NA
        frame["linear_d2"] = pd.NA
        return frame
    return pd.DataFrame(
        [
            {
                "scope": "cohort",
                "sample": pd.NA,
                "rho": run["global_rho"],
                "n_observations": run["n_included_observations"],
                "linear_d1": run["linear_d1"],
                "linear_d2": run["linear_d2"],
            }
        ]
    )


def run_cohort_shared_snv_analysis(
    count_file: str | Path,
    out_file: str | Path,
    *,
    model: CohortSnvModel = "per-donor",
    min_count: int = 10,
    pseudocount: int = 1,
    min_donor_observations: int = 50,
    min_informative_donors: int = 3,
    expected_sha256: str | None = None,
) -> dict[str, Path]:
    """Run one shared-effect test per exact SNV across included donors."""
    if rust_analyze_cohort_snvs is None:
        raise RuntimeError(
            "Rust cohort SNV extension not available. Build it with "
            "`maturin develop --release` in the WASP2 environment."
        )
    if model not in {"single", "linear", "per-donor"}:
        raise ValueError("Cohort SNV model must be 'single', 'linear', or 'per-donor'")
    if min_count < 0 or pseudocount < 0:
        raise ValueError("min_count and pseudocount must be nonnegative")
    if min_donor_observations < 1 or min_informative_donors < 1:
        raise ValueError("donor thresholds must be positive")

    input_path = Path(count_file).expanduser().resolve(strict=True)
    input_signature = _signature(input_path)
    input_record = _file_record(input_path)
    input_sha256 = str(input_record["sha256"])
    if expected_sha256 is not None and input_sha256 != expected_sha256.lower():
        raise ValueError(f"Input SHA-256 {input_sha256} does not match expected {expected_sha256}")
    count_lock = _validate_canonical_lock(input_path, input_sha256)

    result_path = Path(out_file).expanduser().resolve()
    paths = _artifact_paths(result_path)
    if not result_path.parent.is_dir():
        raise ValueError(f"Output parent directory does not exist: {result_path.parent}")
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite analysis artifacts: {existing}")

    staging = Path(tempfile.mkdtemp(prefix=f".{result_path.stem}.staging-", dir=result_path.parent))
    try:
        run = rust_analyze_cohort_snvs(
            str(input_path),
            min_count=min_count,
            pseudocount=pseudocount,
            method=model,
            min_donor_observations=min_donor_observations,
            min_informative_donors=min_informative_donors,
        )
        if _signature(input_path) != input_signature:
            raise RuntimeError("Count input changed while analysis was running")
        results = pd.DataFrame(run["results"])
        if results.empty:
            raise RuntimeError("Rust returned no cohort SNV results")
        results["significant_q05"] = results["fdr_pval"] < 0.05
        results["significant_q10"] = results["fdr_pval"] < 0.10
        results = results.sort_values(["chrom", "pos", "ref", "alt"], kind="mergesort")
        dispersion = _dispersion_frame(run, model)
        qc = pd.DataFrame(run["donor_qc"]).sort_values("sample", kind="mergesort")
        qc["status"] = qc["included"].map({True: "included", False: "excluded_min_observations"})

        staged = {name: staging / path.name for name, path in paths.items()}
        results.to_csv(staged["results"], sep="\t", header=True, index=False)
        dispersion.to_csv(staged["dispersion"], sep="\t", header=True, index=False)
        qc.to_csv(staged["qc"], sep="\t", header=True, index=False)
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "candidate",
            "wasp2_version": _package_version(),
            "scientific_contract": {
                "assay": "bulk ATAC-seq",
                "analysis_unit": "exact biallelic SNV",
                "scope": "cohort-shared",
                "phased": False,
                "shared_effect": "one allelic proportion per SNV across donors",
                "dispersion": model,
                "multiple_testing": "Benjamini-Hochberg cohort-wide within model",
                "primary_q_threshold": 0.05,
                "secondary_q_threshold": 0.10,
            },
            "parameters": {
                "min_count": min_count,
                "pseudocount": pseudocount,
                "min_donor_observations": min_donor_observations,
                "min_informative_donors": min_informative_donors,
            },
            "inputs": {"counts": input_record, "canonical_lock": count_lock},
            "observations": {
                "raw": int(run["n_raw_observations"]),
                "included": int(run["n_included_observations"]),
                "tested_snvs": len(results),
            },
            "donors": {
                "total": len(qc),
                "included": int(qc["included"].sum()),
                "excluded": int((~qc["included"]).sum()),
            },
            "outputs": {
                name: {**_file_record(path), "path": paths[name].name}
                for name, path in staged.items()
                if name != "provenance"
            },
        }
        staged["provenance"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name, destination in paths.items():
            os.replace(staged[name], destination)
        staging.rmdir()
        return paths
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
