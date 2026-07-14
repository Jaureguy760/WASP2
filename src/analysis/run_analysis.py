"""Allelic imbalance analysis pipeline.

Main entry point for running the beta-binomial allelic imbalance analysis
using the Rust-accelerated backend.
"""

from __future__ import annotations

import logging
from csv import reader
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from .run_analysis_cohort_shared import run_cohort_shared_snv_analysis

# Rust analysis (required; no Python fallback)
try:
    from wasp2_rust import analyze_imbalance as rust_analyze_imbalance
except ImportError:
    rust_analyze_imbalance = None

logger = logging.getLogger(__name__)


class WaspAnalysisData:
    """Container for allelic imbalance analysis configuration.

    Attributes
    ----------
    count_file : str | Path
        Path to the count TSV file.
    region_col : str | None
        Column name for grouping variants by region.
    groupby : str | None
        Column name for additional grouping (e.g., parent gene).
    out_file : str
        Output file path for results.
    phased : bool
        Whether to use phased genotype information.
    model : Literal["single", "linear"]
        Dispersion model type.
    min_count : int
        Minimum total allele count threshold.
    pseudocount : int
        Pseudocount to add to allele counts.
    """

    def __init__(
        self,
        count_file: str | Path,
        min_count: int | None = None,
        pseudocount: int | None = None,
        phased: bool | None = None,
        model: str | None = None,
        out_file: str | None = None,
        region_col: str | None = None,
        groupby: str | None = None,
        per_variant: bool = False,
    ) -> None:
        # Per-variant (SNV-solo) mode cannot also name a region column: per-variant tests each
        # SNV independently, whereas region_col groups variants by that column.
        if per_variant and region_col is not None:
            raise ValueError(
                "per_variant cannot be combined with region_col: per-variant tests each SNV "
                "independently, while region_col groups by that column. Choose one."
            )
        self.per_variant: bool = per_variant

        # User input data
        self.count_file = count_file
        self.region_col = region_col
        self.groupby = groupby
        self.out_file = out_file

        self.phased: bool = bool(phased)

        # Default to single dispersion model
        if model == "linear":
            self.model: Literal["single", "linear"] = "linear"
        else:
            self.model = "single"

        # Default min count of 10, pseudocount of 1
        self.min_count: int = 10 if min_count is None else min_count
        self.pseudocount: int = 1 if pseudocount is None else pseudocount

        # Read header only for validation
        with open(self.count_file) as f:
            count_cols = next(reader(f, delimiter="\t"))

        # 7 columns at minimum, 10 at maximum
        # 3required : chr, pos, ref, alt
        # 3 optional: <GT>, <region>, <parent>
        # 3 required: ref_count, alt_count, other_count
        # [chr, pos, ref, alt, <GT>, <region>, <parent>, ref_c, alt_c, other_c]

        if "GT" in count_cols:
            min_cols = 8
            region_idx = 5
        else:
            min_cols = 7
            region_idx = 4

        # Check regions. Skip auto-detect in per-variant mode so region_col stays None
        # (which the Rust backend interprets as per-variant chrom_pos grouping).
        if self.region_col is None and not self.per_variant:
            if len(count_cols) > min_cols:
                self.region_col = count_cols[region_idx]

        # By default group by feature rather than parent?
        if self.groupby is not None:
            # If denoting to group by feature
            if (self.region_col is None) or (self.groupby == self.region_col):
                self.groupby = None

            elif (len(count_cols) > (min_cols + 1)) and self.groupby in {
                count_cols[region_idx + 1],
                "Parent",
                "parent",
            }:
                self.groupby = count_cols[region_idx + 1]
            else:
                logger.warning("%s not found in columns %s", self.groupby, count_cols)
                self.groupby = None

        # Create default outfile
        if self.out_file is None:
            self.out_file = str(Path.cwd() / "ai_results.tsv")


def run_ai_analysis(
    count_file: str | Path,
    min_count: int | None = None,
    pseudocount: int | None = None,
    phased: bool | None = None,
    model: str | None = None,
    out_file: str | None = None,
    region_col: str | None = None,
    groupby: str | None = None,
    per_variant: bool = False,
    scope: str | None = None,
    unit: str | None = None,
    min_donor_observations: int = 50,
    min_informative_donors: int = 3,
    expected_sha256: str | None = None,
) -> dict[str, Path] | None:
    """Run allelic imbalance analysis pipeline.

    Parameters
    ----------
    count_file : str | Path
        Path to TSV file with allele counts.
    min_count : int | None, optional
        Minimum total count threshold, by default 10.
    pseudocount : int | None, optional
        Pseudocount to add, by default 1.
    phased : bool | None, optional
        Use phased genotype information, by default False.
    model : str | None, optional
        Dispersion model. Legacy analysis accepts 'single' or 'linear'; cohort-shared
        SNVs also accept 'per-donor', which is their default.
    out_file : str | None, optional
        Output file path, by default 'ai_results.tsv'.
    region_col : str | None, optional
        Column name for grouping variants.
    groupby : str | None, optional
        Additional grouping column (not supported by the Rust backend; raises if set).
    per_variant : bool, optional
        Test each SNV independently (per-variant) instead of grouping by a region column.
        Forces per-variant even when a region column is present. Default False.

    Raises
    ------
    RuntimeError
        If Rust analysis extension is not available.
    """
    if scope is not None:
        if scope != "cohort-shared":
            raise ValueError("scope must be 'cohort-shared'")
        if unit != "snv":
            raise ValueError("Cohort-shared analysis requires --unit snv")
        if phased:
            raise ValueError("Cohort-shared SNV analysis is unphased; --phased is not valid")
        if region_col is not None or groupby is not None or per_variant:
            raise ValueError("Cohort-shared SNVs are grouped by exact alleles automatically")
        cohort_model = "per-donor" if model is None else model
        if cohort_model not in {"single", "linear", "per-donor"}:
            raise ValueError("Cohort-shared SNV model must be 'single', 'linear', or 'per-donor'")
        output = out_file if out_file is not None else str(Path.cwd() / "ai_results.tsv")
        return run_cohort_shared_snv_analysis(
            count_file,
            output,
            model=cast(Literal["single", "linear", "per-donor"], cohort_model),
            min_count=10 if min_count is None else min_count,
            pseudocount=1 if pseudocount is None else pseudocount,
            min_donor_observations=min_donor_observations,
            min_informative_donors=min_informative_donors,
            expected_sha256=expected_sha256,
        )

    if unit is not None:
        raise ValueError("--unit requires --scope cohort-shared")

    # Fail closed: --groupby is not supported by the Rust backend. It only re-keys the grouping
    # column (region_col = groupby), so the identical result is obtained with --region_col <parent>.
    # Erroring prevents silently returning feature-level results when parent-level was requested.
    if groupby is not None:
        raise RuntimeError(
            "--groupby (parent-level grouping) is not supported by the Rust analysis backend. "
            "Since groupby only re-keys the grouping column, group by the parent column directly "
            "with --region_col <parent_column> instead."
        )

    # Store analysis data and params
    ai_files = WaspAnalysisData(
        count_file,
        min_count=min_count,
        pseudocount=pseudocount,
        phased=phased,
        model=model,
        out_file=out_file,
        region_col=region_col,
        groupby=groupby,
        per_variant=per_variant,
    )

    # Run analysis pipeline (Rust only)
    if rust_analyze_imbalance is None:
        raise RuntimeError(
            "Rust analysis extension not available. Build it with "
            "`maturin develop --release` in the WASP2 env."
        )

    results = rust_analyze_imbalance(
        str(ai_files.count_file),
        min_count=ai_files.min_count,
        pseudocount=ai_files.pseudocount,
        method=ai_files.model,
        phased=ai_files.phased,
        region_col=ai_files.region_col,
    )
    ai_df = pd.DataFrame(results)

    if "fdr_pval" in ai_df.columns:
        ai_df = ai_df.sort_values(by="fdr_pval", ascending=True)

    # Write results
    ai_df.to_csv(ai_files.out_file, sep="\t", header=True, index=False)
