/// WASP2 Analysis Module - Beta-binomial Allelic Imbalance Detection
///
/// Rust implementation of the Python analysis stage (src/analysis/as_analysis.py)
/// Uses beta-binomial model to detect allelic imbalance in ASE data.
///
/// Performance target: 3-5x speedup over Python (2.7s → 0.5-0.9s)
use anyhow::{Context, Result};
use rayon::prelude::*;
use rv::dist::BetaBinomial;
use rv::traits::HasDensity;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// ============================================================================
// Data Structures
// ============================================================================

/// Allele count data for a single variant
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VariantCounts {
    pub chrom: String,
    pub pos: u32,
    pub ref_count: u32,
    pub alt_count: u32,
    pub region: String,
    /// Donor/sample identifier; required only by the per_donor model.
    pub sample: Option<String>,
    /// Genotype phase relative to the reference allele: 0 = ref|alt, 1 = alt|ref.
    /// `None` when genotypes are absent or unphased. Required by phased analysis.
    pub gt: Option<u8>,
}

/// Deduplication key for matching Python's `df[keep_cols].drop_duplicates()`.
/// Used to remove exact duplicate observations before analysis.
#[derive(Hash, Eq, PartialEq)]
struct DedupeKey {
    chrom: String,
    pos: u32,
    region: String,
    ref_count: u32,
    alt_count: u32,
    n: u32,
    gt: Option<u8>, // Include GT when phased=true
}

/// Statistical results for a region
#[derive(Debug, Clone)]
pub struct ImbalanceResult {
    pub region: String,
    pub ref_count: u32,
    pub alt_count: u32,
    pub n: u32,
    pub snp_count: usize,
    pub null_ll: f64,  // Null model log-likelihood
    pub alt_ll: f64,   // Alternative model log-likelihood
    pub mu: f64,       // Estimated imbalance proportion
    pub lrt: f64,      // Likelihood ratio test statistic
    pub pval: f64,     // P-value
    pub fdr_pval: f64, // FDR-corrected p-value
}

/// Configuration for analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub min_count: u32,
    pub pseudocount: u32,
    pub method: AnalysisMethod,
    pub phased: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMethod {
    Single,   // Single global dispersion parameter
    Linear,   // Linear dispersion model: rho = expit(d1 + N*d2)
    PerDonor, // Per-donor dispersion: rho fit separately per sample
}

/// One donor/SNV count observation for cohort-shared SNV inference.
#[derive(Debug, Clone)]
pub struct CohortSnvObservation {
    pub sample: String,
    pub snv_id: String,
    pub chrom: String,
    pub pos: u32,
    pub ref_allele: String,
    pub alt_allele: String,
    pub ref_count: u32,
    pub alt_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortSnvMethod {
    Single,
    Linear,
    PerDonor,
}

#[derive(Debug, Clone)]
pub struct CohortSnvConfig {
    pub min_count: u32,
    pub pseudocount: u32,
    pub min_donor_observations: usize,
    pub min_informative_donors: usize,
    pub method: CohortSnvMethod,
}

#[derive(Debug, Clone)]
pub struct CohortSnvResult {
    pub snv_id: String,
    pub chrom: String,
    pub pos: u32,
    pub ref_allele: String,
    pub alt_allele: String,
    pub ref_count: u64,
    pub alt_count: u64,
    pub n: u64,
    pub donor_count: usize,
    pub null_ll: f64,
    pub alt_ll: f64,
    pub mu: f64,
    pub lrt: f64,
    pub pval: f64,
    pub fdr_pval: f64,
}

#[derive(Debug, Clone)]
pub struct DonorQc {
    pub sample: String,
    pub raw_observations: usize,
    pub eligible_observations: usize,
    pub included: bool,
}

#[derive(Debug, Clone)]
pub struct DonorDispersion {
    pub sample: String,
    pub rho: f64,
    pub n_observations: usize,
}

#[derive(Debug, Clone)]
pub struct CohortSnvOutput {
    pub results: Vec<CohortSnvResult>,
    pub donor_qc: Vec<DonorQc>,
    pub donor_dispersion: Vec<DonorDispersion>,
    pub global_rho: Option<f64>,
    pub linear_params: Option<(f64, f64)>,
    pub n_raw_observations: usize,
    pub n_included_observations: usize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct SnvKey {
    chrom: String,
    pos: u32,
    ref_allele: String,
    alt_allele: String,
}

#[derive(Debug, Clone)]
struct PreparedCohortSnvObservation {
    sample: String,
    snv_id: String,
    key: SnvKey,
    ref_count: u32,
    alt_count: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_count: 10,
            pseudocount: 1,
            method: AnalysisMethod::Single,
            phased: false,
        }
    }
}

// ============================================================================
// Rho Parameter Bounds (Issue #228)
// ============================================================================

/// Epsilon for clamping rho parameter to avoid division by zero.
/// Matches Python's RHO_EPSILON in as_analysis.py.
const RHO_EPSILON: f64 = 1e-10;

/// Clamp rho to safe range (epsilon, 1-epsilon) to prevent division by zero.
///
/// The beta-binomial parameterization uses alpha = mu * (1-rho) / rho, which
/// causes division by zero when rho=0 and produces zero alpha/beta when rho=1.
///
/// # Note on Silent Clamping
///
/// This function does not log warnings when clamping occurs because it is called
/// in tight optimization loops where logging would impact performance. Extreme
/// rho values (at boundaries) may indicate data quality issues. The Python
/// implementation (`as_analysis.py`) supports optional warning via `warn=True`.
#[inline]
fn clamp_rho(rho: f64) -> f64 {
    rho.clamp(RHO_EPSILON, 1.0 - RHO_EPSILON)
}

/// Deduplicate variants to match Python's `df[keep_cols].drop_duplicates()`.
///
/// Python keeps columns: chrom, pos, ref_count, alt_count, N, region (+ GT if phased)
/// and removes exact duplicate rows. This ensures parity with Python's behavior.
fn deduplicate_variants(variants: Vec<VariantCounts>, phased: bool) -> Vec<VariantCounts> {
    let mut seen: HashSet<DedupeKey> = HashSet::new();
    let mut result = Vec::with_capacity(variants.len());

    for v in variants {
        let key = DedupeKey {
            chrom: v.chrom.clone(),
            pos: v.pos,
            region: v.region.clone(),
            ref_count: v.ref_count,
            alt_count: v.alt_count,
            n: v.ref_count + v.alt_count,
            gt: if phased { v.gt } else { None },
        };
        if seen.insert(key) {
            result.push(v);
        }
    }
    result
}

fn effective_phased(variants: &[VariantCounts], requested_phased: bool) -> bool {
    if !requested_phased {
        return false;
    }

    let mut saw_ref_alt = false;
    let mut saw_alt_ref = false;
    for variant in variants {
        match variant.gt {
            Some(0) => saw_ref_alt = true,
            Some(1) => saw_alt_ref = true,
            _ => {}
        }
    }

    if saw_ref_alt && saw_alt_ref {
        true
    } else {
        eprintln!("No informative numeric phased GT found: switching to unphased model");
        false
    }
}

/// Sigmoid function (inverse logit): expit(x) = 1 / (1 + e^-x)
///
/// Numerically stable implementation that avoids overflow for large |x|.
#[inline]
fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ============================================================================
// Core Statistical Functions
// ============================================================================

/// Calculate beta-binomial log-likelihood (negative for optimization)
///
/// Python equivalent: `opt_prob()` in as_analysis.py
///
/// # Arguments
/// * `prob` - Probability parameter (0 to 1)
/// * `rho` - Dispersion parameter (0 to 1), will be clamped to safe range
/// * `k` - Reference allele count
/// * `n` - Total count
///
/// # Returns
/// Negative log-likelihood value (for minimization)
pub fn opt_prob(prob: f64, rho: f64, k: u32, n: u32) -> Result<f64> {
    // Clamp rho to prevent division by zero (Issue #228)
    let rho = clamp_rho(rho);

    // Convert to alpha/beta parameters for beta-binomial
    let alpha = prob * (1.0 - rho) / rho;
    let beta = (1.0 - prob) * (1.0 - rho) / rho;

    // Create beta-binomial distribution (rv uses: n as u32, alpha, beta)
    let bb =
        BetaBinomial::new(n, alpha, beta).context("Failed to create beta-binomial distribution")?;

    // Return negative log-likelihood (rv uses reference for ln_f, k as u64)
    let log_pmf = bb.ln_f(&(k as u64));
    Ok(-log_pmf)
}

/// Calculate beta-binomial log-likelihood for array of counts
///
/// Python equivalent: Used in `single_model()` for null/alt likelihood
pub fn betabinom_logpmf_sum(
    ref_counts: &[u32],
    n_array: &[u32],
    alpha: f64,
    beta: f64,
) -> Result<f64> {
    let mut sum = 0.0;

    for (k, n) in ref_counts.iter().zip(n_array.iter()) {
        let bb = BetaBinomial::new(*n, alpha, beta)
            .context("Failed to create beta-binomial distribution")?;
        sum += bb.ln_f(&(*k as u64));
    }

    Ok(sum)
}

// ============================================================================
// Optimization Functions
// ============================================================================

/// Optimize dispersion parameter using Brent's method
///
/// Python equivalent: `minimize_scalar()` in scipy.optimize
fn optimize_dispersion(ref_counts: &[u32], n_array: &[u32]) -> Result<f64> {
    // Objective function: negative log-likelihood of null model (prob=0.5)
    let objective = |rho: f64| -> f64 {
        // Clamp rho to prevent division by zero (Issue #228)
        let rho = clamp_rho(rho);
        let alpha = 0.5 * (1.0 - rho) / rho;
        let beta = 0.5 * (1.0 - rho) / rho;

        match betabinom_logpmf_sum(ref_counts, n_array, alpha, beta) {
            Ok(ll) => -ll, // Return negative for minimization
            Err(_) => f64::INFINITY,
        }
    };

    // Match scipy.optimize.minimize_scalar(method="bounded") used by AHO.
    // Sparse donors can have an MLE below 0.001; imposing that floor changes
    // the null likelihood and every downstream p-value.
    scipy_bounded_minimize(objective, 0.0, 1.0, 1e-5, 500)
}

/// Optimize linear dispersion parameters using Nelder-Mead
///
/// Python equivalent: `minimize(opt_linear, x0=(0, 0), method="Nelder-Mead")`
/// Fits ρ = expit(d1 + N × d2) to the data.
fn optimize_dispersion_linear(ref_counts: &[u32], n_array: &[u32]) -> Result<(f64, f64)> {
    let objective = |params: [f64; 2]| -> f64 {
        let (d1, d2) = (params[0], params[1]);
        let mut neg_ll = 0.0;

        for (&ref_count, &n) in ref_counts.iter().zip(n_array.iter()) {
            // Clamp to prevent overflow in expit
            let exp_in = (d1 + n as f64 * d2).clamp(-10.0, 10.0);
            let rho = clamp_rho(expit(exp_in));
            let alpha = 0.5 * (1.0 - rho) / rho;

            // Beta-binomial log-pmf with α = β (null: p=0.5)
            if let Ok(bb) = rv::dist::BetaBinomial::new(n, alpha, alpha) {
                use rv::traits::HasDensity;
                neg_ll -= bb.ln_f(&(ref_count as u64));
            } else {
                return f64::INFINITY;
            }
        }

        neg_ll
    };

    // Multi-start Nelder-Mead. A single start at (0,0) can stall on the flat plateau
    // created by the exp_in clamp (when |d1 + N*d2| > 10 for all N, rho is constant, so
    // the surface has no descent direction and the simplex collapses at an unconverged,
    // worse point). Run NM from a neutral grid of starts spanning the plausible (d1,d2)
    // range and keep the best (lowest neg-LL). Deterministic; starts are NOT seeded near
    // any expected answer. Fixes the case where the simple simplex returned e.g. (1.5,-2.0).
    let starts: [[f64; 2]; 8] = [
        [0.0, 0.0],
        [-3.0, 0.0],
        [-6.0, 0.0],
        [3.0, 0.0],
        [0.0, -0.2],
        [-3.0, -0.2],
        [-6.0, -0.2],
        [3.0, -0.2],
    ];
    let mut best: Option<(f64, (f64, f64))> = None;
    for s in starts {
        if let Ok((d1, d2)) = nelder_mead_2d(&objective, s, 1e-6, 1000) {
            let val = objective([d1, d2]);
            if val.is_finite() && best.map_or(true, |(bv, _)| val < bv) {
                best = Some((val, (d1, d2)));
            }
        }
    }
    best.map(|(_, p)| p)
        .ok_or_else(|| anyhow::anyhow!("linear dispersion optimization failed from all starts"))
}

/// Optimize probability parameter for alternative model
///
/// Python equivalent: `parse_opt()` calling `minimize_scalar(opt_prob, ...)`
fn optimize_prob(ref_counts: &[u32], n_array: &[u32], disp: f64) -> Result<(f64, f64)> {
    // For single SNP, optimize directly
    if ref_counts.len() == 1 {
        let objective = |prob: f64| -> f64 {
            match opt_prob(prob, disp, ref_counts[0], n_array[0]) {
                Ok(nll) => nll,
                Err(_) => f64::INFINITY,
            }
        };

        let mu = golden_section_search(objective, 0.0, 1.0, 1e-6)?;
        let alt_ll = -objective(mu);
        return Ok((alt_ll, mu));
    }

    // For multiple SNPs, sum log-likelihoods
    let objective = |prob: f64| -> f64 {
        let mut sum = 0.0;
        for (k, n) in ref_counts.iter().zip(n_array.iter()) {
            match opt_prob(prob, disp, *k, *n) {
                Ok(nll) => sum += nll,
                Err(_) => return f64::INFINITY,
            }
        }
        sum
    };

    let mu = golden_section_search(objective, 0.0, 1.0, 1e-6)?;
    let alt_ll = -objective(mu);
    Ok((alt_ll, mu))
}

/// Optimize probability for phased data using known genotype phase.
///
/// Python equivalent: `parse_opt()` phased branch calling
/// `minimize_scalar(opt_phased_new, ...)` in as_analysis.py (L213-256, L118-140).
///
/// The phase array is first normalized so the first SNP is the reference
/// (`if gt[0] > 0 { gt = 1 - gt }`), matching Python's
/// `if gt_array[0] > 0: gt_array = 1 - gt_array`. The objective then sums
/// `opt_prob(|prob - gt_i|, disp, ref_i, n_i)` across the region and is
/// minimized over `prob` in (0, 1).
///
/// # Arguments
/// * `ref_counts` - Reference allele counts for the region
/// * `n_array` - Total counts for the region
/// * `gt_array` - Genotype phase per variant (0 or 1)
/// * `disp` - Per-SNP dispersion (rho) slice, aligned 1:1 with ref/n/gt. A
///   length-1 slice broadcasts to all SNPs (matches the single/per-region scalar
///   path); a multi-element slice supplies a distinct rho per SNP (linear model's
///   per-row rho, mirroring Python's per-row `df["disp"]`).
///
/// # Returns
/// Tuple of (alt_ll, mu) - alternative-model log-likelihood and imbalance proportion
fn optimize_prob_phased(
    ref_counts: &[u32],
    n_array: &[u32],
    gt_array: &[u8],
    disp: &[f64],
) -> Result<(f64, f64)> {
    // Normalize phase so the first SNP is with respect to ref (matches Python).
    let flip = gt_array.first().is_some_and(|&g| g > 0);
    let gt_norm: Vec<u8> = gt_array
        .iter()
        .map(|&g| if flip { 1 - g } else { g })
        .collect();

    // objective(prob) = Σ opt_prob(|prob - gt_i|, rho_i, ref_i, n_i)
    // rho_i is the per-SNP dispersion (disp[i]) when a per-row slice is supplied,
    // or disp[0] broadcast across all SNPs for the scalar/single case.
    let objective = |prob: f64| -> f64 {
        let mut sum = 0.0;
        for (i, ((&k, &n), &g)) in ref_counts
            .iter()
            .zip(n_array.iter())
            .zip(gt_norm.iter())
            .enumerate()
        {
            let rho = if disp.len() > 1 { disp[i] } else { disp[0] };
            let phased_prob = (prob - g as f64).abs();
            match opt_prob(phased_prob, rho, k, n) {
                Ok(nll) => sum += nll,
                Err(_) => return f64::INFINITY,
            }
        }
        sum
    };

    let mu = golden_section_search(objective, 0.0, 1.0, 1e-6)?;
    let alt_ll = -objective(mu);
    Ok((alt_ll, mu))
}

/// Optimize probability for unphased data using dynamic programming.
///
/// Python equivalent: `opt_unphased_dp()` in as_analysis.py
///
/// This marginalizes over unknown phase using dynamic programming. At each
/// position after the first, we consider both phase possibilities and weight
/// them equally (0.5 each), accumulating likelihood across the region.
///
/// # Arguments
/// * `prob` - Probability parameter (0 to 1)
/// * `disp` - Dispersion parameter(s). If slice length > 1, first element is for
///   first position and remaining elements are for subsequent positions.
/// * `first_ref` - Reference count for first position
/// * `first_n` - Total count for first position
/// * `phase_ref` - Reference counts for subsequent positions
/// * `phase_n` - Total counts for subsequent positions
///
/// # Returns
/// Negative log-likelihood value (for minimization)
pub fn optimize_prob_unphased_dp(
    prob: f64,
    disp: &[f64],
    first_ref: u32,
    first_n: u32,
    phase_ref: &[u32],
    phase_n: &[u32],
) -> Result<f64> {
    // Split per-variant dispersion for first vs subsequent positions (linear model)
    let (first_disp, phase_disp): (f64, &[f64]) = if disp.len() > 1 {
        (disp[0], &disp[1..])
    } else {
        // Scalar dispersion: use same value for all positions
        (disp[0], disp)
    };

    // Get likelihood of first position (negative log-likelihood)
    let first_ll = opt_prob(prob, first_disp, first_ref, first_n)?;

    // If no subsequent positions, just return first_ll
    if phase_ref.is_empty() {
        return Ok(first_ll);
    }

    // Accumulate likelihoods for subsequent positions under both phase hypotheses
    // in LOG-space (mathematically identical to the old linear product, but without
    // underflow). For each position:
    //   log(0.5 * p1 + 0.5 * p2) = ln(0.5) + logaddexp(ln(p1), ln(p2))
    // where lp1 = ln-pmf with prob and lp2 = ln-pmf with (1 - prob).
    let mut log_prev: f64 = 0.0;

    for (i, (&ref_count, &n)) in phase_ref.iter().zip(phase_n.iter()).enumerate() {
        // Get dispersion for this position
        let rho = if phase_disp.len() > 1 {
            clamp_rho(phase_disp[i.min(phase_disp.len() - 1)])
        } else {
            clamp_rho(phase_disp[0])
        };

        // Phase 1: use prob
        let alpha1 = prob * (1.0 - rho) / rho;
        let beta1 = (1.0 - prob) * (1.0 - rho) / rho;
        let bb1 = BetaBinomial::new(n, alpha1, beta1)
            .context("Failed to create beta-binomial for phase1")?;
        let lp1 = bb1.ln_f(&(ref_count as u64)); // log-pmf

        // Phase 2: use (1 - prob)
        let alpha2 = (1.0 - prob) * (1.0 - rho) / rho;
        let beta2 = prob * (1.0 - rho) / rho;
        let bb2 = BetaBinomial::new(n, alpha2, beta2)
            .context("Failed to create beta-binomial for phase2")?;
        let lp2 = bb2.ln_f(&(ref_count as u64)); // log-pmf

        // DP update in log-space: marginalize over both phases with equal weight.
        log_prev += (0.5_f64).ln() + logaddexp(lp1, lp2);
    }

    // Return first_ll + (-log_prev)
    if !log_prev.is_finite() {
        return Ok(f64::INFINITY);
    }

    Ok(first_ll + (-log_prev))
}

/// Numerically stable two-argument log-sum-exp: ln(exp(a) + exp(b)).
///
/// Equivalent to NumPy's `np.logaddexp`. Subtracting the max before exponentiating
/// avoids overflow/underflow. If the max is non-finite (e.g. -inf when both
/// inputs are -inf), the max is returned directly.
#[inline]
fn logaddexp(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if !m.is_finite() {
        m
    } else {
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
}

/// Bounded Brent minimizer matching SciPy's scalar-bounded routine.
fn scipy_bounded_minimize<F>(f: F, x1: f64, x2: f64, xatol: f64, maxfun: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if !x1.is_finite() || !x2.is_finite() || x1 > x2 {
        anyhow::bail!("invalid bounded optimization interval");
    }
    let sqrt_eps = (2.2e-16_f64).sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = f(xf);
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut fu = f64::INFINITY;
    let mut calls = 1_usize;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut golden = true;
        if e.abs() > tol1 {
            let r = (xf - nfc) * (fx - ffulc);
            let q0 = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q0 - (xf - nfc) * r;
            let mut q = 2.0 * (q0 - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let previous_e = e;
            e = rat;
            if p.abs() < (0.5 * q * previous_e).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let candidate = xf + rat;
                if candidate - a < tol2 || b - candidate < tol2 {
                    rat = tol1 * if xm - xf >= 0.0 { 1.0 } else { -1.0 };
                }
                golden = false;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }

        let direction = if rat >= 0.0 { 1.0 } else { -1.0 };
        let x = xf + direction * rat.abs().max(tol1);
        fu = f(x);
        calls += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if calls >= maxfun {
            anyhow::bail!("bounded scalar optimization exceeded {maxfun} evaluations");
        }
    }
    if !xf.is_finite() || !fx.is_finite() || fu.is_nan() {
        anyhow::bail!("bounded scalar optimization produced a non-finite result");
    }
    Ok(xf)
}

/// Golden section search for 1D optimization
///
/// Simple but robust method for bounded scalar optimization.
/// Equivalent to scipy's minimize_scalar with method='bounded'
#[allow(unused_assignments)]
fn golden_section_search<F>(f: F, a: f64, mut b: f64, tol: f64) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    const PHI: f64 = 1.618033988749895; // Golden ratio
    let inv_phi = 1.0 / PHI;
    let inv_phi2 = 1.0 / (PHI * PHI);

    let mut a = a;
    let mut h = b - a;

    // Initial points
    let mut c = a + inv_phi2 * h;
    let mut d = a + inv_phi * h;
    let mut fc = f(c);
    let mut fd = f(d);

    // Iterate until convergence
    while h.abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            h = inv_phi * h;
            c = a + inv_phi2 * h;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            h = inv_phi * h;
            d = a + inv_phi * h;
            fd = f(d);
        }
    }

    Ok(if fc < fd { c } else { d })
}

/// Nelder-Mead simplex optimization for 2D functions
///
/// Used for fitting linear dispersion parameters (d1, d2).
/// Equivalent to scipy's minimize(..., method="Nelder-Mead")
fn nelder_mead_2d<F>(f: F, x0: [f64; 2], tol: f64, max_iter: usize) -> Result<(f64, f64)>
where
    F: Fn([f64; 2]) -> f64,
{
    // Standard Nelder-Mead coefficients
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrinkage

    // Initialize simplex: equilateral triangle around x0
    let step = 1.0;
    let mut simplex = [
        (x0, f(x0)),
        ([x0[0] + step, x0[1]], f([x0[0] + step, x0[1]])),
        ([x0[0], x0[1] + step], f([x0[0], x0[1] + step])),
    ];

    for _iter in 0..max_iter {
        // Sort by function value (ascending)
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (x_best, f_best) = simplex[0];
        let (x_worst, f_worst) = simplex[2];
        let (x_second, f_second) = simplex[1];

        // Check convergence: simplex size
        let size = ((x_worst[0] - x_best[0]).powi(2) + (x_worst[1] - x_best[1]).powi(2)).sqrt();
        if size < tol {
            return Ok((x_best[0], x_best[1]));
        }

        // Centroid of all points except worst
        let centroid = [
            (x_best[0] + x_second[0]) / 2.0,
            (x_best[1] + x_second[1]) / 2.0,
        ];

        // Reflection
        let x_r = [
            centroid[0] + alpha * (centroid[0] - x_worst[0]),
            centroid[1] + alpha * (centroid[1] - x_worst[1]),
        ];
        let f_r = f(x_r);

        if f_r < f_second && f_r >= f_best {
            // Accept reflection
            simplex[2] = (x_r, f_r);
            continue;
        }

        if f_r < f_best {
            // Try expansion
            let x_e = [
                centroid[0] + gamma * (x_r[0] - centroid[0]),
                centroid[1] + gamma * (x_r[1] - centroid[1]),
            ];
            let f_e = f(x_e);

            if f_e < f_r {
                simplex[2] = (x_e, f_e);
            } else {
                simplex[2] = (x_r, f_r);
            }
            continue;
        }

        // f_r >= f_second: try contraction
        let x_c = if f_r < f_worst {
            // Outside contraction
            [
                centroid[0] + rho * (x_r[0] - centroid[0]),
                centroid[1] + rho * (x_r[1] - centroid[1]),
            ]
        } else {
            // Inside contraction
            [
                centroid[0] + rho * (x_worst[0] - centroid[0]),
                centroid[1] + rho * (x_worst[1] - centroid[1]),
            ]
        };
        let f_c = f(x_c);

        if f_c < f_worst {
            simplex[2] = (x_c, f_c);
            continue;
        }

        // Shrink toward best point
        for i in 1..3 {
            let (x_i, _) = simplex[i];
            let x_new = [
                x_best[0] + sigma * (x_i[0] - x_best[0]),
                x_best[1] + sigma * (x_i[1] - x_best[1]),
            ];
            simplex[i] = (x_new, f(x_new));
        }
    }

    // Return best point after max iterations
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    Ok((simplex[0].0[0], simplex[0].0[1]))
}

// ============================================================================
// FDR Correction
// ============================================================================

/// Benjamini-Hochberg FDR correction
///
/// Python equivalent: `false_discovery_control(pvals, method="bh")`
pub fn fdr_correction(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len();
    if n == 0 {
        return vec![];
    }

    // Create indexed p-values for sorting
    let mut indexed_pvals: Vec<(usize, f64)> = pvals.iter().copied().enumerate().collect();

    // Sort by p-value (ascending)
    indexed_pvals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Calculate BH-adjusted p-values
    let mut adjusted = vec![0.0; n];
    let mut prev_adj = 1.0;

    for (rank, (idx, pval)) in indexed_pvals.iter().enumerate().rev() {
        let adj_pval = (pval * n as f64 / (rank + 1) as f64).min(prev_adj).min(1.0);
        adjusted[*idx] = adj_pval;
        prev_adj = adj_pval;
    }

    adjusted
}

fn valid_snv_allele(allele: &str) -> bool {
    allele.len() == 1
        && matches!(
            allele.as_bytes()[0].to_ascii_uppercase(),
            b'A' | b'C' | b'G' | b'T'
        )
}

fn cohort_prob_nll(
    prob: f64,
    indices: &[usize],
    observations: &[PreparedCohortSnvObservation],
    row_rho: &[f64],
) -> f64 {
    indices
        .iter()
        .map(|&index| {
            let observation = &observations[index];
            let n = observation.ref_count + observation.alt_count;
            opt_prob(prob, row_rho[index], observation.ref_count, n).unwrap_or(f64::INFINITY)
        })
        .sum()
}

/// Run an unphased cohort SNV analysis with one shared effect per exact SNV.
///
/// Dispersion is estimated from all eligible observations for `Single` and
/// `Linear`, or independently within each included donor for `PerDonor`.
/// Donor identity is retained throughout filtering and likelihood evaluation.
pub fn analyze_cohort_snvs(
    observations: Vec<CohortSnvObservation>,
    config: &CohortSnvConfig,
) -> Result<CohortSnvOutput> {
    if observations.is_empty() {
        anyhow::bail!("cohort SNV table contains no observations");
    }
    if config.min_donor_observations == 0 || config.min_informative_donors == 0 {
        anyhow::bail!("donor observation thresholds must be positive");
    }

    let n_raw_observations = observations.len();
    let mut seen_observations = HashSet::with_capacity(observations.len());
    let mut id_to_key: HashMap<String, SnvKey> = HashMap::new();
    let mut key_to_id: HashMap<SnvKey, String> = HashMap::new();
    let mut donor_counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut eligible = Vec::with_capacity(observations.len());

    for observation in observations {
        if observation.sample.is_empty()
            || observation.snv_id.is_empty()
            || observation.chrom.is_empty()
        {
            anyhow::bail!("sample, snv_id, and chromosome must be non-empty");
        }
        if observation.pos == 0 {
            anyhow::bail!("SNV positions must be one-based");
        }
        if !valid_snv_allele(&observation.ref_allele)
            || !valid_snv_allele(&observation.alt_allele)
            || observation
                .ref_allele
                .eq_ignore_ascii_case(&observation.alt_allele)
        {
            anyhow::bail!(
                "cohort SNV analysis requires distinct single-base A/C/G/T REF and ALT alleles"
            );
        }

        let key = SnvKey {
            chrom: observation.chrom,
            pos: observation.pos,
            ref_allele: observation.ref_allele.to_ascii_uppercase(),
            alt_allele: observation.alt_allele.to_ascii_uppercase(),
        };
        let observation_key = (
            observation.sample.clone(),
            key.chrom.clone(),
            key.pos,
            key.ref_allele.clone(),
            key.alt_allele.clone(),
        );
        if !seen_observations.insert(observation_key) {
            anyhow::bail!(
                "duplicate donor/SNV observation for {} {}:{} {}>{}",
                observation.sample,
                key.chrom,
                key.pos,
                key.ref_allele,
                key.alt_allele
            );
        }
        if let Some(existing) = id_to_key.get(&observation.snv_id) {
            if existing != &key {
                anyhow::bail!(
                    "snv_id maps to more than one exact SNV: {}",
                    observation.snv_id
                );
            }
        } else {
            id_to_key.insert(observation.snv_id.clone(), key.clone());
        }
        if let Some(existing) = key_to_id.get(&key) {
            if existing != &observation.snv_id {
                anyhow::bail!(
                    "exact SNV maps to conflicting snv_id values: {} and {}",
                    existing,
                    observation.snv_id
                );
            }
        } else {
            key_to_id.insert(key.clone(), observation.snv_id.clone());
        }

        let counts = donor_counts
            .entry(observation.sample.clone())
            .or_insert((0, 0));
        counts.0 += 1;
        let raw_n = observation
            .ref_count
            .checked_add(observation.alt_count)
            .context("allelic count depth overflow")?;
        if raw_n < config.min_count {
            continue;
        }
        counts.1 += 1;
        let ref_count = observation
            .ref_count
            .checked_add(config.pseudocount)
            .context("reference count plus pseudocount overflow")?;
        let alt_count = observation
            .alt_count
            .checked_add(config.pseudocount)
            .context("alternate count plus pseudocount overflow")?;
        eligible.push(PreparedCohortSnvObservation {
            sample: observation.sample,
            snv_id: observation.snv_id,
            key,
            ref_count,
            alt_count,
        });
    }

    let included_samples: BTreeSet<String> = donor_counts
        .iter()
        .filter(|(_, (_, eligible_count))| *eligible_count >= config.min_donor_observations)
        .map(|(sample, _)| sample.clone())
        .collect();
    if included_samples.is_empty() {
        anyhow::bail!("no donors meet the minimum eligible-observation threshold");
    }
    let donor_qc: Vec<DonorQc> = donor_counts
        .iter()
        .map(|(sample, (raw, eligible_count))| DonorQc {
            sample: sample.clone(),
            raw_observations: *raw,
            eligible_observations: *eligible_count,
            included: included_samples.contains(sample),
        })
        .collect();
    eligible.retain(|observation| included_samples.contains(&observation.sample));
    if eligible.is_empty() {
        anyhow::bail!("no eligible observations remain after donor filtering");
    }
    let n_included_observations = eligible.len();

    let ref_counts: Vec<u32> = eligible
        .iter()
        .map(|observation| observation.ref_count)
        .collect();
    let n_array: Vec<u32> = eligible
        .iter()
        .map(|observation| observation.ref_count + observation.alt_count)
        .collect();
    let mut donor_dispersion = Vec::new();
    let (row_rho, global_rho, linear_params) = match config.method {
        CohortSnvMethod::Single => {
            let rho = optimize_dispersion(&ref_counts, &n_array)?;
            (vec![rho; eligible.len()], Some(rho), None)
        }
        CohortSnvMethod::Linear => {
            let (d1, d2) = optimize_dispersion_linear(&ref_counts, &n_array)?;
            let row_rho = n_array
                .iter()
                .map(|&n| clamp_rho(expit((d1 + n as f64 * d2).clamp(-10.0, 10.0))))
                .collect();
            (row_rho, None, Some((d1, d2)))
        }
        CohortSnvMethod::PerDonor => {
            let mut donor_indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
            for (index, observation) in eligible.iter().enumerate() {
                donor_indices
                    .entry(&observation.sample)
                    .or_default()
                    .push(index);
            }
            let mut row_rho = vec![f64::NAN; eligible.len()];
            for (sample, indices) in donor_indices {
                let donor_ref: Vec<u32> = indices.iter().map(|&index| ref_counts[index]).collect();
                let donor_n: Vec<u32> = indices.iter().map(|&index| n_array[index]).collect();
                let rho = optimize_dispersion(&donor_ref, &donor_n)?;
                for &index in &indices {
                    row_rho[index] = rho;
                }
                donor_dispersion.push(DonorDispersion {
                    sample: sample.to_string(),
                    rho,
                    n_observations: indices.len(),
                });
            }
            (row_rho, None, None)
        }
    };

    let mut snv_indices: BTreeMap<SnvKey, Vec<usize>> = BTreeMap::new();
    for (index, observation) in eligible.iter().enumerate() {
        snv_indices
            .entry(observation.key.clone())
            .or_default()
            .push(index);
    }
    snv_indices.retain(|_, indices| indices.len() >= config.min_informative_donors);
    if snv_indices.is_empty() {
        anyhow::bail!("no SNVs meet the minimum informative-donor threshold");
    }

    let chi2 = ChiSquared::new(1.0).context("failed to create chi-squared distribution")?;
    let pseudocount = config.pseudocount as u64;
    let results: Result<Vec<CohortSnvResult>> = snv_indices
        .par_iter()
        .map(|(key, indices)| {
            let objective = |prob: f64| cohort_prob_nll(prob, indices, &eligible, &row_rho);
            let null_ll = -objective(0.5);
            let mu = golden_section_search(objective, 0.0, 1.0, 1e-6)?;
            let alt_ll = -objective(mu);
            if !null_ll.is_finite() || !alt_ll.is_finite() || !mu.is_finite() {
                anyhow::bail!("non-finite likelihood for {}:{}", key.chrom, key.pos);
            }
            let lrt = (2.0 * (alt_ll - null_ll)).max(0.0);
            let pval = 1.0 - chi2.cdf(lrt);
            let donor_count = indices.len();
            let ref_with_pc: u64 = indices
                .iter()
                .map(|&index| eligible[index].ref_count as u64)
                .sum();
            let alt_with_pc: u64 = indices
                .iter()
                .map(|&index| eligible[index].alt_count as u64)
                .sum();
            let total_pc = pseudocount * donor_count as u64;
            let ref_count = ref_with_pc.saturating_sub(total_pc);
            let alt_count = alt_with_pc.saturating_sub(total_pc);
            Ok(CohortSnvResult {
                snv_id: eligible[indices[0]].snv_id.clone(),
                chrom: key.chrom.clone(),
                pos: key.pos,
                ref_allele: key.ref_allele.clone(),
                alt_allele: key.alt_allele.clone(),
                ref_count,
                alt_count,
                n: ref_count + alt_count,
                donor_count,
                null_ll,
                alt_ll,
                mu,
                lrt,
                pval,
                fdr_pval: 0.0,
            })
        })
        .collect();
    let mut results = results?;
    results.sort_by(|left, right| {
        (&left.chrom, left.pos, &left.ref_allele, &left.alt_allele).cmp(&(
            &right.chrom,
            right.pos,
            &right.ref_allele,
            &right.alt_allele,
        ))
    });
    let adjusted = fdr_correction(&results.iter().map(|result| result.pval).collect::<Vec<_>>());
    for (result, fdr_pval) in results.iter_mut().zip(adjusted) {
        result.fdr_pval = fdr_pval;
    }

    Ok(CohortSnvOutput {
        results,
        donor_qc,
        donor_dispersion,
        global_rho,
        linear_params,
        n_raw_observations,
        n_included_observations,
    })
}

// ============================================================================
// Main Analysis Functions
// ============================================================================

/// Single dispersion model analysis
///
/// Python equivalent: `single_model()` in as_analysis.py
pub fn single_model(variants: Vec<VariantCounts>, phased: bool) -> Result<Vec<ImbalanceResult>> {
    if variants.is_empty() {
        return Ok(vec![]);
    }

    // Extract ref_counts and N for all variants
    let ref_counts: Vec<u32> = variants.iter().map(|v| v.ref_count).collect();
    let n_array: Vec<u32> = variants.iter().map(|v| v.ref_count + v.alt_count).collect();

    // Step 1: Optimize global dispersion parameter
    eprintln!("Optimizing dispersion parameter...");
    let disp = optimize_dispersion(&ref_counts, &n_array)?;
    eprintln!("  Dispersion: {:.6}", disp);

    // Step 2: Group by region
    let mut region_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, variant) in variants.iter().enumerate() {
        region_map
            .entry(variant.region.clone())
            .or_default()
            .push(i);
    }

    eprintln!(
        "Optimizing imbalance likelihood for {} regions...",
        region_map.len()
    );

    // Step 3: Calculate null and alternative likelihoods per region (parallel)
    // Clamp disp before calculating null_param (Issue #228)
    let disp = clamp_rho(disp);
    let null_param = 0.5 * (1.0 - disp) / disp;

    let results: Result<Vec<_>> = region_map
        .par_iter()
        .map(|(region, indices)| -> Result<ImbalanceResult> {
            // Extract counts for this region
            let region_ref: Vec<u32> = indices.iter().map(|&i| ref_counts[i]).collect();
            let region_n: Vec<u32> = indices.iter().map(|&i| n_array[i]).collect();

            // Null model: prob = 0.5 (no imbalance)
            let null_ll = betabinom_logpmf_sum(&region_ref, &region_n, null_param, null_param)?;

            // Alternative model: optimize prob (phase-aware routing)
            let all_gt = indices.iter().all(|&i| variants[i].gt.is_some());
            let (alt_ll, mu) = if phased && indices.len() > 1 && all_gt {
                // Phased multi-SNP region: use known genotype phase.
                let region_gt: Vec<u8> = indices
                    .iter()
                    .map(|&i| variants[i].gt.expect("gt present (checked by all_gt)"))
                    .collect();
                // single_model uses one global scalar dispersion; pass it as a
                // 1-element slice so the broadcast fallback keeps results identical.
                optimize_prob_phased(&region_ref, &region_n, &region_gt, &[disp])?
            } else if indices.len() > 1 {
                // Unphased multi-SNP region: marginalize over phase via DP,
                // minimizing opt_unphased_dp over prob in (0, 1)
                // (first = index 0, phase = remaining), matching Python's
                // parse_opt() unphased branch (minimize_scalar(opt_unphased_dp,...)).
                let disp_slice = [disp];
                let first_ref = region_ref[0];
                let first_n = region_n[0];
                let phase_ref = &region_ref[1..];
                let phase_n = &region_n[1..];
                let objective = |prob: f64| -> f64 {
                    match optimize_prob_unphased_dp(
                        prob,
                        &disp_slice,
                        first_ref,
                        first_n,
                        phase_ref,
                        phase_n,
                    ) {
                        Ok(nll) => nll,
                        Err(_) => f64::INFINITY,
                    }
                };
                let mu = golden_section_search(objective, 0.0, 1.0, 1e-6)?;
                let alt_ll = -objective(mu);
                (alt_ll, mu)
            } else {
                optimize_prob(&region_ref, &region_n, disp)?
            };

            // Likelihood ratio test
            let lrt = -2.0 * (null_ll - alt_ll);

            // P-value from chi-squared distribution (df=1)
            let chi2 = ChiSquared::new(1.0).context("Failed to create chi-squared distribution")?;
            let pval = 1.0 - chi2.cdf(lrt);

            // Sum counts for this region
            let total_ref: u32 = region_ref.iter().sum();
            let total_alt: u32 = indices.iter().map(|&i| variants[i].alt_count).sum();
            let total_n = total_ref + total_alt;

            Ok(ImbalanceResult {
                region: region.clone(),
                ref_count: total_ref,
                alt_count: total_alt,
                n: total_n,
                snp_count: indices.len(),
                null_ll,
                alt_ll,
                mu,
                lrt,
                pval,
                fdr_pval: 0.0, // Will be filled later
            })
        })
        .collect();

    let mut results = results?;

    // Sort by region name for deterministic output across runs
    results.sort_by(|a, b| a.region.cmp(&b.region));

    // Step 4: FDR correction
    let pvals: Vec<f64> = results.iter().map(|r| r.pval).collect();
    let fdr_pvals = fdr_correction(&pvals);

    for (result, fdr_pval) in results.iter_mut().zip(fdr_pvals.iter()) {
        result.fdr_pval = *fdr_pval;
    }

    Ok(results)
}

/// Linear dispersion model analysis
///
/// Python equivalent: `linear_model()` in as_analysis.py
/// Uses ρ = expit(d1 + N × d2) where d1, d2 are globally optimized parameters.
pub fn linear_model(variants: Vec<VariantCounts>, phased: bool) -> Result<Vec<ImbalanceResult>> {
    if variants.is_empty() {
        return Ok(vec![]);
    }

    // Extract ref_counts and N for all variants
    let ref_counts: Vec<u32> = variants.iter().map(|v| v.ref_count).collect();
    let n_array: Vec<u32> = variants.iter().map(|v| v.ref_count + v.alt_count).collect();

    // Step 1: Optimize linear dispersion parameters (d1, d2)
    eprintln!("Optimizing linear dispersion parameters...");
    let (d1, d2) = optimize_dispersion_linear(&ref_counts, &n_array)?;
    eprintln!("  d1 = {:.6}, d2 = {:.6}", d1, d2);

    // Step 2: Compute per-variant rho array
    let rho_array: Vec<f64> = n_array
        .iter()
        .map(|&n| {
            let exp_in = (d1 + n as f64 * d2).clamp(-10.0, 10.0);
            clamp_rho(expit(exp_in))
        })
        .collect();

    // Step 3: Group by region
    let mut region_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, variant) in variants.iter().enumerate() {
        region_map
            .entry(variant.region.clone())
            .or_default()
            .push(i);
    }

    eprintln!(
        "Optimizing imbalance likelihood for {} regions...",
        region_map.len()
    );

    // Step 4: Calculate null and alternative likelihoods per region (parallel)
    let results: Result<Vec<_>> = region_map
        .par_iter()
        .map(|(region, indices)| -> Result<ImbalanceResult> {
            // Extract counts and rho for this region
            let region_ref: Vec<u32> = indices.iter().map(|&i| ref_counts[i]).collect();
            let region_n: Vec<u32> = indices.iter().map(|&i| n_array[i]).collect();
            let region_rho: Vec<f64> = indices.iter().map(|&i| rho_array[i]).collect();

            // Null model: prob = 0.5, using per-variant rho
            let mut null_ll = 0.0;
            for ((&ref_count, &n), &rho) in region_ref
                .iter()
                .zip(region_n.iter())
                .zip(region_rho.iter())
            {
                let alpha = 0.5 * (1.0 - rho) / rho;
                let bb = rv::dist::BetaBinomial::new(n, alpha, alpha)
                    .context("Failed to create beta-binomial for null")?;
                use rv::traits::HasDensity;
                null_ll += bb.ln_f(&(ref_count as u64));
            }

            // Alternative model: optimize prob using the per-SNP rho slice
            // (region_rho), which is aligned 1:1 with region_ref/region_n in the
            // region's index order ([0] = first SNP, [1..] = subsequent SNPs).
            // This matches Python's per-row df["disp"] and intentionally differs
            // from single_model's single global scalar dispersion.
            let all_gt = indices.iter().all(|&i| variants[i].gt.is_some());
            let (alt_ll, mu) = if phased && indices.len() > 1 && all_gt {
                // Phased multi-SNP region: use known genotype phase.
                let region_gt: Vec<u8> = indices
                    .iter()
                    .map(|&i| variants[i].gt.expect("gt present (checked by all_gt)"))
                    .collect();
                optimize_prob_phased(&region_ref, &region_n, &region_gt, &region_rho)?
            } else if indices.len() > 1 {
                // Unphased multi-SNP region: marginalize over phase via DP,
                // minimizing opt_unphased_dp over prob in (0, 1)
                // (first = index 0, phase = remaining), matching Python's
                // parse_opt() unphased branch (minimize_scalar(opt_unphased_dp,...)).
                // region_rho is in the region's index order ([0] = first,
                // [1..] = phase), matching opt_unphased_dp's internal split.
                let first_ref = region_ref[0];
                let first_n = region_n[0];
                let phase_ref = &region_ref[1..];
                let phase_n = &region_n[1..];
                let objective = |prob: f64| -> f64 {
                    match optimize_prob_unphased_dp(
                        prob,
                        &region_rho,
                        first_ref,
                        first_n,
                        phase_ref,
                        phase_n,
                    ) {
                        Ok(nll) => nll,
                        Err(_) => f64::INFINITY,
                    }
                };
                let mu = golden_section_search(objective, 0.0, 1.0, 1e-6)?;
                let alt_ll = -objective(mu);
                (alt_ll, mu)
            } else {
                optimize_prob(&region_ref, &region_n, region_rho[0])?
            };

            // Likelihood ratio test (clamped at 0 to avoid tiny negative LRT from
            // optimizer noise; matches Python's chi2.sf behavior on near-zero LRT)
            let lrt = (-2.0 * (null_ll - alt_ll)).max(0.0);

            // P-value from chi-squared distribution (df=1)
            let chi2 = ChiSquared::new(1.0).context("Failed to create chi-squared distribution")?;
            let pval = 1.0 - chi2.cdf(lrt);

            // Sum counts for this region
            let total_ref: u32 = region_ref.iter().sum();
            let total_alt: u32 = indices.iter().map(|&i| variants[i].alt_count).sum();
            let total_n = total_ref + total_alt;

            Ok(ImbalanceResult {
                region: region.clone(),
                ref_count: total_ref,
                alt_count: total_alt,
                n: total_n,
                snp_count: indices.len(),
                null_ll,
                alt_ll,
                mu,
                lrt,
                pval,
                fdr_pval: 0.0,
            })
        })
        .collect();

    let mut results = results?;

    // Step 5: FDR correction
    let pvals: Vec<f64> = results.iter().map(|r| r.pval).collect();
    let fdr_pvals = fdr_correction(&pvals);

    for (result, fdr_pval) in results.iter_mut().zip(fdr_pvals.iter()) {
        result.fdr_pval = *fdr_pval;
    }

    Ok(results)
}

/// Fit per-donor dispersion (rho) via bounded MLE, one value per `sample`.
///
/// Python equivalent: `compute_per_donor_rho()` — each donor's rho is the
/// symmetric beta-binomial MLE over all that donor's observations, bounded to
/// (1e-6, 1-1e-6). No extra floor (matches locked donor_rho.tsv: min ~7e-6).
fn compute_donor_rho(variants: &[VariantCounts]) -> Result<HashMap<String, f64>> {
    // Group observation indices by donor
    let mut donor_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, v) in variants.iter().enumerate() {
        let donor = v.sample.clone().ok_or_else(|| {
            anyhow::anyhow!("per_donor model requires a 'sample' column in the counts TSV")
        })?;
        donor_map.entry(donor).or_default().push(i);
    }

    // Fit rho per donor in parallel (golden-section on symmetric beta-binomial)
    donor_map
        .par_iter()
        .map(|(donor, idx)| -> Result<(String, f64)> {
            let dref: Vec<u32> = idx.iter().map(|&i| variants[i].ref_count).collect();
            let dn: Vec<u32> = idx
                .iter()
                .map(|&i| variants[i].ref_count + variants[i].alt_count)
                .collect();
            let objective = |rho: f64| -> f64 {
                let rho = clamp_rho(rho);
                let a = 0.5 * (1.0 - rho) / rho;
                match betabinom_logpmf_sum(&dref, &dn, a, a) {
                    Ok(ll) => -ll,
                    Err(_) => f64::INFINITY,
                }
            };
            // Bounds (1e-6, 1-1e-6) match scipy minimize_scalar(method="bounded")
            let rho = golden_section_search(objective, 1e-6, 1.0 - 1e-6, 1e-6)?;
            Ok((donor.clone(), rho))
        })
        .collect()
}

/// Per-donor dispersion model analysis.
///
/// Python equivalent: WASP2 `per_donor` extension — fit rho per `sample`, then
/// run the standard per-region beta-binomial LRT using each observation's
/// donor-specific rho (per-observation rho in BOTH null and alternative).
pub fn per_donor_model(variants: Vec<VariantCounts>, phased: bool) -> Result<Vec<ImbalanceResult>> {
    if variants.is_empty() {
        return Ok(vec![]);
    }

    // Step 1: fit per-donor dispersion
    eprintln!("Fitting per-donor dispersion...");
    let donor_rho = compute_donor_rho(&variants)?;
    eprintln!("  Fit rho for {} donors", donor_rho.len());
    // Emit fitted rho (for validation against locked donor_rho.tsv)
    let mut donors_sorted: Vec<(&String, &f64)> = donor_rho.iter().collect();
    donors_sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (donor, rho) in donors_sorted {
        eprintln!("DONOR_RHO\t{}\t{:.8}", donor, rho);
    }

    // Step 2: per-observation rho = each obs's donor rho
    let ref_counts: Vec<u32> = variants.iter().map(|v| v.ref_count).collect();
    let n_array: Vec<u32> = variants.iter().map(|v| v.ref_count + v.alt_count).collect();
    let rho_array: Vec<f64> = variants
        .iter()
        .map(|v| {
            let donor = v
                .sample
                .as_ref()
                .expect("sample present (checked in compute_donor_rho)");
            clamp_rho(
                *donor_rho
                    .get(donor)
                    .expect("donor rho fit for every sample"),
            )
        })
        .collect();

    // Step 3: group by region
    let mut region_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, variant) in variants.iter().enumerate() {
        region_map
            .entry(variant.region.clone())
            .or_default()
            .push(i);
    }
    eprintln!(
        "Optimizing imbalance likelihood for {} regions...",
        region_map.len()
    );

    // Step 4: per-region null/alt LL + LRT, using per-observation rho
    let results: Result<Vec<_>> = region_map
        .par_iter()
        .map(|(region, indices)| -> Result<ImbalanceResult> {
            let region_ref: Vec<u32> = indices.iter().map(|&i| ref_counts[i]).collect();
            let region_n: Vec<u32> = indices.iter().map(|&i| n_array[i]).collect();
            let region_rho: Vec<f64> = indices.iter().map(|&i| rho_array[i]).collect();

            use rv::traits::HasDensity;

            // Null model: prob = 0.5, per-observation rho
            let mut null_ll = 0.0;
            for ((&ref_count, &n), &rho) in region_ref
                .iter()
                .zip(region_n.iter())
                .zip(region_rho.iter())
            {
                let alpha = 0.5 * (1.0 - rho) / rho;
                let bb = rv::dist::BetaBinomial::new(n, alpha, alpha)
                    .context("Failed to create beta-binomial for null")?;
                null_ll += bb.ln_f(&(ref_count as u64));
            }

            // Alternative model: optimize one pi, per-observation rho.
            // Phase-aware routing (mirrors single_model): for a phased multi-SNP
            // region with known GT, use the haplotype-phased likelihood, feeding
            // the per-observation (per-donor) rho slice into optimize_prob_phased's
            // &[f64] disp argument. Otherwise (unphased, or single-SNP, or missing
            // GT) use the unphased shared-pi product UNCHANGED — so phased=false is
            // byte-identical to the pre-phasing per_donor model.
            let all_gt = indices.iter().all(|&i| variants[i].gt.is_some());
            let (alt_ll, mu) = if phased && indices.len() > 1 && all_gt {
                let region_gt: Vec<u8> = indices
                    .iter()
                    .map(|&i| variants[i].gt.expect("gt present (checked by all_gt)"))
                    .collect();
                // Per-observation donor rho flows straight into the &[f64] slice.
                optimize_prob_phased(&region_ref, &region_n, &region_gt, &region_rho)?
            } else {
                // Unphased: optimize one pi over the per-observation beta-binomial
                // product (matches Python _alt_ll: a=pi(1-rho)/rho, b=(1-pi)(1-rho)/rho)
                let alt_obj = |pi: f64| -> f64 {
                    let pi = pi.clamp(1e-6, 1.0 - 1e-6);
                    let mut nll = 0.0;
                    for ((&ref_count, &n), &rho) in region_ref
                        .iter()
                        .zip(region_n.iter())
                        .zip(region_rho.iter())
                    {
                        let a = pi * (1.0 - rho) / rho;
                        let b = (1.0 - pi) * (1.0 - rho) / rho;
                        match rv::dist::BetaBinomial::new(n, a, b) {
                            Ok(bb) => nll -= bb.ln_f(&(ref_count as u64)),
                            Err(_) => return f64::INFINITY,
                        }
                    }
                    nll
                };
                let mu = golden_section_search(alt_obj, 1e-6, 1.0 - 1e-6, 1e-6)?;
                let alt_ll = -alt_obj(mu);
                (alt_ll, mu)
            };

            // Likelihood ratio test (clamped at 0 like Python)
            let lrt = (-2.0 * (null_ll - alt_ll)).max(0.0);
            let chi2 = ChiSquared::new(1.0).context("Failed to create chi-squared distribution")?;
            let pval = 1.0 - chi2.cdf(lrt);

            let total_ref: u32 = region_ref.iter().sum();
            let total_alt: u32 = indices.iter().map(|&i| variants[i].alt_count).sum();
            let total_n = total_ref + total_alt;

            Ok(ImbalanceResult {
                region: region.clone(),
                ref_count: total_ref,
                alt_count: total_alt,
                n: total_n,
                snp_count: indices.len(),
                null_ll,
                alt_ll,
                mu,
                lrt,
                pval,
                fdr_pval: 0.0,
            })
        })
        .collect();

    let mut results = results?;

    // Deterministic output order
    results.sort_by(|a, b| a.region.cmp(&b.region));

    // FDR correction (BH)
    let pvals: Vec<f64> = results.iter().map(|r| r.pval).collect();
    let fdr_pvals = fdr_correction(&pvals);
    for (result, fdr_pval) in results.iter_mut().zip(fdr_pvals.iter()) {
        result.fdr_pval = *fdr_pval;
    }

    Ok(results)
}

/// Main entry point for allelic imbalance analysis
///
/// Python equivalent: `get_imbalance()` in as_analysis.py
pub fn analyze_imbalance(
    variants: Vec<VariantCounts>,
    config: &AnalysisConfig,
) -> Result<Vec<ImbalanceResult>> {
    // Apply filters and pseudocounts
    let filtered: Vec<VariantCounts> = variants
        .into_iter()
        .map(|mut v| {
            v.ref_count += config.pseudocount;
            v.alt_count += config.pseudocount;
            v
        })
        .filter(|v| {
            let n = v.ref_count + v.alt_count;
            n >= config.min_count + (2 * config.pseudocount)
        })
        .collect();

    eprintln!("Processing {} variants after filtering", filtered.len());

    let phased = effective_phased(&filtered, config.phased);

    // Deduplicate variants for Python parity (single and linear models only).
    // Python's get_imbalance() does: df[keep_cols].drop_duplicates()
    // Per-donor model skips dedup as it's a Rust-only feature.
    let deduped = if config.method != AnalysisMethod::PerDonor {
        let before = filtered.len();
        let after = deduplicate_variants(filtered, phased);
        if after.len() < before {
            eprintln!(
                "Deduplicated {} → {} variants ({} duplicates removed)",
                before,
                after.len(),
                before - after.len()
            );
        }
        after
    } else {
        filtered
    };

    // Run analysis based on method
    let mut results = match config.method {
        AnalysisMethod::Single => single_model(deduped, phased)?,
        AnalysisMethod::Linear => linear_model(deduped, phased)?,
        AnalysisMethod::PerDonor => per_donor_model(deduped, phased)?,
    };

    // Remove pseudocounts from results
    for result in results.iter_mut() {
        if result.ref_count < config.pseudocount
            || result.alt_count < config.pseudocount
            || result.n < 2 * config.pseudocount
        {
            eprintln!(
                "[WARN] Counts smaller than pseudocount for region {}: ref={}, alt={}, n={}, pc={}",
                result.region, result.ref_count, result.alt_count, result.n, config.pseudocount
            );
        }
        result.ref_count = result.ref_count.saturating_sub(config.pseudocount);
        result.alt_count = result.alt_count.saturating_sub(config.pseudocount);
        result.n = result.n.saturating_sub(2 * config.pseudocount);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_prob() {
        // Test beta-binomial likelihood calculation
        let result = opt_prob(0.5, 0.1, 10, 20).unwrap();
        assert!(result.is_finite());
        assert!(result > 0.0); // Negative log-likelihood should be positive
    }

    #[test]
    fn test_opt_prob_rho_boundary_zero() {
        // Issue #228: rho=0 should not cause division by zero
        let result = opt_prob(0.5, 0.0, 10, 20).unwrap();
        assert!(
            result.is_finite(),
            "rho=0 should produce finite result after clamping"
        );
        assert!(!result.is_nan(), "rho=0 should not produce NaN");
    }

    #[test]
    fn test_opt_prob_rho_boundary_one() {
        // Issue #228: rho=1 should not produce zero alpha/beta
        let result = opt_prob(0.5, 1.0, 10, 20).unwrap();
        assert!(
            result.is_finite(),
            "rho=1 should produce finite result after clamping"
        );
        assert!(!result.is_nan(), "rho=1 should not produce NaN");
    }

    #[test]
    fn test_opt_prob_rho_near_boundaries() {
        // Test values very close to boundaries
        for rho in [1e-15, 1e-12, 1e-10, 0.999999999, 0.9999999999999] {
            let result = opt_prob(0.5, rho, 10, 20).unwrap();
            assert!(
                result.is_finite(),
                "rho={} should produce finite result",
                rho
            );
        }
    }

    #[test]
    fn test_clamp_rho() {
        // Test clamping function directly
        assert!((clamp_rho(0.0) - RHO_EPSILON).abs() < 1e-15);
        assert!((clamp_rho(1.0) - (1.0 - RHO_EPSILON)).abs() < 1e-15);
        assert!((clamp_rho(0.5) - 0.5).abs() < 1e-15);
        assert!((clamp_rho(-1.0) - RHO_EPSILON).abs() < 1e-15);
        assert!((clamp_rho(2.0) - (1.0 - RHO_EPSILON)).abs() < 1e-15);
    }

    #[test]
    fn test_fdr_correction() {
        let pvals = vec![0.01, 0.05, 0.1, 0.5];
        let fdr = fdr_correction(&pvals);

        // FDR-adjusted p-values should be >= original
        for (orig, adj) in pvals.iter().zip(fdr.iter()) {
            assert!(adj >= orig);
        }
    }

    #[test]
    fn test_golden_section() {
        // Test optimization on simple quadratic
        let f = |x: f64| (x - 0.7).powi(2);
        let min = golden_section_search(f, 0.0, 1.0, 1e-6).unwrap();
        assert!((min - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_scipy_bounded_minimize_preserves_near_zero_solution() {
        let minimum = scipy_bounded_minimize(|rho| rho, 0.0, 1.0, 1e-5, 500).unwrap();
        let scipy_boundary = 5.9608609865491405e-6;
        assert!((minimum - scipy_boundary).abs() < 1e-15);
        assert!(minimum < 0.001);
    }

    #[test]
    fn test_optimize_prob_unphased_dp_single_position() {
        // Test with only first position (no subsequent positions)
        let disp = vec![0.1];
        let result = optimize_prob_unphased_dp(0.5, &disp, 10, 20, &[], &[]).unwrap();
        // Should just return first_ll from opt_prob
        let expected = opt_prob(0.5, 0.1, 10, 20).unwrap();
        assert!(
            (result - expected).abs() < 1e-10,
            "Single position: result={}, expected={}",
            result,
            expected
        );
    }

    #[test]
    fn test_optimize_prob_unphased_dp_multiple_positions() {
        // Test with multiple positions
        let disp = vec![0.1];
        let phase_ref = vec![8, 12];
        let phase_n = vec![20, 25];
        let result = optimize_prob_unphased_dp(0.5, &disp, 10, 20, &phase_ref, &phase_n).unwrap();
        // Result should be finite and positive
        assert!(result.is_finite(), "Result should be finite");
        assert!(result > 0.0, "Negative log-likelihood should be positive");
    }

    #[test]
    fn test_optimize_prob_unphased_dp_per_variant_disp() {
        // Test with per-variant dispersion (linear model)
        let disp = vec![0.1, 0.15, 0.2]; // first pos: 0.1, subsequent: [0.15, 0.2]
        let phase_ref = vec![8, 12];
        let phase_n = vec![20, 25];
        let result = optimize_prob_unphased_dp(0.5, &disp, 10, 20, &phase_ref, &phase_n).unwrap();
        assert!(
            result.is_finite(),
            "Result should be finite with per-variant disp"
        );
        assert!(result > 0.0, "Negative log-likelihood should be positive");
    }

    #[test]
    fn test_optimize_prob_unphased_dp_asymmetric_prob() {
        // Test with asymmetric probability (imbalance)
        let disp = vec![0.1];
        let phase_ref = vec![15, 18]; // More ref alleles
        let phase_n = vec![20, 25];
        let result_balanced =
            optimize_prob_unphased_dp(0.5, &disp, 10, 20, &phase_ref, &phase_n).unwrap();
        let result_imbalanced =
            optimize_prob_unphased_dp(0.7, &disp, 14, 20, &phase_ref, &phase_n).unwrap();
        // Both should be finite
        assert!(result_balanced.is_finite());
        assert!(result_imbalanced.is_finite());
    }

    fn vc_peak(pos: u32, rc: u32, ac: u32, gt: Option<u8>) -> VariantCounts {
        VariantCounts {
            chrom: "chr1".to_string(),
            pos,
            ref_count: rc,
            alt_count: ac,
            region: "peak1".to_string(),
            sample: Some("d1".to_string()),
            gt,
        }
    }

    fn assert_result_close(a: &ImbalanceResult, b: &ImbalanceResult) {
        assert_eq!(a.region, b.region);
        assert_eq!(a.ref_count, b.ref_count);
        assert_eq!(a.alt_count, b.alt_count);
        assert_eq!(a.n, b.n);
        assert!((a.null_ll - b.null_ll).abs() < 1e-10);
        assert!((a.alt_ll - b.alt_ll).abs() < 1e-10);
        assert!((a.mu - b.mu).abs() < 1e-10);
        assert!((a.lrt - b.lrt).abs() < 1e-10);
        assert!((a.pval - b.pval).abs() < 1e-10);
    }

    #[test]
    fn test_effective_phased_requires_both_numeric_orientations() {
        let both = vec![vc_peak(100, 18, 4, Some(0)), vc_peak(200, 5, 15, Some(1))];
        assert!(effective_phased(&both, true));
        assert!(!effective_phased(&both, false));

        let none = vec![vc_peak(100, 18, 4, None), vc_peak(200, 5, 15, None)];
        assert!(!effective_phased(&none, true));

        let same = vec![vc_peak(100, 18, 4, Some(0)), vc_peak(200, 12, 8, Some(0))];
        assert!(!effective_phased(&same, true));

        let partial = vec![vc_peak(100, 18, 4, Some(0)), vc_peak(200, 12, 8, None)];
        assert!(!effective_phased(&partial, true));
    }

    #[test]
    fn test_analyze_imbalance_all_same_numeric_gt_falls_back_to_unphased() {
        let variants = vec![vc_peak(100, 18, 4, Some(0)), vc_peak(200, 12, 8, Some(0))];
        let phased_cfg = AnalysisConfig {
            min_count: 0,
            pseudocount: 0,
            method: AnalysisMethod::Single,
            phased: true,
        };
        let unphased_cfg = AnalysisConfig {
            phased: false,
            ..phased_cfg.clone()
        };

        let phased = analyze_imbalance(variants.clone(), &phased_cfg).unwrap();
        let unphased = analyze_imbalance(variants, &unphased_cfg).unwrap();
        assert_eq!(phased.len(), 1);
        assert_eq!(unphased.len(), 1);
        assert_result_close(&phased[0], &unphased[0]);
    }

    #[test]
    fn test_single_model_missing_gt_uses_unphased_multi_snp_path() {
        let variants = vec![vc_peak(100, 18, 4, None), vc_peak(200, 5, 15, None)];
        let phased = single_model(variants.clone(), true).unwrap();
        let unphased = single_model(variants, false).unwrap();
        assert_eq!(phased.len(), 1);
        assert_eq!(unphased.len(), 1);
        assert_result_close(&phased[0], &unphased[0]);
    }

    // Helper: a per_donor VariantCounts for one donor "d1".
    fn vc_d1(pos: u32, rc: u32, ac: u32, gt: u8) -> VariantCounts {
        VariantCounts {
            chrom: "chr1".to_string(),
            pos,
            ref_count: rc,
            alt_count: ac,
            region: "peak1".to_string(),
            sample: Some("d1".to_string()),
            gt: Some(gt),
        }
    }

    #[test]
    fn test_per_donor_phased_wiring() {
        // Phased per_donor on a single-donor multi-SNP region must equal a DIRECT
        // optimize_prob_phased call with that donor's fitted rho — proving
        // region_rho / region_gt are wired into the phased likelihood correctly.
        let variants = vec![
            vc_d1(100, 18, 4, 0),
            vc_d1(200, 5, 15, 1),
            vc_d1(300, 12, 8, 0),
        ];

        let rho_map = compute_donor_rho(&variants).unwrap();
        let rho = clamp_rho(rho_map["d1"]);
        let region_ref = vec![18u32, 5, 12];
        let region_n = vec![22u32, 20, 20];
        let region_gt = vec![0u8, 1, 0];
        let region_rho = vec![rho, rho, rho];
        let (exp_alt, exp_mu) =
            optimize_prob_phased(&region_ref, &region_n, &region_gt, &region_rho).unwrap();

        let res = per_donor_model(variants, true).unwrap();
        assert_eq!(res.len(), 1);
        assert!(
            (res[0].alt_ll - exp_alt).abs() < 1e-9,
            "phased per_donor alt_ll {} != direct optimize_prob_phased {}",
            res[0].alt_ll,
            exp_alt
        );
        assert!(
            (res[0].mu - exp_mu).abs() < 1e-9,
            "mu {} != {}",
            res[0].mu,
            exp_mu
        );
        assert!(res[0].lrt >= 0.0, "LRT must be clamped >= 0");
    }

    #[test]
    fn test_per_donor_unphased_ignores_gt() {
        // phased=false uses the shared-pi product, which does NOT use GT —
        // flipping GT must leave the unphased result unchanged.
        let a = vec![vc_d1(100, 18, 4, 0), vc_d1(200, 5, 15, 1)];
        let b = vec![vc_d1(100, 18, 4, 1), vc_d1(200, 5, 15, 0)]; // GT flipped
        let ra = per_donor_model(a, false).unwrap();
        let rb = per_donor_model(b, false).unwrap();
        assert!(
            (ra[0].lrt - rb[0].lrt).abs() < 1e-12,
            "unphased per_donor must ignore GT: {} vs {}",
            ra[0].lrt,
            rb[0].lrt
        );
    }

    fn cohort_observation(
        sample: &str,
        snv_id: &str,
        pos: u32,
        ref_count: u32,
        alt_count: u32,
    ) -> CohortSnvObservation {
        CohortSnvObservation {
            sample: sample.to_string(),
            snv_id: snv_id.to_string(),
            chrom: "chr1".to_string(),
            pos,
            ref_allele: "A".to_string(),
            alt_allele: "G".to_string(),
            ref_count,
            alt_count,
        }
    }

    fn cohort_fixture() -> Vec<CohortSnvObservation> {
        vec![
            cohort_observation("d1", "s1", 101, 18, 2),
            cohort_observation("d1", "s2", 202, 10, 10),
            cohort_observation("d2", "s1", 101, 16, 4),
            cohort_observation("d2", "s2", 202, 9, 11),
            cohort_observation("d3", "s1", 101, 19, 1),
            cohort_observation("d3", "s2", 202, 11, 9),
        ]
    }

    fn cohort_config(method: CohortSnvMethod) -> CohortSnvConfig {
        CohortSnvConfig {
            min_count: 10,
            pseudocount: 1,
            min_donor_observations: 2,
            min_informative_donors: 3,
            method,
        }
    }

    #[test]
    fn test_cohort_per_donor_uses_shared_effect_and_preserves_raw_counts() {
        let output =
            analyze_cohort_snvs(cohort_fixture(), &cohort_config(CohortSnvMethod::PerDonor))
                .unwrap();
        assert_eq!(output.results.len(), 2);
        assert_eq!(output.donor_dispersion.len(), 3);
        assert_eq!(output.n_included_observations, 6);
        let imbalanced = output
            .results
            .iter()
            .find(|result| result.snv_id == "s1")
            .unwrap();
        assert_eq!(imbalanced.donor_count, 3);
        assert_eq!(imbalanced.ref_count, 53);
        assert_eq!(imbalanced.alt_count, 7);
        assert!(imbalanced.mu > 0.75);
        assert!(imbalanced.alt_ll >= imbalanced.null_ll);
    }

    #[test]
    fn test_cohort_single_and_linear_return_nuisance_metadata() {
        let single =
            analyze_cohort_snvs(cohort_fixture(), &cohort_config(CohortSnvMethod::Single)).unwrap();
        assert!(single.global_rho.is_some());
        assert!(single.linear_params.is_none());
        assert!(single.donor_dispersion.is_empty());

        let linear =
            analyze_cohort_snvs(cohort_fixture(), &cohort_config(CohortSnvMethod::Linear)).unwrap();
        assert!(linear.global_rho.is_none());
        assert!(linear.linear_params.is_some());
        assert!(linear.donor_dispersion.is_empty());
    }

    #[test]
    fn test_cohort_rejects_duplicate_donor_snv_observation() {
        let mut observations = cohort_fixture();
        observations.push(observations[0].clone());
        let error = analyze_cohort_snvs(observations, &cohort_config(CohortSnvMethod::PerDonor))
            .unwrap_err();
        assert!(error.to_string().contains("duplicate donor/SNV"));
    }

    #[test]
    fn test_cohort_allele_swap_preserves_lrt_and_flips_mu() {
        let config = cohort_config(CohortSnvMethod::Single);
        let original = analyze_cohort_snvs(cohort_fixture(), &config).unwrap();
        let swapped_observations = cohort_fixture()
            .into_iter()
            .map(|mut observation| {
                std::mem::swap(&mut observation.ref_allele, &mut observation.alt_allele);
                std::mem::swap(&mut observation.ref_count, &mut observation.alt_count);
                observation.snv_id = format!("{}_swap", observation.snv_id);
                observation
            })
            .collect();
        let swapped = analyze_cohort_snvs(swapped_observations, &config).unwrap();
        for (left, right) in original.results.iter().zip(swapped.results.iter()) {
            assert!((left.lrt - right.lrt).abs() < 1e-8);
            assert!((left.pval - right.pval).abs() < 1e-8);
            assert!((left.mu + right.mu - 1.0).abs() < 1e-6);
        }
    }
}
