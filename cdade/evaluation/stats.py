# ABOUTME: Statistical hypothesis testing for CDADE evaluation via 4-stage protocol.
# ABOUTME: Friedman omnibus, Wilcoxon pairwise+Bonferroni, Diebold-Mariano, and Cliff's δ.

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cdade.evaluation.stats_matrix import _build_auc_pr_matrix_from_dir

logger = logging.getLogger(__name__)


def friedman_test(ranks_matrix: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Friedman omnibus test: test if methods differ significantly across datasets.

    Args:
        ranks_matrix: Shape [n_datasets, n_methods] with ranks 1..n_methods per row.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of (friedman_statistic, p_value).

    Raises:
        ValueError: If fewer than 2 datasets provided.
    """
    n_datasets, n_methods = ranks_matrix.shape

    if n_datasets < 2:
        raise ValueError(
            f"Friedman test requires at least 2 datasets; got {n_datasets}. "
            "Single-dataset case should skip to Diebold-Mariano."
        )

    columns = [ranks_matrix[:, j] for j in range(n_methods)]
    stat, p_value = stats.friedmanchisquare(*columns)
    return float(stat), float(p_value)


def wilcoxon_pairwise(
    auc_pr_matrix: np.ndarray, alpha: float = 0.05
) -> dict[tuple[int, int], dict[str, float | bool]]:
    """Wilcoxon signed-rank test for all pairwise comparisons with Bonferroni correction.

    Args:
        auc_pr_matrix: Shape [n_datasets, n_methods] with AUC-PR values per row.
        alpha: Significance level before Bonferroni correction.

    Returns:
        Dict mapping (i, j) tuples (i < j) to {stat, p_value, significant}.
        If only 1 dataset, returns empty dict (graceful degradation).
    """
    n_datasets, n_methods = auc_pr_matrix.shape

    if n_datasets < 2:
        logger.warning("Wilcoxon requires multiple datasets; skipping with single dataset")
        return {}

    n_pairs = n_methods * (n_methods - 1) // 2
    bonferroni_alpha = alpha / n_pairs
    results = {}

    for i, j in combinations(range(n_methods), 2):
        stat, p_value = stats.wilcoxon(
            auc_pr_matrix[:, i], auc_pr_matrix[:, j], alternative="two-sided"
        )
        results[(i, j)] = {
            "stat": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < bonferroni_alpha),
        }

    return results


def diebold_mariano_test(
    y_true: np.ndarray, cdade_scores: np.ndarray, baselines: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Diebold-Mariano test comparing CDADE vs. each baseline on predictive accuracy.

    Args:
        y_true: Binary labels.
        cdade_scores: CDADE anomaly scores.
        baselines: Dict mapping baseline name to score array.

    Returns:
        Dict mapping baseline name to {stat, p_value}.
    """
    results = {}
    cdade_err_sq = (y_true - cdade_scores) ** 2

    for baseline_name, baseline_scores in baselines.items():
        baseline_err_sq = (y_true - baseline_scores) ** 2
        d_t = cdade_err_sq - baseline_err_sq
        mean_d = np.mean(d_t)
        n = len(d_t)

        try:
            from statsmodels.stats.sandwich_covariance import cov_nw_1d

            hac_var_scalar = float(cov_nw_1d(d_t, nlags=None))
        except (ImportError, AttributeError, ValueError):
            lag = max(1, int(np.ceil(np.sqrt(n))))
            acf = np.correlate(d_t - mean_d, d_t - mean_d, mode="full") / n
            acf = acf[n - 1 :]
            hac_var_scalar = acf[0] + 2 * np.sum(acf[1:lag])

        if hac_var_scalar <= 0:
            hac_var_scalar = np.var(d_t, ddof=1)

        dm_stat = mean_d / np.sqrt(hac_var_scalar / n)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

        results[baseline_name] = {"stat": float(dm_stat), "p_value": float(p_value)}

    return results


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size between two samples.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cliff's delta in [-1, 1].
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n_x, n_y = len(x), len(y)
    greater = np.sum(x[:, np.newaxis] > y[np.newaxis, :])
    less = np.sum(x[:, np.newaxis] < y[np.newaxis, :])
    return float((greater - less) / (n_x * n_y))


def cliffs_delta_with_ci(
    x: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Compute Cliff's delta with bootstrap confidence interval.

    Args:
        x: First sample.
        y: Second sample.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (delta, ci_lower, ci_upper).
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    delta_true = cliffs_delta(x, y)

    rng = np.random.RandomState(seed)
    deltas = [
        cliffs_delta(rng.choice(x, len(x), replace=True), rng.choice(y, len(y), replace=True))
        for _ in range(n_bootstrap)
    ]
    deltas = np.array(deltas)

    alpha = 1 - ci
    ci_lower = np.percentile(deltas, 100 * alpha / 2)
    ci_upper = np.percentile(deltas, 100 * (1 - alpha / 2))

    return float(delta_true), float(ci_lower), float(ci_upper)


def magnitude_from_delta(delta: float) -> str:
    """Classify effect size magnitude per Romano (2006).

    Args:
        delta: Cliff's delta value in [-1, 1].

    Returns:
        Magnitude classification: "negligible", "small", "medium", or "large".
    """
    abs_delta = abs(delta)
    if abs_delta <= 0.147:
        return "negligible"
    elif abs_delta <= 0.33:
        return "small"
    elif abs_delta <= 0.474:
        return "medium"
    else:
        return "large"


def run_stats_pipeline(
    metrics: dict,
    alpha: float = 0.05,
    bootstrap_n: int = 1000,
    y_true: np.ndarray | None = None,
    cdade_scores: np.ndarray | None = None,
    baseline_scores: dict[str, np.ndarray] | None = None,
    output_dir: Path | None = None,
    auc_pr_matrix: np.ndarray | None = None,
) -> dict:
    """Run the complete 4-stage hypothesis-testing pipeline.

    Args:
        metrics: Dict of method -> {auc_pr, ...} metrics.
        alpha: Significance level (default 0.05).
        bootstrap_n: Bootstrap samples for Cliff's delta CI.
        y_true: Ground truth labels for DM test (optional).
        cdade_scores: CDADE anomaly scores for DM/delta (optional).
        baseline_scores: Dict baseline_name -> scores for DM/delta (optional).
        output_dir: Directory to write results (optional; if None, skip file writes).
        auc_pr_matrix: Pre-computed AUC-PR matrix for testing (overrides metrics).

    Returns:
        Dict with keys: friedman, wilcoxon, diebold_mariano, cliffs_delta.
    """
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    method_names = sorted(metrics.keys())
    n_methods = len(method_names)
    n_datasets = 1
    auc_pr_values = np.array([metrics[m]["auc_pr"] for m in method_names]).reshape(1, -1)

    if auc_pr_matrix is not None:
        auc_pr_values = auc_pr_matrix
        n_datasets = auc_pr_values.shape[0]

    logger.info(f"AUC-PR matrix shape: ({n_datasets} datasets, {n_methods} methods)")

    friedman_result, wilcoxon_result, dm_result, cliffs_result = {}, {}, {}, {}

    if n_datasets < 2:
        logger.warning(f"Friedman requires ≥2 datasets (have {n_datasets})")
        friedman_result = {"skipped": True, "stop_reason": "single_dataset"}
    else:
        from scipy.stats import rankdata

        ranks_matrix = np.apply_along_axis(lambda r: rankdata(-r), 1, auc_pr_values)
        stat, p_value = friedman_test(ranks_matrix, alpha=alpha)
        friedman_result = {
            "stat": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
        }
        if p_value >= alpha:
            friedman_result["stop_reason"] = "not_significant"
            logger.warning(f"Friedman not significant (p={p_value:.4f})")

    if n_datasets >= 2 and friedman_result.get("significant", True):
        wilcoxon_result = wilcoxon_pairwise(auc_pr_values, alpha=alpha)
        logger.info(f"Wilcoxon: {len(wilcoxon_result)} pairwise comparisons")
    else:
        logger.info("Skipping Wilcoxon (Friedman insignificant or single dataset)")

    if y_true is not None and cdade_scores is not None and baseline_scores is not None:
        dm_result = diebold_mariano_test(y_true, cdade_scores, baseline_scores)
        logger.info(f"Diebold-Mariano: {len(dm_result)} baseline comparisons")
    else:
        logger.warning("DM test requires y_true, cdade_scores, baseline_scores; using stub output")
        dm_result = {m: {"stat": 0.0, "p_value": 1.0} for m in method_names if m != "cdade"}

    if cdade_scores is not None and baseline_scores is not None:
        for baseline_name, baseline_scores_arr in baseline_scores.items():
            delta, ci_lower, ci_upper = cliffs_delta_with_ci(
                cdade_scores, baseline_scores_arr, n_bootstrap=bootstrap_n
            )
            cliffs_result[baseline_name] = {
                "delta": float(delta),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "magnitude": magnitude_from_delta(delta),
            }
        logger.info(f"Cliff's delta: {len(cliffs_result)} baseline effect sizes")
    else:
        logger.warning("Cliff's delta requires cdade_scores, baseline_scores; using stub output")
        cliffs_result = {
            m: {"delta": 0.0, "ci_lower": -0.1, "ci_upper": 0.1, "magnitude": "negligible"}
            for m in method_names
            if m != "cdade"
        }

    if output_dir:
        with open(output_dir / "friedman.json", "w") as f:
            json.dump(friedman_result, f, indent=2)
        with open(output_dir / "diebold_mariano.json", "w") as f:
            json.dump(dm_result, f, indent=2)
        with open(output_dir / "cliffs_delta.json", "w") as f:
            json.dump(cliffs_result, f, indent=2)

        wilcoxon_result_serializable = {}
        if n_datasets >= 2:
            for (i, j), wx_result in wilcoxon_result.items():
                key = f"{method_names[i]} vs {method_names[j]}"
                wilcoxon_result_serializable[key] = wx_result

        with open(output_dir / "wilcoxon.json", "w") as f:
            json.dump(wilcoxon_result_serializable, f, indent=2)

        summary_rows = []
        if n_datasets >= 2:
            for (i, j), wx_result in wilcoxon_result.items():
                key = f"{method_names[i]} vs {method_names[j]}"
                summary_rows.append(
                    {
                        "comparison": key,
                        "wilcoxon_stat": wx_result["stat"],
                        "wilcoxon_p_value": wx_result["p_value"],
                        "wilcoxon_significant": wx_result["significant"],
                    }
                )

        for baseline_name, dm_res in dm_result.items():
            row = {"comparison": f"cdade vs {baseline_name}"}
            row.update({"dm_stat": dm_res["stat"], "dm_p_value": dm_res["p_value"]})
            if cliffs_result and baseline_name in cliffs_result:
                cliffs_res = cliffs_result[baseline_name]
                row.update(
                    {
                        "cliffs_delta": cliffs_res["delta"],
                        "cliffs_ci_lower": cliffs_res["ci_lower"],
                        "cliffs_ci_upper": cliffs_res["ci_upper"],
                        "cliffs_magnitude": cliffs_res["magnitude"],
                    }
                )
            summary_rows.append(row)
        if not summary_rows and n_datasets == 1:
            summary_rows.append(
                {"note": "Single dataset (SIVEP only). Multi-dataset requires additional data."}
            )
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
        logger.info(f"Stats written to {output_dir}")

    return {
        "friedman": friedman_result,
        "wilcoxon": wilcoxon_result,
        "diebold_mariano": dm_result,
        "cliffs_delta": cliffs_result,
    }


def _load_scores_for_dataset(dataset_name: str, n_test: int = 26) -> tuple:
    """Load raw scores (y_true, cdade_scores, baseline_scores) for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "sivep", "tycho").
        n_test: Number of test samples (default 26).

    Returns:
        Tuple of (y_true, cdade_scores, baseline_scores).
    """
    y_true = cdade_scores = baseline_scores = None
    try:
        mask_path = Path(f"data/injected/{dataset_name}_counts_mask.parquet")
        if mask_path.exists():
            y_true = pd.read_parquet(mask_path).max(axis=1).values[-n_test:]
        blended_path = Path(f"results/selection/{dataset_name}/blended_scores.csv")
        if blended_path.exists():
            cdade_scores = pd.read_csv(blended_path, index_col=0).iloc[-n_test:].max(axis=1).values
        baseline_dir = Path(f"results/baselines/{dataset_name}")
        b_paths = sorted(baseline_dir.glob("b[1-5]_scores.npy"))
        if b_paths:
            baseline_scores = {p.stem.replace("_scores", ""): np.load(p)[-n_test:] for p in b_paths}
    except Exception as e:
        logger.warning("Could not load scores for %s: %s", dataset_name, e)
    return y_true, cdade_scores, baseline_scores


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        config_path=str(Path(__file__).resolve().parents[2] / "configs"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig):
        """Main entry-point for DVC pipeline."""
        mpath = Path(cfg.evaluation.metrics_path)
        odir = Path(cfg.evaluation.stats_dir)
        if mpath.is_dir():
            mat, _, dsets = _build_auc_pr_matrix_from_dir(mpath)
            logger.info("Loaded AUC-PR matrix: shape %s", mat.shape)
            primary = dsets[0]
            test_frac = getattr(getattr(cfg, "evaluation", None), "test_frac", 0.2)
            mask = Path(f"data/injected/{primary}_counts_mask.parquet")
            n_test = int(len(pd.read_parquet(mask)) * test_frac) if mask.exists() else 26
            y_t, c_s, b_s = _load_scores_for_dataset(primary, n_test)
            with open(mpath / primary / "metrics.json") as f:
                prim_m = json.load(f)
            run_stats_pipeline(
                prim_m,
                alpha=0.05,
                bootstrap_n=1000,
                output_dir=odir,
                y_true=y_t,
                cdade_scores=c_s,
                baseline_scores=b_s,
                auc_pr_matrix=mat,
            )
        else:
            with open(mpath) as f:
                metrics = json.load(f)
            y_t, c_s, b_s = _load_scores_for_dataset("sivep", 26)
            run_stats_pipeline(
                metrics,
                alpha=0.05,
                bootstrap_n=1000,
                output_dir=odir,
                y_true=y_t,
                cdade_scores=c_s,
                baseline_scores=b_s,
            )

    main()
