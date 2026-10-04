import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from typing import Any


class DataProfiler:
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        table_name: str = "dataset",
        alpha: float = 0.05,
    ) -> None:
        self.con = con
        self.table_name = table_name
        self.alpha = alpha

    def run_preflight_check(self) -> dict[str, Any]:
        """Generate a dataset profile and basic statistical diagnostics."""

        row_count = self.con.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()[0]

        schema_info = self.con.execute(
            f"PRAGMA table_info('{self.table_name}')"
        ).fetchall()

        columns_profile: dict[str, dict[str, Any]] = {}
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in schema_info:
            col_name = col[1]
            col_type = col[2].upper()

            series = self.con.execute(
                f'SELECT "{col_name}" FROM {self.table_name}'
            ).fetchdf()[col_name]

            null_count = int(series.isna().sum())
            null_percentage = round(
                (null_count / row_count) * 100, 2
            ) if row_count else 0.0

            clean_series = series.dropna()

            if (
                pd.api.types.is_numeric_dtype(clean_series)
                and clean_series.nunique() > 2
            ):
                numeric_cols.append(col_name)
                columns_profile[col_name] = self._analyze_numeric_column(
                    clean_series,
                    col_type,
                    null_count,
                    null_percentage,
                )
            else:
                categorical_cols.append(col_name)
                columns_profile[col_name] = self._analyze_categorical_column(
                    clean_series,
                    col_type,
                    null_count,
                    null_percentage,
                )

        variance_checks = self._check_variances(
            numeric_cols,
            categorical_cols,
        )
        advanced_diagnostics = self._advanced_diagnostics(
            numeric_cols,
            columns_profile,
            [
                col[1]
                for col in schema_info
                if "DATE" in col[2].upper()
                or "TIME" in col[2].upper()
                or any(token in col[1].lower() for token in ("date", "time", "timestamp"))
            ],
        )

        return {
            "summary": {
                "total_rows": int(row_count),
                "total_columns": len(schema_info),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
            },
            "column_diagnostics": columns_profile,
            "variance_homogeneity_checks": variance_checks,
            "advanced_diagnostics": advanced_diagnostics,
        }

    def _advanced_diagnostics(
        self,
        numeric_cols: list[str],
        columns_profile: dict[str, dict[str, Any]],
        time_cols: list[str],
    ) -> dict[str, Any]:
        """Add descriptive diagnostics while avoiding unsupported assumptions."""
        zero_inflation: list[dict[str, Any]] = []
        for name in numeric_cols:
            details = columns_profile.get(name, {})
            frame = self.con.execute(
                f'SELECT "{name}" FROM {self.table_name}'
            ).fetchdf()
            values = pd.to_numeric(frame[name], errors="coerce").dropna()
            if len(values) and (values >= 0).all() and np.allclose(values, np.round(values)):
                zero_count = int((values == 0).sum())
                zero_inflation.append({
                    "column": name,
                    "zero_count": zero_count,
                    "zero_percentage": round(zero_count / len(values) * 100, 2),
                    "note": "A high zero rate is a flag for review, not a zero-inflation test.",
                })

        vif_results: list[dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            frame = self.con.execute(
                "SELECT " + ", ".join(f'"{name}"' for name in numeric_cols)
                + f" FROM {self.table_name}"
            ).fetchdf()
            frame = frame.apply(pd.to_numeric, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            usable = [name for name in numeric_cols if frame[name].nunique() > 1]
            if len(usable) >= 2 and len(frame) > len(usable) + 1:
                corr = frame[usable].corr().to_numpy()
                inverse = np.linalg.pinv(corr)
                vif_results = [
                    {"column": name, "vif": round(float(inverse[i, i]), 4)}
                    for i, name in enumerate(usable)
                ]

        missingness = {
            name: {
                "missing_count": details.get("null_count", 0),
                "missing_percentage": details.get("null_percentage", 0.0),
            }
            for name, details in columns_profile.items()
            if details.get("null_count", 0)
        }
        autocorrelation: dict[str, Any] = {
            "status": "not_tested",
            "note": "No time/order column was identified; row order alone is not treated as time.",
            "results": [],
        }
        if time_cols:
            temporal_results = []
            time_col = time_cols[0]
            for name in numeric_cols:
                frame = self.con.execute(
                    f'SELECT "{time_col}", "{name}" FROM {self.table_name}'
                ).fetchdf()
                frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
                frame = frame.dropna().sort_values(time_col)
                values = frame[name].to_numpy(dtype=float)
                if len(values) < 4 or np.ptp(values) == 0:
                    continue
                positions = np.arange(len(values), dtype=float)
                residuals = values - np.polyval(np.polyfit(positions, values, 1), positions)
                denominator = float(np.dot(residuals, residuals))
                if denominator > 0:
                    dw = float(np.sum(np.diff(residuals) ** 2) / denominator)
                    temporal_results.append({
                        "column": name,
                        "time_column": time_col,
                        "durbin_watson_trend_residuals": round(dw, 4),
                        "observations": len(values),
                        "note": "Exploratory linear-trend residual diagnostic; model residuals may be more appropriate.",
                    })
            autocorrelation = {
                "status": "screened_with_time_column" if temporal_results else "not_tested_insufficient_usable_data",
                "results": temporal_results,
            }

        return {
            "multicollinearity_vif": {
                "results": vif_results,
                "interpretation": "Screening diagnostic; interpret VIF in model context.",
            },
            "integer_nonnegative_zero_rates": zero_inflation,
            "missingness_summary": missingness,
            "missingness_mechanism": {
                "status": "not_identifiable_from_observed_data_alone",
                "note": "MCAR/MAR/MNAR require study-design knowledge or additional assumptions; no mechanism is inferred.",
            },
            "autocorrelation": autocorrelation,
        }

    def _analyze_numeric_column(
        self,
        series: pd.Series,
        col_type: str,
        null_count: int,
        null_pct: float,
    ) -> dict[str, Any]:
        """Analyze a numeric variable."""

        # Remove infinite values before statistical tests.
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        n = len(series)

        if n == 0:
            return {
                "type": "numeric",
                "duckdb_type": col_type,
                "sample_size": 0,
                "null_count": null_count,
                "null_percentage": null_pct,
            }

        mean_val = float(series.mean())
        std_val = float(series.std())
        median_val = float(series.median())
        iqr_val = float(
            series.quantile(0.75) - series.quantile(0.25)
        )

        # Constant columns cannot be meaningfully tested.
        if series.nunique() > 1:
            skewness = float(stats.skew(series, bias=False))
        else:
            skewness = 0.0

        kurtosis = (
            float(stats.kurtosis(series, fisher=True, bias=False))
            if n >= 4 and series.nunique() > 1
            else 0.0
        )

        # Normality test.
        normality_test = {
            "test": None,
            "is_normal_distribution": None,
            "p_value": None,
        }

        if n >= 3 and series.nunique() > 1:
            if n <= 5000:
                result = stats.shapiro(series)
                test_name = "Shapiro-Wilk"
            elif n >= 8:
                result = stats.normaltest(series)
                test_name = "D'Agostino-Pearson"
            else:
                result = None
                test_name = None

            if result is not None:
                p_value = float(result.pvalue)

                normality_test = {
                    "test": test_name,
                    "is_normal_distribution": p_value > self.alpha,
                    "p_value": round(p_value, 5),
                }

        is_normal = normality_test["is_normal_distribution"]

        # These are recommendations, not automatic decisions.
        recommended_methods = {
            "location": (
                "Parametric methods may be appropriate"
                if is_normal
                else "Consider non-parametric or robust methods"
            ),
            "correlation": (
                "Pearson"
                if is_normal
                else "Spearman / Kendall"
            ),
        }

        return {
            "type": "numeric",
            "duckdb_type": col_type,
            "sample_size": n,
            "null_count": null_count,
            "null_percentage": null_pct,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "median": round(median_val, 4),
            "iqr": round(iqr_val, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "normality_test": normality_test,
            "recommended_methods": recommended_methods,
        }

    def _analyze_categorical_column(
        self,
        series: pd.Series,
        col_type: str,
        null_count: int,
        null_pct: float,
    ) -> dict[str, Any]:
        """Analyze a categorical variable."""

        value_counts = series.value_counts().head(5).to_dict()

        return {
            "type": "categorical",
            "duckdb_type": col_type,
            "unique_values": int(series.nunique()),
            "null_count": null_count,
            "null_percentage": null_pct,
            "top_categories": {
                str(k): int(v)
                for k, v in value_counts.items()
            },
        }

    def _check_variances(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> list[dict[str, Any]]:
        """Check homogeneity of variance using Levene's test."""

        results: list[dict[str, Any]] = []

        for cat_col in categorical_cols:
            for num_col in numeric_cols:

                df = self.con.execute(
                    f'SELECT "{cat_col}", "{num_col}" '
                    f'FROM {self.table_name}'
                ).fetchdf().dropna()

                groups = [
                    group[num_col].to_numpy()
                    for _, group in df.groupby(
                        cat_col,
                        observed=True
                    )
                    if len(group) >= 2
                ]

                if len(groups) < 2:
                    continue

                try:
                    result = stats.levene(
                        *groups,
                        center="median"
                    )

                    p_value = float(result.pvalue)

                    results.append({
                        "numeric_col": num_col,
                        "group_col": cat_col,
                        "equal_variance_assumed": p_value > self.alpha,
                        "levene_p_value": round(p_value, 5),
                    })

                except ValueError:
                    # Skip invalid group combinations.
                    continue

        return results
