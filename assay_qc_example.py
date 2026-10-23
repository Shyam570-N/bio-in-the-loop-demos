"""
Example QC script for in vitro pharmacology assay data.

Two tiers of checks, matching the "syntactic vs semantic" distinction:
  1. SYNTACTIC QC  - pure data hygiene. A script can do this with no biology knowledge.
  2. SEMANTIC QC   - domain rules a pharmacologist has to specify. The script can execute
                      them once defined, but it can't invent them.

Run: python3 assay_qc_example.py assay_results.csv
"""

import sys
import pandas as pd

def syntactic_qc(df):
    """Catches: bad data types, impossible values, missing required fields."""
    issues = []

    # 1. IC50 must be numeric
    bad_ic50 = df[pd.to_numeric(df["ic50_nM"], errors="coerce").isna()]
    for _, row in bad_ic50.iterrows():
        issues.append(f"[SYNTACTIC] Row {row['row_id']}: IC50 is not numeric ('{row['ic50_nM']}')")

    # 2. Percent inhibition (Emax) must be within a physically possible 0-100% range
    numeric_emax = pd.to_numeric(df["emax_pct"], errors="coerce")
    bad_emax = df[(numeric_emax < 0) | (numeric_emax > 100)]
    for _, row in bad_emax.iterrows():
        issues.append(f"[SYNTACTIC] Row {row['row_id']}: Emax {row['emax_pct']}% is outside the physically possible 0-100% range")

    # 3. Every row needs a control_status entry
    missing_control = df[df["control_status"].isna() | (df["control_status"].astype(str).str.strip() == "")]
    for _, row in missing_control.iterrows():
        issues.append(f"[SYNTACTIC] Row {row['row_id']}: missing control_status (can't tell if this well passed QC controls)")

    return issues


def semantic_qc(df):
    """
    Catches: biologically meaningful problems a script can only find if a pharmacologist
    has defined the rule in advance. These three rules are example judgment calls of the kind
    a pharmacologist would specify; the thresholds are illustrative and should be tuned per assay.
    """
    issues = []
    clean = df.copy()
    clean["ic50_nM"] = pd.to_numeric(clean["ic50_nM"], errors="coerce")
    clean["emax_pct"] = pd.to_numeric(clean["emax_pct"], errors="coerce")

    # RULE 1 - "Same IC50, different biology" (Emax/floor-ceiling check)
    # Flag same compound tested in >1 cell type where IC50s are close (<=2x) but
    # Emax differs by more than 20 percentage points.
    for compound, grp in clean.groupby("compound"):
        grp = grp.dropna(subset=["ic50_nM", "emax_pct"])
        rows = grp.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["cell_type"] == b["cell_type"]:
                    continue
                ratio = max(a["ic50_nM"], b["ic50_nM"]) / min(a["ic50_nM"], b["ic50_nM"])
                emax_gap = abs(a["emax_pct"] - b["emax_pct"])
                if ratio <= 2 and emax_gap > 20:
                    issues.append(
                        f"[SEMANTIC] {compound}: near-identical IC50 in {a['cell_type']} "
                        f"({a['ic50_nM']} nM, Emax {a['emax_pct']}%) vs {b['cell_type']} "
                        f"({b['ic50_nM']} nM, Emax {b['emax_pct']}%) - same potency, different "
                        f"maximal effect. Verify against assay floor/ceiling before comparing."
                    )

    # RULE 2 - "Potency vs. cytotoxicity confound"
    # Flag any sub-100nM IC50 in a T cell assay with no paired cytotoxicity entry
    # for the same compound + cell type.
    cytotox_pairs = set(
        clean[clean["assay_type"] == "cytotoxicity"][["compound", "cell_type"]]
        .apply(tuple, axis=1)
    )
    potent_hits = clean[(clean["assay_type"] == "inhibition") & (clean["cell_type"] == "T cell") & (clean["ic50_nM"] < 100)]
    for _, row in potent_hits.iterrows():
        if (row["compound"], row["cell_type"]) not in cytotox_pairs:
            issues.append(
                f"[SEMANTIC] {row['compound']} in T cells: potent IC50 ({row['ic50_nM']} nM) "
                f"has no paired cytotoxicity readout on file - can't rule out the signal is cell death, "
                f"not target inhibition."
            )

    # RULE 3 - "KO background comparability"
    # Flag IC50 comparisons for the same compound across different KO cell-line backgrounds.
    ko_rows = clean[clean["cell_line_background"].str.contains("KO", na=False)]
    for compound, grp in ko_rows.groupby("compound"):
        backgrounds = grp["cell_line_background"].unique()
        if len(backgrounds) > 1:
            issues.append(
                f"[SEMANTIC] {compound}: tested across multiple KO backgrounds ({', '.join(backgrounds)}) "
                f"- IC50s are not directly comparable without normalizing for differing steady-state "
                f"pathway kinetics between genetic backgrounds."
            )

    return issues


def main(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}\n")

    print("=== SYNTACTIC QC (a script can do this with zero biology) ===")
    for issue in syntactic_qc(df):
        print(issue)

    print("\n=== SEMANTIC QC (rules a pharmacologist had to define) ===")
    for issue in semantic_qc(df):
        print(issue)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assay_results.csv")
