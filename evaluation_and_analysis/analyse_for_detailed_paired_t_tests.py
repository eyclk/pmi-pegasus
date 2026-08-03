import os
import re


# ==========================================
# ================ SETTINGS ================
# ==========================================

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALL_paired_t_test_results")

# Subfolders that contain the paired t-test txt outputs.
# "xsum & cnn" files contain the results of two datasets inside a single file.
SUBFOLDERS = ["xsum & cnn", "wikihow"]

# Checkpoint order (pretraining steps), matching the "-<N>M pt-" part of the file names.
CHECKPOINTS = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]

# Metric order used in the printed tables.
METRICS = ["rouge1", "rouge2", "rougeL", "bert", "deberta_f1", "qaeval_f1"]

# Metrics that are parsed out of the txt files but deliberately left out of the tables.
IGNORED_METRICS = ["llama_f1", "qaeval_is_answered"]

# Nicer names for the table headers.
METRIC_DISPLAY_NAMES = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bert": "BERTScore",
    "deberta_f1": "DeBERTa-F1",
    "qaeval_f1": "QAEval-F1",
}

DATASET_ORDER = ["xsum", "cnn", "wikihow"]

SIGNIFICANCE_THRESHOLD = 0.05

# ---- Multiple-comparison correction settings ----

# How the "family" of tests is defined for the correction:
#   "per_dataset" --> each dataset is corrected on its own      (6 metrics x 8 checkpoints = 48 tests)
#   "global"      --> every test in the paper is one family     (3 datasets x 48          = 144 tests)
FAMILY_MODE = "per_dataset"

# Which correction is shown inside the main tables: "bh" (Benjamini-Hochberg FDR) or "holm".
PRIMARY_CORRECTION = "bh"


# ==========================================
# ============ PARSING FUNCTIONS ===========
# ==========================================

P_VALUE_PATTERN = re.compile(
    r"Paired t-test p-value for <(?P<dataset>\w+)> dataset for '(?P<metric>\w+)' metric "
    r"between both models:\s*(?P<p_value>[-+0-9.eE]+)"
)

DIRECTION_PATTERN = re.compile(
    r"The first model \(PMI\) mean is (?P<direction>HIGHER|LOWER) than the second model \(ROUGE\) "
    r"for <(?P<dataset>\w+)> dataset"
)

CHECKPOINT_PATTERN = re.compile(r"-(\d+M) pt-")


def get_checkpoint_from_filename(file_name):
    """Extracts e.g. '3M' out of 'Paired t-test - 13M dataset-3M pt-100K ft models.txt'."""
    match = CHECKPOINT_PATTERN.search(file_name)
    if match is None:
        return None
    return match.group(1)


def parse_single_file(file_path):
    """Returns a list of (dataset, metric, p_value, direction) tuples found in one txt file.

    Inside the txt files, every p-value line is followed a few lines later by the
    HIGHER/LOWER line belonging to the same test, so both are matched in file order
    and then zipped together."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    p_value_matches = list(P_VALUE_PATTERN.finditer(content))
    direction_matches = list(DIRECTION_PATTERN.finditer(content))

    if len(p_value_matches) != len(direction_matches):
        print(f"!!! WARNING: {os.path.basename(file_path)} has {len(p_value_matches)} p-values "
              f"but {len(direction_matches)} direction lines.")

    parsed_rows = []
    for i, p_match in enumerate(p_value_matches):
        direction = None
        if i < len(direction_matches):
            # Sanity check that the direction line indeed belongs to the same dataset.
            if direction_matches[i].group("dataset") == p_match.group("dataset"):
                direction = direction_matches[i].group("direction")
        parsed_rows.append((
            p_match.group("dataset"),
            p_match.group("metric"),
            float(p_match.group("p_value")),
            direction,
        ))
    return parsed_rows


def collect_all_p_values():
    """Walks over every subfolder/txt file and returns a flat list of test records:
    {dataset, metric, checkpoint, p_raw, direction}"""
    test_records = []
    seen_keys = set()

    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(RESULTS_ROOT, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"!!! WARNING: Missing subfolder --> {subfolder_path}")
            continue

        for file_name in sorted(os.listdir(subfolder_path)):
            if not file_name.lower().endswith(".txt"):
                continue

            checkpoint = get_checkpoint_from_filename(file_name)
            if checkpoint is None:
                print(f"!!! WARNING: Could not read a checkpoint name from --> {file_name}")
                continue

            file_path = os.path.join(subfolder_path, file_name)
            for dataset, metric, p_value, direction in parse_single_file(file_path):
                if metric in IGNORED_METRICS:
                    continue

                key = (dataset, metric, checkpoint)
                if key in seen_keys:
                    print(f"!!! WARNING: Duplicate entry for {key} in {file_name}.")
                seen_keys.add(key)

                test_records.append({
                    "dataset": dataset,
                    "metric": metric,
                    "checkpoint": checkpoint,
                    "p_raw": p_value,
                    "direction": direction,
                })

    dataset_names = sorted({r["dataset"] for r in test_records})
    print(f"\n===> Collected {len(test_records)} p-values in total "
          f"from {len(dataset_names)} datasets "
          f"(metrics ignored: {', '.join(IGNORED_METRICS) if IGNORED_METRICS else 'none'}).\n")
    return test_records


# ==========================================
# ===== MULTIPLE-COMPARISON CORRECTION =====
# ==========================================

def holm_bonferroni_adjust(p_values):
    """Holm-Bonferroni step-down adjustment. Controls the family-wise error rate (FWER):
    the probability of making even ONE false rejection inside the family.

    Sort the p-values ascending, multiply the i-th smallest by (m - i + 1), then take a
    running maximum so that the adjusted values stay monotonically increasing.
    A test is significant if its adjusted p-value is below the original alpha."""
    number_of_tests = len(p_values)
    ascending_order = sorted(range(number_of_tests), key=lambda i: p_values[i])

    adjusted = [0.0] * number_of_tests
    running_max = 0.0
    for rank, original_index in enumerate(ascending_order):
        value = (number_of_tests - rank) * p_values[original_index]
        running_max = max(running_max, value)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted


def benjamini_hochberg_adjust(p_values):
    """Benjamini-Hochberg adjustment (a.k.a. FDR-corrected p-values / q-values).
    Controls the false discovery rate: the expected PROPORTION of false positives among
    the tests that are called significant. Less strict than Holm, so it keeps more power.

    Sort the p-values ascending, multiply the i-th smallest by m / i, then take a running
    minimum from the largest p-value downwards to keep the adjusted values monotonic."""
    number_of_tests = len(p_values)
    ascending_order = sorted(range(number_of_tests), key=lambda i: p_values[i])

    adjusted = [0.0] * number_of_tests
    running_min = 1.0
    for rank in range(number_of_tests - 1, -1, -1):
        original_index = ascending_order[rank]
        value = p_values[original_index] * number_of_tests / (rank + 1)
        running_min = min(running_min, value)
        adjusted[original_index] = min(running_min, 1.0)
    return adjusted


def apply_corrections(test_records, family_mode):
    """Adds 'p_holm' and 'p_bh' to every record. The family is either each dataset on its
    own, or all of the tests together."""
    if family_mode == "per_dataset":
        family_key_function = lambda record: record["dataset"]
    elif family_mode == "global":
        family_key_function = lambda record: "ALL"
    else:
        raise ValueError(f"Unknown FAMILY_MODE: {family_mode}")

    families = {}
    for record in test_records:
        families.setdefault(family_key_function(record), []).append(record)

    for family_name, family_records in families.items():
        raw_p_values = [record["p_raw"] for record in family_records]
        holm_values = holm_bonferroni_adjust(raw_p_values)
        bh_values = benjamini_hochberg_adjust(raw_p_values)
        for record, holm_value, bh_value in zip(family_records, holm_values, bh_values):
            record["p_holm"] = holm_value
            record["p_bh"] = bh_value
            record["family_size"] = len(family_records)
            record["family_name"] = family_name

    return families


# ==========================================
# ============ PRINTING FUNCTIONS ==========
# ==========================================

def format_p_value(p_value):
    if p_value < 0.0001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def format_p_value_cell(record, p_value_field):
    """'0.0273 *(+)' --> significant, PMI higher.  '(-)' means PMI is lower."""
    if record is None:
        return "-- missing --"

    p_value = record[p_value_field]
    significance_marker = "*" if p_value < SIGNIFICANCE_THRESHOLD else " "

    if record["direction"] == "HIGHER":
        direction_marker = "(+)"
    elif record["direction"] == "LOWER":
        direction_marker = "(-)"
    else:
        direction_marker = "(?)"

    return f"{format_p_value(p_value)} {significance_marker}{direction_marker}"


def build_lookup(test_records):
    """(dataset, metric, checkpoint) --> record"""
    return {(r["dataset"], r["metric"], r["checkpoint"]): r for r in test_records}


def count_wins_and_losses(records, p_value_field):
    """Splits the significant results into WINS (PMI mean higher than ROUGE) and
    LOSSES (PMI mean lower). Non-significant tests are counted as neither."""
    wins = 0
    losses = 0
    for record in records:
        if record is None or record[p_value_field] >= SIGNIFICANCE_THRESHOLD:
            continue
        if record["direction"] == "HIGHER":
            wins += 1
        elif record["direction"] == "LOWER":
            losses += 1
    return wins, losses


def print_table_for_dataset(dataset_name, lookup, metrics_present, p_value_field, title_suffix):
    metric_column_width = 14
    cell_width = 18

    header_line = "Metric".ljust(metric_column_width) + "".join(
        f"{checkpoint} pt".rjust(cell_width) for checkpoint in CHECKPOINTS
    )
    separator_line = "-" * len(header_line)

    print("=" * len(header_line))
    print(f"<{dataset_name.upper()}>  --  {title_suffix}  (PMI-Pegasus vs. ROUGE-Pegasus)")
    print("=" * len(header_line))
    print(header_line)
    print(separator_line)

    total_cells = 0
    for metric in metrics_present:
        row_text = METRIC_DISPLAY_NAMES.get(metric, metric).ljust(metric_column_width)
        row_records = []
        for checkpoint in CHECKPOINTS:
            record = lookup.get((dataset_name, metric, checkpoint))
            row_text += format_p_value_cell(record, p_value_field).rjust(cell_width)
            row_records.append(record)
            total_cells += 1

        # Per-metric tally of significant wins / losses, appended at the end of each row.
        row_wins, row_losses = count_wins_and_losses(row_records, p_value_field)
        row_text += f"   |  W:{row_wins}  L:{row_losses}"
        print(row_text)

    dataset_records = [lookup.get((dataset_name, metric, checkpoint))
                       for metric in metrics_present for checkpoint in CHECKPOINTS]
    wins, losses = count_wins_and_losses(dataset_records, p_value_field)

    print(separator_line)
    print(f"* = p < {SIGNIFICANCE_THRESHOLD}   |   (+) = PMI mean HIGHER,   (-) = PMI mean LOWER")
    print(f"Significant for <{dataset_name}>: {wins + losses} / {total_cells}   "
          f"-->   significant WINS (PMI higher): {wins}   |   "
          f"significant LOSSES (PMI lower): {losses}   |   "
          f"not significant: {total_cells - wins - losses}\n\n")


def print_lost_significance_report(test_records):
    """Lists the tests that were significant before the correction but are not anymore.
    These are the ones a reviewer will look at, so they are worth checking one by one."""
    lost_under_bh = []
    lost_under_holm = []

    for record in test_records:
        was_significant = record["p_raw"] < SIGNIFICANCE_THRESHOLD
        if not was_significant:
            continue
        if record["p_bh"] >= SIGNIFICANCE_THRESHOLD:
            lost_under_bh.append(record)
        if record["p_holm"] >= SIGNIFICANCE_THRESHOLD:
            lost_under_holm.append(record)

    print("=" * 110)
    print("TESTS THAT LOSE SIGNIFICANCE AFTER CORRECTION")
    print("=" * 110)

    # Only Benjamini-Hochberg is reported. The Holm-Bonferroni line below is kept for
    # reference (its values are still computed), but it is not printed anymore.
    for correction_name, lost_records in [("Benjamini-Hochberg (FDR)", lost_under_bh),
                                          # ("Holm-Bonferroni (FWER)", lost_under_holm),
                                          ]:
        print(f"\n-- {correction_name} --")
        if not lost_records:
            print("   (none -- every originally significant result survives)")
            continue
        for record in sorted(lost_records, key=lambda r: (r["dataset"], r["metric"], r["checkpoint"])):
            metric_name = METRIC_DISPLAY_NAMES.get(record["metric"], record["metric"])
            print(f"   {record['dataset']:<8} {metric_name:<12} {record['checkpoint']:<4} "
                  f"raw p = {format_p_value(record['p_raw']):<10} "
                  f"BH = {format_p_value(record['p_bh'])}"
                  # f"   Holm = {format_p_value(record['p_holm'])}"
                  )
    print()


def print_overall_summary(test_records, families):
    print("=" * 110)
    print("SUMMARY OF THE MULTIPLE-COMPARISON CORRECTION")
    print("=" * 110)
    print(f"Family definition: FAMILY_MODE = '{FAMILY_MODE}'")
    for family_name in sorted(families):
        print(f"   family <{family_name}>: {len(families[family_name])} tests")
    print()

    print("Cells show:  total significant  (W = PMI is HIGHER / L = PMI is LOWER)\n")
    header = (f"{'Dataset':<10}{'tests':>7}"
              f"{'raw p<.05':>22}{'BH q<.05':>22}"
              # f"{'Holm p<.05':>22}"
              )
    print(header)
    print("-" * len(header))

    datasets_present = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in test_records)]
    datasets_present += sorted({r["dataset"] for r in test_records} - set(DATASET_ORDER))

    for dataset_name in datasets_present + ["TOTAL"]:
        if dataset_name == "TOTAL":
            subset = test_records
            print("-" * len(header))
        else:
            subset = [r for r in test_records if r["dataset"] == dataset_name]

        row_text = f"{dataset_name:<10}{len(subset):>7}"
        for p_value_field in ["p_raw", "p_bh"]:  # "p_holm" is computed but not reported

            wins, losses = count_wins_and_losses(subset, p_value_field)
            row_text += f"{wins + losses:>3}  (W:{wins:>2} / L:{losses:>2})".rjust(22)
        print(row_text)
    print()


def main():
    test_records = collect_all_p_values()
    families = apply_corrections(test_records, FAMILY_MODE)
    lookup = build_lookup(test_records)

    datasets_present = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in test_records)]
    datasets_present += sorted({r["dataset"] for r in test_records} - set(DATASET_ORDER))

    metrics_present = [m for m in METRICS if any(r["metric"] == m for r in test_records)]
    metrics_present += sorted({r["metric"] for r in test_records} - set(METRICS))

    p_value_field = "p_bh" if PRIMARY_CORRECTION == "bh" else "p_holm"
    correction_title = ("BENJAMINI-HOCHBERG FDR-CORRECTED p-values (q-values)"
                        if PRIMARY_CORRECTION == "bh"
                        else "HOLM-BONFERRONI CORRECTED p-values")

    print("\n\n##########  RAW (UNCORRECTED) P-VALUES  ##########\n")
    for dataset_name in datasets_present:
        print_table_for_dataset(dataset_name, lookup, metrics_present,
                                "p_raw", "RAW paired t-test p-values")

    print("\n\n##########  AFTER MULTIPLE-COMPARISON CORRECTION  ##########\n")
    for dataset_name in datasets_present:
        print_table_for_dataset(dataset_name, lookup, metrics_present,
                                p_value_field, correction_title)

    print_lost_significance_report(test_records)
    print_overall_summary(test_records, families)


if __name__ == "__main__":
    main()
