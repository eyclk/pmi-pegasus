from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap


# ==========================================
# ======== STATIC RESULTS INPUT AREA =======
# ==========================================

pmi_peg__xsum__rouge1 = [0.3396, 0.3531, 0.3642, 0.3631, 0.3569, 0.3584, 0.3584, 0.3552]

rouge_peg__xsum__rouge1 = [0.3370, 0.3650, 0.3650, 0.3648, 0.3628, 0.3600, 0.3582, 0.3552]

pmi_peg__xsum__rouge2 = [0.1196, 0.1293, 0.1358, 0.1342, 0.1303, 0.1297, 0.1304, 0.1282]

rouge_peg__xsum__rouge2 = [0.1192, 0.1357, 0.1387, 0.1361, 0.1337, 0.1321, 0.1305, 0.1298]

pmi_peg__xsum__rougeL = [0.2607, 0.2743, 0.2867, 0.2851, 0.2805, 0.2806, 0.2808, 0.2778]

rouge_peg__xsum__rougeL = [0.2597, 0.2859, 0.2915, 0.2880, 0.2858, 0.2824, 0.2820, 0.2811]

pmi_peg__xsum__roberta_score = [0.3422, 0.3583, 0.3758, 0.3703, 0.3565, 0.3565, 0.3571, 0.3514]

rouge_peg__xsum__roberta_score = [0.3464, 0.3808, 0.3840, 0.3782, 0.3677, 0.3721, 0.3614, 0.3557]

pmi_peg__xsum__deberta_score = [0.3728, 0.3867, 0.3980, 0.3934, 0.3805, 0.3802, 0.3806, 0.3753]

rouge_peg__xsum__deberta_score = [0.3804, 0.4104, 0.4142, 0.4057, 0.3941, 0.3989, 0.3879, 0.3796]

pmi_peg__xsum__qaeval_f1 = [0.1039, 0.1101, 0.1195, 0.1191, 0.1159, 0.1175, 0.1155, 0.1156]

rouge_peg__xsum__qaeval_f1 = [0.1061, 0.1256, 0.1304, 0.1282, 0.1274, 0.1262, 0.1243, 0.1236]



pmi_peg__cnn__rouge1 = [0.4047, 0.4088, 0.4124, 0.4139, 0.4118, 0.4235, 0.4232, 0.4177]

rouge_peg__cnn__rouge1 = [0.4016, 0.4116, 0.4127, 0.4069, 0.4108, 0.4126, 0.4140, 0.4195]

pmi_peg__cnn__rouge2 = [0.1800, 0.1847, 0.1880, 0.1881, 0.1871, 0.1958, 0.1958, 0.1896]

rouge_peg__cnn__rouge2 = [0.1784, 0.1886, 0.1866, 0.1812, 0.1841, 0.1843, 0.1860, 0.1907]

pmi_peg__cnn__rougeL = [0.2739, 0.2795, 0.2821, 0.2825, 0.2824, 0.2918, 0.2916, 0.2873]

rouge_peg__cnn__rougeL = [0.2740, 0.2849, 0.2809, 0.2750, 0.2779, 0.2775, 0.2806, 0.2854]

pmi_peg__cnn__roberta_score = [0.2405, 0.2472, 0.2515, 0.2522, 0.2520, 0.2691, 0.2686, 0.2602]

rouge_peg__cnn__roberta_score = [0.2407, 0.2506, 0.2508, 0.2420, 0.2472, 0.2476, 0.2517, 0.2578]

pmi_peg__cnn__deberta_score = [0.2716, 0.2786, 0.2814, 0.2826, 0.2813, 0.2932, 0.2932, 0.2854]

rouge_peg__cnn__deberta_score = [0.2690, 0.2775, 0.2817, 0.2752, 0.2778, 0.2776, 0.2811, 0.2851]

pmi_peg__cnn__qaeval_f1 = [0.1670, 0.1731, 0.1746, 0.1764, 0.1747, 0.1902, 0.1895, 0.1801]

rouge_peg__cnn__qaeval_f1 = [0.1672, 0.1751, 0.1800, 0.1703, 0.1735, 0.1749, 0.1777, 0.1830]



pmi_peg__wikihow__rouge1 = [0.3521, 0.3486, 0.3665, 0.3672, 0.3590, 0.3666, 0.3704, 0.3688]

rouge_peg__wikihow__rouge1 = [0.3590, 0.3693, 0.3658, 0.3632, 0.3677, 0.3663, 0.3669, 0.3641]

pmi_peg__wikihow__rouge2 = [0.1346, 0.1354, 0.1413, 0.1397, 0.1345, 0.1379, 0.1388, 0.1386]

rouge_peg__wikihow__rouge2 = [0.1361, 0.1399, 0.1392, 0.1381, 0.1370, 0.1355, 0.1341, 0.1311]

pmi_peg__wikihow__rougeL = [0.2687, 0.2684, 0.2776, 0.2761, 0.2703, 0.2740, 0.2750, 0.2757]

rouge_peg__wikihow__rougeL = [0.2718, 0.2760, 0.2751, 0.2726, 0.2741, 0.2706, 0.2676, 0.2645]

pmi_peg__wikihow__roberta_score = [0.3131, 0.3169, 0.3238, 0.3179, 0.3075, 0.3109, 0.3082, 0.3116]

rouge_peg__wikihow__roberta_score = [0.3180, 0.3125, 0.3192, 0.3115, 0.3094, 0.3031, 0.3004, 0.2957]

pmi_peg__wikihow__deberta_score = [0.243177, 0.2513, 0.2561, 0.2518, 0.2419, 0.2466, 0.2457, 0.2485]

rouge_peg__wikihow__deberta_score = [0.243046, 0.2451, 0.2515, 0.2453, 0.2442, 0.2403, 0.2381, 0.2350]

pmi_peg__wikihow__qaeval_f1 = [0.0705, 0.07447, 0.07939, 0.07848, 0.07461, 0.07885, 0.0805, 0.07929]

rouge_peg__wikihow__qaeval_f1 = [0.0687, 0.07799, 0.07815, 0.07567, 0.07704, 0.07472, 0.0761, 0.07375]


# ==========================================
# PAIRED T-TEST STATISTICS (PMI vs ROUGE)
# ==========================================

t_xsum = [
    [  2.2078,   0.3696,   0.9108,  -6.8150,  -3.7496,  -1.5417],  # 1M
    [-10.1644,  -6.0686, -10.0064, -21.2501, -20.0794, -10.2338],  # 2M
    [ -3.8618,  -2.8849,  -4.2702, -14.6449,  -7.6571,  -7.2322],  # 3M
    [ -1.4461,  -1.8536,  -2.5937, -11.1289,  -7.2944,  -5.9885],  # 4M
    [ -5.3408,  -3.3351,  -4.8155, -12.2793, -10.6004,  -7.5471],  # 5M
    [ -1.4356,  -2.4031,  -1.6889, -17.0313, -14.5838,  -5.7672],  # 6M
    [  0.1295,  -0.0286,  -1.1763,  -6.5773,  -4.0984,  -5.8399],  # 7M
    [ -1.3549,  -1.4935,  -3.0393,  -3.7012,  -4.1592,  -5.2235],  # 8M
]

t_cnn = [
    [  3.0896,   1.6579,  -0.0069,   2.7279,  -0.1703,  -0.1248],  # 1M
    [ -2.8986,  -4.2144,  -5.6999,   1.1617,  -3.2921,  -1.8489],  # 2M
    [ -0.2811,   1.4925,   1.2662,  -0.3211,   0.8061,  -4.6792],  # 3M
    [  7.1739,   7.3094,   7.9291,   7.7300,  10.0851,   5.2495],  # 4M
    [  1.0389,   3.2012,   4.6485,   3.6864,   4.6745,   0.9590],  # 5M
    [ 11.4495,  12.3562,  15.3020,  16.1603,  21.2899,  13.3516],  # 6M
    [  9.8197,  10.6989,  11.9054,  12.8271,  17.1773,  10.1576],  # 7M
    [ -1.9731,  -1.1204,   2.0434,   0.2375,   2.4308,  -2.5164],  # 8M
]

t_wikihow = [
    [ -4.9286,  -1.2748,  -2.7106,   0.1124,  -4.3716,   1.3679],  # 1M
    [-14.3317,  -3.9449,  -6.1794,   5.2300,   3.8878,  -2.7242],  # 2M
    [  0.4965,   1.9403,   2.1671,   4.0682,   4.2367,   0.9453],  # 3M
    [  2.8734,   1.3504,   2.9817,   5.6375,   5.7925,   2.2086],  # 4M
    [ -6.3022,  -2.2853,  -3.2292,  -1.9597,  -1.7196,  -1.8348],  # 5M
    [  0.1591,   2.1724,   2.9578,   5.6052,   7.0256,   3.0064],  # 6M
    [  2.6608,   4.3265,   6.6635,   6.9022,   7.2204,   3.2882],  # 7M
    [  3.5800,   6.8640,   9.7389,  11.9770,  14.4631,   4.1965],  # 8M
]


# ==========================================
# MODEL SIZES
# ==========================================

model_sizes = ["1M","2M","3M","4M","5M","6M","7M","8M"]
x = list(range(len(model_sizes)))

# ==========================================
# COLOR MAP (metric → color)
# ==========================================

metric_colors = {
    "rouge-1": "teal",
    "rouge-2": "orange",
    "rouge-L": "navy",
    "BERT_score_DeBERTa": "green",
    "BERT_score_RoBERTa": "purple",
    "QAeval_F1": "red"
}

line_styles = {
    "pmi_pegasus": "-",
    "rouge_pegasus": "--"
}


# ==========================================
# MULTI-PANEL PLOTTING FUNCTION
# ==========================================

def plot_multi_panel(dataset_name, metric_dict):

    metrics = list(metric_dict.keys())
    n = len(metrics)

    idx, ax = None, None

    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):

        ax = axes[idx]

        pmi_scores = metric_dict[metric]["pmi_pegasus"]
        rouge_scores = metric_dict[metric]["rouge_pegasus"]

        color = metric_colors.get(metric, "black")

        if len(pmi_scores) > 0:
            ax.plot(x, pmi_scores, marker="o", linestyle="-",
                    color=color, label="PMI-Pegasus")

        if len(rouge_scores) > 0:
            ax.plot(x, rouge_scores, marker="^", linestyle="--",
                    color=color, label="ROUGE-Pegasus")

        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.grid(True)

    # remove empty panels
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    # add legend to panels (upper right). Make its color black to look more general. Also, line should be longer because it is hard to see dashed line otherwise.
    handles, labels = ax.get_legend_handles_labels()

    import copy
    black_handles = [copy.copy(h) for h in handles]
    for h in black_handles:
        h.set_color("black")

    fig.legend(black_handles, labels, loc="upper right", fontsize=9, frameon=True, edgecolor="black", facecolor="white", framealpha=1, ncol=1, handlelength=3)

    fig.suptitle(dataset_name, fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_llm_judge_grouped_columns():
    """
    Draw grouped (side-by-side) column graphs for LLM-as-judge results.
    Produces a 3-panel figure:
        - XSUM
        - CNN
        - WikiHow

    Each model size (1M–8M) has three adjacent bars:
        PMI wins | ROUGE wins | Tie
    """

    model_labels = ["1M","2M","3M","4M","5M","6M","7M","8M"]
    x = np.arange(len(model_labels))

    bar_width = 0.25

    # =========================
    # XSUM percentages
    # =========================
    xsum_pmi =  [43.0122,41.0711,41.5299,42.3063,43.8062,39.8447,42.3504,45.6326]
    xsum_rouge = [42.0681,44.6621,44.3533,43.1710,41.0623,46.2679,43.1269,39.3506]
    xsum_tie = [14.9197,14.2668,14.1168,14.5227,15.1315,13.8874,14.5227,15.0168]

    # =========================
    # CNN percentages
    # =========================
    cnn_pmi =  [45.9182,46.5535,47.4761,47.4500,47.4761,46.8030,46.0313,48.7554]
    cnn_rouge = [46.6057,46.6667,45.4743,45.5962,45.4917,46.4926,47.0409,44.4560]
    cnn_tie = [7.4761,6.7798,7.0496,6.9539,7.0322,6.7044,6.9278,6.7885]

    # =========================
    # WikiHow percentages
    # =========================
    wiki_pmi =  [44.5401,47.7318,42.8546,42.8845,46.2973,45.6697,46.9428,49.0228]
    wiki_rouge=[39.6629,37.7802,42.1553,42.4003,38.9995,40.7387,38.0491,36.9912]
    wiki_tie = [15.7970,14.4881,14.9901,14.7152,14.7032,13.5915,15.0081,13.9860]

    # =========================
    # plotting helper
    # =========================
    def plot_single(ax, pmi, rouge, tie, title):

        ax.bar(x - bar_width, pmi, width=bar_width, label="PMI-Pegasus wins")
        ax.bar(x, rouge, width=bar_width, label="ROUGE-Pegasus wins")
        ax.bar(x + bar_width, tie, width=bar_width, label="Tie")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("Percentage (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # =========================
    # multi-panel figure
    # =========================
    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    plot_single(axes[0], xsum_pmi, xsum_rouge, xsum_tie, "XSUM")
    plot_single(axes[1], cnn_pmi, cnn_rouge, cnn_tie, "CNN/DailyMail")
    plot_single(axes[2], wiki_pmi, wiki_rouge, wiki_tie, "WikiHow")

    axes[2].legend(loc="upper right")

    fig.suptitle("LLM-as-Judge Win Distribution", fontsize=16)
    fig.tight_layout()
    plt.show()


# ==========================================
# BUILD DATA STRUCTURES
# ==========================================

xsum_metrics = {
    "rouge-1": {"pmi_pegasus": pmi_peg__xsum__rouge1, "rouge_pegasus": rouge_peg__xsum__rouge1},
    "rouge-2": {"pmi_pegasus": pmi_peg__xsum__rouge2, "rouge_pegasus": rouge_peg__xsum__rouge2},
    "rouge-L": {"pmi_pegasus": pmi_peg__xsum__rougeL, "rouge_pegasus": rouge_peg__xsum__rougeL},
    "BERT_score_DeBERTa": {"pmi_pegasus": pmi_peg__xsum__deberta_score, "rouge_pegasus": rouge_peg__xsum__deberta_score},
    "BERT_score_RoBERTa": {"pmi_pegasus": pmi_peg__xsum__roberta_score, "rouge_pegasus": rouge_peg__xsum__roberta_score},
    "QAeval_F1": {"pmi_pegasus": pmi_peg__xsum__qaeval_f1, "rouge_pegasus": rouge_peg__xsum__qaeval_f1},
}

cnn_metrics = {
    "rouge-1": {"pmi_pegasus": pmi_peg__cnn__rouge1, "rouge_pegasus": rouge_peg__cnn__rouge1},
    "rouge-2": {"pmi_pegasus": pmi_peg__cnn__rouge2, "rouge_pegasus": rouge_peg__cnn__rouge2},
    "rouge-L": {"pmi_pegasus": pmi_peg__cnn__rougeL, "rouge_pegasus": rouge_peg__cnn__rougeL},
    "BERT_score_DeBERTa": {"pmi_pegasus": pmi_peg__cnn__deberta_score, "rouge_pegasus": rouge_peg__cnn__deberta_score},
    "BERT_score_RoBERTa": {"pmi_pegasus": pmi_peg__cnn__roberta_score, "rouge_pegasus": rouge_peg__cnn__roberta_score},
    "QAeval_F1": {"pmi_pegasus": pmi_peg__cnn__qaeval_f1, "rouge_pegasus": rouge_peg__cnn__qaeval_f1},
}

wikihow_metrics = {
    "rouge-1": {"pmi_pegasus": pmi_peg__wikihow__rouge1, "rouge_pegasus": rouge_peg__wikihow__rouge1},
    "rouge-2": {"pmi_pegasus": pmi_peg__wikihow__rouge2, "rouge_pegasus": rouge_peg__wikihow__rouge2},
    "rouge-L": {"pmi_pegasus": pmi_peg__wikihow__rougeL, "rouge_pegasus": rouge_peg__wikihow__rougeL},
    "BERT_score_DeBERTa": {"pmi_pegasus": pmi_peg__wikihow__deberta_score, "rouge_pegasus": rouge_peg__wikihow__deberta_score},
    "BERT_score_RoBERTa": {"pmi_pegasus": pmi_peg__wikihow__roberta_score, "rouge_pegasus": rouge_peg__wikihow__roberta_score},
    "QAeval_F1": {"pmi_pegasus": pmi_peg__wikihow__qaeval_f1, "rouge_pegasus": rouge_peg__wikihow__qaeval_f1},
}


# ==========================================
# HEATMAP PLOTTING FUNCTION
# ==========================================


def plot_multi_dataset_heatmaps():
    model_sizes = ["1M","2M","3M","4M","5M","6M","7M","8M"]

    def build_matrix(dataset_prefix):
        rows = []
        for i in range(8):
            pmi_row = [
                globals()[f"pmi_peg__{dataset_prefix}__rouge1"][i],
                globals()[f"pmi_peg__{dataset_prefix}__rouge2"][i],
                globals()[f"pmi_peg__{dataset_prefix}__rougeL"][i],
                globals()[f"pmi_peg__{dataset_prefix}__roberta_score"][i],
                globals()[f"pmi_peg__{dataset_prefix}__deberta_score"][i],
                globals()[f"pmi_peg__{dataset_prefix}__qaeval_f1"][i],
            ]

            rouge_row = [
                globals()[f"rouge_peg__{dataset_prefix}__rouge1"][i],
                globals()[f"rouge_peg__{dataset_prefix}__rouge2"][i],
                globals()[f"rouge_peg__{dataset_prefix}__rougeL"][i],
                globals()[f"rouge_peg__{dataset_prefix}__roberta_score"][i],
                globals()[f"rouge_peg__{dataset_prefix}__deberta_score"][i],
                globals()[f"rouge_peg__{dataset_prefix}__qaeval_f1"][i],
            ]

            rows.append(pmi_row)
            rows.append(rouge_row)

        return np.array(rows)

    xsum = build_matrix("xsum")
    cnn = build_matrix("cnn")
    wikihow = build_matrix("wikihow")

    model_labels = []
    for m in model_sizes:
        model_labels.append(f"{m} PMI")
        model_labels.append(f"{m} ROUGE")

    metric_labels = ["R-1", "R-2", "R-L",
                     "RoBERTa", "DeBERTa", "QA-F1"]

    yor_cmap = LinearSegmentedColormap.from_list(
        "yor",
        ["#ffffcc", "#fd8d3c", "#b10026"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(30, 18))

    datasets = [
        ("XSUM", xsum),
        ("CNN/DailyMail", cnn),
        ("WikiHow", wikihow)
    ]

    im = None

    for ax, (title, matrix) in zip(axes, datasets):

        im = ax.imshow(matrix, aspect="auto", cmap="Greens")

        ax.set_yticks(np.arange(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=8)
        for tick, label in zip(ax.get_yticklabels(), model_labels):
            tick.set_color("red" if "PMI" in label else "blue")

        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels,
                           rotation=45,
                           ha="right",
                           fontsize=9)

        ax.set_title(title, fontsize=16, pad=12)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i,
                        f"{matrix[i, j]:.4f}",
                        ha="center", va="center",
                        fontsize=9)

        for sep in range(1, 16):
            if sep % 2 == 0:
                # between model groups (e.g. 2M PMI / 2M ROUGE | 3M PMI ...)
                ax.axhline(sep - 0.5, color="black", linewidth=2.2, linestyle="-")
            else:
                # between PMI and ROUGE rows of the same model size
                ax.axhline(sep - 0.5, color="black", linewidth=0.9, linestyle="--")

    plt.subplots_adjust(left=0.05, right=0.90, wspace=0.20)

    # ---- Proper external colorbar ----
    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.04
    )

    cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Model Performance Heatmaps Across Datasets",
                 fontsize=20, y=0.97)

    # plt.subplots_adjust(wspace=0.20)
    plt.show()


# ==========================================
# HEATMAP PLOTTING FUNCTION FOR T-STATISTICS
# ==========================================

def plot_t_stat_heatmaps():

    model_labels = ["1M","2M","3M","4M","5M","6M","7M","8M"]

    metric_labels = ["R-1","R-2","R-L",
                     "DeBERTa","RoBERTa","QA-F1"]

    # Convert to numpy
    xsum = np.array(t_xsum)
    cnn = np.array(t_cnn)
    wikihow = np.array(t_wikihow)

    # Global symmetric normalization
    max_abs = max(
        np.abs(xsum).max(),
        np.abs(cnn).max(),
        np.abs(wikihow).max()
    )

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    datasets = [
        ("XSUM (PMI − ROUGE t-stat)", xsum),
        ("CNN/DailyMail (PMI − ROUGE t-stat)", cnn),
        ("WikiHow (PMI − ROUGE t-stat)", wikihow)
    ]

    im = None

    for ax, (title, matrix) in zip(axes, datasets):

        im = ax.imshow(matrix,
                       aspect="auto",
                       cmap="RdBu_r",   # red=positive, blue=negative
                       norm=norm)

        ax.set_yticks(np.arange(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=10)

        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels,
                           rotation=45,
                           ha="right",
                           fontsize=10)

        ax.set_title(title, fontsize=14)

        # annotate
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i,
                        f"{matrix[i,j]:.4f}",
                        ha="center",
                        va="center",
                        fontsize=9)

    # external colorbar
    plt.subplots_adjust(left=0.05, right=0.90, wspace=0.20)

    # ---- Proper external colorbar ----
    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.04
    )

    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Paired t-statistic (PMI − ROUGE)", fontsize=11)

    fig.suptitle("PMI vs ROUGE Pegasus Paired t-Test Statistics",
                 fontsize=16)

    plt.show()


# ==========================================
# DRAW FIGURES
# ==========================================

"""plot_multi_panel("XSUM Results", xsum_metrics)
plot_multi_panel("CNN/DailyMail Results", cnn_metrics)
plot_multi_panel("WikiHow Results", wikihow_metrics)"""

plot_llm_judge_grouped_columns()

#  plot_multi_dataset_heatmaps()

#  plot_t_stat_heatmaps()
