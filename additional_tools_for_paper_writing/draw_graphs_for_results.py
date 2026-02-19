from matplotlib import pyplot as plt
import numpy as np


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
    cnn_pmi =  [45.9182,46.5535,47.4761,47.4500,47.4761,46.5883,46.0313,48.7554]
    cnn_rouge = [46.6057,46.6667,45.4743,45.5962,45.4917,46.9104,47.0409,44.4560]
    cnn_tie = [7.4761,6.7798,7.0496,6.9539,7.0322,6.5013,6.9278,6.7885]

    # =========================
    # WikiHow percentages
    # =========================
    wiki_pmi =  [44.5401,47.7318,42.8546,42.1732,46.2973,45.6697,46.9428,49.0228]
    wiki_rouge=[39.6629,37.7802,42.1553,43.1056,38.9995,40.7387,38.0491,36.9912]
    wiki_tie = [15.7970,14.4881,14.9901,14.7212,14.7032,13.5915,15.0081,13.9860]

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
# DRAW FIGURES
# ==========================================

plot_multi_panel("XSUM Results", xsum_metrics)
plot_multi_panel("CNN/DailyMail Results", cnn_metrics)
plot_multi_panel("WikiHow Results", wikihow_metrics)


plot_llm_judge_grouped_columns()
