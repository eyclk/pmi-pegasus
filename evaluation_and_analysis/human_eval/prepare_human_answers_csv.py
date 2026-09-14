"""
ANNOTATOR PACKETS -> the votes CSV that the correlation check reads.

Each annotator gets back the packet they were sent with their verdicts written
into the brackets:

    BEST ACCORDING TO FAITHFULNESS  (A / B / tie):  [ A ]
    BEST ACCORDING TO INFORMATIVENESS  (A / B / tie):  [ tie ]

GPT_API_corralation_check_with_human_eval.py wants those same verdicts as one
CSV (see its load_human_answers):

    item,annotator,faithfulness,informativeness
    1,ege,A,A
    2,ege,B,tie
    ...

This script is the bridge. It reads one filled-in packet per annotator, checks
each one against the blank packet, and writes the merged CSV.

WHY THE CHECK MATTERS
---------------------
A vote is only meaningful together with the arrangement it was cast on. "A"
means PMI in one item and ROUGE in the next, because build_items() randomised
the slots per item; the correlation script resolves that through answer_key.txt
against human_eval_set.txt. So a returned packet whose CANDIDATE A text is not
the CANDIDATE A text of the blank packet would silently invert that annotator's
votes on those items, and nothing downstream could detect it -- the CSV carries
only "A" and "B".

So every returned packet is compared, item by item, against the blank one:

  * different CANDIDATE A / CANDIDATE B text is a HARD ERROR. That is the
    arrangement itself, and a mismatch means the votes cannot be mapped.
  * different SOURCE DOCUMENT / REFERENCE SUMMARY text is a WARNING. The
    annotator saw a slightly different document -- worth knowing, and worth
    chasing -- but it does not change what their A/B refers to, so it does not
    invalidate the vote.

That split is the whole reason this is a separate script rather than a
hand-made CSV.

USAGE
-----
    python prepare_human_answers_csv.py                  # every annotator found
    python prepare_human_answers_csv.py --out votes.csv
    python prepare_human_answers_csv.py "human_eval_set - Ege.txt"

With no arguments it picks up every "human_eval_set - NAME.txt" beside this
file and takes NAME as the annotator id, so adding the second and third
annotator later means dropping their file in and re-running. The blank
"human_eval_set.txt" is not matched by that pattern and is used as the
reference copy instead.

Then:

    python GPT_API_corralation_check_with_human_eval.py --report-only \
           --human-answers human_answers.csv

The correlation section needs at least two annotators before its inter-rater
ceiling and its majority vote mean anything -- with one annotator every item
has a "majority" of one. It still runs, and the agreement figures are still
that annotator vs. the judge, which is worth having early.
"""

import argparse
import csv
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
JUDGE_SCRIPT = SCRIPT_DIR / "GPT_API_corralation_check_with_human_eval.py"

BLANK_PACKET = SCRIPT_DIR / "human_eval_set.txt"
ANSWER_KEY = SCRIPT_DIR / "answer_key.txt"
DEFAULT_OUT = SCRIPT_DIR / "human_answers.csv"

# "human_eval_set - Ege.txt" -> annotator "ege". The blank packet has no " - "
# and so is never mistaken for a returned one.
#
# Anything after a SECOND " - " is a note to self, not part of the name:
# "human_eval_set - st - cnn & wikihow completed.txt" is annotator "st". People
# label their working copies, and the label should not leak into the CSV and
# the report as though it were who they are.
PACKET_GLOB = "human_eval_set - *.txt"
ANNOTATOR_FROM_NAME = re.compile(r"^human_eval_set - (.+?)(?: - .*)?$")

# The dimension columns of the CSV, and the packet headings they come from.
# Order fixed here because it is the column order of the output.
DIMENSION_HEADINGS = {
    "faithfulness": "BEST ACCORDING TO FAITHFULNESS",
    "informativeness": "BEST ACCORDING TO INFORMATIVENESS",
}

# What an annotator may write in the brackets. Deliberately the same mapping as
# the correlation script's VALID_VOTES, so anything this script accepts that
# script also accepts -- an inconsistency between the two would show up as a
# crash halfway through a report rather than here.
VALID_VOTES = {"a": "A", "b": "B", "tie": "tie", "t": "tie"}


def load_judge_module():
    """
    Imports the correlation script as a module, for its packet parser.

    Its filename is not an importable identifier, hence the explicit spec. The
    point is to share ITEM_SPLIT and parse_eval_set() rather than re-implement
    them: if the packet format ever changes, both this script and the judge
    should break together, not drift apart quietly.
    """

    if not JUDGE_SCRIPT.exists():
        raise SystemExit(f"cannot find {JUDGE_SCRIPT.name} beside this script")

    spec = importlib.util.spec_from_file_location("judge_script", JUDGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotator_name_from(path):
    """'human_eval_set - Ege Yigit.txt' -> 'ege_yigit'.

    Lowercased and de-spaced so the id is stable whatever the file is called;
    it ends up in a CSV column and in the report, not in any filename. A
    trailing " - note" is dropped, so
    'human_eval_set - st - cnn & wikihow completed.txt' is just 'st'.
    """

    match = ANNOTATOR_FROM_NAME.match(path.stem)
    if not match:
        raise SystemExit(
            f"cannot read an annotator name from {path.name!r}; expected "
            f"'human_eval_set - NAME.txt'"
        )
    return "_".join(match.group(1).lower().split())


def parse_votes(path, judge):
    """
    -> {item number: {dimension: "A"/"B"/"tie"}} for one returned packet.

    Items are split with the judge's own ITEM_SPLIT, so this script and the
    judge always agree on where one item ends and the next begins.

    An empty bracket is "not answered": it is skipped and counted, because an
    annotator who ran out of time should cost their unanswered items, not the
    whole file. Anything else in the brackets is a hard error -- a typo silently
    dropped here would quietly shrink n in the correlation report.
    """

    text = path.read_text(encoding="utf-8-sig")
    chunks = judge.ITEM_SPLIT.split(text)
    if len(chunks) < 3:
        raise SystemExit(f"{path.name}: no items found -- is this a packet?")

    votes = {}
    blank = Counter()
    for i in range(1, len(chunks), 2):
        number = int(chunks[i])
        body = chunks[i + 1]

        answers = {}
        for dimension, heading in DIMENSION_HEADINGS.items():
            # The heading, then anything up to the first bracket on that line.
            match = re.search(
                r"%s[^\[\n]*\[([^\]]*)\]" % re.escape(heading), body)
            if match is None:
                raise SystemExit(
                    f"{path.name}: item {number} has no "
                    f"'{heading}' answer bracket"
                )

            cell = match.group(1).strip().lower()
            if not cell:
                blank[dimension] += 1
                continue
            if cell not in VALID_VOTES:
                raise SystemExit(
                    f"{path.name}: item {number}, {dimension}: cannot read the "
                    f"vote {match.group(1)!r} (expected A, B or tie)"
                )
            answers[dimension] = VALID_VOTES[cell]

        votes[number] = answers

    if blank:
        for dimension, count in sorted(blank.items()):
            print(f"  note: {count} unanswered {dimension} item(s) "
                  f"-- skipped, not guessed")

    return votes


def check_against_blank(path, votes, returned_texts, blank_texts):
    """
    Confirms this packet is the one that was sent out.

    Hard-errors on a changed candidate arrangement (which would invert the
    votes), warns on changed source text (which would not). See the module
    docstring.
    """

    missing = set(blank_texts) - set(returned_texts)
    extra = set(returned_texts) - set(blank_texts)
    if missing or extra:
        raise SystemExit(
            f"{path.name} does not cover the same items as "
            f"{BLANK_PACKET.name}: missing {sorted(missing)[:10]}, "
            f"unexpected {sorted(extra)[:10]}"
        )

    swapped = []
    edited = []
    for number, blank_item in blank_texts.items():
        returned = returned_texts[number]
        if (returned["candidate_a"] != blank_item["candidate_a"]
                or returned["candidate_b"] != blank_item["candidate_b"]):
            swapped.append(number)
        elif (returned["document"] != blank_item["document"]
              or returned["reference"] != blank_item["reference"]):
            edited.append(number)

    if swapped:
        raise SystemExit(
            f"{path.name}: the candidate texts differ from "
            f"{BLANK_PACKET.name} on item(s) {swapped[:10]}"
            f"{' ...' if len(swapped) > 10 else ''}.\n"
            f"That is the A/B arrangement itself, so these votes cannot be "
            f"mapped onto a system and nothing is written. Get the original "
            f"packet back from this annotator, or re-issue it."
        )

    if edited:
        print(f"  WARNING: source document or reference summary differs from "
              f"{BLANK_PACKET.name} on {len(edited)} item(s): "
              f"{edited[:10]}{' ...' if len(edited) > 10 else ''}")
        print("           the A/B arrangement is intact, so the votes are "
              "still usable, but this annotator did not read exactly the same "
              "text as the judge will.")

    answered = sum(1 for a in votes.values() if a)
    print(f"  {answered}/{len(blank_texts)} items with at least one vote")


def summarise(rows):
    """Per-annotator vote counts -- a quick eyeball on the CSV about to be
    written, and on whether an annotator leaned hard on one slot."""

    print()
    print("=" * 60)
    print("VOTES COLLECTED")
    print("=" * 60)

    annotators = sorted({row["annotator"] for row in rows})
    for annotator in annotators:
        mine = [row for row in rows if row["annotator"] == annotator]
        print(f"\n{annotator}  ({len(mine)} items)")
        for dimension in DIMENSION_HEADINGS:
            counts = Counter(row[dimension] for row in mine if row.get(dimension))
            total = sum(counts.values())
            if not total:
                print(f"  {dimension:<16} no votes")
                continue
            print(f"  {dimension:<16} "
                  f"A {counts['A']:>3} ({counts['A']/total:5.1%})   "
                  f"B {counts['B']:>3} ({counts['B']/total:5.1%})   "
                  f"tie {counts['tie']:>3} ({counts['tie']/total:5.1%})   "
                  f"n={total}")

    if len(annotators) < 2:
        print("\nOnly one annotator. The correlation report will run, but its\n"
              "inter-annotator ceiling and its majority vote need at least two\n"
              "-- with one rater every item trivially has a majority.")


def decoded_summary(rows, key, judge):
    """
    The same votes with the slots resolved into systems -- what the annotator
    actually said about PMI vs. ROUGE.

    This is not needed to build the CSV, and the correlation report gives it
    too (as the "PMI-wins h/g" column). It is here because that report cannot
    be produced until the paid judge run exists, and there is no reason to make
    someone buy a GPT run to find out what their own annotation concluded. It
    reads answer_key.txt, which the annotators never see.

    Per annotator, not pooled: pooling raters before taking a majority would
    double-count whoever answered more items. The correlation script takes the
    majority properly, and that is the number to quote.
    """

    print()
    print("=" * 60)
    print("WHAT THESE VOTES SAY ABOUT PMI vs. ROUGE")
    print("=" * 60)
    print("slots resolved through answer_key.txt; per annotator, not a "
          "majority vote")

    for annotator in sorted({row["annotator"] for row in rows}):
        print(f"\n{annotator}")
        mine = [row for row in rows if row["annotator"] == annotator]
        for dimension in DIMENSION_HEADINGS:
            decoded = []
            for row in mine:
                slot = row.get(dimension)
                if not slot:
                    continue
                meta = key.get(row["item"])
                if meta is None:
                    continue
                decoded.append((
                    meta["dataset"],
                    judge.decode_winner(slot, meta["system_a"], meta["system_b"])
                    if slot in ("A", "B") else "tie",
                ))
            if not decoded:
                continue

            print(f"  {dimension.upper()}")

            def line(label, subset):
                counts = Counter(winner for _, winner in subset)
                n = len(subset)
                print(f"    {label:<8} n={n:>3}  "
                      f"PMI {counts['pmi']:>3} ({counts['pmi']/n:5.1%})   "
                      f"ROUGE {counts['rouge']:>3} ({counts['rouge']/n:5.1%})   "
                      f"tie {counts['tie']:>3} ({counts['tie']/n:5.1%})")

            line("ALL", decoded)
            for dataset in sorted({d for d, _ in decoded}):
                line(dataset, [(d, w) for d, w in decoded if d == dataset])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Turn filled-in annotator packets into the votes CSV that "
                    "GPT_API_corralation_check_with_human_eval.py reads."
    )
    parser.add_argument("packets", nargs="*", type=Path,
                        help="filled-in packets; default is every "
                             f"'{PACKET_GLOB}' beside this script")
    parser.add_argument("--blank", type=Path, default=BLANK_PACKET,
                        help="the unfilled packet to check against "
                             f"(default {BLANK_PACKET.name})")
    parser.add_argument("--answer-key", type=Path, default=ANSWER_KEY,
                        help="answer_key.txt, used only to show what the "
                             "votes say about PMI vs ROUGE "
                             f"(default {ANSWER_KEY.name})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"CSV to write (default {DEFAULT_OUT.name})")
    return parser.parse_args()


def main():
    args = parse_args()
    judge = load_judge_module()

    packets = args.packets or sorted(SCRIPT_DIR.glob(PACKET_GLOB))
    if not packets:
        raise SystemExit(
            f"no annotator packets found. Put the returned files beside this "
            f"script named '{PACKET_GLOB}', or pass them as arguments."
        )

    blank_texts = judge.parse_eval_set(args.blank)
    print(f"{len(blank_texts)} items in the blank packet {args.blank.name}\n")

    rows = []
    seen = {}
    for path in packets:
        if not path.exists():
            raise SystemExit(f"packet not found: {path}")

        annotator = annotator_name_from(path)
        if annotator in seen:
            raise SystemExit(
                f"two packets map to the annotator id {annotator!r}: "
                f"{seen[annotator].name} and {path.name}"
            )
        seen[annotator] = path

        print(f"{path.name}  ->  annotator {annotator!r}")
        votes = parse_votes(path, judge)
        check_against_blank(path, votes, judge.parse_eval_set(path), blank_texts)

        for number in sorted(votes):
            answers = votes[number]
            if not answers:
                continue          # nothing answered on this item; omit the row
            row = {"item": number, "annotator": annotator}
            row.update(answers)
            rows.append(row)

    rows.sort(key=lambda row: (row["item"], row["annotator"]))

    columns = ["item", "annotator", *DIMENSION_HEADINGS]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # A dimension left unanswered stays an empty cell, which
            # load_human_answers() skips rather than guessing at.
            writer.writerow({column: row.get(column, "") for column in columns})

    summarise(rows)

    # Nice-to-have, not required to write the CSV: without the key the votes
    # are still perfectly valid, they just cannot be read as PMI vs ROUGE here.
    if args.answer_key.exists():
        decoded_summary(rows, judge.parse_answer_key(args.answer_key), judge)
    else:
        print(f"\n{args.answer_key.name} not found -- skipping the PMI/ROUGE "
              f"breakdown (the CSV is unaffected)")

    # Read the CSV back with the consumer's own parser. Writing a file the
    # correlation script then refuses is the one failure this script exists to
    # prevent, so it is worth proving here rather than discovering later.
    reloaded = judge.load_human_answers(args.out)
    pairs = sum(len(v) for v in reloaded.values())
    print(f"\n{len(rows)} rows -> {args.out}")
    print(f"re-read by load_human_answers(): {pairs} votes over "
          f"{len({item for item, _ in reloaded})} items, "
          f"{len({name for v in reloaded.values() for name in v})} annotator(s)")
    print("\nNext:")
    print(f"  python {JUDGE_SCRIPT.name} --report-only \\")
    print(f"         --human-answers {args.out.name}")


if __name__ == "__main__":
    sys.exit(main())
