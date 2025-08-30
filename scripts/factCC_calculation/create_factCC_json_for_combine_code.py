import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_from_disk
import nltk
from tqdm import tqdm


pretrain_dataset_path = "D:\\PMI-Pegasus Project - EXTRAS\\PREPROCESSED_DATASETS_BACKUPS - 26.08.2025\\dataset_for_ROUGE_pegasus_1_MIL"


def create_factCC_json():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str, default=pretrain_dataset_path)
    parser.add_argument("--output_json", type=str, default="factcc_for_1MIL_set.json")
    parser.add_argument("--split", type=str, default="train")  # train/validation/test
    # parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load dataset
    dataset = load_from_disk(args.processed_data)[args.split]

    model_path = "manueldeprada/FactCC"

    # Load FactCC model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    predictions = []

    print_for_analysis = True
    counter = 0

    for ex in tqdm(dataset, desc="Processing examples"):
        doc = ex["document"]
        preds = []

        # Replace <mask> in the doc with ex['summary'].
        doc = doc.replace("<mask>", ex['summary'])

        # Split doc into its individual sentences and store them in a list
        nltk.download('punkt', quiet=True)
        sentences_of_doc = nltk.sent_tokenize(doc)
        sentences_of_doc = [s.strip() for s in sentences_of_doc if s.strip()]

        if print_for_analysis:
            print(f"\n---------->> Document \"{counter+1}\":\n{doc}\n")
            # Print the sentences line by line
            print("---------->> Sentences of the document:\n")
            for i, sentence in enumerate(sentences_of_doc):
                print(f"{i+1}: {sentence}")

            print(f"\n---------->> The principal sentence selected by ROUGE metric:   {ex['summary']}\n")

        for summary in sentences_of_doc:  # multiple candidates
            inputs = tokenizer(
                doc, summary,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()  # 0 = incorrect, 1 = correct
                preds.append(pred)
        predictions.append(preds)

        if print_for_analysis:
            print(f"---------->> Predictions for the sentences: {preds}\n\n\n")
            counter += 1
        if counter >= 10:
            print_for_analysis = False

    # Save like factcc_dummy.json
    with open(args.output_json, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    create_factCC_json()

