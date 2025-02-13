from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
from rouge_score import rouge_scorer
import re
# import numpy as np
# from itertools import combinations
# import itertools
import torch.nn.functional as F


def conditional_probability(target_sentences, context_sentences):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentences, return_tensors='pt', padding=True, truncation=True,
                               max_length=512).to(device)

    # Tokenize the target sentence
    target_inputs = tokenizer(target_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
        device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(context_inputs['input_ids'], labels=target_inputs['input_ids'],
                            return_dict=True)
    # print("Probability of target sentence given context sentence:", losses.loss.item())

    #   print("\n\n", outputs.loss, "\n\n")

    # Extract logits
    logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

    # Shift labels left to align with logits
    labels_shifted = target_inputs["input_ids"][:, 1:].contiguous()
    logits_shifted = logits[:, :-1, :].contiguous()

    # Step 5: Compute Per-Token Loss Correctly
    loss_per_token = F.cross_entropy(
        logits_shifted.view(-1, model.config.vocab_size),
        labels_shifted.view(-1),
        ignore_index=-100,
        reduction="none"
    )

    # Reshape back to (batch_size, seq_length - 1)
    loss_per_token = loss_per_token.view(labels_shifted.shape)

    # Mask out padding (-100) and count valid tokens per example
    valid_token_mask = (labels_shifted != -100).float()
    valid_token_count_per_example = valid_token_mask.sum(dim=1)  # Count valid tokens per example

    # Step 6: Compute Corrected Per-Example Loss
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_count_per_example

    # Normalize Loss to Match Forward Pass Scaling
    # total_valid_tokens = valid_token_mask.sum()  # Sum of all valid tokens in batch

    """batch_avg_loss = loss_per_example.mean().item()

    # Print per-example loss
    for i in loss_per_example:
        print(f"Loss: {i.item()}\n")

    # Optional: Compute batch average loss (matches model.loss)
    # batch_avg_loss = loss_per_example.mean().item()
    print(f"\nTotal batch loss (averaged): {batch_avg_loss}")"""

    # print("\n\n", context_inputs['input_ids'].shape, " ---->> ", len(losses), "\n\n")

    #   pipe_output = pipe(context_sentences, labels=target_sentences, batch_size=16)

    #  return math.exp(-losses.loss.item())
    # return [math.exp(-p['loss'].item()) for p in outputs.loss]
    return [math.exp(-l.item()) for l in loss_per_example]


def marginal_probability(context_sentences):
    # Step 1: Tokenize the context sentence (input)
    context_inputs = tokenizer(
        context_sentences, return_tensors='pt', padding=True, truncation=True, max_length=512
    ).to(device)

    # Step 2: Generate output sentences
    generated_output = model.generate(
        context_inputs['input_ids'],
        max_new_tokens=40,
        num_beams=4,   #  5 --> 4
        no_repeat_ngram_size=3,  # 2 --> 3
        early_stopping=True
    )
    """max_length = 100,
    min_length = 10,
    num_beams = 2,
    do_sample = True,
    top_k = 50,
    top_p = 0.9,
    no_repeat_ngram_size = 3,
    early_stopping = False,
    return_dict_in_generate = True"""

    # Step 3: Decode & Tokenize the Generated Sentences Properly
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_output]

    generated_labels = tokenizer(
        generated_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
    ).to(device)

    # Ensure padding tokens (-100) are ignored in loss computation
    generated_labels["input_ids"][generated_labels["input_ids"] == tokenizer.pad_token_id] = -100

    # Step 4: Forward pass with context_inputs and generated_labels
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=context_inputs['input_ids'], labels=generated_labels['input_ids'])

    # print("\n\n", outputs.loss, "\n\n")

    # Extract logits
    logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

    # Shift labels left to align with logits
    labels_shifted = generated_labels["input_ids"][:, 1:].contiguous()
    logits_shifted = logits[:, :-1, :].contiguous()

    # Step 5: Compute Per-Token Loss Correctly
    loss_per_token = F.cross_entropy(
        logits_shifted.view(-1, model.config.vocab_size),
        labels_shifted.view(-1),
        ignore_index=-100,
        reduction="none"
    )

    # Reshape back to (batch_size, seq_length - 1)
    loss_per_token = loss_per_token.view(labels_shifted.shape)

    # **Mask out padding (-100) and count valid tokens per example**
    valid_token_mask = (labels_shifted != -100).float()
    valid_token_count_per_example = valid_token_mask.sum(dim=1)  # Count valid tokens per example

    # Step 6: Compute Corrected Per-Example Loss
    loss_per_example = loss_per_token.sum(dim=1) / valid_token_count_per_example

    # Print per-example loss
    """for i in loss_per_example:
        print(f"Loss: {i.item()}\n")"""

    # Optional: Compute batch average loss (matches model.loss)
    # batch_avg_loss = loss_per_example.mean().item()
    # print(f"\nTotal batch loss (averaged): {batch_avg_loss}")

    # pipe_output = pipe(context_sentences, labels=generated_output, batch_size=16)

    # Print probability
    # print("Probability of target sentence given context sentence:", p_x_given_y)

    # return math.exp(-losses.loss.item())
    # return [math.exp(-p['loss'].item()) for p in outputs.loss]

    return [math.exp(-l.item()) for l in loss_per_example]


def calculate_pmi(target_sentences_per_text, docs_without_target_sentences_per_text):
    p_x_given_y_list = conditional_probability(target_sentences_per_text, docs_without_target_sentences_per_text)
    p_x_list = marginal_probability(docs_without_target_sentences_per_text)

    pmi_list = [math.log2(a / b) for a, b in zip(p_x_given_y_list, p_x_list)]
    #  pmi = math.log2(p_x_given_y / p_x)
    return pmi_list

# Load pre-trained model tokenizer (vocabulary)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load pre-trained model (weights)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Context sentence
context_sentence_1 = "I love to read books."
context_sentence_2 = "Apple's lawyers said they were willing to publish a clarification."
context_sentence_3 = "Michelle, of South Shields, Tyneside, says she feels like a new woman after dropping from dress size 30 to size 12."

# Target sentence
target_sentence_1 = "I also enjoy reading articles."
target_sentence_2 = "However the company does not accept that it misled customers."
target_sentence_3 = "Michelle weighed 25st 3lbs when she joined the group in April 2013 and has since dropped to 12st " \
                  "10lbs."

"""p_x_given_y_temp = conditional_probability(target_sentence_3, context_sentence_3)
p_x_temp = marginal_probability(context_sentence_3)
pmi_temp = calculate_pmi(p_x_given_y_temp, p_x_temp)
print(pmi_temp)"""


def split_into_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


"""def get_pmi_list_from_doc(doc):
    sentences = split_into_sentences(doc)
    pmi_list = []
    for idx in sentences:
        temp_sentences_without_i = [x for x in sentences if x != idx]
        merged_temp_sentences_without_i = ' '.join(temp_sentences_without_i)
        p_x_given_y = conditional_probability(idx, merged_temp_sentences_without_i)
        p_x = marginal_probability(merged_temp_sentences_without_i)
        pmi = calculate_pmi(p_x_given_y, p_x)
        pmi_list.append(pmi)
    return pmi_list"""


"""def select_principal_sentence(doc):
    pmi_list = get_pmi_list_from_doc(doc)
    sentences = split_into_sentences(doc)

    # Select sentence with maximum PMI using argmax
    principal_sentence = sentences[np.argmax(pmi_list)]
    return principal_sentence"""


def calc_pmi_list_for_paragraph(text):
    sentences = split_into_sentences(text)

    temp_texts = [text] * len(sentences)
    docs = [t.replace(s, "", 1) for t, s in zip(temp_texts, sentences)]
    #  docs = [t.replace(s, "", 1).replace(". .", ".") for t, s in zip(temp_texts, sentences)]

    pmi_list = calculate_pmi(sentences, docs)

    # print elements of docs
    for d in range(len(docs)):
        print(f"Docs {d+1} ---> {docs[d]}\n")

    return pmi_list


def calc_rouge_list_for_paragraph(text):
    sentences = split_into_sentences(text)

    temp_texts = [text] * len(sentences)
    docs = [t.replace(s, "", 1) for t, s in zip(temp_texts, sentences)]
    #  docs = [t.replace(s, "", 1).replace(". .", ".") for t, s in zip(temp_texts, sentences)]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', "rougeLsum"], use_stemmer=True)

    rouge_scores = []
    for s, d in zip(sentences, docs):
        scores = scorer.score(s, d)
        rouge_scores.append(scores['rouge1'].fmeasure)
    return rouge_scores


doc1 = "US Treasury Secretary Robert Rubin arrived in Malaysia Sunday for a two-day visit to discuss the " \
            "regional economic situation, the US Embassy said. Rubin, on a tour of Asia's economic trouble spots, " \
            "arrived from Beijing, where he had accompanied US President Bill Clinton on a visit. Rubin was " \
            "scheduled to meet and have dinner with Finance Minister Anwar Ibrahim on Sunday. On Monday, Rubin will " \
            "meet privately with Prime Minister Mahathir Mohamad and separately with senior Malaysian and American " \
            "business leaders, the embassy said in a statement. Rubin will leave Monday for Thailand and South Korea."


"""doc1 = ("Artificial intelligence (AI) is transforming various industries, from healthcare to finance. Machine learning "
        "algorithms analyze vast amounts of data, uncovering patterns that humans might overlook. In healthcare, AI "
        "helps diagnose diseases early, improving patient outcomes. Meanwhile, in finance, AI-driven models detect "
        "fraudulent transactions with high accuracy. Despite these advancements, ethical concerns remain. Issues like "
        "data privacy and bias in AI systems continue to challenge researchers. As AI technology evolves, balancing "
        "innovation with responsible use becomes crucial for society.")"""

"""doc1 = ("Climate change is one of the most pressing global challenges of our time. Rising temperatures contribute to "
        "extreme weather events, such as hurricanes and wildfires, which devastate communities. Melting glaciers lead "
        "to rising sea levels, threatening coastal cities worldwide. Scientists emphasize the importance of reducing "
        "carbon emissions to slow down these effects. Renewable energy sources like solar and wind power offer "
        "sustainable alternatives to fossil fuels. However, transitioning to green energy requires significant "
        "investment and policy support. Collective action from governments, businesses, and individuals is essential "
        "to mitigate climate change's long-term impact.")"""

"""doc1 = ("Space exploration has expanded humanity’s understanding of the universe. Missions to Mars and the Moon provide "
        "valuable insights into planetary formation and the potential for extraterrestrial life. Advances in rocket "
        "technology have made space travel more cost-effective, enabling private companies to participate alongside "
        "government agencies. The International Space Station serves as a hub for scientific research, fostering "
        "global collaboration. Despite these achievements, challenges such as space debris and the high costs of "
        "exploration remain significant hurdles. As technology progresses, future missions may pave the way for human "
        "colonization of other planets.")"""

pmi_list_doc1 = calc_pmi_list_for_paragraph(doc1)

sentences_doc1 = split_into_sentences(doc1)

print()
# Print the sentences and their PMI values
for i in range(len(sentences_doc1)):
    print(f"PMI score of sentence {i+1} = {pmi_list_doc1[i]}\n\t{sentences_doc1[i]}\n")

# Select the principal sentence
"""principal_sentence_doc1 = select_principal_sentence(doc1)
print(f"\nPrincipal sentence --->  {principal_sentence_doc1}\n")"""

print("\n")

rouge_list_doc1 = calc_rouge_list_for_paragraph(doc1)
# Print the sentences and their ROUGE scores
for i in range(len(sentences_doc1)):
    print(f"ROUGE score of sentence {i+1} = {rouge_list_doc1[i]}\n\t{sentences_doc1[i]}\n")
