from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
# from rouge_score import rouge_scorer
import re
import numpy as np


def conditional_probability(target_sentence, context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt').to(device)

    # Tokenize the target sentence
    target_inputs = tokenizer(target_sentence, return_tensors='pt').to(device)

    losses = model(context_inputs['input_ids'], labels=target_inputs['input_ids'])
    #print("Probability of target sentence given context sentence:", losses.loss.item())
    return math.exp(-losses.loss.item())


def marginal_probability(context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt').to(device)

    # Generate output
    output = model.generate(context_inputs['input_ids'], max_new_tokens=40, num_beams=5, no_repeat_ngram_size=2,
                            early_stopping=True)

    losses = model(context_inputs['input_ids'], labels=output)

    # Print probability
    # print("Probability of target sentence given context sentence:", p_x_given_y)
    return math.exp(-losses.loss.item())


def calculate_pmi(target_sentence, doc_without_target_sentence):
    """
    Calculate pointwise mutual information (PMI) between two random variables X and Y.

    Parameters:
    p_x (float): Probability of X
    p_y (float): Probability of Y
    p_xy (float): Joint probability of X and Y

    Returns:
    pmi (float): Pointwise mutual information between X and Y
    """
    p_x_given_y = conditional_probability(target_sentence, doc_without_target_sentence)
    p_x = marginal_probability(doc_without_target_sentence)

    pmi = math.log2(p_x_given_y / p_x)
    return pmi

# Load pre-trained model tokenizer (vocabulary)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load pre-trained model (weights)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def split_into_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def get_pmi_list_from_doc(doc):
    sentences = split_into_sentences(doc)
    pmi_list = []
    for i in sentences:
        temp_sentences_without_i = [x for x in sentences if x != i]
        merged_temp_sentences_without_i = ' '.join(temp_sentences_without_i)
        # p_x_given_y = conditional_probability(i, merged_temp_sentences_without_i)
        # p_x = marginal_probability(merged_temp_sentences_without_i)
        pmi = calculate_pmi(i, merged_temp_sentences_without_i)
        pmi_list.append(pmi)
    return pmi_list


def select_principal_sentence(doc):
    pmi_list = get_pmi_list_from_doc(doc)
    sentences = split_into_sentences(doc)

    # Select sentence with maximum PMI using argmax
    principal_sentence = sentences[np.argmax(pmi_list)]
    return principal_sentence


doc1 = "US Treasury Secretary Robert Rubin arrived in Malaysia Sunday for a two-day visit to discuss the " \
            "regional economic situation, the US Embassy said. Rubin, on a tour of Asia's economic trouble spots, " \
            "arrived from Beijing, where he had accompanied US President Bill Clinton on a visit. Rubin was " \
            "scheduled to meet and have dinner with Finance Minister Anwar Ibrahim on Sunday. On Monday, Rubin will " \
            "meet privately with Prime Minister Mahathir Mohamad and separately with senior Malaysian and American " \
            "business leaders, the embassy said in a statement. Rubin will leave Monday for Thailand and South Korea."

pmi_list_doc1 = get_pmi_list_from_doc(doc1)

sentences_doc1 = split_into_sentences(doc1)

print()
# Print the sentences and their PMI values
for idx in range(len(sentences_doc1)):
    print(f"PMI score of sentence {idx+1} = {pmi_list_doc1[idx]}\n\t{sentences_doc1[idx]}\n")

# Select the principal sentence
principal_sentence_doc1 = select_principal_sentence(doc1)
print(f"\nPrincipal sentence --->  {principal_sentence_doc1}\n")
