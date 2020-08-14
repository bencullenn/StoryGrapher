"""
Created by Ben Cullen on August 5th 2020

This script processes text data using Open AI's OPEN IE Model to extract meaningful triples from given text data
"""

import pandas as pd
import random
import time
import spacy
import os
import torch
import re
import networkx as nx
import matplotlib.pyplot as plt
from spacy.lang.en import English
from allennlp.predictors.predictor import Predictor


def get_all_sentences(predictor, data_path):
    sentences = []


    print("Extracting sentences from file....")

    # Open the file for the text that we want to read
    with open(data_path) as f:
        text = f.read()

    # Create a Spacy doc object to store information about the text
    doc = predictor(text)

    # Get the desired sentence indexes
    for sent in doc.sents:
        # Make sure sentences contain extractable text
        if re.search("[a-zA-Z]", sent.text) != None:
            sentences.append(sent)

    print("Extracted ", len(sentences), " sentences from file")
    return sentences


def get_random_sentences(predictor, data_path, num_of_sent=5):
    sliced_sentences = random.choices(get_all_sentences(predictor, data_path), k=num_of_sent)

    return sliced_sentences


def create_openie_triple(predictor, sent):
    start_time = time.clock()
    result = (predictor.predict(
        sentence=sent
    ))
    delta = time.clock() - start_time
    print("OpenIE Time: {:1.2f}".format(delta))
    print("OpenIE Result:", result)

    return result


def create_srl_triple(predictor, sent):
    start_time = time.clock()
    result = (predictor.predict(
        sentence=sent
    ))
    delta = time.clock() - start_time
    print("SRL Prediction Time: {:1.2f}".format(delta))
    print("SRL Result:", result)

    return result


def get_coref_prediction(predictor, text):
    print(predictor.model)
    start_time = time.clock()
    result = (predictor.predict(
        document=text
    ))
    delta = time.clock() - start_time
    print("Coref Prediction Time: {:1.2f}".format(delta))
    print("Coref Result:", result)


def get_triple_string_from_json(json):
    result_list = []
    verb_dict = json["verbs"]
    for item in verb_dict:
        extraction = item["description"]
        word_list = re.findall(r"\[.*?\]", extraction)
        word_str = "{" + ",".join(word_list) + "}"
        result_list.append(word_str)

    result_str = ",".join(result_list)
    return result_str


def get_sent_pos_string(sent):
    pos_string = ""
    for t in sent:
        pos_string = pos_string + t.text + ":" + t.tag_ + " "
    return pos_string.strip()


def get_sent_pos_dict(sent):
    pos_dict = {}
    for t in sent:
        pos_dict[t.text] = t.tag_
    return pos_dict


def get_sent_dep(sent):
    dep_string = ""
    for t in sent:
        dep_string = dep_string + t.text + ":" + t.dep_ + " "
    return dep_string.strip()


def get_root_verb(sent):
    root = "NA"
    for t in sent:
        if t.dep_ == "ROOT":
            root = t.text
            return root
    return root


def get_root_token(sent):
    for t in sent:
        if t.dep_ == "ROOT":
            return t


def get_verb_tense(sent, relevant_triple):
    verb_tense = "NA"
    triple_verb = ""
    for word in relevant_triple:
        if "[V:" in word:
            start_index = word.find('[V:') + 3
            end_index = word.find(']')
            triple_verb = word[start_index:end_index].strip()

    for t in sent:
        if t.text == triple_verb:
            verb_tense = t.tag_
            return verb_tense
    return verb_tense


def get_relevant_triple(json, root_verb):
    verb_dict = json['verbs']
    triple_list = []

    # If no triples were extracted then return an empty list
    if len(verb_dict) == 0:
        return triple_list

    # If there is only one triple then extract and return
    if len(verb_dict) == 1:
        first_triple = next(iter(verb_dict))
        extraction = first_triple["description"]
        triple_list = re.findall(r"\[.*?\]", extraction)
        return triple_list

    # If there is multiple triples then extract the one that contains the root_verb
    for item in verb_dict:
        if item['verb'] == root_verb:
            extraction = item["description"]
            triple_list = re.findall(r"\[.*?\]", extraction)
            return triple_list

    # If none of the triples contain the root verb then just return the first triple
    first_triple = next(iter(verb_dict))
    extraction = first_triple["description"]
    triple_list = re.findall(r"\[.*?\]", extraction)
    return triple_list


def convert_string_to_sent(predictor, string):
    doc = predictor(string)
    sent = next(doc.sents)

    return sent


def trim_triple(predictor, triple):
    trimmed_triple = [None] * 3
    isFirstARG0 = True
    for part in triple:

        # print("Processing part:", part)

        # Extract the tag from the triple
        result = re.search('\[.*?:', part)
        if result.group() != None:
            tag = result.group()[1:-1]
        # print("Extracted Tag:", tag, type(tag))
        # else:
        # print("No tag found in ", part)

        # Extract the test from the triple
        result = re.search(':.*?\]', part)
        if result.group() != None:
            text = result.group()[1:-1]
            # print("Extracted Text:", text, type(text))
        # else:
        # print("No text found in ", part)

        # Get the root verb from part if tag is V, ARG0, or ARG1
        if tag == "V" or tag == "ARG0" or tag == "ARG1":
            sent = convert_string_to_sent(predictor, text)
            pos = get_sent_pos_dict(sent)
            root = get_root_token(sent)
            # print(tag, "'s Root:", root)

            if pos[root.text] == "IN":
                print("IN Root Detected...")
                for child in root.children:
                    print("Child:", child.text)

        if tag == "ARG0" and isFirstARG0 == True:
            trimmed_triple[0] = root.text
            isFirstARG0 = False
        elif tag == "V":
            trimmed_triple[1] = root.text
        elif tag == "ARG1":
            trimmed_triple[2] = root.text

    return trimmed_triple

def main():

    # Setup Cuda if available, otherwise use the CPU
    device = -1

    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    # Put data path here
    data_path = "data/raw/anne_bonnie.txt"
    save_path = "data/triples/"
    data_name = data_path.split('/')[-1]

    # Generate Models
    print("Generating models...")
    openie_model_url = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
    openie_predictor = Predictor.from_path(openie_model_url, cuda_device=device)
    print("Generated openie predictor")

    spacy_sent = English()
    spacy_sent = spacy.load('en_core_web_sm')
    spacy_sent.add_pipe(spacy_sent.create_pipe('sentencizer'))
    print("Generated Spacy Sentencizer")

    print("Finished generating models")

    sentences = []
    trimmed_triples = []

    # Split text data into sentences
    all_sentences = get_all_sentences(spacy_sent, data_path)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%y_%H:%M', t)

    remove_bad_triples = True
    good_triples = 0
    total_triples = len(all_sentences)

    # print("Doing co-reference analysis")
    # coref_data = get_coref_prediction(coref_predictor, text_data)

    for sent in all_sentences:
        print('Processing sentence:', sent.text)
        sentences.append(sent)
        
        # Get the root of the sentence
        sent_root = get_root_verb(sent)
        # print("Root Verb:", sent_root)

        # Extract a triple using OpenIE
        openie_result = create_openie_triple(openie_predictor, sent.text.strip())

        # Get releveant triple
        relevant_triple = get_relevant_triple(openie_result, sent_root)
        # print("Selected Triple", str(relevant_triple))

        # Tri the triple
        trimmed = trim_triple(spacy_sent, relevant_triple) 
        trimmed_triples.append(trimmed)
        print("Trimmed Triple:", trimmed, "\n")

        if remove_bad_triples == True:
            if None in trimmed:
                sentences.pop()  
                trimmed_triples.pop()
            else:
                good_triples += 1

    # Put sentence and triple data into a pandas dataframe for exporting
    triples_data = pd.DataFrame({'Sentence': sentences,
                                 'Trimmed Triple': trimmed_triples})

    print(good_triples, " triples of total ", total_triples, " triples were extracted")

    # Store the DataFrame into a csv file for examination
    triples_data.to_csv(os.path.join(save_path + data_name + '_triples_ ' + timestamp + '.csv'))

    # Create graph object
    G = nx.Graph()

    file_name = data_name + ' Graph ' + timestamp
    # Add nodes to graph and connect images
    for triple in trimmed_triples:
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    # Create graph picture
    pos = nx.spring_layout(G)
    fig = plt.figure(figsize=(45, 45))
    fig.suptitle(file_name)
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=1000, node_color='seagreen', alpha=0.9,
            labels={node: node for node in G.nodes()})

    # Save the graph as a picture
    plt.savefig(os.path.join(save_path + data_name + '_graph_' + timestamp + '.png'))


if __name__ == '__main__':
    main()
