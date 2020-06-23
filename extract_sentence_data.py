"""
Created by Ben Cullen on May 28th 2020

This script processes text data using stanford's OpenIE project to extract meaningful triples from sentences
"""

import pandas as pd
import random
import time
import spacy
import os
import json
import re
from spacy.lang.en import English
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction


def get_all_sentences(data_path):
    sentences = []

    # Create a Spacy object and the sentencizer to the pipeline
    nlp = English()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    # Open the file for the text that we want to read
    with open(data_path) as f:
        text = f.read()

    # Create a Spacy doc object to store information about the text
    doc = nlp(text)

    # Get the desired sentence indexes
    for sent in doc.sents:
        if not sent.text.isspace():
            sentences.append(sent)

    return sentences


def get_random_sentences(data_path, num_of_sent=5):
    sliced_sentences = random.choices(get_all_sentences(data_path), k=num_of_sent)

    return sliced_sentences


def append_word(original, word):
    return original + ' ' + word


def get_children(token):
    children = []
    for child in token.children:
        # Add each child node that is not punctuation or spacing to the children array
        if child.is_punct is False and child.is_space is False:
            children.append(child)
    return children


def create_spacy_triple(sent):
    root = sent.root
    subject = ''
    object = ''

    # Get the children of the root node
    children = get_children(root)

    for child in children:
        if child.dep_ == 'nsubj' and len(subject) == 0:
            subject = child.text
            subject_descriptors = get_children(child)
            for descriptor in subject_descriptors:
                subject = append_word(subject, descriptor.text)
        elif child.dep_ == 'prep' and len(object) == 0:
            object = child.text
            object_descriptors = get_children(child)
            for descriptor in object_descriptors:
                object = append_word(object, descriptor.text)

    if len(subject) > 0 and len(object) > 0:
        print("Created triple: ", subject.strip(), ",", root.text.strip(), ",", object.strip())
        return (subject.strip(), root.text.strip(), object.strip())


def create_stanfordNLP_triple(sent, npl_client):
    triples = npl_client.annotate(sent.text)
    return triples


def create_allennlp_triple(sent):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    result = (predictor.predict(
        sentence=sent
    ))
    print("Allen NLP Result:", result)

    return result


def main():
    # Put data path here
    data_path = "/Users/bencullen/Projects/StoryGrapher/text_data/fading_light_of_sundown.txt"
    save_path = "/Users/bencullen/Projects/StoryGrapher/output/triples/"
    data_name = data_path.split('/')[-1]
    sentences = []
    allennlp_triples = []
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%y_%H:%M', t)
    all_sentences = get_all_sentences(data_path)
    for sent in all_sentences:
        print('Processing sentence:', sent.text)
        sentences.append(sent)
        allennlp_dict = create_allennlp_triple(sent.text)['verbs']

        triples_list = []
        for item in allennlp_dict:
            extraction = item["description"]
            word_list = re.findall(r"\[.*?\]", extraction)
            word_str = "{" + ",".join(word_list) + "}"
            triples_list.append(word_str)

        triples_str = ",".join(triples_list)
        print("Extracted triples:", triples_str + '\n')
        allennlp_triples.append(triples_str)

    # Put sentence and triple data into a pandas dataframe
    triples_data = pd.DataFrame({'Sentences': sentences, 'Allen NLP Triples': allennlp_triples})

    # Store the DataFrame into a csv file for better reading
    triples_data.to_csv(os.path.join(save_path + data_name + '_triples_ ' + timestamp + '.csv'))


if __name__ == '__main__':
    main()