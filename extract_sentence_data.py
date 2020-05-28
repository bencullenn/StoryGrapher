"""
Created by Ben Cullen on May 28th 2020

This script processes text data using stanford's OpenIE project to extract meaningful triples from sentences
"""

import pandas as pd
import spacy
from spacy.lang.en import English
from openie import StanfordOpenIE


def get_random_sentences(data_path):
    start_index = 0
    end_index = 20
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
        sentences.append(sent)

    sliced_sentences = sentences[start_index:end_index + 1]

    return sliced_sentences

def main():
    data_path = ""

    with StanfordOpenIE() as client:
        for sent in get_random_sentences(data_path):
            print('Processing sentence:', sent.text)
            for triple in client.annotate(sent.text):
                print(triple)



if __name__ == '__main__':
    main()