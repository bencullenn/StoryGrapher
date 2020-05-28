"""
Created by Ben Cullen on May 28th 2020

This script processes text data using stanford's OpenIE project to extract meaningful triples from sentences
"""

import pandas as pd
import random
import spacy
from spacy.lang.en import English
from openie import StanfordOpenIE

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
        sentences.append(sent)

    return sentences


def get_random_sentences(data_path, num_of_sent=5):

    sliced_sentences = random.choices(get_all_sentences(data_path), k=num_of_sent)

    return sliced_sentences

def main():
    # Put data path here
    data_path = "/Users/bencullen/Projects/StoryGrapher/text_data/Ghost_Chimes.txt"
    sentences = []
    triples = []

    with StanfordOpenIE() as client:
        for sent in get_random_sentences(data_path):
            print('Processing sentence:', sent.text)
            sentences.append(sent.text)
            triples.append(client.annotate(sent.text))

    # Put sentence and triple data into a pandas dataframe
    triples_data = pd.DataFrame({'Sentences': sentences, 'Triples': triples})

    print(triples_data)



if __name__ == '__main__':
    main()