"""
Coded by Ben Cullen on April 30 2020

This script uses a spacy object to give more info about the story to determine the best parcing methods
"""

import spacy
from spacy.lang.en import English
import pandas as pd

if __name__ == "__main__":

    # Create a Spacy object and the sentencizer to the pipeline
    nlp = English()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    # Open the file for the text that we want to read
    with open("Ghost_Chimes.txt") as f:
        text = f.read()

    # Create a spacy doc object to store information about the text
    doc = nlp(text)

    # Create column labels for Pandas dataframe and gather row data
    basic_info_cols = ("text", 'head', 'POS', 'POS Def', 'DEP', 'DEP Def', 'Children')
    basic_info_rows = []

    for token in doc:
        children = []
        for t in token.children:
            children.append(t.text)
        row = [token.text, token.head, token.pos_, spacy.explain(token.pos_), token.dep_, spacy.explain(token.dep_), children]
        basic_info_rows.append(row)

    basic_doc_info = pd.DataFrame(basic_info_rows, columns=basic_info_cols)

    print(basic_doc_info.to_string())

    dep_info_cols = ("Text", 'head', 'children')
    dep_info_rows = []

    for token in doc:
        children = []
        for t in token.children:
            children.append(t.text)
        row = [token.text, token.head, children]
        dep_info_rows.append(row)

    dep_info_doc = pd.DataFrame(dep_info_rows, columns=dep_info_cols)

    print(dep_info_doc.to_string())




