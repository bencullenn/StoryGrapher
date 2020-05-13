'''
Created by Ben Cullen on May 13th 2020

This script will create graphs of a sentence's structure starting with the root word
and then traversing each of it's children's children.
'''

import spacy
import sys
from spacy.lang.en import English
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def getChildren(graph, current, parent=None):
    print("Current:", current, " Parent:", parent)
    children = []
    for t in current.children:
        # Add each child node that is not punctuation or spacing to the children array
        if t.is_punct is False and t.is_space is False:
            children.append(t)

    print("Children:", children)
    if len(children) == 0:
        if parent is not None:
            print("Connecting", current, " and ", parent)
            current_label = "\"" + current.text + "\"" + ":" + spacy.explain(current.pos_)
            parent_label = "\"" + parent.text + "\"" + ":" + spacy.explain(parent.pos_)

            """"
            if spacy.explain(current.dep_) is not None:
                current_label += ":" + spacy.explain(current.dep_)
            if spacy.explain(parent.dep_) is not None:
                parent_label += ":" + spacy.explain(parent.dep_)
            """

            graph.add_edge(current_label, parent_label)
        return graph
    else:
        for child in children:
            if parent is not None:
                print("Connecting", current, " and ", parent)
                current_label = "\"" + current.text + "\"" + ":" + spacy.explain(current.pos_)
                parent_label = "\"" + parent.text + "\"" + ":" + spacy.explain(parent.pos_)

                """"
                if spacy.explain(current.dep_) is not None:
                    current_label += ":" + spacy.explain(current.dep_)
                if spacy.explain(parent.dep_) is not None:
                    parent_label += ":" + spacy.explain(parent.dep_)
                """

                graph.add_edge(current_label, parent_label)
            getChildren(graph, child, current)


if __name__ == "__main__":
    start_index = 0
    end_index = 5
    text_filepath = "Ghost_Chimes.txt"
    sentences = []

    # Create a Spacy object and the sentencizer to the pipeline
    nlp = English()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    # Open the file for the text that we want to read
    with open(text_filepath) as f:
        text = f.read()

    # Create a spacy doc object to store information about the text
    doc = nlp(text)

    # Get the desired sentence indexes
    for sent in doc.sents:
        print(sent)
        sentences.append(sent)

    np.set_printoptions(threshold=sys.maxsize)

    sliced_sentences = sentences[start_index:end_index + 1]

    print("The sentences to be graphed are:")
    print(sliced_sentences)

    # Create a graph for each sentence starting with the root and then traversing each of it's children
    for sent in sliced_sentences:
        G = nx.Graph()
        root = sent.root
        getChildren(G, root)

        # Display the graph using matlibplot
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                node_size=1000, node_color='seagreen', alpha=0.9,
                labels={node: node for node in G.nodes()})
        plt.axis('off')
        plt.show()
