'''
Created by Ben Cullen on May 13th 2020

This script will create graphs of a sentence's structure starting with the root word
and then traversing each of it's children's children.
This can be used to view the relationship between parts of a sentence.
'''

import spacy
import sys
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt


def get_children(graph, current, parent=None):
    children = []
    for t in current.children:
        # Add each child node that is not punctuation or spacing to the children array
        if t.is_punct is False and t.is_space is False:
            children.append(t)

    if len(children) == 0:
        if parent is not None:
            current_label = "\"" + current.text + "\"" + ":" + current.pos_
            parent_label = "\"" + parent.text + "\"" + ":" + parent.pos_

            if current.dep_ is not None:
                current_label += ":" + current.dep_
            if parent.dep_ is not None:
                parent_label += ":" + parent.dep_

            graph.add_edge(current_label, parent_label)
        return graph
    else:
        for child in children:
            if parent is not None:
                current_label = "\"" + current.text + "\"" + ":" + current.pos_
                parent_label = "\"" + parent.text + "\"" + ":" + parent.pos_

                if current.dep_ is not None:
                    current_label += ":" + current.dep_
                if parent.dep_ is not None:
                    parent_label += ":" + parent.dep_

                graph.add_edge(current_label, parent_label)
            get_children(graph, child, current)


if __name__ == "__main__":
    # Add your filepath here
    text_filepath = ""
    start_index = 0
    end_index = 20
    sentences = []

    # Create a Spacy object and the sentencizer to the pipeline
    nlp = English()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    # Open the file for the text that we want to read
    with open(text_filepath) as f:
        text = f.read()

    # Create a Spacy doc object to store information about the text
    doc = nlp(text)

    # Get the desired sentence indexes
    for sent in doc.sents:
        sentences.append(sent)

    sliced_sentences = sentences[start_index:end_index + 1]

    print("The sentences to be graphed are sentences ", start_index + 1, " to ", end_index)

    # Create a graph for each sentence starting with the root and then traversing each of it's children
    for sent in sliced_sentences:
        print("Creating graph for sentence:", sent)
        G = nx.Graph()
        root = sent.root
        get_children(G, root)

        # Display the graph using matlibplot
        pos = nx.spring_layout(G)
        plt.figure(figsize=(20, 20))
        nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                node_size=1000, node_color='seagreen', alpha=0.9,
                labels={node: node for node in G.nodes()})
        plt.axis('off')
        plt.show()
