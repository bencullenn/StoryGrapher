'''
Created by Ben Cullen on May 13th 2020

This script will create graphs of a sentence's structure starting with the root word
and then traversing each of it's children's children.
This can be used to view the relationship between parts of a sentence.
'''

import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import time


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
        # Make sure sentences contain graphable text
        if re.search("[a-zA-Z]", sent.text) != None:
            sentences.append(sent)

    return sentences

def get_sliced_sentences(data_path, start_index, end_index):
    sentences = get_all_sentences(data_path)

    sliced_sentences = sentences[start_index:end_index + 1]

    print("Sentences sliced from ", start_index + 1, " to ", end_index)

    return  sliced_sentences

if __name__ == "__main__":

    # Add path for the data here
    data_path = ""
    save_path = ""

    # Use this code to graph all sentence in document.
    # Also can be changed to use get_sliced_sentence to graph specific sentences.
    sentences = get_all_sentences(data_path)

    # Create a graph for each sentence starting with the root and then traversing each of it's children
    sent_index = 0
    data_name = data_path.split('/')[-1]
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%y_%H:%M', t)

    for sent in sentences:
        print("Creating graph for sentence:", sent)
        G = nx.Graph()
        root = sent.root
        get_children(G, root)
            
        # Display the graph using matlibplot
        pos = nx.spring_layout(G)
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('Sentence Text: ' + str(sent))
        nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                node_size=1000, node_color='seagreen', alpha=0.9,
                labels={node: node for node in G.nodes()})

        plt.savefig(os.path.join(save_path + data_name + '_sentence_' + str(sent_index) + '_graph_'+timestamp+'.png'))

        sent_index += 1
