
"""
Created by Ben Cullen on May 13th 2020
"""

import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt

def getSentences(text):
    # Create a Spacy object called npl
    nlp = spacy.load('en_core_web_sm')
    # Add a pipeline object to it that will split up sentences and create a Spacy document object
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    sentences = []
    for sent in document.sents:
        sentences.append(sent)
    return sentences

def append_word(original, word):
    return original + ' ' + word

def get_children(token):
    children = []
    for child in token.children:
        # Add each child node that is not punctuation or spacing to the children array
        if child.is_punct is False and child.is_space is False:
            children.append(child)
    return children

def create_triple(sent):
    print("Creating triple for sentence:", sent, " with root:", sent.root)
    root = sent.root
    subject = ''
    object = ''

    # Get the children of the root node
    children = get_children(root)
    print("The root's children are", children, " and their dependencies are.")
    for child in children:
        print(child.text, " is a ", child.dep_)

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
    else:
        print("Could not create a meaningful triple for this sentence")


def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        print('Graphing triple:', triple)
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(45, 45))
    nx.draw(G, pos, edge_color='black', width=1, linewidths=2,
            node_size=2000, node_color='seagreen', labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    # Add your filepath here
    text_filepath = ""

    # Open up the story file and use the read function to convert the text to a string
    with open(text_filepath) as f:
        text = f.read()

    sentences = getSentences(text)
    triples = []

    for sent in sentences:
        triple = create_triple(sent)
        if triple is not None:
            triples.append(create_triple(sent))

    printGraph(triples)
    printGraph(triples)
    printGraph(triples)
