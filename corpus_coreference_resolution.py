# Coreference Resolution Script
""" SCRIPT USAGE:
    ALWAYS outputs to data/resolved!
    This script is specifically designed to be used with the a corpus which each line is a new sentence. 
    For text data that is not formatted like this use the other corefernece resolution script 
"""

# NOTE/ TODO: we're losing some data with belonging and and's with current replacement scheme.


#%% Imports
from allennlp.predictors.predictor import Predictor
import torch
from time import time
from sys import argv
from os import listdir, stat
from os.path import isfile, join
import allennlp_models.coref
import spacy
import math
import re

IGNORE = {
    "a", "an", "the", "that", "this", "another",
    "her",  "him", "she", "he", "it",  "we",  "us",   "you",           "they",  "them",
    "hers", "his",              "its", "our", "ours", "your", "yours", "their", "theirs",
    "where", "there", "when", "then", "who", "what", "whose",
} # how why because

# Keywords that would indicate line is a title so we know to exclude them
TITLE_INDICATORS = { "copyright", "isbn" }

#% Setup
# Load and save loction for test stories
"""
path_in = "data/non_resolved/"
if __name__ == "__main__" and len(argv) > 1:
    paths = [argv[1]]
else: paths = [f for f in listdir(path_in) if isfile(join(path_in, f))]
path_out = "data/resolved/resolved_"
"""
# Load data from books corpus
path_in = "/mnt/pccfs/not_backed_up/data/bookcorpus/"
paths = ["books_large_total.txt"]
path_out = "data/resolved/"

used_refs = set()

def get_coref_prediction(predictor, text):
    start_time = time()
    result = (predictor.predict(
        document=text
    ))
    delta = time() - start_time
    print("Coref Prediction Time: {:1.2f}".format(delta))
    return result

# This function finds something to call each reference
def get_name(cluster, document):
    tokens = []
    for span in cluster:
        for i in range(span[0], span[1]+1):
            tokens.append(document[i])
    name = None
    for t in tokens:
        if t and t.lower() not in IGNORE:
            name = t
            break
    if name is None:
        name = tokens[0]
    k = 0
    while name + str(k) in used_refs: k += 1
    used_refs.add(name + str(k))
    return name + str(k)

def chunkText(doc):
    print("Chunking Text...")
    sentences = []
    chunks = []
    text_size = 0
    max_size_per_chunk = 10000
    
    for sent in doc.sents:
        sentences.append(sent)
        text_size += len(sent.text.encode('utf-8'))
    
    chunk_num = math.ceil(text_size/max_size_per_chunk)
    print("The length of text was ", text_size,  ". Splitting into ", chunk_num, " chunks of max length ", max_size_per_chunk) 
    
    index = 0
    for i in range(chunk_num):
        print("Processing chunk ", i+1)
        chunk = []
        text = ""
        chunk_size = 0

        while chunk_size <= max_size_per_chunk: 
            # Make sure index is in bounds
            if index >= len(sentences):
                #print("End of sentence array reached. Current chunk size is ", len(text.encode('utf-8')))
                chunks.append(text)
                break
            else:
                sent = sentences[index].text
            
            #print("Chunk Size ", chunk_size, " Sent Size ", len(sent.encode('utf-8')), "Index ", index, "Array Length ", len(sentences))
            
            # Make sure the chunk is below the max size
            if chunk_size + len(sent.encode('utf-8')) >= max_size_per_chunk: 
                #print("Max chunk size reached. Current chunk size is ", len(text.encode('utf-8')))
                chunks.append(text)
                break
            else:
                #print("Adding sent to chunk")
                text += sent
                chunk_size += len(sent.encode('utf-8'))
                index += 1

    print("Quality Check")
    
    chunk_index = 1
    for chunk in chunks:
        print("Chunk #", chunk_index, "of ", len(chunks), " has an length of ", len(chunk), " and size ", len(chunk.encode('utf-8')))
        chunk_index += 1
    
    return chunks   


#%% Initialization
print("Cuda enabled:", torch.cuda.current_device() == 0)
print("Loading models...")
print("Generating Coreference Model...")
coref_model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
coref_predictor = Predictor.from_path(coref_model_url, cuda_device=torch.cuda.current_device())

"""
print("Generating Spacy...")
spacy_sent = spacy.load('en_core_web_sm')
spacy_sent.add_pipe(spacy_sent.create_pipe('sentencizer'))
"""

#%%
print("PATHS DEBUG:", paths)
for p in paths:
    print("Processing file ", p)
    print(f"File Size: {stat(path_in + p).st_size / (1024 * 1024 * 1024)} GB")
    

    """
    line_count = 0

    for line in open(path_in + p):
        line_count += 1
        print("Line:", line_count)

    print("Line count:", line_count)

    exit()
    """

    # Iterate through each line of the file
    # Check that max text size is not exceeded
    # Continue until max text size is reached
    # Run all sentences through spacy as a doc and process for coreference resoltyion 

    
    file = open(path_in +p, "r")
    line_index = 1
    #line_count = len(lines)
    chunk_text = ""
    max_chunk_size = 10000
    chunk_size = 0
    #placeholder_line = "\n***************************************************************\n"
    
    for line in file:
        print("Line ", line_index, ":", line)

        sentence_size = len(line.encode('utf-8'))

        if chunk_size + sentence_size < max_chunk_size:
            if any(([True if indicator in line else False for indicator in TITLE_INDICATORS])):
                print("Line was not added to chunk because it is likely a title or non-story text")
                #print("Inserting placeholder text")
                #chunk_size += len(placeholder_line.encode('utf-8'))
                #chunk_text += placeholder_line
            else:
                print("Chunk Size: ", chunk_size, " is less than max size of ", max_chunk_size)
                print("Adding sentence to chunk")
                chunk_text += line
                chunk_size += sentence_size
        else:
            print("Chunk max size reached")

            print("Creating prediction")
            results = []
            total_result = ""
            result = get_coref_prediction(coref_predictor, chunk_text)
            clusters, result = result['clusters'], result['document'].copy()
            refs = [get_name(c, result) for c in clusters]
            # print(f"Coref Clusters: {clusters}")

            for j, c in enumerate(clusters):
                for span in c:
                    for m in range(span[0], span[1]+1):
                        result[m] = None
                    result[span[0]] = refs[j]
            result = filter(lambda s: s is not None, result)
            result = ' '.join(result)
            results.append(result)
        
            print("Saving results to file...")
            total_result = total_result.join(results)
            with open(path_out + p, "a") as f:
                f.write(total_result)

            #Reset data structures
            chunk_size = 0
            chunk_text = ""

            chunk_text += line
            chunk_size += sentence_size

        line_index += 1

    print("End of file reached")
    exit()


    """
    with open(path_in + p) as f:
        text = f.read()
    print("Opened file successfully")

    print("Splitting into senteneces using spacy")
    doc = spacy_sent(text)
    print("Finished splitting into sentences")
    
    print("Chunking text...")
    chunks = chunkText(doc)
    print("Finished chunking text")
    
    total_result = ""
    results = []
    chunkIndex = 1
    # After how many chunks should files be written to the save file
    saveFrequency = 5

        Vprint("Generating result for chunk ", chunkIndex, " of ", len(chunks))
        chunkIndex += 1

        result = get_coref_prediction(coref_predictor, chunk)
        clusters, result = result['clusters'], result['document'].copy()
        refs = [get_name(c, result) for c in clusters]
        # print(f"Coref Clusters: {clusters}")

        for j, c in enumerate(clusters):
            for span in c:
                for m in range(span[0], span[1]+1):
                    result[m] = None
                result[span[0]] = refs[j]
        result = filter(lambda s: s is not None, result)
        result = ' '.join(result)
        results.append(result)
        
        # Check the current index against the save frequency
        if chunkIndex % saveFrequency  == 0:
            print("Saving file at index ", chunkIndex)
            total_result = total_result.join(results)
            with open(path_out + p, "a") as f:
                f.write(total_result)

            total_result = ""
            results = []
            
        # If we have reached the end of the chunks before reaching the next save frequency 
        elif chunkIndex + 1 == len(chunks):
            print("End of chunks reached. Saving file...")
            total_result = total_result.join(results)
            with open(path_out + p, "a") as f:
                f.write(total_result)

    print("\n")
    """
