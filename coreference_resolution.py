# Coreference Resolution Script
""" SCRIPT USAGE:
    ALWAYS outputs to data/resolved!
    No argument: automatically runs through all files in data/non_resolved.
    Filename as argument: only runs for the given file found in data/non_resolved.
"""

# NOTE/ TODO: we're losing some data with belonging and and's with current replacement scheme.


#%% Imports
from allennlp.predictors.predictor import Predictor
import torch
from time import time
from sys import argv
from os import listdir
from os.path import isfile, join
import allennlp_models.coref
import spacy
import math

IGNORE = {
    "a", "an", "the", "that", "this", "another",
    "her",  "him", "she", "he", "it",  "we",  "us",   "you",           "they",  "them",
    "hers", "his",              "its", "our", "ours", "your", "yours", "their", "theirs",
    "where", "there", "when", "then", "who", "what", "whose",
} # how why because



#%% Setup
path_in = "data/non_resolved/"
if __name__ == "__main__" and len(argv) > 1:
    paths = [argv[1]]
else: paths = [f for f in listdir(path_in) if isfile(join(path_in, f))]
path_out = "data/resolved/resolved_"
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

print("Generating Spacy...")
spacy_sent = spacy.load('en_core_web_sm')
spacy_sent.add_pipe(spacy_sent.create_pipe('sentencizer'))

#%%
print("PATHS DEBUG:", paths)
for p in paths:
    print("Processing file ", p)
    with open(path_in + p) as f:
        text = f.read()
    
    doc = spacy_sent(text)
    chunks = chunkText(doc)
    total_result = ""
    results = []
    chunkIndex = 1
    for chunk in chunks:
        print("Generating result for chunk ", chunkIndex, " of ", len(chunks))
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
        
    total_result = total_result.join(results)
    with open(path_out + p, "w+") as f:
        f.write(total_result)

    print("\n")
