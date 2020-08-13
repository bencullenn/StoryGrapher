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

def chunkText(text):
    spacy_sent = spacy.load('en_core_web_sm')
    spacy_sent.add_pipe(spacy_sent.create_pipe('sentencizer'))
    print("Generated Spacy")

    doc = spacy_sent(text)
    sentences = []
    chunks = []
    sent_num = 0
    max_sent_per_chunk = 250
    
    for sent in doc.sents:
        sentences.append(sent)
        sent_num += 1
    
    chunk_num = math.ceil(sent_num/max_sent_per_chunk)
    chunk_size = math.ceil(sent_num/chunk_num)
    print(sent_num, " sentences were extracted. Splitting into ", chunk_num, " chunks of ", chunk_size, " sentences each.") 
    
    slice_index = 0
    for i in range(chunk_num):
        chunk = []
        
        if slice_index + chunk_size < len(sentences):
            chunk = sentences[slice_index:slice_index + chunk_size]
            slice_index += chunk_size
        else:
            chunk = sentences[slice_index:-1]

        text = ""
        for sent in chunk:
            text += sent.text
        
        chunks.append(text)

    print("Quality Check")
    
    chunk_index = 1
    for chunk in chunks:
        print("Chunk #", chunk_index, " has an length of ", len(chunk))

    return chunks   


#%% Initialization
print("Cuda enabled:", torch.cuda.current_device() == 0)
print("Loading models...")
coref_model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
coref_predictor = Predictor.from_path(coref_model_url, cuda_device=torch.cuda.current_device())

#%%
print("PATHS DEBUG:", paths)
for p in paths:
    with open(path_in + p) as f:
        text = f.read()
    
    chunks = chunkText(text)
    total_result = ""
    results = []
    for chunk in chunks:
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
