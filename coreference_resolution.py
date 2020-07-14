# Coreference Resolution Script

#%%
from allennlp.predictors.predictor import Predictor
import torch
from time import time


#%% Setup
path_in = "data/non_resolved/Ghost_Chimes.txt"
path_out = "data/resolved/resolved_ghost_chimes.txt"
ignore = {"her", "your", "their", "they", "his", "he", "she", "it", "its", "you", "we", "our", "us", "that", "this", "there", "where", "then", "when", "who", "what", "them"} # how why because
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
        if t not in ignore:
            name = t
            break
    if name is None:
        name = tokens[0]
    k = 0
    while name + str(k) in used_refs: k += 1
    used_refs.add(name + str(k))
    return name + str(k)

#%% Initialization
print("Cuda enabled:", torch.cuda.current_device() == 0)
print("Loading models...")
coref_model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
coref_predictor = Predictor.from_path(coref_model_url, cuda_device=torch.cuda.current_device())

#%%
with open(path_in) as f:
    text = f.read()

result = get_coref_prediction(coref_predictor, text)
clusters, result = result['clusters'], result['document'].copy()
refs = [get_name(c, result) for c in clusters]
# print(f"Coref Clusters: {clusters}")

for j, c in enumerate(clusters):
    ref = result[c[0][0]] # Name # TODO: make this unique by checking if in set
    for span in c:
        for m in range(span[0], span[1]+1):
            result[m] = None
        result[span[0]] = refs[j]
result = filter(lambda s: s is not None, result)
result = ' '.join(result)

with open(path_out) as f:
    f.write(result)