# Coreference Resolution Script

#%%
from allennlp.predictors.predictor import Predictor
import torch
from time import time

print("Cuda enabled:", torch.cuda.current_device() == 0)
print("Loading models...")
coref_model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
coref_predictor = Predictor.from_path(coref_model_url, cuda_device=torch.cuda.current_device())

#%%
def get_coref_prediction(predictor, text):
    start_time = time()
    result = (predictor.predict(
        document=text
    ))
    delta = time() - start_time
    print("Coref Prediction Time: {:1.2f}".format(delta))
    print("Coref Result:", result)
    return result

#%%
path_in = "data/non_resolved/Ghost_Chimes.txt"
path_out = "data/resolved/resolved_ghost_chimes.txt"

#%%
# with open(path_in) as f:
#     text = f.read()
text = "Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their high school, Lakeside, to develop their programming skills on several time-sharing computer systems."

spans = get_coref_prediction(coref_predictor, text)
print(spans)

# # DO STUFF
# result = text

# with open(path_out) as f:
#     f.write(result)