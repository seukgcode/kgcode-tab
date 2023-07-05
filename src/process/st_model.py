from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cuda", cache_folder="./.models")
print(model, model._target_device)