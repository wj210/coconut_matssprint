import kagglehub

# Download latest version
path = kagglehub.model_download("victorumesiobi/gemma-2-japanese-english-reasoning/transformers/1")

print("Path to model files:", path)