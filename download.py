import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_extended.csv")
df.to_csv("data/gpqa_extended.csv", index=False)

