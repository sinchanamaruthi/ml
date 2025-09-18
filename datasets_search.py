import openml
import pandas as pd
from datasets import load_dataset

def load_openml_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.concat([X, y], axis=1)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def load_huggingface_dataset(name):
    try:
        dataset = load_dataset(name)
        df = dataset['train'].to_pandas()
        return df.head(2000)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
