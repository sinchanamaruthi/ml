import openml
import pandas as pd

def load_dataset(name):
    try:
        dataset = openml.datasets.get_dataset(name)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        return pd.concat([X, y], axis=1)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})
