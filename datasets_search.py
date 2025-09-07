import openml

def search_datasets(query, limit=5):
    try:
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        matches = datasets[datasets['name'].str.contains(query, case=False, na=False)]
        return matches['name'].head(limit).tolist()
    except Exception as e:
        return [f"Error: {e}"]
