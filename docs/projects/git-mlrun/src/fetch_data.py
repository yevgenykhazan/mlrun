import mlrun
@mlrun.handler(outputs=["dataset","label_column"])
def fetch_data(dataset):
    """
    A function which fetches data to MLRun
    """
    df = dataset.as_df()
    
    return df, "label"
