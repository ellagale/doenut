import pandas as pd


class DataSet:
    """A dataset that has had all it's modifiers applied.

    Parameters
    ----------
    inputs : pd.DataFrame
        The dataset's inputs
    responses: pd.DataFrame
        The dataset's responses

    """

    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame):
        self.inputs = inputs
        self.responses = responses

    def get_inputs(self) -> pd.DataFrame:
        return self.inputs

    def get_responses(self) -> pd.DataFrame:
        return self.responses
