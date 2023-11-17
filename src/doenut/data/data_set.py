import pandas as pd


class DataSet:
    """
    A dataset that has had all it's modifiers applied.
    """

    def __init__(self, inputs, responses):
        self.inputs = inputs
        self.responses = responses

    def get_inputs(self) -> pd.DataFrame:
        return self.inputs

    def get_responses(self) -> pd.DataFrame:
        return self.responses
