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
        # pd has a nasty habit of converting single column df into series
        if isinstance(inputs, pd.Series):
            self.inputs = inputs.to_frame(name="inputs")
        if isinstance(responses, pd.Series):
            self.responses = responses.to_frame(name="responses")

    def get_inputs(self) -> pd.DataFrame:
        return self.inputs

    def get_responses(self) -> pd.DataFrame:
        return self.responses
