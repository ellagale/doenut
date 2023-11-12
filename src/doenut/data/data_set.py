import pandas as pd

from doenut.data.data_set_modifier import DataSetModifier


class DataSet(DataSetModifier):
    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame) -> None:
        super().__init__()
        if inputs is None or len(inputs) == 0:
            raise ValueError("Inputs must not be empty")
        if responses is None or len(responses) == 0:
            raise ValueError("Responses must not be empty")
        if len(inputs) != len(responses):
            raise ValueError("Inputs and Responses must have the same length")

        self.inputs = inputs
        self.responses = responses

    def get_inputs(self) -> pd.DataFrame:
        # Override the base class to just hand the raw data
        return self.inputs

    def get_responses(self) -> pd.DataFrame:
        # override the base class to just hand the raw data
        return self.responses

    def get_raw_inputs(self) -> pd.DataFrame:
        return self.inputs

    def get_raw_responses(self) -> pd.DataFrame:
        return self.responses

    def _parse_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        # Base class does nothing.
        return data

    def _parse_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        # Base class does nothing.
        return data