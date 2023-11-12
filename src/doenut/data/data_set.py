from typing import Iterable
import pandas as pd

from doenut.data import FilteredDataFrame


class DataSet():
    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame) -> None:
        if inputs is None or len(inputs) == 0:
            raise ValueError("Inputs must not be empty")
        if responses is None or len(responses) == 0:
            raise ValueError("Responses must not be empty")
        if len(inputs) != len(responses):
            raise ValueError("Inputs and Responses must have the same length")

        self.inputs = FilteredDataFrame(inputs)
        self.responses = FilteredDataFrame(responses)

    def set_selector(self, selector: Iterable[str]) -> "DataSet":
        self.inputs.set_filter(selector)
        return self

    def set_selector_by_indices(self, indices: Iterable[int]) -> "DataSet":
        self.inputs.filter_by_indices(indices)
        return self

    def set_response_selector(self, selector: Iterable[str]) -> "DataSet":
        self.responses.set_filter(selector)
        return self

    def set_response_selector_by_indices(self, indices: Iterable[int]) -> "DataSet":
        self.responses.filter_by_indices(indices)
        return self

    def get_filtered_inputs(self) -> pd.DataFrame:
        return self.inputs.get()

    def get_filtered_responses(self) -> pd.DataFrame:
        return self.responses.get()

    def get_inputs_without_duplicates(self) -> pd.DataFrame:
        return self.inputs.get_without_duplicates()

    def get_responses_without_duplicates(self) -> pd.DataFrame:
        duplicate_dict = self.inputs.get_duplicate_rows()
        return self.responses.get_without_duplicates(duplicate_dict)

    def get_inputs_with_averaged_duplicates(self) -> pd.DataFrame:
        return self.inputs.get_with_average_duplicates()

    def get_responses_with_averaged_duplicates(self) -> pd.DataFrame:
        duplicate_dict = self.inputs.get_duplicate_rows()
        return self.responses.get_without_duplicates(duplicate_dict)