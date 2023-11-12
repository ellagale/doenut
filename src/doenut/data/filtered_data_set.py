from typing import Iterable, List
import pandas as pd

from doenut.data import FilteredDataFrame
from doenut.data.DataSet import DataSet
from doenut.data.DataSetModifier import DataSetModifier


class FilteredDataSet(DataSetModifier):
    def __init__(self,
                 data: DataSet,
                 input_selector: Iterable[str|int],
                 response_selector: Iterable[str|int] = None
                 ):
        super().__init__(data)
        if input_selector is None:
            raise ValueError("Input selector must select at least one column!")
        self.input_selector = 





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