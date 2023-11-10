from typing import List, Dict, Set, Tuple

import pandas as pd

from doenut.data import FilteredDataFrame


class DataFrameSet(FilteredDataFrame):
    def __init__(self, data: pd.DataFrame, responses: pd.DataFrame) -> None:
        super().__init__(data)
        if responses is None:
            raise ValueError("responses must not be null")
        if len(data) != len(responses):
            raise ValueError("Data and responses must have the same length")
        self.responses = FilteredDataFrame(responses)

    def filter_responses(self, selector: List[str]) -> "DataFrameSet":
        self.responses.filter(selector)
        return self

    def filter_responses_by_indices(
        self, indices: List[int]
    ) -> "FilteredDataFrame":
        self.responses.filter_by_indices(indices)
        return self

    def get_filtered_responses(self):
        return self.responses.get_filtered()

    def remove_duplicates(
        self, duplicates_dict: Dict[int, Set[int]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if duplicates_dict is None:
            duplicates_dict = self.get_duplicate_rows()
        return super().remove_duplicates(
            duplicates_dict
        ), self.responses.remove_duplicates(duplicates_dict)

    def average_duplicates(
        self, duplicates_dict: Dict[int, Set[int]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        duplicates_dict = self.get_duplicate_rows()
        return super().average_duplicates(
            duplicates_dict
        ), self.responses.average_duplicates(duplicates_dict)
