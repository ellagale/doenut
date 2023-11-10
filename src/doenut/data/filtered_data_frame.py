from typing import List

import pandas as pd


class FilteredDataFrame:
    """
    A piece of data, that may have a selector applied to filter it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if data is None:
            raise ValueError("Data must not be null")
        self.data = data
        self.selector = None
        self.indices = None

    def filter(self, selector: List[str]) -> "FilteredDataFrame":
        """
        Defines the filter of what columns we want to use from the data.
        @param selector: List of colunn names to filter with
        @return: self for use as a builder
        """
        # First validate it
        for col in selector:
            if not col in self.data.columns:
                raise ValueError(f"Data lacks column {col}")
        self.selector = selector
        self.indices = [
            i for i, j in enumerate(self.data.columns) if j in self.selector
        ]
        return self

    def filter_by_indices(self, indices: List[int]) -> "FilteredDataFrame":
        """
        Defines the filter of what columsn we want to use from the data
        @param indices: list of column indices to filter with
        @return: self for use as a builder
        """
        # first validate it
        for idx in indices:
            if idx >= len(self.data):
                raise IndexError(f"Index {idx} is out of range for data")
        self.indices = indices
        self.selector = [self.data.columns[i] for i in self.indices]
        return self

    def get_filtered(self) -> pd.DataFrame:
        """
        Get a (shallow) copy of the data, applying the filter (if present)
        @return: the data, filtered
        """
        if not self.indices:
            return self.data
        return self.data.iloc[:, self.indices]