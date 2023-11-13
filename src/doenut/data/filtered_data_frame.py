from typing import List, Dict, Set, Iterable

import pandas as pd


class FilteredDataFrame:
    """
    A piece of data, that may have a selector applied to set_filter it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if data is None:
            raise ValueError("Data must not be null")
        self.data = data
        self.selector = None
        self.indices = None

    def set_filter(self, selector: Iterable[str]) -> "FilteredDataFrame":
        """
        Defines the set_filter of what columns we want to use from the data.
        @param selector: List of colunn names to set_filter with
        @return: self for use as a builder
        """
        # First validate it
        for col in selector:
            if col not in self.data.columns:
                raise ValueError(f"Data lacks column {col}")
        self.selector = selector
        self.indices = [
            i for i, j in enumerate(self.data.columns) if j in self.selector
        ]
        return self

    def filter_by_indices(self, indices: Iterable[int]) -> "FilteredDataFrame":
        """
        Defines the set_filter of what columsn we want to use from the data
        @param indices: list of column indices to set_filter with
        @return: self for use as a builder
        """
        # first validate it
        for idx in indices:
            if idx >= len(self.data):
                raise IndexError(f"Index {idx} is out of range for data")
        self.indices = indices
        self.selector = [self.data.columns[i] for i in self.indices]
        return self

    def get(self) -> pd.DataFrame:
        """
        Get a view of the data, applying the set_filter (if present)
        @return: the data, filtered
        """
        if not self.indices:
            return self.data
        return self.data.iloc[:, self.indices]

    def get_duplicate_rows(self) -> Dict[int, Set[int]]:
        """
        Finds which rows are duplicated in the data frame.
        @return:A set of sets of duplicate row's indices,
        """
        whole_inputs = self.get()
        duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index]
        non_duplicate_list = [
            x for x in whole_inputs.index if x not in duplicates
        ]
        results = {}
        for duplicate in duplicates:
            # find the first row that is a duplicate of
            for index in self.data.index:
                duplicate_row = self.data.iloc[duplicate]
                if index >= duplicate:
                    # pandas ensures the duplicate is later in the
                    # dataframe than the row it is a duplicate of.
                    raise OverflowError(
                        f"Duplicate is missing for {duplicate}"
                    )
                if self.data.iloc[index].equals(duplicate_row):
                    try:
                        results[index].add(duplicate)
                    except KeyError:
                        results[index] = {duplicate}
                    break
        return results

    def _get_non_duplicate_rows(
        self, duplicates_dict: Dict[int, Iterable[int]] = None
    ) -> List[int]:
        if duplicates_dict is None:
            # assume we are removing according to this dataset
            duplicates_dict = self.get_duplicate_rows()

        # build the list of rows we don't want.
        duplicate_indices = set()
        for duplicate_set in duplicates_dict.values():
            duplicate_indices = duplicate_indices.union(duplicate_set)
        non_duplicates = [
            x for x in self.data.index if x not in duplicate_indices
        ]
        return non_duplicates

    def get_without_duplicates(
        self, duplicates_dict: Dict[int, Iterable[int]] = None
    ) -> pd.DataFrame:
        """
        Returns a view of the data with duplicate rows having their values removed
        The lowest indexed instance of each duplicate set will be retained.

        @param duplicates_dict: Optional Set of Sets of indices of duplicates.
        @return: the data, with all bar one copy of each duplicate removed
        """
        non_duplicates = self._get_non_duplicate_rows(duplicates_dict)
        return self.get().iloc[non_duplicates]

    def remove_duplicates(
        self, duplicates_dict: Dict[int, Iterable[int]] = None
    ) -> None:
        non_duplicates = self._get_non_duplicate_rows(duplicates_dict)
        self.data = self.data.iloc[non_duplicates]

    def get_with_average_duplicates(
        self, duplicates_dict: Dict[int, Iterable[int]] = None
    ) -> pd.DataFrame:
        """
        Returns a view of the data with duplicate rows having their values averaged.
        The expectation is this list of duplicates is coming from a different DataFrame
        @param duplicates_dict: dict of sets of duplicates, keyed by the index
        of the first copy of that duplicate
        @return: the dataframe, with each set of duplicate rows replaced with
        a single row whose value is the average of those it is replacing.
        """
        # first build a copy of the data with the duplicates removed
        results = self.get_without_duplicates(duplicates_dict).copy()

        # now figure out the averages for the ones that need averaging
        filtered_data = self.get()
        for idx, dupes in duplicates_dict.items():
            to_average = [filtered_data.iloc[dupe] for dupe in dupes]
            to_average.append(results.iloc[idx])
            results.iloc[idx] = pd.concat(to_average).mean(axis=0)

        return results
