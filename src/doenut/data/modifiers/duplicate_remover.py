from typing import TYPE_CHECKING, Tuple, Dict, Set, Iterable, List

import pandas as pd

from doenut.data.modifiers.data_set_modifier import DataSetModifier

if TYPE_CHECKING:
    from doenut.data.data_set import DataSet


class DuplicateRemover(DataSetModifier):
    @classmethod
    def _get_duplicate_rows(cls, data: pd.DataFrame) -> Dict[int, Set[int]]:
        duplicates = [x for x in data[data.duplicated()].index]
        non_duplicates = [x for x in data.index if x not in duplicates]
        results = {}
        for duplicate in duplicates:
            # find first row it is a dupe of
            for index in data.index:
                duplicate_row = data.iloc[duplicate]
                if index >= duplicate:
                    # pandas ensures the duplicate is later in the
                    # dataframe than the row it is a duplicate of.
                    raise OverflowError(
                        f"Duplicate is missing for {duplicate}"
                    )
                if data.iloc[index].equals(duplicate_row):
                    try:
                        results[index].add(duplicate)
                    except KeyError:
                        results[index] = {duplicate}
                    break
                return results

    @classmethod
    def _get_non_duplicate_rows(
        cls,
        data: pd.DataFrame,
        duplicates_dict: Dict[int, Iterable[int]] = None,
    ) -> List[int]:
        if duplicates_dict is None:
            # assume we are removing according to this dataset
            duplicates_dict = cls._get_duplicate_rows(data)

        # build the list of rows we don't want.
        duplicate_indices = set()
        for duplicate_set in duplicates_dict.values():
            duplicate_indices = duplicate_indices.union(duplicate_set)
        non_duplicates = [x for x in data.index if x not in duplicate_indices]
        return non_duplicates

    def __init__(self, data: "DataSet") -> None:
        super().__init__(data)
        # use input data to determine which rows are duplicates
        input_data = data.get_inputs()
        self.duplicate_dict = self._get_duplicate_rows(input_data)
        self.non_duplicate_rows = self._get_non_duplicate_rows(
            input_data, self.duplicate_dict
        )

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.iloc[self.non_duplicate_rows]

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.iloc[self.non_duplicate_rows]
