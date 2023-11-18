from typing import Dict, Iterable, List
import pandas as pd
from doenut.data.modifiers.duplicate_remover import DuplicateRemover


class DuplicateAverager(DuplicateRemover):
    """
    Parses a dataset and removes all but the _first_ instance of any row that
    has duplicate values for the _inputs_. Will also remove the corresponding
    row in the responses, replacing the remaining response with the averages
    of the duplicates' values.
    """

    @classmethod
    def _apply(
        cls,
        data: pd.DataFrame,
        duplicate_dict: Dict[int, Iterable[int]],
        non_duplicate_rows: List[int],
    ) -> pd.DataFrame:
        # first build a copy of the data with the duplicates removed
        results = data.iloc[non_duplicate_rows].copy(True)
        # now figure out the averages for the ones that need averaging
        for idx, dupes in duplicate_dict.items():
            to_average = [data.iloc[dupe] for dupe in dupes]
            to_average.append(results.iloc[idx])
            results.iloc[idx] = pd.concat(to_average, axis=1).T.mean(axis=0)
        return results

    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame) -> None:
        """
        This modifier will remove all rows from the dataset which have
        identical values for the _inputs_, and set the response value to be
         the average of all the duplicates. The first instance in the dataset
        of a given set of values will be the one retained.
        @param inputs: The inputs of the dataset
        @param responses: The responses of the dataset
        """
        super().__init__(inputs, responses)

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(data, self.duplicate_dict, self.non_duplicate_rows)

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(data, self.duplicate_dict, self.non_duplicate_rows)
