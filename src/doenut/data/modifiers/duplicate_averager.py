from typing import TYPE_CHECKING, Tuple, Dict, Set, Iterable, List

import pandas as pd

from doenut.data.modifiers.duplicate_remover import DuplicateRemover

if TYPE_CHECKING:
    from doenut.data.data_set import DataSet


class DuplicateAverager(DuplicateRemover):
    @classmethod
    def _apply(
        cls,
        data: pd.DataFrame,
        duplicate_dict: Dict[int, Iterable[int]],
        non_duplicate_rows: List[int],
    ) -> pd.DataFrame:
        # first build a copy of the data with the duplicates removed
        results = data.iloc[non_duplicate_rows].copy()
        # now figure out the averages for the ones that need averaging
        for idx, dupes in duplicate_dict.items():
            to_average = [data.iloc[dupe] for dupe in dupes]
            to_average.append(results.iloc[idx])
            results.iloc[idx] = pd.concat(to_average).mean(axis=0)
        return results

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(data, self.duplicate_dict, self.non_duplicate_rows)

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(data, self.duplicate_dict, self.non_duplicate_rows)
