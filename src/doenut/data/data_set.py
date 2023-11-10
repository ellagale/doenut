from typing import List

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

    # def get_average
    # whole_inputs = inputs
    # averaged_responses = pd.DataFrame()
    # averaged_inputs = pd.DataFrame()
    #
    # duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index]
    # duplicates_for_averaging = {}
    # non_duplicate_list = [x for x in whole_inputs.index if x not in duplicates]
    # for non_duplicate in non_duplicate_list:
    #     this_duplicate_list = []
    #     non_duplicate_row = whole_inputs.loc[[non_duplicate]]
    #     for duplicate in duplicates:
    #         duplicate_row = whole_inputs.loc[[duplicate]]
    #         if non_duplicate_row.equals(duplicate_row):
    #             this_duplicate_list.append(duplicate)
    #             if verbose:
    #                 print(
    #                     f"found duplicate pairs: {non_duplicate}, {duplicate}"
    #                 )
    #     if len(this_duplicate_list) > 0:
    #         duplicates_for_averaging[non_duplicate] = this_duplicate_list
    #     else:
    #         averaged_inputs = pd.concat([averaged_inputs, non_duplicate_row])
    #         averaged_responses = pd.concat(
    #             [averaged_responses, responses.iloc[[non_duplicate]]]
    #         )
    #
    # for non_duplicate, duplicates in duplicates_for_averaging.items():
    #     # print(f"nd: {non_duplicate}")
    #     to_average = whole_inputs.loc[[non_duplicate]]
    #     to_average_responses = responses.loc[[non_duplicate]]
    #     for duplicate in duplicates:
    #         to_average = pd.concat([to_average, whole_inputs.loc[[duplicate]]])
    #         to_average_responses = pd.concat(
    #             [to_average_responses, responses.loc[[duplicate]]]
    #         )
    #     meaned = to_average.mean(axis=0)
    #     meaned_responses = to_average_responses.mean(axis=0)
    #     try:
    #         averaged_inputs = pd.concat(
    #             [averaged_inputs, pd.DataFrame(meaned).transpose()],
    #             ignore_index=True,
    #         )
    #         averaged_responses = pd.concat(
    #             [
    #                 averaged_responses,
    #                 pd.DataFrame(meaned_responses).transpose(),
    #             ],
    #             ignore_index=True,
    #         )
    #     except TypeError:
    #         averaged_inputs = pd.DataFrame(meaned).transpose()
    #         averaged_responses = pd.DataFrame(meaned_responses).transpose()
    #
    # return averaged_inputs, averaged_responses
    #
