from doenut.data import FilteredDataFrame


class DataSet:
    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame) -> None:
        if inputs is None or len(inputs) == 0:
            raise ValueError("Inputs must not be empty")
        if responses is None or len(responses) == 0:
            raise ValueError("Responses must not be empty")
        if len(inputs) != len(responses):
            raise ValueError("Inputs and Responses must have the same length")

        self.inputs = FilteredDataFrame(inputs)
        self.responses = FilteredDataFrame(responses)