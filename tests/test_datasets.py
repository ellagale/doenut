import pandas as pd
import os

from doenut.data import DataSet, FilteredDataSet

os.chdir(os.path.dirname(__file__))
df = pd.read_csv("solar_cells_1.csv")
inputs = pd.DataFrame(
    {
        "Donor %": [float(x) for x in df.iloc[1:-1, 1]],
        "Conc.": [float(x) for x in df.iloc[1:-1, 2]],
        "Spin": [float(x) for x in df.iloc[1:-1, 3]],
        "Add.": [float(x) for x in df.iloc[1:-1, 4]],
    }
)

responses = pd.DataFrame({"PCE": [float(x) for x in df["PCE"][1:-1]]})


def test_dataset():
    x = DataSet(inputs, responses)
    assert x.get_inputs().equals(inputs)
    assert x.get_responses().equals(responses)
    assert x.get_raw_inputs().equals(inputs)
    assert x.get_raw_responses().equals(responses)


def test_filtered_dataset():
    x = DataSet(inputs, responses)
    y = FilteredDataSet(x, [0,1,3])
    assert list(y.get_inputs().columns) == ['Donor %', "Conc.", "Add."]
    assert y.get_responses().equals(responses)
    assert y.get_raw_inputs().equals(inputs)
    assert y.get_raw_responses().equals(responses)
