import numpy as np
import pandas as pd
import os

from doenut.data import DataSet

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
    y = DataSet(inputs, responses).filter([0, 1, 3])
    assert list(y.get_inputs().columns) == ["Donor %", "Conc.", "Add."]
    assert y.get_responses().equals(responses)
    assert y.get_raw_inputs().equals(inputs)
    assert y.get_raw_responses().equals(responses)
    z = DataSet(inputs, responses).filter(["Donor %", "Conc.", "Add."])
    assert z.get_inputs().equals(y.get_inputs())


def test_scaled_dataset():
    x = DataSet(inputs, responses).scale()
    xi = x.get_inputs()
    xr = x.get_inputs()
    assert np.min(xi) >= -1
    assert np.max(xi) <= 1
    assert np.min(xr) >= -1
    assert np.max(xr) <= 1


def test_scaled_filtered_ordering():
    x = DataSet(inputs, responses).scale().filter([0, 1, 3])
    y = DataSet(inputs, responses).filter([0, 1, 3]).scale()
    assert x.get_inputs().equals(y.get_inputs())
    assert x.get_responses().equals(y.get_responses())
