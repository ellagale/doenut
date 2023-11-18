import numpy as np
import pandas as pd
import os

from doenut.data import ModifiableDataSet

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

df2 = pd.read_csv("solar_cells_2.csv")

inputs_2 = pd.DataFrame(
    {
        "Donor %": [float(x) for x in df2.iloc[1:-1, 1]],
        "Conc.": [float(x) for x in df2.iloc[1:-1, 2]],
        "Spin": [float(x) for x in df2.iloc[1:-1, 3]],
    }
)

responses_2 = pd.DataFrame({"PCE": [float(x) for x in df2["PCE"][1:-1]]})

new_inputs = pd.concat(
    [inputs[["Donor %", "Conc.", "Spin"]], inputs_2], axis=0, ignore_index=True
)
new_responses = pd.concat([responses, responses_2], axis=0, ignore_index=True)


def test_dataset():
    x = ModifiableDataSet(inputs, responses).get()
    assert x.get_inputs().equals(inputs)
    assert x.get_responses().equals(responses)


def test_filtered_dataset():
    y = ModifiableDataSet(inputs, responses).filter([0, 1, 3]).get()
    assert list(y.get_inputs().columns) == ["Donor %", "Conc.", "Add."]
    assert y.get_responses().equals(responses)
    z = (
        ModifiableDataSet(inputs, responses)
        .filter(["Donor %", "Conc.", "Add."])
        .get()
    )
    assert z.get_inputs().equals(y.get_inputs())


def test_scaled_dataset():
    x = ModifiableDataSet(inputs, responses).scale(True).get()
    xi = x.get_inputs()
    xr = x.get_responses()
    assert np.min(xi) >= -1
    assert np.max(xi) <= 1
    assert np.min(xr) >= -1
    assert np.max(xr) <= 1


def test_scaled_filtered_ordering():
    x = ModifiableDataSet(inputs, responses).scale().filter([0, 1, 3]).get()
    y = ModifiableDataSet(inputs, responses).filter([0, 1, 3]).scale().get()
    assert x.get_inputs().equals(y.get_inputs())
    assert x.get_responses().equals(y.get_responses())


def test_drop_duplicates():
    x = ModifiableDataSet(new_inputs, new_responses).drop_duplicates().get()
    assert len(x.get_inputs()) == 26
    assert len(x.get_responses()) == 26
    assert x.get_responses().iloc[7].equals(new_responses.iloc[7])


def test_avg_duplicates():
    x = ModifiableDataSet(new_inputs, new_responses).average_duplicates().get()
    assert len(x.get_inputs()) == 26
    assert len(x.get_responses()) == 26
    # check the rounding has occurred
    assert round(x.get_responses().iloc[7]["PCE"], 2) == 7.22
