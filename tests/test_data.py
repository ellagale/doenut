import os
import pandas as pd

from doenut.data import FilteredDataFrame

# since pytest runs from an arbitrary location, fix that.
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


def test_terms():
    data = FilteredDataFrame(inputs)
    assert data.data.size == 60
    assert len(data.data.columns) == 4
    assert data.get_filtered().equals(inputs)


def test_filter_columns():
    selector = ["Spin", "Add."]
    data = FilteredDataFrame(inputs).filter(selector)
    filtered_data = data.get_filtered()
    assert filtered_data.size == 30
    assert len(filtered_data.columns) == 2


def test_filter_columns_by_index():
    indices = [2, 3]
    data = FilteredDataFrame(inputs).filter_by_indices(indices)
    filtered_data = data.get_filtered()
    assert filtered_data.size == 30
    assert len(filtered_data.columns) == 2


def test_filter_both_ways():
    selector = ["Spin", "Add."]
    data = FilteredDataFrame(inputs).filter(selector)
    filtered_data = data.get_filtered()
    indices = [2, 3]
    indexed_data = FilteredDataFrame(inputs).filter_by_indices(indices)
    filtered_indexed_data = data.get_filtered()
    assert filtered_data.equals(filtered_indexed_data)


def test_get_duplicates():
    data = FilteredDataFrame(new_inputs)
    duplicates = data.get_duplicate_rows()
    assert duplicates == {7: {26}}


def test_remove_duplicates():
    frame = FilteredDataFrame(new_inputs)
    data = frame.remove_duplicates()
    frame2 = FilteredDataFrame(new_responses)
    data2 = frame2.remove_duplicates(frame.get_duplicate_rows())
    assert len(data) == 26
    assert max(data.index) == 25
    assert len(data2) == 26
    assert max(data2.index) == 25
    assert data2.iloc[7][0] == 7.21


def test_average_duplicates():
    duplicates = FilteredDataFrame(new_inputs).get_duplicate_rows()
    data = FilteredDataFrame(new_responses).average_duplicates(duplicates)
    assert len(data) == 26
    assert max(data.index) == 25
    assert round(data.iloc[7][0], 2) == 7.22
