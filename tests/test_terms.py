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
