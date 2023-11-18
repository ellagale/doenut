"""
DoENUT

Design of Experiments Numerical Utility Toolkit

DoENUT is a set of classes and functions designed to make Design of Experiments
easier in Python.

To get started, see the workbooks under Tutorials or look at AveragedModel
and ModifiableDataSet.

As a very quick start, assuming your data is split into a pair of pandas
DataFrames, one for the input data and one for the responses, the following
code will create a standard model and generate some stats on how good it is.

dataset = doenut.data.ModifiableDataSet(inputs, responses)
model = doenut.models.AveragedModel(dataset)
r2, q2 = model.r2, model.q2
print(f"R2 is {r2}, Q2 is {q2}")
doenut.plot.plot_summary_of_fit_small(r2, q2)
doenut.plot.coeff_plot(model.coeffs,
                       labels=list(dataset.get().inputs.columns),
                       errors='p95',
                       normalise=True)
"""

from importlib.metadata import version

__version__ = version("doenut")

from . import doenut
from . import designer
from . import plot
from . import models

from .doenut import (
    orthogonal_scaling,
    add_higher_order_terms,
    autotune_model,
    scale_by,
    Calculate_Q2,
    calc_ave_coeffs_and_errors,
    map_chemical_space,
    predict_from_model,
)
