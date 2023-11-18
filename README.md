<img align="right" src="images/doenut_small.jpg">

# Design of Experiments Numerical Utility Toolkit DoENUT 

Package to conveniently combine all necessary functions required to do Design of Experiments.

Code for a forthcoming paper:
"DoENUT: Design of Experiments Numerical Utility Toolbox, for use in research and teaching," Ella M. Gale

Also has tutorials and examples

## About

A python module to do DoE from start to
finish for the main methodologies used by chemists and thus
explain the methods of DoE to the interested chemist.

Design of experiments[1] (DoE) is a technique widely used in industry
to optimise processes, formulations and materials or to
quickly explore chemical space. The standard approach to scientific
research is the *one variable at a time* (OVAT), where a single variable
is changed and the effect measured, or the related ‘trail and error’
approach where an experimentor does repeated OVAT experiments to
understand a process. DoE changes more than one variable at a time, and
uses regression to fit simple models to the data, which shows up where
there is interaction between factors, which can improve (synergistic) or
reduce (antagonistic) outcomes. DoE is commonly used to optimise a
process. The DoE method is:

1.  identify all possible factors that could affect an
    experiment/process deciding which ones to investigate

2.  setting up a series of experiments to fully explore the variable
    landscape by changing more than one variable at a time

3.  including repeat experiments to get a measure of the variance in the
    data

4.  using multi-variate linear regression to fit models to the data

5.  using leave one out approaches to test the predictability of the
    models

6.  using cost terms (often associated with the number of terms in the
    model

7.  choosing the best model using predictability and cost

8.  using the model to predict which area of variable space to explore
    next to continue to optimise.

There are several software programs to make this easy (e.g. Modde,SAS etc). For example, in optimsing a new recipie for Pringles (a
type of potatoe based chip) that reduced the amount of potato flour by
including rice, corn and wheat flour, Kelloggs performed a DoE with the
constraints that the potato flour content must be at least 40% and that
the water content must be 40%, but the amount of other flours were
varied simultaneously. DoE experimentation tends to give some
understanding as to what is important in a system, and has the advantage
over OVAT that you know exactly how many experiments you will need to do
a priori. Giving the requirement for several experiments, DoE approaches
go very well with automated synthesis machines (like ChemSpeed etc),
allowing for easy gathering of data.

[1] Also called ‘experimental design’ and ‘statistical experimental
design’

## Installation
The easiest way is via pip `pip install doenut`. 

Alternatively, get the source and use poetry via `poetry install`

## Getting started
As a very quick start, assuming your data is split into a pair of pandas
DataFrames, one for the input data and one for the responses, the following
code will create a standard model and generate some stats on how good it is.

```python
dataset = doenut.data.ModifiableDataSet(inputs, responses)
model = doenut.models.AveragedModel(dataset)
r2, q2 = model.r2, model.q2
print(f"R2 is {r2}, Q2 is {q2}")
doenut.plot.plot_summary_of_fit_small(r2, q2)
doenut.plot.coeff_plot(model.coeffs,
                       labels=list(dataset.get().inputs.columns),
                       errors='p95',
                       normalise=True)
```

For a more comprehensive look, open up a workbook in Tutorials.



