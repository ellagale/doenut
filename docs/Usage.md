# Usage
DoENUT provides a set of modules and functions for doing different tasks needed
by Design of Experiments.

Generally you are going to need an array-like set of inputs -
pandas dataframes are preferred as they give you column names, but plain
numpy arrays work fine. You will also need another set of the correct responses
for those inputs (also called the ground truth). For the purposes of these
instructions we will assume you have called these `inputs` and `responses`.

Most functions will also take an *input selector*. This is a list of column
_numbers_ that any generated model should pay attention to. 



# Making a basic model.
`doenut.train_model` will train a simple model, and then run it against the
provided inputs. It returns the following tuple:

`(model, inputs, R2, predictions`

Where:
- `model` is the trained model.
- `inputs` was the effective inputs 
the model was trained with (this may be different to the passed inputs if
you have asked it to scale the inputs - see the function's help string for 
more details). 
- `R2` is the R2 scoring coefficient of the model against the inputs.
- `predictions` is what the trained model gives as output for the inputs.

## Example
TODO (sorry)

# Making an averaged model
`doenut.calc_average_model` Will generate a set of models from the inputs using
a _leave one out_ approach to generate each model (using `train_model`). 
It returns the following tuple:

`(ortho_model, predictions, inputs, coefficients, R2s, R2, Q2)`

Where:
- `ortho_model` is the averaged model
- `predictions` is what the trained model gives as output for the inputs
- `inputs` were the effective inputs used (as per `train_model`)
- `coefficients` are the IDK. Ella to fill in this.
- `R2s` is the R2 values of each generated model.
- `R2` is the R2 value of the averaged model
- `Q2` is the Q2 value of the averaged model

# Adding higher order terms.
Assuming your input data is all separate terms, you can use 
`doenut.add_higher_order_terms` to calculate them for you. In its simplest
usage you can simply pass it your inputs and it will add both square and
interaction terms. If you want, you can control this behaviour using the
`add_squares` and `add_interactions` parameters. If you only want to add terms
for a subset of your columns, specify `column_list`

## Example
Given a pandas array with the following columns: `Temp`, `Pressure`, `Conc`,
`add_higher_order_terms` will add the columns `Temp**2`, `Pressure**2` and
`Conc**2` if `add_squares` is True, and the columns `Temp*Pressure`,
`Temp*Conc` and `Pressure*Conc` if `add_interactions` is True.