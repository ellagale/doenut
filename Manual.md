A linear model for 3 factors which is capable of distinguishing main
effects:
*y* = *β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + *β*<sub>3</sub>*x*<sub>3</sub> .

A saturated model up to the power of two would be
*y* = *β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + *β*<sub>3</sub>*x*<sub>3</sub>
 + *β*<sub>4</sub>*x*<sub>1</sub>*x*<sub>2</sub> + *β*<sub>5</sub>*x*<sub>2</sub>*x*<sub>3</sub> + *β*<sub>6</sub>*x*<sub>1</sub>*x*<sub>3</sub>
 + *β*<sub>7</sub>*x*<sub>1</sub><sup>2</sup> + *β*<sub>8</sub>*x*<sub>2</sub><sup>2</sup> + *β*<sub>9</sub>*x*<sub>3</sub><sup>2</sup>

If a model only contains the cross terms and main effects (2 and 3 in
the equation above) it is an interaction model, if it only contains the
main and square terms it is a squared model (2 and 4 in the above
equation). The models are hierarchical: if a higher order interaction or
square term is included in the model, the linear term for that factor
must also be present (regardless of whether it is considered
statistically significant). The parsimonious model is defined as a model
in containing as few terms of any type that describe that describe the
data well.

## Checking the response plot

The first thing to do is check for experimental error.

types of error. bias error etc.

        doenut.replicate_plot(inputs, # the input dataframe
            responses, # the response dataframe
            key="ortho")

<img src="response_plot.png" style="width:40.0%" alt="image" />
<img src="head_foot/bias_response_plot.png" style="width:40.0%"
alt="image" />

# Modelling

### Choosing a model

The best *R*<sup>2</sup> is usually found by using a saturated model,
however, this model is often overfit and thus is worse at predicting on
new data. Often there are terms that are insignificantly different from
0, these are easily identified from the figure as if the error bars
cross the x-axis then the value of that coefficient can be replaced by
zero, and doing this reduces the overfit of the model, and thus
increases the *Q*<sup>2</sup>, and improves the efficiency of the model.
See figure <a href="#fig:coeff_egA" data-reference-type="ref"
data-reference="fig:coeff_egA">[fig:coeff_egA]</a> for an example of a
coefficient plot. The model is generally considered to be invalid if
*Q*<sup>2</sup> and *R*<sup>2</sup> are less than 0.5, and we expect the
difference between them to be less than 0.2 for a good model. \[\[cite
MODDE manual\]\]

We can use degrees of freedom (DoF) as another metric for choosing a
good model, higher DoF models are better (equivalently, lower numbers of
terms). DoF is given by:
*D**o**F* = *N* − *N*<sub>*β*<sub>*x*</sub></sub> + 1
where *N* is the number of experiments, *N*<sub>*β*<sub>*x*</sub></sub>
is the number of model terms (the +1 is due to the intercept,
*β*<sub>0</sub>).

Note that qualitative factors (i.e. solvent) have only one *β*
associated with them, but are plotted as two separate bars in the
coefficients plots in MODDE, to allow you to see which qualitative
factor is the best.

The AIC can be used be used as a metric for the model’s efficiency and
choosing the model with the lower AIC is the best.

Parsimonious models are created by successively removing the
statistically insignificantly higher order terms from a saturated model
and taking the model with the best *Q*<sup>2</sup> (N.B. *R*<sup>2</sup>
is always higher than the *Q*<sup>2</sup>). With lower resolution models
(such as resolution IV models) it is not possible to de-confound the
higher order terms, and thus the worker must use their judgement as to
which terms are more chemically irrelevant and removable, test the
possible models or do a follow up experiment with a higher resolution.
In this work, we used our chemical reasoning and tested the model, as
our aim is to get a good enough model to optimise the system, not to
create the best model to describe the system. Predictions were then
verified using follow up experiments.

# Fit a simple first order model

The main effects in the system are those relating to the input
variables. For example, in a system with 3 input variables,
*x*<sub>1</sub>, *x*<sub>2</sub>, *x*<sub>3</sub>, the value of these
factors are the main effects on the system.

A model relates input factors to output responses.
*i**n**p**u**t**s* → *m**o**d**e**l* → *o**u**t**p**u**t**s*

A very simple model would be something like this:
*o**u**t**p**u**t**s* = *β*<sub>0</sub> + *β*<sub>1</sub>*i**n**p**u**t*
which is simply the equation for a straight line:
*y* = *m**x* + *c*
where *c* and *β*<sub>0</sub> are the intercept and *m* and
*β*<sub>1</sub> are the gradient.

When you fit the equation of a line to 2D data, you are building a very
simple model to relate the output, *y*, to the input, *x*.

You can extend this to an arbitrary size of dimensions of your data. For
example, if you have 3D input data, you could build a model from this
equation:
*y* = *β*<sub>0</sub> + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + *β*<sub>3</sub>*x*<sub>3</sub>
which would require you to produce 4D plots to look at it. (As an aside,
the field of machine learning is concerned with building models for very
high dimensional data).

This is a model of order 1 as all the variables are *x*<sup>1</sup> or
below. (N.B. the first term is
*β*<sub>0</sub>*x*<sup>0</sup> = *β*<sub>0</sub>, and is called the
\*intercept\*).

## Implementation in DoENUT

Based on scikit-learn so could add in any model here. Simpler linear
regression is the best etc.

\[\[To-DoENUT! perhaps add in an option to put in any model?\]\]

        original_model, inputs_used, original_model_R2, predictions = doenut.train_model(
        inputs, 
        responses['ortho'], 
        test_responses=None,
        do_scaling_here=False,
        fit_intercept=True,
        verbose=True)

# Prediction

<figure>
<img src="linear_model.png" id="fig:linear_model_prediction"
alt="4D contour plot for the linear model in section [[]]" />
<figcaption aria-hidden="true">4D contour plot for the linear model in
section [[]]</figcaption>
</figure>
