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
