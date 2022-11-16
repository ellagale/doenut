# doenut
# Design of Experiments Numerical Utility Toolkit

Package to conveniently combine all necessary functions required to do Design of Experiments. 

Also has tutorials and examples

This paper will present a novel python module to do DoE from start to
finish for the \[\[x\]\] main methodologies used by chemists and thus
explain the methods of DoE to the interested chemist.

Design of experiments[1] (DoE) is a technique widely used in industry
\[\[cite!\]\] to optimise processes, formulations and materials or to
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

There are several software programs to make this easy \[\[cite Modde,SAS
etc\]\]. As an example \[\[cite MArcus\]\] wanted to determine the best
a model to predict the best formulation of a novel material to enable
cell attachment for use as a cell scaffold. The inputs chosen were
surface zeta potential, capacitive coupling, shear modulus and surface
roughness (the things that could be measured in the lab) and the output
the cell attachement %. With cross terms this gave 22 models to
investigate. The differences between DoE and linear regression are: that
you are generally optimising a known system and that constraints are
usually applied. For example, in optimsing a new recipie for Pringles (a
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

Code for a forthcoming paper:
"DoENUT: Design of Experiments Numerical Utility Toolbox, for use in research and teaching," Ella M. Gale
