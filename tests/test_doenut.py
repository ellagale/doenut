import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import doenut

df = pd.read_csv('solar_cells_1.csv')
inputs = pd.DataFrame({
    'Donor %': [float(x) for x in df.iloc[1:-1, 1]],
    'Conc.': [float(x) for x in df.iloc[1:-1, 2]],
    'Spin': [float(x) for x in df.iloc[1:-1, 3]],
    'Add.': [float(x) for x in df.iloc[1:-1, 4]]})

responses = pd.DataFrame({'PCE': [float(x) for x in df['PCE'][1:-1]]})

input_selector = range(len(inputs.columns))

def test_calulate_r2_and_q2_for_models():
    this_model, R2, temp_tuple, _ = doenut.calulate_R2_and_Q2_for_models(
        inputs,
        responses,
        input_selector=input_selector,
        response_selector=[0],
        use_scaled_inputs=True,
        do_scaling_here=True
        )

    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2= temp_tuple
    assert round(R2, 3) == 0.604
    assert round(Q2, 3) == 0.170
