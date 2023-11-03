import pandas as pd
import doenut
import pytest

df = pd.read_csv('solar_cells_1.csv')
inputs = pd.DataFrame({
    'Donor %': [float(x) for x in df.iloc[1:-1, 1]],
    'Conc.': [float(x) for x in df.iloc[1:-1, 2]],
    'Spin': [float(x) for x in df.iloc[1:-1, 3]],
    'Add.': [float(x) for x in df.iloc[1:-1, 4]]})

responses = pd.DataFrame({'PCE': [float(x) for x in df['PCE'][1:-1]]})


df2 = pd.read_csv('solar_cells_2.csv')

inputs_2 = pd.DataFrame({
    'Donor %': [float(x) for x in df2.iloc[1:-1,1]],
    'Conc.': [float(x) for x in df2.iloc[1:-1,2]],
    'Spin': [float(x) for x in df2.iloc[1:-1,3]]})

responses_2 = pd.DataFrame({'PCE': [float(x) for x in df2['PCE'][1:-1]]})

new_inputs = pd.concat([inputs[['Donor %', 'Conc.', 'Spin']], inputs_2], axis=0).reset_index(drop=True)
new_responses = pd.concat([responses, responses_2], axis=0).reset_index(drop=True)


def pytest_namespace():
    """
    Helper function to store calculated values that are passed from one test
    to another for consecutive steps.
    """
    return {'sat_inputs_orig': None,
            'sat_inputs_2': None}


def test_calulate_r2_and_q2_for_models():
    input_selector = range(len(inputs.columns))
    this_model, R2, temp_tuple, _ = doenut.calulate_R2_and_Q2_for_models(
        inputs,
        responses,
        input_selector=input_selector,
        response_selector=[0],
        use_scaled_inputs=True,
        do_scaling_here=True
    )

    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.604
    assert round(Q2, 3) == 0.170


def test_add_higher_order_terms():
    sat_inputs_orig, sat_source_list = doenut.add_higher_order_terms(
        inputs,
        add_squares=True,
        add_interactions=True,
        column_list=[])
    assert sat_inputs_orig.size == 210
    assert len(sat_source_list) == 14
    pytest.sat_inputs_orig = sat_inputs_orig


def test_tune_model_fully_quad():
    input_selector = [0, 1, 2, 3,
                      4, 5, 6, 7]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_orig,
        responses,
        input_selector=input_selector,
        response_selector=[0]
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.815
    assert round(Q2, 3) == -0.176


def test_tune_model_parsimonious():
    input_selector = [0, 1, 2,
                      4, 5, 6]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_orig,
        responses,
        input_selector=input_selector,
        response_selector=[0]
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.813
    assert round(Q2, 3) == 0.332


def test_saturated_models():
    sat_inputs_2, sat_source_list = doenut.add_higher_order_terms(
        new_inputs,
        add_squares=True,
        add_interactions=True,
        column_list=[])
    assert sat_inputs_2.size == 243
    assert len(sat_source_list) == 9
    pytest.sat_inputs_2 = sat_inputs_2


def test_saturated_9_terms():
    input_selector = [0, 1, 2,
                      3, 4, 5,
                      6, 7, 8]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_2,
        new_responses,
        input_selector=input_selector,
        response_selector=[0]
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.895
    assert round(Q2, 3) == -0.204


def test_saturated_parsnip_terms():
    input_selector = [0, 1, 2,
                      3, 4, 5]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_2,
        new_responses,
        input_selector=input_selector,
        response_selector=[0]
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.871
    assert round(Q2, 3) == 0.716
