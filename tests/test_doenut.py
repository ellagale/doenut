import pandas as pd
import doenut
import doenut.models
import pytest
import os

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


def _get_column_names_by_number(input, numbers):
    return [input.columns[x] for x in numbers]


def pytest_namespace():
    """
    Helper function to store calculated values that are passed from one test
    to another for consecutive steps.
    """
    return {
        "sat_inputs_orig": None,
        "sat_inputs_orig_source_list": None,
        "sat_inputs_2": None,
        "sat_inputs_2_source_list": None,
        "scaled_model": None,
        "scaled_model_2": None,
    }


def test_calculate_r2_and_q2_for_models():
    input_selector = range(len(inputs.columns))
    this_model, R2, temp_tuple, _ = doenut.calculate_R2_and_Q2_for_models(
        inputs,
        responses,
        input_selector=input_selector,
        response_selector=[0],
        use_scaled_inputs=True,
        do_scaling_here=True,
    )

    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.604
    assert round(Q2, 3) == 0.170


def test_selective_model():
    model = doenut.models.SelectiveModel(inputs, responses)
    assert round(model.r2, 3) == 0.604
    assert round(model.q2, 3) == 0.170


def test_averaged_model_set():
    modelset = doenut.models.AveragedModelSet.multiple_response_columns(
        inputs, responses
    )
    model = modelset.models[0]
    assert round(model.r2, 3) == 0.604
    assert round(model.q2, 3) == 0.170


def test_add_higher_order_terms():
    sat_inputs_orig, sat_source_list = doenut.add_higher_order_terms(
        inputs, add_squares=True, add_interactions=True, column_list=[]
    )
    assert sat_inputs_orig.size == 210
    assert len(sat_source_list) == 14
    pytest.sat_inputs_orig = sat_inputs_orig
    pytest.sat_inputs_orig_source_list = sat_source_list


def test_tune_model_fully_quad():
    input_selector = [0, 1, 2, 3, 4, 5, 6, 7]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_orig,
        responses,
        input_selector=input_selector,
        response_selector=[0],
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.815
    assert round(Q2, 3) == -0.176


def test_tune_model_parsimonious():
    input_selector = [0, 1, 2, 4, 5, 6]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_orig,
        responses,
        input_selector=input_selector,
        response_selector=[0],
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.813
    assert round(Q2, 3) == 0.332


def test_hand_tune_fully_quad():
    input_selector = _get_column_names_by_number(
        pytest.sat_inputs_orig, range(8)
    )
    model = doenut.models.AveragedModel(
        pytest.sat_inputs_orig,
        responses,
        input_selector=input_selector,
        scale_data=True,
        scale_run_data=True,
        drop_duplicates="no",
    )
    assert round(model.r2, 3) == 0.815
    assert round(model.q2, 3) == -0.176


def test_hand_tune_parsnip():
    # Index(['Donor %', 'Conc.', 'Spin', 'Add.', 'Donor %**2', 'Conc.**2', 'Spin**2',
    #'Add.**2', 'Donor %*Conc.', 'Donor %*Spin', 'Donor %*Add.',
    #'Conc.*Spin', 'Conc.*Add.', 'Spin*Add.'],
    # dtype='object')
    # input_selector = ['Donor %', 'Conc.', 'Spin', 'Donor %**2', 'Conc.**2', 'Spin**2']
    input_selector = _get_column_names_by_number(
        pytest.sat_inputs_orig, [0, 1, 2, 4, 5, 6]
    )
    model = doenut.models.AveragedModel(
        pytest.sat_inputs_orig,
        responses,
        input_selector=input_selector,
        scale_data=True,
        scale_run_data=True,
        drop_duplicates="no",
    )
    assert round(model.r2, 3) == 0.813
    assert round(model.q2, 3) == 0.332


def test_saturated_models():
    sat_inputs_2, sat_source_list = doenut.add_higher_order_terms(
        new_inputs, add_squares=True, add_interactions=True, column_list=[]
    )
    assert sat_inputs_2.size == 243
    assert len(sat_source_list) == 9
    pytest.sat_inputs_2 = sat_inputs_2
    pytest.sat_inputs_2_source_list = sat_source_list


def test_saturated_9_terms():
    input_selector = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_2,
        new_responses,
        input_selector=input_selector,
        response_selector=[0],
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.895
    assert round(Q2, 3) == -0.204


def test_saturated_9_terms_2():
    input_selector = _get_column_names_by_number(pytest.sat_inputs_2, range(9))
    model = doenut.models.AveragedModel(
        pytest.sat_inputs_2,
        new_responses,
        input_selector=input_selector,
        drop_duplicates="no",
    )
    assert round(model.r2, 3) == 0.895
    assert round(model.q2, 3) == -0.204


def test_saturated_parsnip_terms():
    input_selector = [0, 1, 2, 3, 4, 5]
    scaled_model, R2, temp_tuple, _ = doenut.tune_model(
        pytest.sat_inputs_2,
        new_responses,
        input_selector=input_selector,
        response_selector=[0],
    )
    new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    assert round(R2, 3) == 0.871
    assert round(Q2, 3) == 0.716
    pytest.scaled_model = scaled_model


def test_saturated_parsnip_terms_2():
    input_selector = _get_column_names_by_number(
        pytest.sat_inputs_2, [0, 1, 2, 3, 4, 5]
    )
    model = doenut.models.AveragedModel(
        pytest.sat_inputs_2,
        new_responses,
        input_selector=input_selector,
        drop_duplicates="no",
    )
    filtered_inputs = pytest.sat_inputs_2[input_selector]
    # model.model.fit(filtered_inputs, new_responses)
    assert round(model.r2, 3) == 0.871
    assert round(model.q2, 3) == 0.716

    pytest.scaled_model_2 = model


def test_run_model():
    runs = pd.DataFrame(
        {
            "A": {"Donor %": 20, "Conc.": 12, "Spin": 500},
            "B": {"Donor %": 40, "Conc.": 16, "Spin": 1500},
            "C": {"Donor %": 35, "Conc.": 22, "Spin": 1500},
            "D": {"Donor %": 45, "Conc.": 18, "Spin": 2500},
            "E": {"Donor %": 20, "Conc.": 17, "Spin": 2500},
        }
    ).T
    sat_inputs, sat_sources = doenut.add_higher_order_terms(
        runs,
        add_squares=True,
        add_interactions=True,
        column_list=[],
        verbose=False,
    )
    expected_results = [2.11, 6.11, 7.61, 4.65, 3.97]
    input_selector = [0, 1, 2, 3, 4, 5]
    results, _ = doenut.predict_from_model(
        pytest.scaled_model, sat_inputs, input_selector
    )
    actual_results = [round(x[0], 2) for x in results]
    term_list = _get_column_names_by_number(sat_inputs, input_selector)
    filtered_inputs = sat_inputs[term_list]
    results_test = pytest.scaled_model.predict(filtered_inputs)
    results_test2 = pytest.scaled_model_2.model.predict(filtered_inputs)
    results2 = pytest.scaled_model_2.get_predictions_for(
        filtered_inputs
    ).reshape(
        -1,
    )
    actual_results_2 = [round(x, 2) for x in results2]

    assert expected_results == actual_results
    assert expected_results == actual_results_2


def test_autotune():
    # input_selector = _get_column_names_by_number(pytest.sat_inputs_2, range(9))
    # model = doenut.models.AveragedModel(
    #     pytest.sat_inputs_2,
    #     new_responses,
    #     input_selector=input_selector,
    #     drop_duplicates="no",
    # )

    (
        output_indices,
        new_model,
    ) = doenut.autotune_model(
        pytest.sat_inputs_2,
        new_responses,
        pytest.sat_inputs_2_source_list,
        verbose=True,
    )

    assert round(new_model.r2, 3) == 0.886
    assert round(new_model.q2, 3) == 0.486
