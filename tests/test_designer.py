import doenut

inputs = {
    "Pressure": [50, 60, 70],
    "Temperature": [290, 320, 350],
    "Flow rate": [0.9, 1.0],
}


def test_full_fact():
    """Test full factorial output"""
    actual = doenut.designer.full_fact(inputs)
    # TODO:: validate this properly
    assert actual is not None
    assert actual.size == 54


def test_fract_fact():
    actual = doenut.designer.frac_fact(inputs, 2)
    assert actual is not None
    assert actual.size == 24


def test_fact_designer():
    levels = {'Pressure': [550, 850],
              'Temperature': [325, 450],
              'Flow': [12000, 16000],
              }
    factory_design_full_fact = doenut.designer.fact_designer(levels,
                                                             do_midpoints=True,
                                                             shuffle=False,
                                                             repeats=1,
                                                             num_midpoints=1)
    assert factory_design_full_fact is not None
    assert factory_design_full_fact.size == 27
