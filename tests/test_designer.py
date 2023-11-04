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
    input_ranges = doenut.designer.get_ranges(inputs)
    actual = doenut.designer.frac_fact_res(input_ranges, 2)
    assert actual is not None
    assert actual.size == 24
