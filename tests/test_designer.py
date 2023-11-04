import doenut


def test_full_fact():
    """Test full factorial output"""
    actual = doenut.designer.full_fact({
            'Pressure':[50, 60, 70],
            'Temperature':[290, 320, 350],
            'Flow rate':[0.9, 1.0],
    })
    # TODO:: validate this properly
    assert actual is not None
    assert actual.size == 54
