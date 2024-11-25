from pytest import raises
from numpy import isnan
from shapls.params.PesResult import PesResult


def test_PesResult():
    # Test degraded
    # Cannot init empty
    with raises(TypeError):
        PesResult()
    # end with

    val = 1.0
    err = 2.0
    err_default = 0.0

    # test nominal (no error)
    res0 = PesResult(val)
    assert res0.get_value() == val
    assert res0.get_error() == err_default

    # Test nominal (with error)
    res1 = PesResult(val, err)
    assert res1.get_value() == val
    assert res1.get_error() == err
    assert res1.get_result()[0] == val
    assert res1.get_result()[1] == err

    # Test degraded (Nan value)
    res2 = PesResult(None, err)
    assert isnan(res2.get_value())
    assert res2.get_error() == 0.0

    # Test degraded (Nan value/error)
    res2 = PesResult(None, None)
    assert isnan(res2.get_value())
    assert res2.get_error() == 0.0

# end def
