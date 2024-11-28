from pytest import raises
from stalk.io.GeometryLoader import GeometryLoader


def test_GeometryLoader():
    # Test degraded
    with raises(AssertionError):
        GeometryLoader(None)
    # end with

    loader = GeometryLoader()
    with raises(TypeError):
        # 'path' argument is required
        loader.load()
    # end with

    with raises(NotImplementedError):
        loader.load('')
    # end with

    # Nominal functionality is tested in derived classes
# end def
