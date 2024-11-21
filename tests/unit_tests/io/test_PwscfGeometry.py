from shapls.io.PwscfGeometry import PwscfGeometry
from shapls.io.XyzLoader import XyzLoader
from shapls.params import GeometryResult
from shapls.util.util import match_to_tol


def test_PwscfGeometry():
    # Test with empty args
    loader = PwscfGeometry()

    # Use XyzLoader for reference
    pos_ref = XyzLoader().load(
        'tests/unit_tests/assets/pwscf_relax', suffix='relax_bohr.xyz')

    # default suffix: relax.in; only path is needed
    res0 = loader.load('tests/unit_tests/assets/pwscf_relax')
    assert isinstance(res0, GeometryResult)
    assert res0.get_axes() is None
    assert res0.get_elem() is None
    assert match_to_tol(res0.get_pos(), pos_ref.get_pos(), 1e-6)

    # Test by providing args (c_pos multiplication by 2)
    loader1 = PwscfGeometry({'suffix': 'pwscf_relax/relax.in', 'c_pos': 0.0})
    res1 = loader1.load('tests/unit_tests/assets', c_pos=2.0)
    assert isinstance(res1, GeometryResult)
    assert match_to_tol(res1.get_pos(), 2.0 * pos_ref.get_pos(), 1e-6)

    # Test reading axes
    res2 = loader1.load(
        'tests/unit_tests/assets/pwscf_relax', suffix='relax_axes.in')
    assert isinstance(res2, GeometryResult)
    # Only test superficially for now
    assert res2.get_axes() is not None
# end def
