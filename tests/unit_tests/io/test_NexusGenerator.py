from pytest import raises
from stalk.io.NexusGenerator import NexusGenerator
from stalk.params import ParameterStructure


def generator(structure, path, arg0='', arg1=''):
    # Test that the structure is nexus-friendly
    assert structure.forward_func is None
    assert structure.backward_func is None
    # Return something to test
    return [structure.label + path + arg0 + arg1]
# end def


def test_NexusGenerator():

    s = ParameterStructure(label='label')
    path = '_path'

    # Test empty (should fail)
    with raises(TypeError):
        NexusGenerator()
    # end with

    # Test nominal, no args
    gen_noargs = NexusGenerator(generator)
    assert gen_noargs.generate(s, path)[0] == 'label_path'

    # Test nominal, args
    gen_args = NexusGenerator(generator, {'arg0': '_arg0', 'arg1': '_arg1'})
    assert gen_args.generate(s, path)[0] == 'label_path_arg0_arg1'
    # Test nominal, arg overridden
    assert gen_args.generate(s, path, arg1='_over')[0] == 'label_path_arg0_over'

# end def
