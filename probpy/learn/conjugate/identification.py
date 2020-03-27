from probpy.core import RandomVariable


def _check_no_none_parameters(rv: RandomVariable):
    for parameter in rv.parameters.values():
        if parameter.value is None:
            return False
    return True


def _check_only_none_is(rv: RandomVariable, none_name: [str]):
    for name, parameter in rv.parameters.items():
        if name in none_name:
            if parameter.value is not None:
                return False
        else:
            if parameter.value is None:
                return False
    return True
