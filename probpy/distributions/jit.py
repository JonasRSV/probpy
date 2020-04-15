from probpy.core import RandomVariable


def jit_probability(rv: RandomVariable):
    if not hasattr(rv.cls, "jit_probability"):
        raise Exception(f"{rv.cls} has not implemented jit_probability")

    return rv.cls.jit_probability(rv)

