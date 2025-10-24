from compehndly.core import register_function


@register_function(name="add_one", version="0.0.1")
def _add_one_v0_0_1(x: int):
    return x + 1


@register_function(name="add_one", version="0.1.0")
def _add_one_v0_1_0(x):
    if x > 0:
        return x + 1
    else:
        return 0
