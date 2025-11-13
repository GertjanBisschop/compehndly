from compehndly.core import FunctionRegistry


class TestBuildRegistry:
    def test_build(self):
        to_register = ["tests.utils"]
        registry = FunctionRegistry.build_registry(to_register)
        f = registry.get("add_one", version="0.0.1")
        x = 10
        assert f(x) == 11
