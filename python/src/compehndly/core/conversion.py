def arrowize_arguments(func, adapter):
    def wrapper(*args, **kwargs):
        # Convert args → arrow
        arr_args = [adapter.to_arrow(a) for a in args]

        # Convert kwargs → arrow (only values!)
        arr_kwargs = {k: adapter.to_arrow(v) for k, v in kwargs.items()}

        result = func(*arr_args, **arr_kwargs)

        # Convert result → library-specific type
        return adapter.from_arrow(result)

    return wrapper
