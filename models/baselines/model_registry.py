def register_model(*args, **kwargs):
    """Compatibility decorator for copied reference models."""
    def decorator(cls):
        return cls
    return decorator
