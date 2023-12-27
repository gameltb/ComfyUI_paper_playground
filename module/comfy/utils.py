import re

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_node(identifier: str = None, display_name: str = None):
    def decorator(cls):
        nonlocal identifier, display_name
        if identifier == None:
            identifier = cls.__name__
        if display_name == None:
            display_name = " ".join(re.split("(?<=[a-z0-9])(?=[A-Z])", identifier))

        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name

        return cls

    return decorator
