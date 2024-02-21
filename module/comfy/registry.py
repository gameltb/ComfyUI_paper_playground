import inspect
import re
from functools import wraps

from .types import ComfyWidgetType

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

PACK_BASE_CATEGORY = None
PACK_UID = None


class NodeTemplate:
    _INPUT_TYPES = None
    FUNCTION = "exec"

    @classmethod
    def INPUT_TYPES(cls):
        return cls._INPUT_TYPES


def set_pack_options(uid: str, category: str = None):
    global PACK_BASE_CATEGORY, PACK_UID
    PACK_BASE_CATEGORY = category
    PACK_UID = uid


def get_nodes():
    return {k: v.exec.__doc__ for k, v in NODE_CLASS_MAPPINGS.items()}


def register_node(category=None, version=0, identifier=None, display_name=None, output=False):
    def decorator(f):
        if PACK_UID is None:
            raise Exception("PACK_UID is not set. Call set_pack_options in __init__.py to set it.")

        node_identifier = identifier if identifier is not None else f.__name__
        unique_name = f"{PACK_UID}_{version}_{node_identifier}"

        if inspect.isfunction(f):
            node_attrs = {}
            node_attrs["OUTPUT_NODE"] = output
            node_attrs["_INPUT_TYPES"] = {"required": {}, "optional": {}}

            sig = inspect.signature(f)

            node_attrs["RETURN_TYPES"] = tuple(
                x.type if isinstance(x, ComfyWidgetType) else x for x in sig.return_annotation
            )

            for k, v in sig.parameters.items():
                t = v.annotation

                assert isinstance(t, ComfyWidgetType)

                opts = {}
                req = "required" if t.required else "optional"
                opts = t.opts()
                t = t.type
                if v.default is inspect._empty:
                    opts["forceInput"] = True
                else:
                    opts["default"] = v.default

                node_attrs["_INPUT_TYPES"][req][k] = (t, opts)

            cat_list = []
            if PACK_BASE_CATEGORY is not None:
                cat_list.append(PACK_BASE_CATEGORY)
            if category is not None:
                cat_list.append(category)
            if cat_list:
                node_attrs["CATEGORY"] = "/".join(cat_list)
            else:
                print(
                    f"WARNING: No category specified for {node_identifier} and no base category. It won't be shown in menus."
                )

            @wraps(f)
            def exec(**kwargs):
                for k, v in kwargs.items():
                    if isinstance(sig.parameters[k].annotation, ComfyWidgetType):
                        # Look up Combo value from mapping
                        kwargs[k] = sig.parameters[k].annotation[v]
                return f(**kwargs)

            node_attrs["exec"] = staticmethod(exec)
            node_class = type(unique_name, (NodeTemplate,), node_attrs)
        elif inspect.isclass(f):
            node_class = f
        else:
            raise Exception(f"Unknow object {node_identifier} for register_node.")

        NODE_CLASS_MAPPINGS[unique_name] = node_class
        if display_name is not None:
            NODE_DISPLAY_NAME_MAPPINGS[unique_name] = display_name
        else:
            NODE_DISPLAY_NAME_MAPPINGS[unique_name] = " ".join(re.split("(?<=[a-z0-9])(?=[A-Z])", node_identifier))
        return f

    return decorator


def scrape_module(m, sig_len):
    for name in dir(m):
        v = getattr(m, name)
        if callable(v):
            try:
                sig = inspect.signature(v)
                if len(sig.parameters) == sig_len:
                    yield v
            except ValueError:
                pass
