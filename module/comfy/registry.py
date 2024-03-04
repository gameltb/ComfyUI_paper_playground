import inspect
import re
from functools import wraps
import typing

from .types import find_comfy_widget_type_annotation, ReturnType

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

PACK_BASE_CATEGORY = None
PACK_UID = None


class NodeTemplate:
    _FUNCTION_SIG: inspect.Signature = None
    FUNCTION = "exec"

    @classmethod
    def INPUT_TYPES(cls):
        input_types = {"required": {}, "optional": {}}
        for k, v in cls._FUNCTION_SIG.parameters.items():
            comfy_widget = find_comfy_widget_type_annotation(v.annotation)

            assert comfy_widget is not None
            assert comfy_widget.required or v.default is not inspect._empty

            opts = {}
            req = "required" if comfy_widget.required else "optional"
            opts = comfy_widget.opts()

            if v.default is not inspect._empty:
                opts["default"] = v.default

            tp = comfy_widget.type
            input_types[req][k] = (tp, opts)
        return input_types


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

        node_identifier: str = identifier if identifier is not None else f.__name__
        unique_name = f"{PACK_UID}_{version}_{node_identifier}"

        if inspect.isfunction(f):
            node_attrs = {}
            node_attrs["OUTPUT_NODE"] = output
            if f.__doc__ is not None:
                node_attrs["DESCRIPTION"] = f.__doc__.strip()

            sig = inspect.signature(f)
            node_attrs["_FUNCTION_SIG"] = sig

            return_annotation = sig.return_annotation
            if return_annotation == inspect._empty or return_annotation is None:
                return_annotation = tuple()
            elif isinstance(return_annotation, tuple):
                pass
            elif typing.get_origin(return_annotation) == tuple:
                return_annotation = typing.get_args(return_annotation)
            else:
                print(f"WARNING: Unknow object {return_annotation} for RETURN_TYPES.")

            node_attrs["RETURN_TYPES"] = tuple(find_comfy_widget_type_annotation(x).type for x in return_annotation)

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
                    comfy_widget_type_annotation = find_comfy_widget_type_annotation(sig.parameters[k].annotation)
                    if comfy_widget_type_annotation is not None:
                        # Look up Combo value from mapping
                        kwargs[k] = comfy_widget_type_annotation[v]
                results = f(**kwargs)
                if results is None:
                    results = {}
                elif isinstance(results, ReturnType):
                    results = results.model_dump()
                return results

            node_attrs["exec"] = staticmethod(exec)
            node_class = type(unique_name, (NodeTemplate,), node_attrs)
        elif inspect.isclass(f):
            node_class = f
        else:
            raise Exception(f"WARNING: Unknow object {node_identifier} for register_node.")

        # just check at setup
        node_class.INPUT_TYPES()

        assert unique_name not in NODE_CLASS_MAPPINGS

        NODE_CLASS_MAPPINGS[unique_name] = node_class
        node_display_name = display_name
        if node_display_name is None:
            if identifier is None and inspect.isfunction(f):
                node_display_name = " ".join(x.capitalize() for x in node_identifier.lower().split("_"))
            else:
                node_display_name = " ".join(re.split("(?<=[a-z0-9])(?=[A-Z])", node_identifier))
        NODE_DISPLAY_NAME_MAPPINGS[unique_name] = node_display_name
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
