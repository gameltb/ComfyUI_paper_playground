import inspect
import re
import typing

from docstring_parser import parse

from .types import ComfyWidget, ComfyWidgetInputType, ReturnType, find_comfy_widget_type_annotation

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

PACK_BASE_CATEGORY = None
PACK_UID = None


class NodeTemplate:
    _COMFY_WIDGET_MAP: dict[str, ComfyWidget] = None
    _WARP_FUNCTION: typing.Callable = None
    FUNCTION = "exec"

    @classmethod
    def INPUT_TYPES(cls):
        input_types = {}
        for k, comfy_widget in cls._COMFY_WIDGET_MAP.items():
            opts = {}
            input_type = comfy_widget.input_type.value
            opts = comfy_widget.opts()

            tp_str = comfy_widget.type
            if input_type not in input_types:
                input_types[input_type] = {}
            input_types[input_type][k] = (tp_str, opts)
        return input_types

    @classmethod
    def exec(cls, **kwargs):
        for k, v in kwargs.items():
            comfy_widget = cls._COMFY_WIDGET_MAP.get(k, None)
            if comfy_widget is not None:
                # Look up Combo value from mapping
                kwargs[k] = comfy_widget[v]
        results = cls._WARP_FUNCTION(**kwargs)
        if results is None:
            results = {}
        elif isinstance(results, ReturnType):
            results = results.model_dump()
        return results


def set_pack_options(uid: str, category: str = None):
    global PACK_BASE_CATEGORY, PACK_UID
    PACK_BASE_CATEGORY = category
    PACK_UID = uid


def get_nodes():
    return {k: v.exec.__doc__ for k, v in NODE_CLASS_MAPPINGS.items()}


T = typing.TypeVar("T")


def register_node(category=None, version=0, identifier=None, display_name=None, output=False):
    def decorator(f: T) -> T:
        if PACK_UID is None:
            raise Exception("PACK_UID is not set. Call set_pack_options in __init__.py to set it.")

        node_identifier: str = identifier if identifier is not None else f.__name__
        unique_name = f"{PACK_UID}_{version}_{node_identifier}"

        if inspect.isfunction(f):
            node_attrs = {}
            node_attrs["OUTPUT_NODE"] = output

            sig = inspect.signature(f)

            input_description = None
            if f.__doc__ is not None:
                parse_doc = parse(f.__doc__)
                input_description = {p.arg_name: p.description for p in parse_doc.params}
                node_attrs["DESCRIPTION"] = parse_doc.description

            comfy_widget_map = {}
            for k, v in sig.parameters.items():
                comfy_widget = find_comfy_widget_type_annotation(v.annotation)

                assert comfy_widget is not None
                assert comfy_widget.input_type is not ComfyWidgetInputType.OPTIONAL or v.default is not inspect._empty

                opts = {}

                if v.default is not inspect._empty:
                    opts["default"] = v.default

                if comfy_widget.tooltip is None and input_description is not None:
                    opts["tooltip"] = input_description.get(k, None)

                comfy_widget_map[k] = comfy_widget.model_copy(update=opts)

            node_attrs["_COMFY_WIDGET_MAP"] = comfy_widget_map

            return_annotation = sig.return_annotation
            if return_annotation == inspect._empty or return_annotation is None:
                return_annotation = tuple()
            elif isinstance(return_annotation, tuple):
                pass
            elif issubclass(return_annotation, tuple) and hasattr(return_annotation, "_fields"):
                # assume it's typing.NamedTuple
                rt_sig = inspect.signature(return_annotation)
                rt_names, rt_parameters = zip(*rt_sig.parameters.items())
                node_attrs["RETURN_NAMES"] = tuple(rt_names)
                return_annotation = map(lambda p: p.annotation, rt_parameters)
            elif typing.get_origin(return_annotation) is tuple:
                return_annotation = typing.get_args(return_annotation)
            else:
                raise Exception(f"WARNING: Unknow object {return_annotation} for RETURN_TYPES.")

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

            node_attrs["_WARP_FUNCTION"] = f
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
