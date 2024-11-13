import os
from functools import cache

try:
    import folder_paths

    PLAYGROUND_HUGGINGFACE_MODEL_PATH = os.path.join(folder_paths.models_dir, "huggingface")
    PLAYGROUND_MODEL_PATH = os.path.join(folder_paths.models_dir, "playground")
    PLAYGROUND_OUTPUT_DIRECTORY_PATH = folder_paths.output_directory
except ModuleNotFoundError:
    PLAYGROUND_HUGGINGFACE_MODEL_PATH = os.getenv("PLAYGROUND_HUGGINGFACE_MODEL_PATH")
    PLAYGROUND_MODEL_PATH = os.getenv("PLAYGROUND_MODEL_PATH")
    PLAYGROUND_OUTPUT_DIRECTORY_PATH = os.getenv("PLAYGROUND_OUTPUT_DIRECTORY_PATH")

REPO_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def _split_abs_memo(file_name: str):
    return file_name[:13], file_name[13:]


@cache
def _gen_abs_map(dir_path: str):
    abs_map = {}
    for file_name in os.listdir(dir_path):
        if file_name.startswith("abs"):
            file_base_name = file_name.split(".")[0]
            abs_str, memo = _split_abs_memo(file_base_name)
            abs_map[abs_str] = file_base_name
    return abs_map


@cache
def _gen_sub_path_list_from_module_name(module_name: str):
    if "module.comfy.node." in module_name:
        sub_key = module_name.split("node.")[-1]
    elif "module.paper." in module_name:
        sub_key = module_name.split("module.")[-1]
    else:
        raise NotImplementedError()
    return sub_key.split(".")


def get_data_path(self_name, *paths):
    sub_paths = _gen_sub_path_list_from_module_name(self_name)
    _replace_abs_memo(sub_paths, PLAYGROUND_MODEL_PATH)
    return os.path.join(PLAYGROUND_MODEL_PATH, *sub_paths, *paths)


def _replace_abs_memo(sub_paths, root_path):
    if "arxiv" in sub_paths:
        arxiv_index = sub_paths.index("arxiv")
        abs_str, memo = _split_abs_memo(sub_paths[arxiv_index + 1])
        abs_map = _gen_abs_map(os.path.join(root_path, *sub_paths[: arxiv_index + 1]))
        if abs_str in abs_map:
            sub_paths[arxiv_index + 1] = abs_map[abs_str]


def get_data_path_file_list(self_name, *paths):
    full_path = get_data_path(self_name, *paths)
    return os.listdir(full_path)


def get_paper_repo_path(self_name, *paths):
    sub_paths = _gen_sub_path_list_from_module_name(self_name)
    _replace_abs_memo(sub_paths, REPO_ROOT_PATH)
    return os.path.join(REPO_ROOT_PATH, *sub_paths, *paths)


def get_output_path(save_path):
    file_name = os.path.basename(save_path)
    folder_path = os.path.dirname(save_path)
    if not os.path.isabs(save_path):
        folder_path = os.path.join(PLAYGROUND_OUTPUT_DIRECTORY_PATH, folder_path)

    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, file_name)


def get_local_huggingface_path(repo_id):
    return os.path.join(PLAYGROUND_HUGGINGFACE_MODEL_PATH, repo_id)


def gen_default_category_path_by_module_name(self_name):
    sub_paths = _gen_sub_path_list_from_module_name(self_name)
    if "arxiv" in sub_paths:
        arxiv_index = sub_paths.index("arxiv")
        abs_str, memo = _split_abs_memo(sub_paths[arxiv_index + 1])
        return f"arxiv/{abs_str} ({memo.lstrip('_')})"
    else:
        raise NotImplementedError()
