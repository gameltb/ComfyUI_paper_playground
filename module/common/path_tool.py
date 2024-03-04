import os

import folder_paths

PLAYGROUND_MODEL_PATH = os.path.join(folder_paths.models_dir, "playground")
REPO_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def get_model_dir(self_name, key=""):
    if ".node." in self_name:
        sub_key = self_name.split("node.")[-1]
    elif ".paper." in self_name:
        sub_key = self_name.split("module.")[-1]
    sub_path = os.path.join(*sub_key.split("."), key)
    full_path = os.path.join(PLAYGROUND_MODEL_PATH, sub_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)
    return full_path


def get_model_filename_list(self_name, key=""):
    sub_key = self_name.split("node.")[-1]
    sub_path = os.path.join(*sub_key.split("."), key)
    full_path = os.path.join(PLAYGROUND_MODEL_PATH, sub_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)
    return os.listdir(full_path)


def get_model_full_path(self_name, key, filename):
    sub_key = self_name.split("node.")[-1]
    sub_path = os.path.join(*sub_key.split("."), key)
    return os.path.join(PLAYGROUND_MODEL_PATH, sub_path, filename)


def get_paper_repo_path(self_name):
    sub_key = self_name.split("node.")[-1]
    sub_path = os.path.join(*sub_key.split("."))
    full_path = os.path.join(REPO_ROOT_PATH, sub_path)
    return full_path


def get_output_path(save_path):
    file_name = os.path.basename(save_path)
    folder_path = os.path.dirname(save_path)
    if not os.path.isabs(save_path):
        folder_path = os.path.join(folder_paths.output_directory, folder_path)

    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, file_name)
