import os

import folder_paths

PLAYGROUND_MODEL_PATH = os.path.join(folder_paths.models_dir, "playground")


def get_model_filename_list(self_name, key):
    sub_key = self_name.split("node.")[-1]
    sub_path = os.path.join(*sub_key.split("."), key)
    full_path = os.path.join(PLAYGROUND_MODEL_PATH, sub_path)
    return os.listdir(full_path)


def get_model_full_path(self_name, key, filename):
    sub_key = self_name.split("node.")[-1]
    sub_path = os.path.join(*sub_key.split("."), key)
    return os.path.join(PLAYGROUND_MODEL_PATH, sub_path, filename)
