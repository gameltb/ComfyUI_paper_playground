import os
from dataclasses import dataclass
from typing import Optional

from .path_tool import get_local_huggingface_path


@dataclass(kw_only=True)
class FileSource:
    name: Optional[str] = None
    loacal_folder: Optional[str] = None
    hash_sha256: Optional[str] = None

    def get_local_path(self):
        if self.loacal_folder is None:
            raise ValueError(f"cannt find requred file {self.name} without loacal folder.")
        if self.name is None:
            return self.loacal_folder
        return os.path.join(self.loacal_folder, self.name)


@dataclass(kw_only=True)
class FileSourceUrl(FileSource):
    url: str


@dataclass(kw_only=True)
class FileSourceHuggingface(FileSource):
    repo_id: str
    subfolder: Optional[str] = None
    repo_type: Optional[str] = None
    revision: Optional[str] = None

    def get_local_path(self):
        local_dir = get_local_huggingface_path(self.repo_id)
        if self.name is None:
            return local_dir
        if self.subfolder is None:
            return os.path.join(local_dir, self.name)
        else:
            return os.path.join(local_dir, self.subfolder, self.name)


def find_or_download_file(sources: list[FileSource]):
    for source in sources:
        local_path = source.get_local_path()
        if os.path.exists(local_path):
            return local_path
    raise FileNotFoundError(f"cannt find requred file from sources : {sources}")


def find_or_download_huggingface_repo(sources: list[FileSource]):
    help_list = []
    for source in sources:
        if not isinstance(source, (FileSourceHuggingface, FileSource)):
            raise ValueError("find_or_download_huggingface_repo requre Huggingface source.")

        if type(source) == FileSource:
            local_path = source.get_local_path()
            if os.path.exists(local_path):
                return local_path
            else:
                continue

        local_path = get_local_huggingface_path(source.repo_id)
        if os.path.exists(local_path):
            return local_path
        help_list.append(
            f"mkdir -p {os.path.dirname(local_path)};cd {os.path.dirname(local_path)};git clone https://huggingface.co/{source.repo_id}"
        )
    # TODO: download or just raise?
    help_text = "\nor\n".join(help_list)
    raise FileNotFoundError(f"""can not find requred repo from sources : {sources}.
you can try use 
{help_text}
to get it.
""")
