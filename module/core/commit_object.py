import dataclasses
import logging
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar, Union

_logger = logging.getLogger(__name__)

T = TypeVar("T")
UNKNOWN_STATE_UUID = None


class CommitABC(ABC, Generic[T]):
    def __init__(self):
        self.commit_uuid = uuid.uuid4()

    @abstractmethod
    def apply(self, base_object: T, **kwargs):
        pass

    @abstractmethod
    def revert(self, base_object: T):
        pass

    def get_revert_callable(self):
        return self.revert

    def release_revert_callable(self):
        pass


class CallableCommit(CommitABC, Generic[T]):
    def __init__(self, commit_callable: Callable[[T], Callable[[T], None]]):
        super().__init__()
        self.commit_callable = commit_callable

    def apply(self, base_object: T):
        self.revert_callable = self.commit_callable(base_object)

    def revert(self, base_object: T):
        self.revert_callable(base_object)

    def get_revert_callable(self):
        return self.revert_callable

    def release_revert_callable(self):
        self.revert_callable = None


@dataclasses.dataclass
class AppliedCommitRefItem:
    commit_ref: weakref.ref[CommitABC]
    commit_uuid: uuid.UUID
    revert_callable: Optional[Callable]
    snapshot: bool = False

    @classmethod
    def from_commit(cls, commit: CommitABC):
        return cls(
            commit_ref=weakref.ref(commit),
            commit_uuid=commit.commit_uuid,
            revert_callable=commit.get_revert_callable(),
        )


class CommitSnapshotItem(Generic[T]):
    def __init__(self):
        self.commit_stack_states: list[uuid.UUID] = []
        self.revert_callable_list: list[Callable] = []

    def append(self, applied_commit_ref: AppliedCommitRefItem):
        self.commit_stack_states.append(applied_commit_ref.commit_uuid)
        self.revert_callable_list.append(applied_commit_ref.revert_callable)
        applied_commit_ref.revert_callable = None

    def get_commit_stack_states(self):
        return self.commit_stack_states

    def revert(self, base_object: T):
        for revert_callable in reversed(self.revert_callable_list):
            revert_callable(base_object)


class BaseCommitObjectRef(Generic[T]):
    def __init__(self, base_object: T):
        self.object_uuid = uuid.uuid4()
        self.state_uuid = self.object_uuid
        self.base_object = base_object
        self.applied_commit_ref_stack: list[AppliedCommitRefItem] = []

        self._doing_snapshot = False
        self.active_commit_snapshot: Optional[CommitSnapshotItem] = None
        self.commit_snapshot_stack: list[CommitSnapshotItem] = []

    def apply_commit(self, commit_list: list[CommitABC]):
        for commit in commit_list:
            try:
                commit.apply(self.base_object)
                self.state_uuid = commit.commit_uuid
                applied_commit_ref = AppliedCommitRefItem.from_commit(commit)
                self.applied_commit_ref_stack.append(applied_commit_ref)
                if self.doing_snapshot:
                    self._do_commit_snapshot(applied_commit_ref)
            except Exception as e:
                self.state_uuid = UNKNOWN_STATE_UUID
                raise e

    def revert_commit(self, commit_count: int = 1):
        if commit_count == 0:
            return

        if commit_count > 0:
            assert len(self.applied_commit_ref_stack) >= commit_count

        if self._expand_revert_stack_with_snapshot(commit_count) > commit_count:
            raise Exception(f"revert {commit_count} commit but some commit is not full apart of an snapshot.")

        while commit_count > 0:
            applied_commit = self.applied_commit_ref_stack[-1]
            assert applied_commit.commit_uuid == self.state_uuid

            revert_callable_commit_count = self._expand_revert_stack_with_snapshot(1)
            commit_count -= revert_callable_commit_count

            revert_commit = self.applied_commit_ref_stack[-revert_callable_commit_count]

            try:
                if revert_commit.snapshot:
                    self.commit_snapshot_stack[-1].revert(self.base_object)
                    self.commit_snapshot_stack.pop()
                else:
                    revert_commit.revert_callable(self.base_object)
                self.applied_commit_ref_stack = self.applied_commit_ref_stack[:-revert_callable_commit_count]
                if len(self.applied_commit_ref_stack) > 0:
                    self.state_uuid = self.applied_commit_ref_stack[-1].commit_uuid
                else:
                    self.state_uuid = self.object_uuid
            except Exception as e:
                self.state_uuid = UNKNOWN_STATE_UUID
                raise e

    def rebase(self, commit_stack: list[CommitABC]):
        current_state_uuid = self.state_uuid
        target_state_uuid = self.object_uuid
        if len(commit_stack) != 0:
            target_state_uuid = commit_stack[-1].commit_uuid

        if current_state_uuid == target_state_uuid:
            return

        if len(commit_stack) == 0:
            self.reset()
            return

        pawn_state_uuid = current_state_uuid

        commit_stack_states = [self.object_uuid] + [p.commit_uuid for p in commit_stack]
        stack_need_apply = commit_stack
        stack_need_revert_count = 0

        _logger.debug(f"commit_stack_states : {commit_stack_states}")

        if pawn_state_uuid not in commit_stack_states:
            applied_commit_stack = self.applied_commit_ref_stack
            applied_commit_stack_states = [self.object_uuid] + [p.commit_uuid for p in applied_commit_stack]
            _logger.debug(f"applied_commit_stack_states : {applied_commit_stack_states}")
            common_path_len = 0
            for applied_state, need_apply_state in zip(applied_commit_stack_states, commit_stack_states):
                if applied_state != need_apply_state:
                    break
                else:
                    common_path_len += 1
            _logger.debug(f"common_path_len : {common_path_len}")
            assert common_path_len != 0
            stack_need_revert_count = self._expand_revert_stack_with_snapshot(
                len(applied_commit_stack_states[common_path_len:])
            )
            pawn_state_uuid = applied_commit_stack_states[-(stack_need_revert_count + 1)]

        assert pawn_state_uuid in commit_stack_states

        stack_need_apply = commit_stack[commit_stack_states.index(pawn_state_uuid) :]

        self.revert_commit(stack_need_revert_count)
        self.apply_commit(stack_need_apply)

    def do_snapshot(self, snapshot=None):
        if snapshot is None:
            snapshot = CommitSnapshotItem()
        self.active_commit_snapshot = snapshot
        self._doing_snapshot = True

    def stop_snapshot(self):
        self._doing_snapshot = False

    def get_active_snapshot(self):
        assert self.doing_snapshot
        return self.active_commit_snapshot

    def _do_commit_snapshot(self, commit_ref: AppliedCommitRefItem):
        active_snapshot = self.get_active_snapshot()
        active_snapshot.append(commit_ref)
        commit_ref.snapshot = True

        if len(self.commit_snapshot_stack) == 0 or self.commit_snapshot_stack[-1] is not active_snapshot:
            self.commit_snapshot_stack.append(active_snapshot)

    def _expand_revert_stack_with_snapshot(self, revert_commit_count: int):
        assert revert_commit_count > 0

        applied_commit_count = len(self.applied_commit_ref_stack)

        assert applied_commit_count >= revert_commit_count

        if len(self.commit_snapshot_stack) == 0:
            return revert_commit_count

        if not self.applied_commit_ref_stack[-revert_commit_count].snapshot:
            return revert_commit_count

        for commit in reversed(self.applied_commit_ref_stack[:-revert_commit_count]):
            if commit.snapshot:
                revert_commit_count += 1
            else:
                break

        return revert_commit_count

    @property
    def doing_snapshot(self):
        return self._doing_snapshot and self.active_commit_snapshot is not None

    def reset(self):
        self._reset()
        self.applied_commit_ref_stack = []
        self.state_uuid = self.object_uuid
        self.commit_snapshot_stack = []
        self.stop_snapshot()

    def _reset(self):
        self.revert_commit(len(self.applied_commit_ref_stack))


class CommitObjectProxy(Generic[T]):
    def __init__(self, base_object: Union[T, BaseCommitObjectRef[T]] = None):
        self.manager_uuid = uuid.uuid4()
        self.commit_stack: list[CommitABC[T]] = []

        self.base_object_ref: BaseCommitObjectRef[T] = None
        if isinstance(base_object, BaseCommitObjectRef):
            self.base_object_ref = base_object
        else:
            self.base_object_ref = BaseCommitObjectRef(base_object)

    def clone(self):
        new_manager = CommitObjectProxy[T]()
        new_manager.base_object_ref = self.base_object_ref
        new_manager.commit_stack = [*self.commit_stack]
        return new_manager

    def add_commit(self, commit: CommitABC[T]):
        self.commit_stack.append(commit)

    def clone_and_add_commit(self, commit: CommitABC[T]):
        new_self = self.clone()
        new_self.add_commit(commit)
        return new_self

    def clone_and_add_callable_commit(self, commit_callable: Callable[[T], Callable[[T], None]]):
        return self.clone_and_add_commit(CallableCommit(commit_callable))

    def apply_commit_stack(self):
        if self.base_object_ref.state_uuid is UNKNOWN_STATE_UUID:
            self.base_object_ref.reset()

        return self.base_object_ref.rebase(self.commit_stack)

    @property
    def base_object(self):
        return self.base_object_ref.base_object

    def __enter__(self):
        self.apply_commit_stack()
        return self.base_object

    def __exit__(self, exc_type, exc_value, traceback):
        pass


if __name__ in ("__main__", "<run_path>"):
    base = CommitObjectProxy(None)

    def a(o):
        def r(o):
            pass

        return r

    s1 = base.clone_and_add_callable_commit(a)
    s2 = s1.clone_and_add_callable_commit(a)
    s3 = s2.clone_and_add_callable_commit(a)

    base.base_object_ref.do_snapshot()
    with s3:
        print(s3.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_snapshot()

    base.base_object_ref.do_snapshot()
    with s2:
        print(s2.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_snapshot()

    s11 = s1.clone_and_add_callable_commit(a)

    base.base_object_ref.do_snapshot()
    with s11:
        print(s11.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_snapshot()
