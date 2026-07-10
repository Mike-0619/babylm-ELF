from __future__ import annotations

from huggingface_hub import HfApi

from .artifacts import RevisionTree, smoke_hub_revision


def publish_revision_tree(
    tree: RevisionTree,
    repo_id: str,
    *,
    private: bool = False,
    smoke: bool = True,
    api: HfApi | None = None,
) -> None:
    api = api or HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    _upload(api, repo_id, "main", tree.main, "Publish BabyLM final checkpoint")
    if smoke:
        smoke_hub_revision(repo_id, "main")
    for revision, artifact in tree.revisions:
        api.create_branch(
            repo_id,
            repo_type="model",
            branch=revision,
            revision="main",
            exist_ok=True,
        )
        _upload(api, repo_id, revision, artifact, f"Publish BabyLM checkpoint {revision}")
        if smoke:
            smoke_hub_revision(repo_id, revision)


def _upload(api: HfApi, repo_id: str, revision: str, artifact, message: str) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        folder_path=artifact,
        commit_message=message,
    )
