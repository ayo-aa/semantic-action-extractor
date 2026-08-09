"""Pinned identities for the public research archives used by the project."""

from __future__ import annotations

from dataclasses import dataclass
import re


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """One immutable upstream archive identity."""

    key: str
    dataset: str
    release: str
    url: str
    sha256: str
    archive_format: str

    def __post_init__(self) -> None:
        for field_name in ("key", "dataset", "release", "url"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"artifact {field_name} cannot be empty")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("artifact sha256 must contain 64 lowercase hex digits")
        if self.archive_format not in {"tar", "zip"}:
            raise ValueError("archive_format must be tar or zip")


QA_SRL_BANK_2_1 = DatasetArtifact(
    key="qa-srl-bank-2.1",
    dataset="qa-srl-bank",
    release="2.1",
    url="https://qasrl.org/data/qasrl-v2_1.tar",
    sha256="407c4f5e554fbfcc446c2dbfee093312a6f443c53901bda0d0ce64ae3e7301cf",
    archive_format="tar",
)

QA_SRL_GOLD = DatasetArtifact(
    key="qa-srl-gold-standard",
    dataset="qa-srl-gold-standard",
    release="f7c64ae9b6fe48ff3910c3e59850a12ec278bf83",
    url="https://qasrl.org/data/qasrl-gs.tar",
    sha256="2fdbe4274141842cdb22efc04b6330e01455e843dd1a3d6c779bd799e7110eda",
    archive_format="tar",
)

QANOM = DatasetArtifact(
    key="qanom-2020",
    dataset="qanom",
    release="2bce70e8a39b40157ba97f38e1a8ae7619b30162",
    url=(
        "https://raw.githubusercontent.com/kleinay/QANom/"
        "2bce70e8a39b40157ba97f38e1a8ae7619b30162/qanom_dataset.zip"
    ),
    sha256="165c699ba0f8f9e4d093f7b879836e2a169f4fdf8d309fbeafc4f2d03e515cf5",
    archive_format="zip",
)

ARTIFACTS = {
    artifact.key: artifact
    for artifact in (QA_SRL_BANK_2_1, QA_SRL_GOLD, QANOM)
}


def get_artifact(key: str) -> DatasetArtifact:
    try:
        return ARTIFACTS[key]
    except KeyError as error:
        choices = ", ".join(sorted(ARTIFACTS))
        raise KeyError(f"unknown dataset artifact {key!r}; choose from: {choices}") from error
