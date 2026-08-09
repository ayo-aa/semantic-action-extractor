"""Dataset preparation and adaptation for the research pipeline."""

from .common import DatasetFormatError
from .leakage import (
    DatasetPartition,
    TrainingQuarantineReport,
    build_training_quarantine,
    check_cross_partition_leakage,
    iter_after_document_quarantine,
    load_training_quarantine_report,
)
from .qanom import (
    QANOM_ADAPTER_VERSION,
    QANOM_RELEASE,
    iter_qanom_csv,
    load_qanom_csv,
)
from .qasrl import (
    QASRL_ADAPTER_VERSION,
    QASRL_BANK_RELEASE,
    QASRLDocument,
    QASRL_GOLD_RELEASE,
    QASRLIndex,
    adapt_qasrl_record,
    iter_qasrl_records,
    load_qasrl_index,
)
from .serialization import (
    annotation_record_from_dict,
    iter_adapted_jsonl,
    load_adapted_jsonl,
    load_preparation_manifest,
)

__all__ = [
    "DatasetFormatError",
    "DatasetPartition",
    "QANOM_ADAPTER_VERSION",
    "QANOM_RELEASE",
    "QASRL_ADAPTER_VERSION",
    "QASRL_BANK_RELEASE",
    "QASRLDocument",
    "QASRL_GOLD_RELEASE",
    "QASRLIndex",
    "TrainingQuarantineReport",
    "adapt_qasrl_record",
    "annotation_record_from_dict",
    "build_training_quarantine",
    "check_cross_partition_leakage",
    "iter_after_document_quarantine",
    "iter_adapted_jsonl",
    "iter_qanom_csv",
    "iter_qasrl_records",
    "load_qanom_csv",
    "load_adapted_jsonl",
    "load_preparation_manifest",
    "load_training_quarantine_report",
    "load_qasrl_index",
]
