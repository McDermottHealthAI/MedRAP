import torch
from meds_torchdata import MEDSTorchBatch

from medrap.retrieval_logging import count_unique_retrieved_documents, model_diagnostic_scalars
from medrap.types import ModelOutput, QueryOutput, RetrieverOutput


def _batch() -> MEDSTorchBatch:
    return MEDSTorchBatch(
        code=torch.LongTensor([[1, 0, 0], [2, 3, 0]]),
        numeric_value=torch.zeros(2, 3),
        numeric_value_mask=torch.zeros(2, 3, dtype=torch.bool),
        time_delta_days=torch.zeros(2, 3),
    )


def test_model_diagnostic_scalars_are_grouped_and_curated() -> None:
    predictions = ModelOutput(
        logits=torch.FloatTensor([[0.0], [1.0]]),
        metadata={
            "query_output": QueryOutput(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
            "retriever_output": RetrieverOutput(
                doc_tokens=torch.ones(2, 1, 2, 3, dtype=torch.long),
                doc_attention_mask=torch.ones(2, 1, 2, 3, dtype=torch.bool),
                doc_ids=torch.LongTensor([[[1, 2]], [[2, 3]]]),
                doc_scores=torch.FloatTensor([[[2.0, 1.0]], [[0.5, 0.25]]]),
            ),
        },
    )

    logs = model_diagnostic_scalars(predictions, _batch(), stage="train")

    assert "prediction/train/logits_mean" in logs
    assert "prediction/train/prob_mean" in logs
    assert "query/train/norm_mean" in logs
    assert "retrieval/train/unique_doc_ratio" in logs
    assert "retrieval/train/score_mean" in logs
    assert "retrieval/train/top1_score_mean" in logs
    assert "retrieval/train/score_entropy_mean" not in logs
    assert "retrieval/train/top1_top2_margin_mean" not in logs
    assert "mask/train/pad_fraction" in logs
    assert not any(name.startswith("train/") for name in logs)


def test_model_diagnostic_scalars_include_differentiable_retrieval_scores() -> None:
    predictions = ModelOutput(
        logits=torch.FloatTensor([[0.0, 1.0], [1.0, 0.0]]),
        metadata={"differentiable_doc_scores": torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])},
    )

    logs = model_diagnostic_scalars(predictions, _batch(), stage="val")

    assert "val_diagnostics/retrieval/differentiable/score_mean" not in logs
    assert "val_diagnostics/retrieval/differentiable/score_entropy_mean" in logs
    assert "val_diagnostics/retrieval/differentiable/top1_top2_margin_mean" in logs
    assert "val_diagnostics/prediction/entropy_mean" in logs


def test_validation_diagnostics_use_separate_group_and_prune_score_summaries() -> None:
    predictions = ModelOutput(
        logits=torch.FloatTensor([[0.0], [1.0]]),
        metadata={
            "query_output": QueryOutput(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
            "retriever_output": RetrieverOutput(
                doc_tokens=torch.ones(2, 1, 2, 3, dtype=torch.long),
                doc_attention_mask=torch.ones(2, 1, 2, 3, dtype=torch.bool),
                doc_ids=torch.LongTensor([[[1, 2]], [[2, 3]]]),
                doc_scores=torch.FloatTensor([[[2.0, 1.0]], [[0.5, 0.25]]]),
            ),
        },
    )

    logs = model_diagnostic_scalars(predictions, _batch(), stage="val")

    assert "val_diagnostics/prediction/logits_std" in logs
    assert "val_diagnostics/query/offdiag_cos_mean" in logs
    assert "val_diagnostics/retrieval/unique_doc_ratio" in logs
    assert "val_diagnostics/retrieval/top1_mode_frac" in logs
    assert "val_diagnostics/retrieval/score_mean" not in logs
    assert "val_diagnostics/retrieval/top1_score_mean" not in logs
    assert "val_diagnostics/mask/pad_fraction" in logs
    assert not any("/val/" in name for name in logs)


def test_count_unique_retrieved_documents_handles_missing_identifiers() -> None:
    retrieval = RetrieverOutput(
        doc_tokens=torch.empty(0, 1, 2, 3, dtype=torch.long),
        doc_attention_mask=torch.empty(0, 1, 2, 3, dtype=torch.bool),
    )

    assert count_unique_retrieved_documents(retrieval) is None
