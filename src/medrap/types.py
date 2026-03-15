"""Shared API contract types.

This module defines the stage-by-stage outputs used by the RAP pipeline.

Shape notation:
    - ``B``: batch size.
    - ``S_ehr``: EHR sequence length (1 in tabular mode).
    - ``D_ehr``: EHR hidden / embedding dimension (1 for scaffold encoders).
    - ``S_fused``: fused sequence length (equals ``S_ehr``).
    - ``D_fused``: fused hidden dimension (``D_ehr + D_mem`` for concat,
      ``D_mem`` for replace).
"""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(slots=True)
class EncoderOutput:
    """Output of encode.

    Attributes:
        patient_state: Encoded patient representation with shape
            ``(B, S_ehr, D_ehr)``.
    """

    patient_state: Tensor


@dataclass(slots=True)
class QueryOutput:
    """Output of query projection.

    Attributes:
        query_embeddings: Query tensor with shape ``(B, R, D_ret)``.
        retrieval_step_ids: Optional retrieval step mapping ``(B, S_ehr)``.
            Present in sequence mode (Config A), absent in tabular modes (B/C).
    """

    query_embeddings: Tensor
    retrieval_step_ids: Tensor | None = None


@dataclass(slots=True)
class RetrieverOutput:
    """Output of retrieval.

    Attributes:
        doc_tokens: Retrieved token ids with shape ``(B, R, K, S_doc)``.
        doc_attention_mask: Retrieved token mask with shape ``(B, R, K, S_doc)``.
        doc_scores: Optional retrieval scores with shape ``(B, R, K)``.
        doc_ids: Optional retrieval identifiers with shape ``(B, R, K)``.
        doc_key_embeddings: Optional retrieved key embeddings with shape
            ``(B, R, K, D_ret)`` for in-graph query-key scoring losses.
    """

    doc_tokens: Tensor
    doc_attention_mask: Tensor
    doc_scores: Tensor | None = None
    doc_ids: Tensor | None = None
    doc_key_embeddings: Tensor | None = None


@dataclass(slots=True)
class RetrievalEncoderOutput:
    """Output of retrieval encoding.

    Attributes:
        retrieval_memory: Encoded retrieval memory.
            Supported scaffold shapes include:
            - sequence-style token features: ``(B, R, K, S_doc, D_mem)``
            - tabular pooled memory: ``(B, 1, 1, 1, D_mem)``
    """

    retrieval_memory: Tensor


@dataclass(slots=True)
class FusionInput:
    """Input payload for fusion stages.

    Attributes:
        patient_state: Encoded patient representation with shape
            ``(B, S_ehr, D_ehr)``.
        retrieval_memory: Encoded retrieval memory (for example
            ``(B, R, K, S_doc, D_mem)``).
        retrieval_step_ids: Optional retrieval step mapping ``(B, S_ehr)``.
        doc_attention_mask: Optional retrieval token mask.
    """

    patient_state: Tensor
    retrieval_memory: Tensor
    retrieval_step_ids: Tensor | None = None
    doc_attention_mask: Tensor | None = None


@dataclass(slots=True)
class FusionOutput:
    """Output of fusion.

    Attributes:
        fused_state: Fused representation with shape
            ``(B, S_ehr, D_fused)``.
    """

    fused_state: Tensor


@dataclass(slots=True)
class ModelOutput:
    """Output of full model forward pass.

    Attributes:
        logits: Task logits with shape ``(B, C)``.
        metadata: Optional side outputs (for diagnostics and training utilities).
    """

    logits: Tensor
    metadata: dict[str, object] = field(default_factory=dict)
