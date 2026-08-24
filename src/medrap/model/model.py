"""RAP API model orchestration."""

import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn
from torch.nn import functional as nn_functional

from ..types import FusionInput, ModelOutput
from .heads import LinearHead
from .retrieval_scoring import differentiable_retrieval_scores


def _marginal_class_probabilities(per_doc_logits: Tensor, doc_scores: Tensor) -> Tensor:
    """Marginalize a single ``C``-way mutually-exclusive class distribution over documents.

    Softmaxes ``per_doc_logits`` over its last (class) dimension, so the ``C``
    output probabilities compete and sum to 1 -- correct for a single
    ``MarginalizedBinaryClassificationTask`` (``C=2``: positive vs. negative),
    but wrong for ``C`` *independent* binary tasks, where one task being likely
    says nothing about another. Use :func:`_marginal_binary_logits` for that case
    (see ``RetrievalAugmentedModel``'s ``marginalized_output_mode``).

    Computed in log-space via ``logsumexp`` (mirroring :func:`_marginal_binary_logits`)
    rather than by multiplying softmax outputs directly, which underflows when ``K``
    is large or the per-document/retrieval distributions are sharply peaked.

    Examples:
        With ``K=1``, marginalization is a no-op and the output matches a plain softmax:

        >>> logits = torch.FloatTensor([[[1.5, -2.0, 0.5]]])
        >>> scores = torch.zeros(1, 1)
        >>> torch.allclose(
        ...     _marginal_class_probabilities(logits, scores),
        ...     torch.softmax(logits.squeeze(1), dim=-1),
        ...     atol=1e-5,
        ... )
        True

        Matches direct probability-space marginalization on well-scaled inputs:

        >>> _ = torch.manual_seed(0)
        >>> per_doc = torch.randn(4, 5, 3)
        >>> scores = torch.randn(4, 5)
        >>> stable = _marginal_class_probabilities(per_doc, scores)
        >>> p_ret = torch.softmax(scores, dim=-1)
        >>> p_pred = torch.softmax(per_doc, dim=-1)
        >>> direct = (p_ret.unsqueeze(-1) * p_pred).sum(dim=1)
        >>> torch.allclose(stable, direct, atol=1e-5)
        True
    """
    log_p_ret = nn_functional.log_softmax(doc_scores, dim=-1)
    log_p_pred = nn_functional.log_softmax(per_doc_logits, dim=-1)
    return torch.logsumexp(log_p_ret.unsqueeze(-1) + log_p_pred, dim=1).exp()


def _marginal_binary_logits(per_doc_logits: Tensor, doc_scores: Tensor) -> Tensor:
    """Marginalize ``C`` independent per-task binary probabilities over documents.

    For each of the ``C`` tasks independently, marginalizes
    ``sigmoid(per_doc_logits)`` over the ``K`` retrieved documents, weighted by
    ``softmax(doc_scores)`` over the *document* dimension only -- unlike
    :func:`_marginal_class_probabilities`, tasks never compete for probability
    mass. Returns the corresponding logit (``torch.logit`` of the marginal
    probability) so the result stays compatible with BCE-based losses/metrics
    that expect independent per-task logits.

    Examples:
        >>> logits = torch.zeros(2, 3, 4)
        >>> scores = torch.zeros(2, 3)
        >>> _marginal_binary_logits(logits, scores).shape
        torch.Size([2, 4])
        >>> torch.allclose(_marginal_binary_logits(logits, scores), torch.zeros(2, 4))
        True

        Unlike the categorical marginalization, per-task probabilities don't
        need to sum to 1:

        >>> logits = torch.FloatTensor([[[3.0, 3.0, 3.0]]])
        >>> scores = torch.zeros(1, 1)
        >>> probs = torch.sigmoid(_marginal_binary_logits(logits, scores))
        >>> bool((probs.sum(dim=-1) > 1.5).all())
        True

        With ``K=1`` (a single retrieved document), marginalization is a
        no-op and the output logit equals the input logit:

        >>> logits = torch.FloatTensor([[[1.5, -2.0]]])
        >>> scores = torch.zeros(1, 1)
        >>> torch.allclose(_marginal_binary_logits(logits, scores), logits.squeeze(1), atol=1e-5)
        True

        ``sigmoid`` of the result matches ``MultiTaskBCEMarginalizedLoss``'s
        own (log-space) marginal-probability computation exactly -- this is
        the same quantity the loss already trains against, just now also
        exposed as ``predictions.logits`` for AUROC/accuracy/inference:

        >>> import torch.nn.functional as F
        >>> _ = torch.manual_seed(0)
        >>> per_doc = torch.randn(4, 5, 3)
        >>> scores = torch.randn(4, 5)
        >>> our_probs = torch.sigmoid(_marginal_binary_logits(per_doc, scores))
        >>> log_p_ret = F.log_softmax(scores, dim=-1).unsqueeze(-1)
        >>> loss_probs = torch.logsumexp(log_p_ret + F.logsigmoid(per_doc), dim=1).exp()
        >>> torch.allclose(our_probs, loss_probs, atol=1e-5)
        True
    """
    p_ret = nn_functional.softmax(doc_scores, dim=-1)
    p_pos = (p_ret.unsqueeze(-1) * per_doc_logits.sigmoid()).sum(dim=1)
    eps = torch.finfo(p_pos.dtype).eps
    return torch.logit(p_pos.clamp(eps, 1.0 - eps))


class RetrievalAugmentedModel(nn.Module):
    """Composable pipeline orchestrator for RAP.

    This class wires the RAP stage flow and delegates stage-specific logic to
    injected modules:
    ``encode -> query -> retrieve -> retrieval-encode -> fuse -> pool -> predict``.

    Args:
        encoder: Module implementing patient encoding.
        query_projector: Module mapping patient state to retrieval queries.
        retriever: Module retrieving document payloads from query embeddings.
        retrieval_encoder: Module encoding retrieved payloads into retrieval memory.
        fusion: Module combining patient state and retrieval memory.
        pooling: Module reducing fused state to prediction features.
        head: Module mapping pooled features to task outputs.
        marginalized_retrieval: If true, predict per retrieved document then form
            marginal class logits ``(B, C)``; metadata includes ``per_doc_logits``
            and ``differentiable_doc_scores`` for :class:`MarginalizedRetrievalSupervisedLoss`.
        marginalized_score_similarity: ``\"dot\"`` or ``\"cosine\"`` for recomputed
            query--key scores (should match the retriever's notion of similarity
            when possible).
        marginalized_output_mode: ``\"categorical\"`` (default) marginalizes a
            single ``C``-way mutually-exclusive class distribution with
            ``softmax`` over the class dimension -- correct for
            :class:`MarginalizedBinaryClassificationTask` (``C=2``: positive vs.
            negative), wrong for ``C`` independent binary tasks.
            ``\"binary\"`` marginalizes each of the ``C`` tasks' ``sigmoid``
            probabilities independently (softmax only over the retrieved
            *documents*) and returns BCE-compatible logits -- use this for
            :class:`MultiTaskBinaryClassificationTask`
            (``training/loss=multitask_binary_bce_marginalized``).
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        query_projector: nn.Module,
        retriever: nn.Module,
        retrieval_encoder: nn.Module,
        fusion: nn.Module,
        pooling: nn.Module,
        head: nn.Module,
        marginalized_retrieval: bool = False,
        marginalized_score_similarity: str = "dot",
        marginalized_output_mode: str = "categorical",
    ) -> None:
        super().__init__()
        if marginalized_output_mode not in ("categorical", "binary"):
            raise ValueError(
                "marginalized_output_mode must be 'categorical' or 'binary', "
                f"got {marginalized_output_mode!r}"
            )
        if marginalized_retrieval and not isinstance(head, LinearHead):
            raise ValueError("marginalized_retrieval requires a LinearHead for per-document logits")
        self.encoder = encoder
        self.query_projector = query_projector
        self.retriever = retriever
        self.retrieval_encoder = retrieval_encoder
        self.fusion = fusion
        self.pooling = pooling
        self.head = head
        self.marginalized_retrieval = bool(marginalized_retrieval)
        self.marginalized_score_similarity = str(marginalized_score_similarity)
        self.marginalized_output_mode = str(marginalized_output_mode)

    def forward(self, batch: MEDSTorchBatch) -> ModelOutput:
        """Run the end-to-end RAP pipeline.

        Args:
            batch: ``MEDSTorchBatch`` input from ``meds_torchdata``.

        Returns:
            ``ModelOutput`` with:
                - ``logits``: task output tensor
                - ``metadata['query_output']``: ``QueryOutput`` from query projection
                - ``metadata['retriever_output']``: ``RetrieverOutput`` from retrieval
                - ``metadata['retrieval_encoder_output']``: ``RetrievalEncoderOutput``
                  from retrieval encoding
                - ``metadata['fusion_output']``: ``FusionOutput`` from fusion

        Examples:
            >>> import torch
            >>> from medrap.model.encoders import MEDSCodeEncoder
            >>> from medrap.model.fusion import ReplaceFusion
            >>> from medrap.model.heads import LinearHead
            >>> from meds_torchdata import MEDSTorchBatch
            >>> from medrap.model.pooling import IdentityPooling
            >>> from medrap.model.query_projection import SequenceMeanQueryProjector
            >>> from medrap.model.retrieval_encoder import MeanPooledRetrievalEncoder
            >>> from medrap.model.retrievers import InMemoryRetriever
            >>> model = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True]]),
            ...     ),
            ...     retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
            ...     fusion=ReplaceFusion(),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=2, out_dim=2),
            ... )
            >>> batch = MEDSTorchBatch(
            ...     code=torch.LongTensor([[101, 0], [42, 7]]),
            ...     numeric_value=torch.zeros((2, 2), dtype=torch.float32),
            ...     numeric_value_mask=torch.zeros((2, 2), dtype=torch.bool),
            ...     time_delta_days=torch.zeros((2, 2), dtype=torch.float32),
            ... )
            >>> out = model.forward(batch=batch)
            >>> tuple(out.logits.shape)
            (2, 2)
            >>> sorted(out.metadata)
            ['fusion_output', 'query_output', 'retrieval_encoder_output', 'retriever_output']

        Marginalized retrieval (per-doc logits and differentiable scores):

            >>> import torch
            >>> from meds_torchdata import MEDSTorchBatch
            >>> from medrap.model.encoders import MEDSCodeEncoder
            >>> from medrap.model.fusion import ReplaceFusion
            >>> from medrap.model.heads import LinearHead
            >>> from medrap.model.pooling import IdentityPooling
            >>> from medrap.model.query_projection import SequenceMeanQueryProjector
            >>> from medrap.model.retrieval_encoder import KeyEmbeddingRetrievalEncoder
            >>> from medrap.model.retrievers import InMemoryRetriever
            >>> from medrap.train.losses import MarginalizedRetrievalSupervisedLoss
            >>> m = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor(
            ...             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ...         ),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, True]]),
            ...         k=2,
            ...         similarity="dot",
            ...     ),
            ...     retrieval_encoder=KeyEmbeddingRetrievalEncoder(),
            ...     fusion=ReplaceFusion(),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=4, out_dim=2),
            ...     marginalized_retrieval=True,
            ... )
            >>> mb = MEDSTorchBatch(
            ...     code=torch.LongTensor([[101, 0], [42, 7]]),
            ...     numeric_value=torch.zeros((2, 2), dtype=torch.float32),
            ...     numeric_value_mask=torch.zeros((2, 2), dtype=torch.bool),
            ...     time_delta_days=torch.zeros((2, 2), dtype=torch.float32),
            ... )
            >>> mb.boolean_value = torch.BoolTensor([True, False])
            >>> mo = m.forward(mb)
            >>> tuple(mo.logits.shape)
            (2, 2)
            >>> tuple(mo.metadata["per_doc_logits"].shape)
            (2, 2, 2)
            >>> mo.metadata["differentiable_doc_scores"].shape
            torch.Size([2, 2])
            >>> loss = MarginalizedRetrievalSupervisedLoss()(mo, torch.FloatTensor([1.0, 0.0]))
            >>> loss.backward()
            >>> (
            ...     m.query_projector.linear.weight.grad is not None
            ...     and m.query_projector.linear.weight.grad.abs().sum() > 0
            ... )
            tensor(True)

        Marginalized retrieval with a fusion that attends to each document
        independently (works at any K, unlike ``CrossAttentionFusion`` below):

            >>> from medrap.model.fusion import PerDocCrossAttentionFusion
            >>> from medrap.model.retrieval_encoder import TokenFeatureRetrievalEncoder
            >>> m_perdoc = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor(
            ...             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ...         ),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, True]]),
            ...         k=2,
            ...     ),
            ...     retrieval_encoder=TokenFeatureRetrievalEncoder(vocab_size=8, embedding_dim=6),
            ...     fusion=PerDocCrossAttentionFusion(
            ...         d_model=4, num_heads=2, ff_dim=8, num_layers=1, d_in_patient=1, d_in_doc=6
            ...     ),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=4, out_dim=2),
            ...     marginalized_retrieval=True,
            ... )
            >>> out_perdoc = m_perdoc(mb)
            >>> tuple(out_perdoc.logits.shape)
            (2, 2)
            >>> tuple(out_perdoc.metadata["per_doc_logits"].shape)
            (2, 2, 2)

        ``marginalized_output_mode="binary"`` for independent multi-task
        targets (e.g. :class:`MultiTaskBinaryClassificationTask`): same
        per-document fusion setup, but ``logits`` no longer sums to a
        probability distribution across tasks -- each task's probability is
        independent:

            >>> m_binary = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor(
            ...             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ...         ),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, True]]),
            ...         k=2,
            ...     ),
            ...     retrieval_encoder=TokenFeatureRetrievalEncoder(vocab_size=8, embedding_dim=6),
            ...     fusion=PerDocCrossAttentionFusion(
            ...         d_model=4, num_heads=2, ff_dim=8, num_layers=1, d_in_patient=1, d_in_doc=6
            ...     ),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=4, out_dim=3),
            ...     marginalized_retrieval=True,
            ...     marginalized_output_mode="binary",
            ... )
            >>> out_binary = m_binary(mb)
            >>> tuple(out_binary.logits.shape)
            (2, 3)
            >>> probs = torch.sigmoid(out_binary.logits)
            >>> bool((probs.sum(dim=-1) - 1.0).abs().gt(1e-4).any())
            True

        Invalid ``marginalized_output_mode`` is rejected eagerly, at
        construction time:

            >>> RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor([[1.0, 0.0, 0.0, 0.0]]),
            ...         doc_tokens=torch.LongTensor([[1, 2]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True]]),
            ...     ),
            ...     retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=8, embedding_dim=2),
            ...     fusion=ReplaceFusion(),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=2, out_dim=2),
            ...     marginalized_output_mode="invalid",
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: marginalized_output_mode must be 'categorical' or 'binary'...

        Marginalized retrieval validation errors:

            >>> from unittest.mock import patch
            >>> from torch import nn
            >>> class _NH(nn.Module):
            ...     def __init__(self) -> None:
            ...         super().__init__()
            ...         self.linear = nn.Linear(4, 2)
            ...
            ...     def forward(self, x):
            ...         return self.linear(x)
            >>> RetrievalAugmentedModel(  # doctest: +ELLIPSIS
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor(
            ...             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ...         ),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, True]]),
            ...         k=2,
            ...         similarity="dot",
            ...     ),
            ...     retrieval_encoder=KeyEmbeddingRetrievalEncoder(),
            ...     fusion=ReplaceFusion(),
            ...     pooling=IdentityPooling(),
            ...     head=_NH(),
            ...     marginalized_retrieval=True,
            ... )
            Traceback (most recent call last):
            ...
            ValueError: marginalized_retrieval requires a LinearHead...
            >>> from medrap.model.fusion import CrossAttentionFusion
            >>> m_wrong_fusion = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor(
            ...             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ...         ),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, True]]),
            ...         k=2,
            ...         similarity="dot",
            ...     ),
            ...     retrieval_encoder=TokenFeatureRetrievalEncoder(vocab_size=8, embedding_dim=4),
            ...     fusion=CrossAttentionFusion(
            ...         d_model=4, num_heads=2, ff_dim=8, num_layers=1, d_in_patient=1, d_in_doc=4
            ...     ),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=4, out_dim=2),
            ...     marginalized_retrieval=True,
            ... )
            >>> m_wrong_fusion(mb)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...fusion module...
            >>> class _NoKeyRet(nn.Module):
            ...     def forward(self, _q):
            ...         from medrap.types import RetrieverOutput  # types stays at top level
            ...
            ...         return RetrieverOutput(
            ...             doc_tokens=torch.zeros(2, 1, 2, 2, dtype=torch.long),
            ...             doc_attention_mask=torch.ones(2, 1, 2, 2, dtype=torch.bool),
            ...             doc_key_embeddings=None,
            ...         )
            >>> from medrap.model.retrieval_encoder import MeanPooledRetrievalEncoder
            >>> m_no_key = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=_NoKeyRet(),
            ...     retrieval_encoder=MeanPooledRetrievalEncoder(vocab_size=16, embedding_dim=4),
            ...     fusion=ReplaceFusion(),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=4, out_dim=2),
            ...     marginalized_retrieval=True,
            ... )
            >>> m_no_key(mb)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...doc_key_embeddings...
            >>> class _BadFus(nn.Module):
            ...     produces_per_document_state = True
            ...
            ...     def forward(self, fusion_input):
            ...         from medrap.types import FusionOutput  # types stays at top level
            ...
            ...         return FusionOutput(fused_state=torch.zeros(2, 4))
            >>> m_bad_fus = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=InMemoryRetriever(
            ...         doc_key_embeddings=torch.FloatTensor(
            ...             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ...         ),
            ...         doc_tokens=torch.LongTensor([[1, 2], [3, 4], [5, 6]]),
            ...         doc_attention_mask=torch.BoolTensor([[True, True], [True, True], [True, True]]),
            ...         k=2,
            ...         similarity="dot",
            ...     ),
            ...     retrieval_encoder=KeyEmbeddingRetrievalEncoder(),
            ...     fusion=_BadFus(),
            ...     pooling=IdentityPooling(),
            ...     head=LinearHead(in_dim=4, out_dim=2),
            ...     marginalized_retrieval=True,
            ... )
            >>> m_bad_fus(mb)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...fused_state...
            >>> with patch(
            ...     "medrap.model.model.differentiable_retrieval_scores", lambda *a, **k: torch.zeros(2, 99)
            ... ):
            ...     m(mb)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: ...differentiable scores...
        """
        encoder_out = self.encoder(batch)
        query_out = self.query_projector(encoder_out, batch)
        retrieval_out = self.retriever(query_out.query_embeddings)
        retrieval_encoded = self.retrieval_encoder(retrieval_out)
        fusion_out = self.fusion(
            FusionInput(
                patient_state=encoder_out.patient_state,
                retrieval_memory=retrieval_encoded.retrieval_memory,
                retrieval_step_ids=query_out.retrieval_step_ids,
                doc_attention_mask=retrieval_out.doc_attention_mask,
            )
        )
        meta: dict[str, object] = {
            "query_output": query_out,
            "retriever_output": retrieval_out,
            "retrieval_encoder_output": retrieval_encoded,
            "fusion_output": fusion_out,
        }
        if self.marginalized_retrieval:
            if not getattr(self.fusion, "produces_per_document_state", False):
                raise ValueError(
                    "marginalized_retrieval requires a fusion module that produces per-document "
                    f"fused_state (B, K, D); {type(self.fusion).__name__} does not "
                    "(set produces_per_document_state=True on it, e.g. ReplaceFusion or "
                    "PerDocCrossAttentionFusion)"
                )
            if retrieval_out.doc_key_embeddings is None:
                raise ValueError(
                    "marginalized_retrieval requires retriever outputs with doc_key_embeddings "
                    "(enable doc_key_embeddings_column for hf_dataset retriever)"
                )
            fused = fusion_out.fused_state
            if fused.ndim != 3:
                raise ValueError(
                    f"marginalized_retrieval expects fused_state shaped (B, K, D); got {tuple(fused.shape)}"
                )
            b, k_docs, d_mem = fused.shape
            num_classes = self.head.linear.out_features
            per_doc_logits = self.head(fused.reshape(-1, d_mem)).view(b, k_docs, num_classes)
            doc_scores = differentiable_retrieval_scores(
                query_out.query_embeddings,
                retrieval_out.doc_key_embeddings,
                similarity=self.marginalized_score_similarity,
            )
            if doc_scores.ndim != 2 or doc_scores.shape[1] != k_docs:
                raise ValueError(
                    "differentiable scores must be (B, K) matching fused documents; "
                    f"got scores={tuple(doc_scores.shape)}, K={k_docs}"
                )
            if self.marginalized_output_mode == "binary":
                logits = _marginal_binary_logits(per_doc_logits, doc_scores)
            else:
                marginal_probs = _marginal_class_probabilities(per_doc_logits, doc_scores)
                logits = marginal_probs.clamp_min(1e-8).log()
            meta["per_doc_logits"] = per_doc_logits
            meta["differentiable_doc_scores"] = doc_scores
        else:
            pooled = self.pooling(fusion_out.fused_state)
            logits = self.head(pooled)

        return ModelOutput(logits=logits, metadata=meta)
