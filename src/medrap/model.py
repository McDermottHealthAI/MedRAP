"""RAP API model orchestration."""

from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn
from torch.nn import functional as nn_functional

from .heads import LinearHead
from .retrieval_scoring import differentiable_retrieval_scores
from .types import FusionInput, ModelOutput


def _marginal_class_probabilities(per_doc_logits: Tensor, doc_scores: Tensor) -> Tensor:
    p_ret = nn_functional.softmax(doc_scores, dim=-1)
    p_pred = nn_functional.softmax(per_doc_logits, dim=-1)
    return (p_ret.unsqueeze(-1) * p_pred).sum(dim=1)


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
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.query_projector = query_projector
        self.retriever = retriever
        self.retrieval_encoder = retrieval_encoder
        self.fusion = fusion
        self.pooling = pooling
        self.head = head
        self.marginalized_retrieval = bool(marginalized_retrieval)
        self.marginalized_score_similarity = str(marginalized_score_similarity)

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
            >>> from medrap.encoders import MEDSCodeEncoder
            >>> from medrap.fusion import ReplaceFusion
            >>> from medrap.heads import LinearHead
            >>> from meds_torchdata import MEDSTorchBatch
            >>> from medrap.pooling import IdentityPooling
            >>> from medrap.query_projection import SequenceMeanQueryProjector
            >>> from medrap.retrieval_encoder import MeanPooledRetrievalEncoder
            >>> from medrap.retrievers import InMemoryRetriever
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
            >>> from medrap.encoders import MEDSCodeEncoder
            >>> from medrap.fusion import ReplaceFusion
            >>> from medrap.heads import LinearHead
            >>> from medrap.pooling import IdentityPooling
            >>> from medrap.query_projection import SequenceMeanQueryProjector
            >>> from medrap.retrieval_encoder import KeyEmbeddingRetrievalEncoder
            >>> from medrap.retrievers import InMemoryRetriever
            >>> from medrap.losses import MarginalizedRetrievalSupervisedLoss
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
        """
        encoder_out = self.encoder(batch)
        query_out = self.query_projector(encoder_out.patient_state)
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
            if not isinstance(self.head, LinearHead):
                raise ValueError("marginalized_retrieval requires a LinearHead for per-document logits")
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
            marginal_probs = _marginal_class_probabilities(per_doc_logits, doc_scores)
            logits = marginal_probs.clamp_min(1e-8).log()
            meta["per_doc_logits"] = per_doc_logits
            meta["differentiable_doc_scores"] = doc_scores
        else:
            pooled = self.pooling(fusion_out.fused_state)
            logits = self.head(pooled)

        return ModelOutput(logits=logits, metadata=meta)
