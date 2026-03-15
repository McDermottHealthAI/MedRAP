"""RAP API model orchestration."""

from meds_torchdata import MEDSTorchBatch
from torch import nn

from .types import FusionInput, ModelOutput


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
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.query_projector = query_projector
        self.retriever = retriever
        self.retrieval_encoder = retrieval_encoder
        self.fusion = fusion
        self.pooling = pooling
        self.head = head

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
            >>> from medrap.retrievers import TopKPayloadRetriever
            >>> model = RetrievalAugmentedModel(
            ...     encoder=MEDSCodeEncoder(),
            ...     query_projector=SequenceMeanQueryProjector(in_dim=1, out_dim=4),
            ...     retriever=TopKPayloadRetriever(
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
        pooled = self.pooling(fusion_out.fused_state)
        logits = self.head(pooled)

        return ModelOutput(
            logits=logits,
            metadata={
                "query_output": query_out,
                "retriever_output": retrieval_out,
                "retrieval_encoder_output": retrieval_encoded,
                "fusion_output": fusion_out,
            },
        )
