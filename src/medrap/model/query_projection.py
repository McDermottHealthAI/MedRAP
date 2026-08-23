"""Query projection modules for mapping patient state into retrieval space.

These components convert encoded patient representations into retrieval queries.
"""

from abc import ABC, abstractmethod

import torch
from meds_torchdata import MEDSTorchBatch
from torch import nn

from ..types import EncoderOutput, QueryOutput


class QueryProjector(nn.Module, ABC):
    """Abstract base for all query projectors.

    Subclasses must implement :meth:`project`, which maps an ``EncoderOutput``
    to a ``QueryOutput``.  The ``forward`` method delegates to ``project`` so
    that the projector can be used as a standard ``nn.Module``.
    """

    @abstractmethod
    def project(self, encoder_out: EncoderOutput, batch: MEDSTorchBatch | None = None) -> QueryOutput:
        """Project encoded patient state into retrieval query space.

        Args:
            encoder_out: ``EncoderOutput`` with ``patient_state`` shaped
                ``(B, S_ehr, D_ehr)`` and an optional ``attention_mask``
                shaped ``(B, S_ehr)``.
            batch: Optional raw ``MEDSTorchBatch`` for projectors that need
                the original code ids (for example to render event codes as
                text). Projectors that only need ``encoder_out`` ignore this.

        Returns:
            A ``QueryOutput`` with ``query_embeddings`` shaped
            ``(B, R, D_ret)``.
        """

    def forward(self, encoder_out: EncoderOutput, batch: MEDSTorchBatch | None = None) -> QueryOutput:
        """Call ``project``."""
        return self.project(encoder_out, batch)


class _LinearQueryProjectorBase(QueryProjector):
    """Shared ``__init__`` for query projectors backed by a single linear layer.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim)


class LinearQueryProjector(_LinearQueryProjectorBase):
    """Tabular query projector producing one retrieval query per patient.

    Projects the last dimension of a ``(B, 1, D_ehr)`` patient state through a
    linear layer, producing ``(B, 1, D_ret)`` with ``R = 1``.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def project(self, encoder_out: EncoderOutput, batch: MEDSTorchBatch | None = None) -> QueryOutput:
        """Project a tabular patient state into a single retrieval query.

        Args:
            encoder_out: ``EncoderOutput`` with ``patient_state`` shaped
                ``(B, 1, D_ehr)``.

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> from medrap.types import EncoderOutput
            >>> projector = LinearQueryProjector(in_dim=2, out_dim=3)
            >>> patient_state = torch.FloatTensor([[[1.0, 2.0]], [[3.0, 4.0]]])
            >>> out = projector.project(EncoderOutput(patient_state=patient_state))
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 3)
            >>> out.query_embeddings.dtype
            torch.float32
            >>> out.retrieval_step_ids is None
            True
            >>> tuple(projector(EncoderOutput(patient_state=patient_state)).query_embeddings.shape)
            (2, 1, 3)
            >>> projector.project(EncoderOutput(patient_state=torch.randn(2, 4)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: LinearQueryProjector expects patient_state shaped (B, 1, D_ehr), ...
        """
        patient_state = encoder_out.patient_state
        if patient_state.ndim != 3 or patient_state.shape[1] != 1:
            raise ValueError(
                "LinearQueryProjector expects patient_state shaped (B, 1, D_ehr), "
                f"got {tuple(patient_state.shape)}"
            )
        return QueryOutput(query_embeddings=self.linear(patient_state.float()))


class SequenceMeanQueryProjector(_LinearQueryProjectorBase):
    """Sequence query projector that mean-pools over the EHR sequence.

    This is a minimal sequence baseline. It reduces ``patient_state`` across the
    sequence dimension and emits a single retrieval query per patient (``R = 1``).
    When ``encoder_out.attention_mask`` is present, padding positions are excluded
    from the mean so retrieval quality does not depend on sequence length.

    Args:
        in_dim: Input patient-state size ``D_ehr``.
        out_dim: Retrieval query size ``D_ret``.
    """

    def project(self, encoder_out: EncoderOutput, batch: MEDSTorchBatch | None = None) -> QueryOutput:
        """Mean-pool over sequence positions and emit one query per sample.

        Args:
            encoder_out: ``EncoderOutput`` with ``patient_state`` shaped
                ``(B, S_ehr, D_ehr)`` and an optional ``attention_mask``
                shaped ``(B, S_ehr)`` (``True`` = valid, non-padding position).

        Returns:
            ``QueryOutput`` with:
                - ``query_embeddings`` shaped ``(B, 1, D_ret)``
                - ``retrieval_step_ids=None``

        Examples:
            >>> from medrap.types import EncoderOutput
            >>> projector = SequenceMeanQueryProjector(in_dim=2, out_dim=2)
            >>> patient_state = torch.FloatTensor(
            ...     [
            ...         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ...         [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
            ...     ]
            ... )
            >>> out = projector.project(EncoderOutput(patient_state=patient_state))
            >>> tuple(out.query_embeddings.shape)
            (2, 1, 2)
            >>> out.query_embeddings.dtype
            torch.float32
            >>> out.retrieval_step_ids is None
            True
            >>> tuple(projector(EncoderOutput(patient_state=patient_state)).query_embeddings.shape)
            (2, 1, 2)
            >>> projector.project(EncoderOutput(patient_state=torch.randn(2, 4)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: SequenceMeanQueryProjector expects patient_state shaped ...

            Trailing padding does not change the query vector once masked:

            >>> padded_state = torch.cat([patient_state, torch.zeros(2, 1, 2)], dim=1)
            >>> mask = torch.BoolTensor([[True, True, True, False], [True, True, True, False]])
            >>> out_padded = projector.project(EncoderOutput(patient_state=padded_state, attention_mask=mask))
            >>> torch.allclose(out_padded.query_embeddings, out.query_embeddings)
            True
        """
        patient_state = encoder_out.patient_state
        if patient_state.ndim != 3:
            raise ValueError(
                "SequenceMeanQueryProjector expects patient_state shaped "
                f"(B, S_ehr, D_ehr), got {tuple(patient_state.shape)}"
            )
        attention_mask = encoder_out.attention_mask
        if attention_mask is None:
            pooled = patient_state.mean(dim=1)
        else:
            mask = attention_mask.bool().unsqueeze(-1)
            counts = mask.sum(dim=1).clamp_min(1).to(dtype=patient_state.dtype)
            pooled = (patient_state.float() * mask).sum(dim=1) / counts
        return QueryOutput(query_embeddings=self.linear(pooled).unsqueeze(1))


def _load_code_text_vocab(code_metadata_path: str) -> list[str]:
    """Load a ``code/vocab_index -> text`` lookup from a MEDS ``codes.parquet``.

    Prefers each code's ``description`` (when present and non-null), falling
    back to the raw ``code`` string. Index 0 (the padding id used throughout
    ``meds_torchdata``) is never a real code, so it is left as ``""``.

    Args:
        code_metadata_path: Path to ``<tensorized_cohort_dir>/metadata/codes.parquet``.

    Returns:
        A list indexed by ``code/vocab_index`` mapping each id to display text.

    Examples:
        >>> import polars as pl
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "codes.parquet"
        ...     pl.DataFrame(
        ...         {
        ...             "code/vocab_index": [1, 2, 3],
        ...             "code": ["DIAG//A", "DIAG//B", "DIAG//C"],
        ...             "description": ["Diabetes", None, "Hypertension"],
        ...         }
        ...     ).write_parquet(fp)
        ...     vocab = _load_code_text_vocab(str(fp))
        >>> vocab
        ['', 'Diabetes', 'DIAG//B', 'Hypertension']

        Missing ``description`` column falls back to ``code`` for every row:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "codes.parquet"
        ...     pl.DataFrame({"code/vocab_index": [1], "code": ["DIAG//A"]}).write_parquet(fp)
        ...     _load_code_text_vocab(str(fp))
        ['', 'DIAG//A']
    """
    import polars as pl

    df = pl.read_parquet(code_metadata_path)
    text_expr = (
        pl.col("description").fill_null(pl.col("code")) if "description" in df.columns else pl.col("code")
    )
    df = df.select(pl.col("code/vocab_index").alias("idx"), text_expr.alias("text"))
    vocab = [""] * (int(df["idx"].max()) + 1)
    for idx, text in df.iter_rows():
        vocab[idx] = text
    return vocab


def _render_batch_text(
    code: torch.Tensor,
    code_text: list[str],
    *,
    max_codes: int,
    separator: str,
) -> list[str]:
    """Render each patient's most recent codes into a display-text string.

    Args:
        code: Integer code ids with shape ``(B, S)``. ``0`` marks padding
            (see :class:`medrap.model.encoders.TimeDeltaRoPEPatientEncoder`).
        code_text: ``code/vocab_index -> text`` lookup from
            :func:`_load_code_text_vocab`.
        max_codes: Maximum number of most-recent (rightmost) valid codes
            rendered per patient.
        separator: String joining rendered code texts.

    Returns:
        One rendered string per row of ``code``.

    Examples:
        >>> import torch
        >>> vocab = ["", "Diabetes", "Hypertension", "Aspirin"]
        >>> code = torch.LongTensor([[1, 2, 3, 0], [0, 0, 0, 0]])
        >>> _render_batch_text(code, vocab, max_codes=2, separator=", ")
        ['Hypertension, Aspirin', '']

        An id outside the vocab falls back to its raw integer:

        >>> _render_batch_text(torch.LongTensor([[9]]), vocab, max_codes=8, separator=", ")
        ['9']
    """
    rows: list[str] = []
    for row in code.tolist():
        valid = [c for c in row if c != 0][-max_codes:]
        texts = [code_text[c] if 0 <= c < len(code_text) else str(c) for c in valid]
        rows.append(separator.join(texts))
    return rows


class Qwen3TextQueryProjector(QueryProjector):
    """Serializes recent codes to text and embeds them with a frozen Qwen3 encoder.

    Each patient's ``max_codes`` most recent event codes are rendered into a
    short text string (via :func:`_render_batch_text`), then embedded with
    the same frozen ``sentence-transformers`` model used to build the
    retrieval document corpus (``medrap-prepare-retrieval-dataset``). Query
    and document embeddings therefore live in the same space by
    construction -- unlike the learned linear projectors above, there is no
    randomly-initialized layer that has to discover this alignment on its
    own from a weak downstream-task gradient.

    Args:
        model_name_or_path: ``sentence-transformers`` model id, matching the
            model used to prepare the retrieval corpus (e.g.
            ``"Qwen/Qwen3-Embedding-0.6B"``).
        code_metadata_path: Path to the tensorized cohort's
            ``metadata/codes.parquet``, used to render code ids as text.
        max_codes: Maximum number of most-recent codes rendered per patient.
        separator: String joining rendered code texts.
        device: Device for the frozen embedder. ``None`` uses the
            ``sentence-transformers`` default placement.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        code_metadata_path: str,
        max_codes: int = 32,
        separator: str = ", ",
        device: str | None = None,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.max_codes = int(max_codes)
        self.separator = separator
        self._embedder = SentenceTransformer(model_name_or_path, device=device)
        self._embedder.eval()
        for parameter in self._embedder.parameters():
            parameter.requires_grad_(False)
        self._code_text = _load_code_text_vocab(code_metadata_path)

    def project(self, encoder_out: EncoderOutput, batch: MEDSTorchBatch | None = None) -> QueryOutput:
        """Render recent codes to text and embed them with the frozen encoder.

        Args:
            encoder_out: Unused except for its device/dtype (this projector
                queries off raw codes, not the learned patient state).
            batch: Raw ``MEDSTorchBatch`` with ``code`` shaped ``(B, S_ehr)``.

        Returns:
            ``QueryOutput`` with ``query_embeddings`` shaped ``(B, 1, D_ret)``
            and ``retrieval_step_ids=None``.

        Raises:
            ValueError: If ``batch`` is ``None``.
        """
        if batch is None:
            raise ValueError("Qwen3TextQueryProjector requires the raw MEDSTorchBatch, got batch=None")
        texts = _render_batch_text(
            batch.code,
            self._code_text,
            max_codes=self.max_codes,
            separator=self.separator,
        )
        with torch.no_grad():
            embeddings = self._embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings = embeddings.to(device=encoder_out.patient_state.device, dtype=torch.float32)
        return QueryOutput(query_embeddings=embeddings.unsqueeze(1))


class ResidualAdapterQueryProjector(QueryProjector):
    """Wraps a base query projector with a learned low-rank residual adapter.

    :class:`Qwen3TextQueryProjector` aligns queries to the frozen retrieval
    doc-key space by construction, but is entirely untrainable -- the model
    has no way to learn which similarity structure actually matters for the
    downstream task, since both sides of retrieval (query and doc keys) are
    fixed pretrained vectors. This wraps any base projector with a small
    trainable residual, ``query = base + up(relu(down(base)))``, so the
    model can learn a task-specific deviation from the base alignment
    instead of nothing at all. The up-projection is zero-initialized, so at
    the start of training this is a pure no-op and behaves identically to
    ``base`` alone -- training starts exactly at the validated alignment and
    only departs from it as gradients justify it.

    Args:
        base: Base query projector (for example a frozen
            :class:`Qwen3TextQueryProjector`).
        dim: Query embedding dimension ``D_ret``.
        rank: Bottleneck dimension of the residual adapter.

    Examples:
        >>> import torch
        >>> from medrap.types import EncoderOutput, QueryOutput
        >>> class _ConstantBase(QueryProjector):
        ...     def project(self, encoder_out, batch=None):
        ...         return QueryOutput(query_embeddings=torch.ones(2, 1, 4))
        >>> adapter = ResidualAdapterQueryProjector(base=_ConstantBase(), dim=4, rank=2)
        >>> encoder_out = EncoderOutput(patient_state=torch.zeros(2, 1, 4))
        >>> out = adapter.project(encoder_out)
        >>> tuple(out.query_embeddings.shape)
        (2, 1, 4)

        Zero-initialized up-projection means the adapter starts as a no-op:

        >>> torch.allclose(out.query_embeddings, torch.ones(2, 1, 4))
        True

        Once the adapter has learned a nonzero residual, the output departs
        from the base projector's embeddings:

        >>> with torch.no_grad():
        ...     _ = adapter.up.weight.fill_(0.1)
        >>> perturbed = adapter.project(encoder_out)
        >>> torch.allclose(perturbed.query_embeddings, torch.ones(2, 1, 4))
        False
    """

    def __init__(self, *, base: QueryProjector, dim: int, rank: int = 64) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Linear(int(dim), int(rank))
        self.up = nn.Linear(int(rank), int(dim))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def project(self, encoder_out: EncoderOutput, batch: MEDSTorchBatch | None = None) -> QueryOutput:
        """Add a learned residual to the base projector's query embeddings.

        Args:
            encoder_out: Passed through to ``base``.
            batch: Passed through to ``base``.

        Returns:
            ``QueryOutput`` with the same shape as ``base``'s output.
        """
        base_out = self.base(encoder_out, batch)
        residual = self.up(torch.relu(self.down(base_out.query_embeddings)))
        return QueryOutput(
            query_embeddings=base_out.query_embeddings + residual,
            retrieval_step_ids=base_out.retrieval_step_ids,
        )
