"""Course-provided skeleton for Part B: BiLSTM slot filling.

Same fixed-interface-with-student-fill pattern as `crf.py`: the
constructor and `predict()` are fully implemented, wiring the model
together per the README's specified architecture (embeddings → BiLSTM →
per-token softmax). The one thing left to implement is `forward()` — the
actual embedding → BiLSTM → tag-space projection pass. Look for the
`# TODO: implement` block below.
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
    ) -> None:
        """Define the model's layers per the README's architecture:
        embeddings -> BiLSTM -> per-token softmax.

        Args:
            vocab_size: Number of distinct tokens in the input vocabulary.
            tagset_size: Number of distinct BIO tags in the output space.
            embedding_dim: Dimensionality of token embeddings.
            hidden_dim: Hidden size of the LSTM (per direction; BiLSTM
                output is `2 * hidden_dim`).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

        # Optional +5 bonus: a handwritten CRF decoding layer (forward/Viterbi)
        # on top of the BiLSTM emissions could be added here, e.g. as
        # `self.crf = CRFLayer(tagset_size)` applied to the output of
        # `hidden2tag` in `forward()`/`predict()`. Not built here — the base
        # BiLSTM already produces per-token softmax logits as specified.

    def forward(
        self, sentences_batch: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Run embedding -> BiLSTM -> linear projection to tag space.

        This is the part you implement.

        Args:
            sentences_batch: `(batch_size, max_seq_len)` tensor of token ids,
                padded to the longest sequence in the batch.
            lengths: `(batch_size,)` tensor of each sequence's true (unpadded)
                length. Typically used with `nn.utils.rnn.pack_padded_sequence`
                / `pad_packed_sequence` so the LSTM ignores padding.

        Returns:
            `(batch_size, max_seq_len, tagset_size)` tensor of per-token,
            per-tag logits (unnormalized — softmax/cross-entropy is applied outside this method).
        """
        # ------------------------------------------------------------------
        # TODO: implement
        # ------------------------------------------------------------------
        raise NotImplementedError("forward() is left for you to implement.")

    def viterbi_decode(
        self, emissions: torch.Tensor, lengths: torch.Tensor
    ) -> list[list[str]]:
        """Hand-written Viterbi decode over BiLSTM emissions.

        STUDENT-IMPLEMENTED, OPTIONAL — backs the README's "+5 bonus: add a
        CRF decoding layer on top of BiLSTM (handwritten forward/Viterbi)."
        Not implemented by default. If you implement this, `predict()`
        picks it up automatically (it calls this first and only falls back
        to plain argmax decoding if this raises `NotImplementedError`) —
        no flag or declaration needed anywhere else. Must be a genuinely
        hand-written forward/Viterbi pass over learned tag-transition
        scores; don't call out to an external CRF library here (e.g.
        `sklearn-crfsuite`) — that's Part A's job.

        Args:
            emissions: `(batch_size, max_seq_len, tagset_size)` tensor of
                per-token, per-tag logits — the same tensor `forward()`
                returns.
            lengths: `(batch_size,)` tensor of each sequence's true
                (unpadded) length.

        Returns:
            A list of predicted BIO label sequences (as tag strings), one
            per input sequence, aligned token-for-token and truncated to
            each sequence's true length.
        """
        # ------------------------------------------------------------------
        # TODO: implement (optional bonus)
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "viterbi_decode() is an optional +5 bonus, not implemented by default."
        )

    def predict(self, sentences: list[list[str]], vocab: Any) -> list[list[str]]:
        """Predict BIO label sequences for tokenized sentences.

        Gives `BiLSTMModel` the same `.predict()` signature as `CRFModel`,
        so `autograder_interface.py` can call either model uniformly.

        Args:
            sentences: A list of tokenized sentences (raw token strings).
            vocab: Vocabulary/label object exposing the token<->id and
                tag<->id mappings built by `data_loader.py` (e.g.
                `vocab.token_to_id`, `vocab.id_to_tag`). Adjust the exact
                attribute names below to match `data_loader.py`'s actual
                interface once it exists.

        Returns:
            A list of predicted BIO label sequences, one per sentence,
            aligned token-for-token with `sentences`.
        """
        # --- Integration point: `data_loader.py`'s vocab interface. ---------
        unk_id = vocab.token_to_id.get("<UNK>", 0)
        id_sequences = [
            [vocab.token_to_id.get(token, unk_id) for token in sentence]
            for sentence in sentences
        ]
        # ---------------------------------------------------------------------

        lengths = torch.tensor([len(seq) for seq in id_sequences], dtype=torch.long)
        max_len = int(lengths.max().item())

        padded = torch.zeros((len(id_sequences), max_len), dtype=torch.long)
        for idx, seq in enumerate(id_sequences):
            padded[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        self.eval()
        with torch.no_grad():
            logits = self.forward(padded, lengths)

            try:
                return self.viterbi_decode(logits, lengths)
            except NotImplementedError:
                pass

            predicted_ids = logits.argmax(dim=-1)

        predictions: list[list[str]] = []

        for idx, length in enumerate(lengths.tolist()):
            tag_ids = predicted_ids[idx, :length].tolist()
            predictions.append([vocab.id_to_tag[tag_id] for tag_id in tag_ids])

        return predictions
