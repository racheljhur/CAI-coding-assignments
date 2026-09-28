"""Course-provided evaluation script for A1a (Slot Filling on ATIS).

Model loading and data loading are wired up for you. The one thing left
to implement is `compute_metrics()` — scoring predictions against gold
labels via `scorer.py`. Look for the `# TODO: implement` block below.

Note: `load_model()` is used directly by Gradescope for reproducibility check —
loading a submitted checkpoint and re-running inference.
`load_eval_data()` remains local-dev-only — autograder's prediction-file scoring path —
`crf_predictions.jsonl` / `bilstm_predictions.jsonl` doesn't use it, since it operates on
already-generated predictions, not the underlying data split directly.

Usage:
    python evaluate.py --model crf    --checkpoint-dir checkpoints/crf_1.00    --split dev
    python evaluate.py --model bilstm --checkpoint-dir checkpoints/bilstm_1.00 --split test
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from types import SimpleNamespace
from typing import Any, Literal
from scorer import *
from train import *


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for an evaluation run."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model type to evaluate.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing the trained model to load.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training set to use, e.g. 0.05 / 0.10 / 0.25 / 1.00.",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="Which split to evaluate on (default: dev).",
    )

    parser.add_argument(
        "--saveDir",
        type=str,
        required=True,
        help="directory to write test f1 scores to.",
    )


    return parser.parse_args()


def load_model(model_type: str, checkpoint_dir: str, trainFrac: float, ) -> Any:
    """Construct a model and load its trained weights/state from `checkpoint_dir`.

    Checkpoint format contract (what `train_loop()` in `train.py` must produce):
      - "crf":    `<checkpoint_dir>/crf_model.pkl`, a joblib-dumped
                  `sklearn_crfsuite.CRF` estimator.

      - "bilstm": `<checkpoint_dir>/bilstm_model.pt`, a torch-saved dict with
                  keys `state_dict`, `vocab_size`, `tagset_size`, and
                  `vocab` (itself a dict with `token_to_id` / `id_to_tag`,
                  matching the mapping `BiLSTMModel.predict()` expects).
    """
    model: Any
    
    if model_type == "crf":
        # --- Integration point: `models/crf.py` ------------------------------
        # Same dispatch pattern as `train.py`'s `build_model()`.
        from models.crf import CRFModel
        import joblib
        
        # note that joblib speeds up disk-based caching
        
        model = CRFModel()
        model.model = joblib.load(os.path.join(checkpoint_dir, f"crf_model_{trainFrac}.pkl"))
        return model
        # ---------------------------------------------------------------------

    if model_type == "bilstm":
        # --- Integration point: `models/bilstm.py` ----------------------------
        # Same dispatch pattern as `train.py`'s `build_model()`.
        from models.bilstm import BiLSTMModel
        import torch

        checkpoint = torch.load(
            os.path.join(checkpoint_dir, f"bilstm_model_{trainFrac}.pt"), map_location="cpu"
        )
        
        model = BiLSTMModel(
            vocab_size = checkpoint["vocab_size"],
            tagset_size = checkpoint["tagset_size"],
            word2idx = checkpoint["vocab"], 
            embedding_dim = 300
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.vocab = checkpoint["vocab"]
        return model
        # ---------------------------------------------------------------------

    raise ValueError(f"Unknown model type: {model_type!r}")


def load_eval_data(
    split: Literal["dev", "test"],
) -> tuple[list[list[str]], list[list[str]]]:
    """Load tokenized sentences and gold BIO labels for the requested `split`.

    Returns:
        (`sentences`, `gold_labels`)
    """
    import data_loader

    examples = data_loader.load_split(split)
    sentences = [example.tokens for example in examples]
    gold_labels = [example.slots for example in examples]
    return sentences, gold_labels


def compute_metrics(
    predictions: list[list[str]], gold: list[list[str]]
) -> dict[str, float]:
    """Score predicted BIO tag sequences against gold BIO tag sequences.

    This is the part you implement.

    Args:
        predictions: Predicted BIO label sequences, one per sentence.
        gold: Gold BIO label sequences, one per sentence, aligned
            sentence-for-sentence (and token-for-token within each
            sentence) with `predictions`.

    Returns:
        A dict with at least the keys `"token_f1"` and `"span_f1"`.
    """
    # ------------------------------------------------------------------
    # grab and store f1 metrics.
    # um. so scorer.py exposes the function call needed. you just need to
    # yank the correct key values. 
    return {
        "token_f1": token_scores(gold, predictions)["f1"],
        "span_f1": span_scores(gold, predictions)["f1"],
    }
 
    # `scorer.py` exposes (confirmed, not speculative):
    #     token_scores(y_true, y_pred) -> {"precision": ..., "recall": ..., "f1": ...}
    #     span_scores(y_true, y_pred)  -> {"precision": ..., "recall": ..., "f1": ...}
    # Note the arg order is (y_true, y_pred) i.e. (gold, predictions), not
    # (predictions, gold). Pull `["f1"]` out of each to populate this
    # function's `token_f1`/`span_f1` return keys.
    # ------------------------------------------------------------------
    raise NotImplementedError("compute_metrics() is left for you to implement.")

def show_failures(
    sentences: list[list[str]],
    gold: list[list[str]],
    predictions: list[list[str]],
    n: int = 10,
    seed: int = 0,
) -> str:
    """Print n sentences where at least one predicted tag is wrong,
    token by token, with the wrong tags marked. Returns the text too."""
    import random

    failures = [i for i, (g, p) in enumerate(zip(gold, predictions)) if list(g) != list(p)]
    picked = sorted(random.Random(seed).sample(failures, min(n, len(failures))))

    lines = [f"{len(failures)}/{len(gold)} sentences have at least one wrong tag. "
             f"Showing {len(picked)}:"]
    for i in picked:
        lines.append(f"\n--- Example {i}: {' '.join(sentences[i])}")
        w = max(len(t) for t in sentences[i])
        for tok, g, p in zip(sentences[i], gold[i], predictions[i]):
            mark = "   <-- WRONG" if g != p else ""
            lines.append(f"  {tok:<{w}}  gold={g:<30} pred={p}{mark}")
        if len(predictions[i]) != len(gold[i]):
            lines.append(f"  (length mismatch: gold {len(gold[i])}, pred {len(predictions[i])})")

    text = "\n".join(lines)
    print(text)
    return text

def write_predictions(
    sentences: list[list[str]],
    predictions: list[list[str]],
    path: str,
) -> None:
    """Write one JSON object per line: {"index", "tokens", "slots"}."""
    import json

    assert len(sentences) == len(predictions), "sentence/prediction count mismatch"
    with open(path, "w", encoding="utf-8") as f:
        for i, (toks, preds) in enumerate(zip(sentences, predictions)):
            preds = [str(p) for p in preds]  # guard against numpy strings
            if len(preds) != len(toks):
                raise ValueError(
                    f"Sentence {i}: {len(toks)} tokens but {len(preds)} predicted slots"
                )
            f.write(json.dumps({"index": i, "tokens": list(toks), "slots": preds}) + "\n")
    print(f"Wrote {len(predictions)} predictions to {path}")

def main() -> None:
    args = parse_args()

    model = load_model(args.model, args.checkpoint_dir, args.train_fraction)
    sentences, gold_labels = load_eval_data(args.split)

    # `CRFModel.predict()` ignores `vocab`; `BiLSTMModel.predict()` uses it.
    # `getattr()` covers `CRFModel` instances, which never set a `.vocab` attribute.
    vocab = getattr(model, "vocab", None)
    predictions = model.predict(sentences, vocab)

    metrics = compute_metrics(predictions, gold_labels)
    
    tokenF1 = metrics['token_f1']
    spanF1 = metrics['span_f1'] 

    print(f"Evaluation results — model={args.model}, split={args.split}")
    print(f"  Token-level slot F1: {tokenF1:.4f}")
    print(f"  Span-level slot F1:  {spanF1:.4f}")

    os.makedirs(args.saveDir, exist_ok = True)

    failure_text = show_failures(sentences, gold_labels, predictions, n=10)
    with open(os.path.join(args.saveDir, f"failures_{args.model}_trainFrac_{args.train_fraction}.txt"), "w") as f:
        f.write(failure_text)

    f1Both = np.array((tokenF1, spanF1))
    print(f1Both.shape)

    os.makedirs(args.saveDir, exist_ok = True)

    if args.split == "test":
            write_predictions(
                sentences, predictions,
                os.path.join(args.saveDir, f"{args.model}_predictions.jsonl"),
            )

    # write to common npy file.
    #np.save(os.path.join(args.saveDir, f'tokenSpanF1_trainFrac_{args.train_fraction}_bilstm.npy'), f1Both)

if __name__ == "__main__":
    main()
