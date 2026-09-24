"""Course-provided skeleton for training Part A (CRF) and Part B (BiLSTM) models.

In this training script, CLI parsing, data loading, and model construction are wired up for you.
The only one thing left to implement is `train_loop()` — actual training loop,
for both Part A CRF and Part B BiLSTM. Look for the `# TODO: implement` block below.

Usage:
    python train.py --model crf    --train-fraction 1.00 --checkpoint-dir checkpoints/crf_1.00
    python train.py --model bilstm --train-fraction 0.25 --checkpoint-dir checkpoints/bilstm_0.25 \
        --tensorboard-logdir runs/bilstm_0.25
"""

from __future__ import annotations
import argparse
import random
from collections import Counter
from typing import Any
import numpy as np
import torch
import os

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Train a CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model to train.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training set to use, e.g. 0.05 / 0.10 / 0.25 / 1.00.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save the trained model to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default=None,
        help="TensorBoard log directory. Only used when --model BiLSTM; "
        "ignored for --model crf (CRF training doesn't need TensorBoard or a GPU).",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch — CPU + CUDA — for reproducibility.
        ignoreMe: this function syntax uses type hinting.
                  seed: int hints that the kwarg dtype is an int
                  -> None hints the return type.

                  this is done not to actually declare dtypes (as in cpp)
                  and accelerate the compile process, but for documentation
                  and error detection.
                  if you're using an IDE, if the actual return variable does
                  not match the return type hint, it will throw an error.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(fraction: float, seed: int) -> tuple[Any, Any, Any, Any]:
    """Load the ATIS train/dev/test splits, subsampling the train split.

    ignoreMe: tuple[Any, Any, Any, Any] is suggesting that the tuple
              has 4 elements (not ndims), where each element can be any type.
              e.g., (1,2,3,4) is int, int, int, int
                    ("a", 2, 3.5, True) is str, int, float, bool

    Returns:
        (`train_data`, `dev_data`, `test_data`, `vocab`)

        `train_data`/`dev_data`/`test_data` are `list[data_loader.Example]`.
        `vocab` is the dict returned by `build_vocab()` (see below),
        augmented with `tagset_size` (via `data_loader.slot_label_set()`) —
        needed to construct `BiLSTMModel` in `build_model()`.
    """
    import data_loader

    all_splits = data_loader.load_atis()
    train_data = data_loader.subsample(
        all_splits["train"], fraction=fraction, seed=seed
    )

    dev_data = all_splits["dev"]  # dev data is used to make hyperparameter decisions. this is akin to a validation dataset.
    test_data = all_splits["test"]

    train_sentences = [example.tokens for example in train_data]
    vocab = build_vocab(train_sentences)
    vocab["tagset_size"] = len(data_loader.slot_label_set(train_data))

    return train_data, dev_data, test_data, vocab


# Built here rather than in data_loader.py to avoid touching a
# teammate-owned file; if data_loader.py later grows its own vocab
# utility, this should be removed in favor of that to avoid two
# divergent implementations.
def build_vocab(train_sentences: list[list[str]]) -> dict[str, Any]:
    """Build a token vocabulary from tokenized training sentences.

    Fully implemented — this is infrastructure, not the student-fill part
    of the assignment.

    Reserves index 0 for "<PAD>" and index 1 for "<UNK>". Tokens are then
    ordered by descending frequency, ties broken alphabetically. This
    determinism is required: `token_to_id` must be identical across
    machines/Python versions/runs, or it risks breaking the ±0.3 F1
    reproducibility tolerance in the grading rubric. Do NOT rely on dict or
    set iteration order for this — the explicit `sorted()` key below is
    what makes it deterministic.

    Args:
        train_sentences: Tokenized training sentences.

    Returns:
        A dict with `token_to_id` (Dict[str, int]), `id_to_token`
        (Dict[int, str]), and `vocab_size` (int).
    """
    counts = Counter(token for sentence in train_sentences for token in sentence) # count the number of times a token appears in all sentences.
                                                                                  # Counter(str) is indexable by the str. returns the count.

    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token)) # key = lambda is an argument to invoke unnamed functions.
                                                                               # the colon separates the input from the output.
                                                                               # whatever comes before the colon is the argument 
                                                                               # and whatever comes after is what gets returned.
                                                                               # thus, here, we return a tuple of negated counts, indexed by
                                                                               # a given token.
                                                                               # sorted by default orders in increasing order. now, sorted
                                                                               # on negated counts will return the highest count first.
                                                                               # if there is a tie, the token is ordered by alphabetical order.
    token_to_id = {"<PAD>": 0, "<UNK>": 1} # we are starting an integer vocabulary that maps each token to an
                                           # integer id. the first id is reserved for padding (for shorter sentences)
                                           # and the second id is reserved for unknown tokens, previously unseen or discared due to rarity.
    for token in ordered_tokens:
        token_to_id[token] = len(token_to_id) # we are adding dictionary keys (token) and assigning a value equal to the length
                                              # of the token_to_id dict (which i think should always be 2).
                                              # we are appending a new value for each iteration, so the id updates by +1 each time.
                                              # because we have ordered this already, we now know that increasing integers correspond
                                              # to tokens with decreasing frequency.
    id_to_token = {index: token for token, index in token_to_id.items()} # reverse the items we've constructed.

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(token_to_id),
    }


def build_model(
    model_type: str,
    vocab_size: int | None = None,
    tagset_size: int | None = None,
) -> Any:
    """Construct an untrained model instance for the given model type.

       ignoreMe: int | None = None means "or" between types in Python
                 type annotation. This means that vocab_size could be
                 either an int or None.

    `vocab_size`/`tagset_size` (from `load_data()`'s `vocab`) are required
    for `model_type == "bilstm"` and ignored for `model_type == "crf"` —
    `CRFModel` works on raw token features, not a fixed vocabulary, so
    there's nothing for it to size itself against. This is intentional,
    not a bug, same as `CRFModel.predict()`'s unused `vocab` parameter.
    """
    if model_type == "crf":
        from models.crf import CRFModel

        return CRFModel()

    if model_type == "bilstm":
        from models.bilstm import BiLSTMModel

        if vocab_size is None or tagset_size is None:
            raise ValueError(
                "vocab_size and tagset_size are required for model_type='bilstm'"
            )

        return BiLSTMModel(vocab_size=vocab_size, tagset_size=tagset_size)

    # ignoreMe: note that this is a sequence of early exits, since
    # the return line is nested in each condition.
    # otherwise, the ValueError is returned.

    raise ValueError(f"Unknown model type: {model_type!r}")


def train_loop(
    model: Any,
    xTrain: Any,
    yTrain: Any,
    dev_data: Any,
    args: argparse.Namespace,
) -> float:
    """Train `model` on `train_data`, validating against `dev_data`.

    This is the part you implement.

    Your implementation must:
      - Train `model` on `train_data`.

      - Use `dev_data` for validation (and early stopping, if you choose to implement it).

      - When `args.model == "BiLSTM"`, log training progress to
        TensorBoard via `args.tensorboard_logdir` (e.g. using
        `torch.utils.tensorboard.SummaryWriter`). Not applicable for `args.model == "crf"`.

      - Save the final trained model to `args.checkpoint_dir`. See
        `evaluate.py`'s `load_model()` docstring for the exact checkpoint
        format your saved model must match.

    Returns:
        Final dev-set slot F1 (float). `main()` will print what you return.
    """
    # ------------------------------------------------------------------
    # train loop (v1)
    #
    checkPointDir = args.checkpoint_dir
    trainFrac = args.train_fraction
    os.makedirs(checkPointDir, exist_ok=True)

    if model.model_type == 'crf': 
        # model instance is passed in as an argument. override the default filename.
        model.model.model_filename = os.path.join(f'{checkPointDir}', f'crf_{trainFrac}.pkl')
        return model.fit(xTrain, yTrain)
        
    else:
        tmp = 0 
        # TODO - implement bilstm training loop.

    # ------------------------------------------------------------------ 

    raise NotImplementedError("train_loop() is left for you to implement.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_data, dev_data, test_data, vocab = load_data(args.train_fraction, args.seed)
    # Example(tokens: list[str], slots: list[str], intent: str)

    xTrain = [ex.tokens for ex in train_data]
    yTrain = [ex.slots for ex in train_data]

    # i think you'd need to split the training data such that the first list of list of dicts are for the tokens
    # and the second is for the labels (slots).
    
    # so, a given datapoint is a tuple containing tokens as a list, slots as a list, and the intent as a string.

    if args.model == "bilstm":
        model = build_model(
            args.model, vocab_size=vocab["vocab_size"], tagset_size=vocab["tagset_size"]
        )

    else:
        model = build_model(args.model)

    dev_score = train_loop(model, xTrain, yTrain, dev_data, args)

    import os

    if not os.path.isdir(args.checkpoint_dir) or not os.listdir(args.checkpoint_dir):
        raise RuntimeError(
            f"Expected train_loop() to save a checkpoint to {args.checkpoint_dir!r}, "
            "but the directory is missing or empty."
        )

    print("Training complete.")
    print(f"  Model:          {args.model}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    if dev_score is not None:
        print(f"  Final dev F1:   {dev_score:.4f}")


if __name__ == "__main__":
    main()
