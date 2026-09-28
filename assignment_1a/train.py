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
from evaluate import *
from typing import Any
import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
import json
import numpy as np
import joblib
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
        "--epochs",
        type=int,
        default=100,
        help="numEpochs (for biLSTM only) (default=100)",
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

    # adding some extra keys for the bilstm model. 
    tags = sorted(data_loader.slot_label_set(train_data))
    vocab["tag_to_id"] = {t: i for i, t in enumerate(tags)} 
    vocab["id_to_tag"] = {i: t for t, i in vocab["tag_to_id"].items()}

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
    # note - i don't think you should make the vocab more granular (by tokenizing at the character level)
    # instead, you should check how the word-level embeddings can be improved.
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
    embedding_dim: int | None = None,
    vocab: dict | None = None
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

        return BiLSTMModel(vocab_size=vocab_size, tagset_size=tagset_size, embedding_dim = 300, word2idx = vocab)

    # ignoreMe: note that this is a sequence of early exits, since
    # the return line is nested in each condition.
    # otherwise, the ValueError is returned.

    raise ValueError(f"Unknown model type: {model_type!r}")

# Source - https://stackoverflow.com/a/73704579
# Posted by isle_of_gods, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-26, License - CC BY-SA 4.0

class EarlyStopper:
    def __init__(self, patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    # the idea behind this logic is that we set a max count (patience) for
    # our validation loss to go below some a priori set minimum. if it falls
    # below this, we reset the counter and we keep learning.
    # if the validation loss is strictly greater than the minimum plus some
    # we increase the counter. since the counter is greater than the patience
    # value, the early_stop returns True, which can be used in a break condition.

    # tldr: we stop if the validation loss is above a set value.
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_loop(
    model: Any,
    xTrain: Any,
    yTrain: Any,
    xDev: Any,
    yDev: Any,
    vocab: Any,
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
    num_epochs = args.epochs
    
    os.makedirs(checkPointDir, exist_ok = True)
    
    cwd = os.getcwd()
    metaDataPath = os.path.join(cwd, 'metaData')
    os.makedirs(metaDataPath, exist_ok = True)

    if model.model_type == 'crf': 
        # add meta-data to recall model training details.
        config = {
                'model': 'crf',
                'optim': 'lbfgs',
                'numIter': 100,
                'tokenization': 'manual parse',
                'notes': 'the tokenToVec was done manually. context windows were built with a max perception of one word (i.e, t-1, t+1). all that was added about its neighborhood was about the grammar of the word (e.g., word.isupper).',
                }
        with open(metaDataPath + "/crf_config.json", "w") as f:
            json.dump(config, f, indent=2) 

        # model instance is passed in as an argument. override the default filename.
        #model.model.model_filename = os.path.join(f'{checkPointDir}', f'crf_model_{args.train_fraction}.pkl')        
        model.fit(xTrain, yTrain)

        yPred = model.predict(xDev)
        # compute_metrics kwargs:
        #   predictions: list[list[str]], gold: list[list[str]] 
        #   returns token level and span level f1 scores.
        # 
        #   NOTE - you pass in the ground truth first!
        dev_f1_span = compute_metrics(yDev, yPred)['span_f1']
        dev_f1_token = compute_metrics(yDev, yPred)['token_f1']

        print(f"CRF dev F1 (span level): {dev_f1_span:.4f}")
        print(f"CRF dev F1 (token level): {dev_f1_token:.4f}")
        
        joblib.dump(model.model, os.path.join(checkPointDir, f'crf_model_{args.train_fraction}.pkl'))

        return dev_f1_span, dev_f1_token 

    else:
        tensorBoardDir = args.tensorboard_logdir or os.path.join("runs", f"bilstm_{trainFrac}")
        os.makedirs(tensorBoardDir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorBoardDir)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        early_stopper = EarlyStopper(patience=10, min_delta=0.0)

        # add meta-data to recall model training details.
        config = {
                'model': 'bilstm',
                'numHidden': 1,
                'optim': 'Adam',
                'numIter': num_epochs,
                'tokenization': 'word-level manual parse',
                'token_to_id': 'ordered by frequency, such that increasing integers corresponded to lower frequency',
                'notes on featurization': 'used static word embedding GloVe',
                'notes on tags': 'because nn.functional.cross_entropy() needs logit to logit computations, the tags themselves must also get numerical embeddings. for these, we just assigned each tag an integer. the final layer of the bilstm model maps from h > tag, but this tag might be continuous - there is nothing enfocring that these tags are in any way similar to the tags that we embedded for the ground truth.',
                }
        with open(metaDataPath + "/bilstm_config.json", "w") as f:
            json.dump(config, f, indent=2) 

        best_span, best_token, best_state = -1.0, None, None

        # the following block takes the known token > id map
        # and transforms every token in xTrain to a tensor of ids.
        # this is a necessary step, prior to the random encoding 
        # done by the BiLSTM.nn.embedding() method.
        unk = vocab['token_to_id']["<UNK>"]
        lengths = torch.tensor([len(sent) for sent in xTrain], dtype = torch.long)
        xTrain_ids = torch.zeros((len(xTrain), int(lengths.max())), dtype=torch.long) 
        for i, s in enumerate(xTrain):
            xTrain_ids[i, :len(s)] = torch.tensor([vocab['token_to_id'].get(w, unk) for w in s])

        # i think we need to convert the tags in yTrain into something passable into
        # the cross entropy loss.
        # the -100 is a safeproof measure so the loss ignores missing ids, but i think tag_to_id
        # should map every training tag to some value.
        yTrain_ids = torch.full((len(yTrain), int(lengths.max())), -100, dtype=torch.long)
        for i, slots in enumerate(yTrain):
            yTrain_ids[i, :len(slots)] = torch.tensor([vocab['tag_to_id'][slot] for slot in slots])

        #print(xTrain_ids.size()), print(yTrain_ids.size())
        
        #model.to(device)
        #xTrain_ids = xTrain_ids.to(device)
        #yTrain_ids = yTrain_ids.to(device)
        print(f'Training...')
        for epoch in range(num_epochs): 
            model.train()
            optimizer.zero_grad() 

            yPred = model(xTrain_ids, lengths)  

            loss = criterion(yPred.reshape(-1, yPred.size(-1)), yTrain_ids.reshape(-1))
            loss.backward()
            optimizer.step()

            m = compute_metrics(model.predict(xDev, vocab), yDev)

            writer.add_scalar("loss/train", loss.item(), epoch)
            writer.add_scalar("f1_span/dev", m["span_f1"], epoch)
            writer.add_scalar("f1_token/dev", m["token_f1"], epoch)

            if m["span_f1"] > best_span:
                best_span, best_token = m["span_f1"], m["token_f1"]
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] loss {loss.item():.4f} "
                      f"dev span F1 {m['span_f1']:.4f}")

            # EarlyStopper minimizes, so feed it negative F1
            #if early_stopper.early_stop(-m["span_f1"]):
                #break

        model.load_state_dict(best_state)
        print(f"biLSTM dev F1 (span level): {best_span:.4f}")
        print(f"biLSTM dev F1 (token level): {best_token:.4f}")

        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_size": vocab["vocab_size"],
                "tagset_size": vocab["tagset_size"],
                "vocab": vocab,
            },
            os.path.join(checkPointDir, f"bilstm_model_{trainFrac}.pt"),
        )
        
        writer.close()
        return best_span, best_token

    # ------------------------------------------------------------------ 

    raise NotImplementedError("train_loop() is left for you to implement.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    train_data, dev_data, test_data, vocab = load_data(args.train_fraction, args.seed)
    # Example(tokens: list[str], slots: list[str], intent: str)
 
    xTrain = [ex.tokens for ex in train_data]
    yTrain = [ex.slots for ex in train_data]
    
    xDev = [ex.tokens for ex in dev_data]
    yDev = [ex.slots for ex in dev_data]

    # i think you'd need to split the training data such that the first list of list of dicts are for the tokens
    # and the second is for the labels (slots).
    
    # so, a given datapoint is a tuple containing tokens as a list, slots as a list, and the intent as a string.

    if args.model == "bilstm":
        model = build_model(
            args.model, vocab_size=vocab["vocab_size"], tagset_size=vocab["tagset_size"], embedding_dim = 300, vocab = vocab
        )

    else:
        model = build_model(args.model)

    dev_score_span, dev_score_token = train_loop(model, xTrain, yTrain, xDev, yDev, vocab, args)

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

    if dev_score_span is not None:
        print(f"  Final dev F1:   {dev_score_span:.4f}")


if __name__ == "__main__":
    main()
