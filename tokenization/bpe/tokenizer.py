import json
import heapq
import base64
import argparse
from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Self,
    Iterable,
    Iterator
)
import numpy as np
from tqdm.auto import tqdm
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from ..pretokenization import pretokenize_chunk

class Tokenizer:

    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[Tuple[bytes, bytes]], 
                 special_tokens: Optional[List[str]] = None):
        """Initialize tokenizer with vocab, merges and special tokens"""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else list()

         # Append special tokens to the vocab if not already there
        existing_values = set(self.vocab.values())
        next_id = max(self.vocab.keys()) + 1
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in existing_values:
                self.vocab[next_id] = special_token_bytes
                existing_values.add(special_token_bytes)
                next_id += 1

        # Fast lookup from bytes token to id
        self._token_to_id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Map raw bytes of special token text -> its id for single-token emission
        self._special_token_bytes_to_id: Dict[bytes, int] = {}
        for s in self.special_tokens:
            b = s.encode("utf-8")
            tok_id = self._token_to_id.get(b)
            if tok_id is not None:
                self._special_token_bytes_to_id[b] = tok_id

        # Rank of each merge pair (lower index = higher priority)
        self._merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_file(cls, tokenizer_filepath: str) -> Self:
        with open(tokenizer_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "vocab" not in data or "merges" not in data:
            raise ValueError("Invalid tokenizer file: expected keys 'vocab' and 'merges'.")

        enc = data.get("byte_encoding", "base64").lower()
        if enc != "base64":
            raise ValueError(f"Unsupported byte_encoding: {enc!r}; expected 'base64'.")

        vocab_items = data["vocab"].items()
        vocab: Dict[int, bytes] = {int(id_): base64.b64decode(tok) for tok, id_ in vocab_items}

        merges_raw = data["merges"]
        merges: List[Tuple[bytes, bytes]] = []
        for pair in merges_raw:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid merge pair: {pair!r}")
            left = base64.b64decode(pair[0])
            right = base64.b64decode(pair[1])
            merges.append((left, right))

        special_tokens = data.get("special_tokens") or []
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> Iterator[int]:
        # Emit special token as a single id if matched exactly
        st_id = self._special_token_bytes_to_id.get(pretoken)
        if st_id is not None:
            yield st_id
            return

        n = len(pretoken)
        if n == 0:
            return
        if n == 1:
            yield self._token_to_id[pretoken]
            return

        # Doubly-linked list: each node is [token_bytes, prev_idx, next_idx]
        # Using parallel arrays for speed
        tok = [bytes([b]) for b in pretoken]
        prev_node = [i - 1 for i in range(n)]
        next_node = [i + 1 for i in range(n)]
        alive = [True] * n

        # Min-heap of (merge_rank, left_idx) — lower rank = higher priority
        heap: List[Tuple[int, int]] = []
        merge_ranks = self._merge_ranks

        for i in range(n - 1):
            pair = (tok[i], tok[i + 1])
            rank = merge_ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, left = heapq.heappop(heap)
            # Lazy deletion: skip if left was consumed or its right neighbor changed
            if not alive[left]:
                continue
            right = next_node[left]
            if right >= n or not alive[right]:
                continue
            # Verify the pair still matches the rank we recorded
            pair = (tok[left], tok[right])
            actual_rank = merge_ranks.get(pair)
            if actual_rank is None or actual_rank != rank:
                continue

            # Merge: combine left and right into left node
            tok[left] = tok[left] + tok[right]
            alive[right] = False

            # Update linked list: skip right
            nxt = next_node[right]
            next_node[left] = nxt
            if nxt < n:
                prev_node[nxt] = left

            # Check new pair with left's predecessor
            prv = prev_node[left]
            if prv >= 0 and alive[prv]:
                new_pair = (tok[prv], tok[left])
                r = merge_ranks.get(new_pair)
                if r is not None:
                    heapq.heappush(heap, (r, prv))

            # Check new pair with left's new successor
            if nxt < n and alive[nxt]:
                new_pair = (tok[left], tok[nxt])
                r = merge_ranks.get(new_pair)
                if r is not None:
                    heapq.heappush(heap, (r, left))

        # Emit tokens by walking the linked list
        i = 0
        while i < n:
            if alive[i]:
                yield self._token_to_id[tok[i]]
            i = next_node[i] if alive[i] else i + 1

    def encode(self, text: str) -> List[int]:
        pretokens: List[bytes] = pretokenize_chunk(chunk=text, 
                                                   special_tokens=self.special_tokens,
                                                   training=False)
        ids: List[int] = []
        for pretoken in pretokens:
            ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for pretoken in pretokenize_chunk(chunk=text, 
                                              special_tokens=self.special_tokens,
                                              training=False):
                yield from self._encode_pretoken(pretoken)

    def decode(self, ids: List[int]) -> str:
        byte_stream = b"".join(self.vocab[id] for id in ids)
        return byte_stream.decode("utf-8", errors="replace")


def _iter_texts(dataset_path: str) -> Iterator[str]:
    """Yield text lines from a file or all .txt files in a directory, lazily."""
    p = Path(dataset_path)
    if p.is_dir():
        files = sorted(fp for fp in p.rglob("*.txt") if fp.is_file())
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    yield line
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                yield line


def _read_special_tokens(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_special_id(tokenizer: "Tokenizer", token_text: str) -> int:
    ids = tokenizer.encode(token_text)
    if len(ids) != 1:
        raise ValueError(f"Special token {token_text!r} did not resolve to a single id.")
    return ids[0]


def _ensure_uint16_capacity(tokenizer: "Tokenizer"):
    max_id = max(tokenizer.vocab.keys())
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer vocab ids exceed uint16 capacity (max id {max_id}). "
            f"Use a smaller vocab or a wider dtype."
        )


def _tokenize_to_bin(tokenizer: "Tokenizer",
                     dataset_path: str,
                     out_path: str,
                     append_eot: Optional[str],
                     buffer_size: int = 1_000_000,
                     show_progress: bool = True) -> int:
    """Stream token ids to a raw .bin file as uint16; returns total token count."""
    eot_id: Optional[int] = None
    if append_eot:
        eot_id = _resolve_special_id(tokenizer, append_eot)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    buf: List[int] = []

    itr = _iter_texts(dataset_path)
    if show_progress:
        itr = tqdm(itr, desc="Tokenizing (bin)", unit="lines")

    with open(out_path, "wb") as f:
        for text in itr:
            for tok in tokenizer.encode_iterable([text]):
                buf.append(tok)
                if len(buf) >= buffer_size:
                    np.asarray(buf, dtype=np.uint16).tofile(f)
                    total += len(buf)
                    buf.clear()
            if eot_id is not None:
                buf.append(eot_id)
                if len(buf) >= buffer_size:
                    np.asarray(buf, dtype=np.uint16).tofile(f)
                    total += len(buf)
                    buf.clear()

        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
            buf.clear()

    return total

def _write_meta(meta_path: str, meta: dict):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# Global tokenizer reference for worker processes
_G_TOKENIZER = None

def _worker_init_tokenizer(tokenizer_path: str, special_tokens: Optional[List[str]]):
    global _G_TOKENIZER
    tok = Tokenizer.from_file(tokenizer_path)
    if special_tokens is not None:
        tok.special_tokens = special_tokens
        tok._special_token_bytes_to_id = {}
        for s in tok.special_tokens:
            b = s.encode("utf-8")
            tok_id = tok._token_to_id.get(b)
            if tok_id is not None:
                tok._special_token_bytes_to_id[b] = tok_id
    _G_TOKENIZER = tok

def _encode_lines_worker(lines: List[str], eot_id: Optional[int]) -> np.ndarray:
    out: List[int] = []
    for text in lines:
        for tok in _G_TOKENIZER.encode_iterable([text]):
            out.append(tok)
        if eot_id is not None:
            out.append(eot_id)
    return np.asarray(out, dtype=np.uint16)

def _batched_iter_lines(dataset_path: str, batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for line in _iter_texts(dataset_path):
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def _tokenize_to_bin_parallel(tokenizer_path: str,
                              special_tokens: Optional[List[str]],
                              dataset_path: str,
                              out_path: str,
                              eot_id: Optional[int],
                              workers: int,
                              batch_lines: int,
                              show_progress: bool = True) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init_tokenizer,
                             initargs=(tokenizer_path, special_tokens)) as ex:
        batches = _batched_iter_lines(dataset_path, batch_lines)
        results = ex.map(_encode_lines_worker, batches, repeat(eot_id))
        itr = results if not show_progress else tqdm(results, desc="Tokenizing (bin, parallel)", unit="batches")
        with open(out_path, "wb") as f:
            for arr in itr:
                arr.tofile(f)
                total += int(arr.size)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset with a given tokenizer JSON (containing vocab and merges).")
    parser.add_argument(
        "--dataset_path", "-d", type=str, required=True,
        help="Path to the input dataset (a .txt file or a directory containing .txt files)."
    )
    parser.add_argument(
        "--tokenizer_path", "-t", type=str, required=True,
        help="Path to the tokenizer JSON (vocab+merges in one file)."
    )
    parser.add_argument(
        "--special_tokens_path", "-sp", type=str, required=False,
        help="Optional path to a file with special tokens (one per line)."
    )
    parser.add_argument(
        "--out_path", "-o", type=str, required=False,
        help="Output .bin path. Defaults to data/<dataset_name>-tokens.bin"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--batch_lines", "-b", type=int, default=1024,
        help="Number of lines per parallel batch"
    )
    parser.add_argument(
        "--append_eot", action="store_true",
        help="Append an end-of-text special token after each input line."
    )
    parser.add_argument(
        "--eot_token", type=str, default="<|endoftext|>",
        help="Which special token string to append when --append_eot is set."
    )
    parser.add_argument(
        "--no_tqdm", action="store_true",
        help="Disable progress bars."
    )
    args = parser.parse_args()

    # Load tokenizer
    special_tokens = _read_special_tokens(args.special_tokens_path)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    # If explicit special tokens are provided, override
    if special_tokens is not None:
        tokenizer.special_tokens = special_tokens
        # rebuild special bytes->id map
        tokenizer._special_token_bytes_to_id = {}
        for s in tokenizer.special_tokens:
            b = s.encode("utf-8")
            tok_id = tokenizer._token_to_id.get(b)
            if tok_id is not None:
                tokenizer._special_token_bytes_to_id[b] = tok_id

    _ensure_uint16_capacity(tokenizer)

    # Default output path
    base = Path(args.dataset_path)
    name = base.name if base.is_dir() else base.stem
    out_path = args.out_path or str(Path("data") / f"{name}-tokens.bin")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    show_progress = not args.no_tqdm
    eot_id = _resolve_special_id(tokenizer, args.eot_token) if args.append_eot else None

    # Parallel or sequential
    if args.workers > 1:
        total = _tokenize_to_bin_parallel(
            tokenizer_path=args.tokenizer_path,
            special_tokens=special_tokens,
            dataset_path=args.dataset_path,
            out_path=out_path,
            eot_id=eot_id,
            workers=args.workers,
            batch_lines=args.batch_lines,
            show_progress=show_progress,
        )
    else:
        total = _tokenize_to_bin(
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            out_path=out_path,
            append_eot=(args.eot_token if args.append_eot else None),
            show_progress=show_progress,
        )

    meta = {
        "total_tokens": int(total),
        "dtype": "uint16",
        "dataset_path": args.dataset_path,
        "tokenizer_path": args.tokenizer_path,
        "special_tokens_path": args.special_tokens_path,
        "append_eot": bool(args.append_eot),
        "eot_token": args.eot_token if args.append_eot else None,
        "workers": args.workers,
        "batch_lines": args.batch_lines,
    }
    _write_meta(out_path + ".meta.json", meta)
    print(f"Wrote {total} tokens to {out_path} (uint16). Meta: {out_path}.meta.json")