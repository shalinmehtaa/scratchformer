import time
import json
import heapq
import base64
import pstats
import cProfile
import argparse
from tqdm.auto import tqdm
from multiprocessing import Pool
from contextlib import contextmanager
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

from ..pretokenization import pretokenize


# Context manager for timing specific sections
@contextmanager
def timer(description: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{description}: {end - start:.3f} seconds")


class Trainer:
    """An efficient implementation of the Byte Pair Encoding tokenization algorithm"""
    
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        """Initialize the class"""
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        num_special_tokens = len(self.special_tokens)
        self.num_merges = self.vocab_size - (256 + num_special_tokens)

        if self.num_merges < 0:
            raise ValueError("vocab_size must be greater than 256 + the number of special tokens")
        
    def _pairs_in(self, pretoken: List[int]) -> List[Tuple[int, int]]:
        """Get the token pairs in a list of tokens"""
        if len(pretoken) < 2:
            return []
        # List of pairs
        return [(pretoken[i], pretoken[i+1]) for i in range(len(pretoken)-1)]
    
    def _build_pair_index(self, 
                          pretokens: List[List[int]],
                          freqs: List[int]) -> Tuple[Dict[Tuple[int, int], int],
                                                     Dict[Tuple[int, int], set]]:
        """Count token pairs for each unique pretoken weighted by frequency, and build index of pairs to pretokens."""
        pair_counts = Counter()
        pair_index: Dict[Tuple[int, int], set] = defaultdict(set)
        
        for idx, (pretoken, freq) in tqdm(enumerate(zip(pretokens, freqs)), 
                                          total=len(pretokens), 
                                          desc="Indexing pairs"):
            pairs = self._pairs_in(pretoken)
            if not pairs:
                continue
            local_counts = Counter(pairs)
            for p, c in local_counts.items():
                pair_counts[p] += c * freq
                pair_index[p].add(idx)
        
        return dict(pair_counts), pair_index
            
    def _build_heap(self, pair_counts: Dict[Tuple[int, int], int],
                    vocab: Dict[int, bytes]) -> list:
        """Build a max-heap of (-count, tiebreaker, pair) tuples."""
        heap = []
        for pair, count in pair_counts.items():
            tb = (vocab[pair[0]], vocab[pair[1]])
            heapq.heappush(heap, (-count, tb, pair))
        return heap

    def _heap_pop_valid(self, heap: list,
                        pair_counts: Dict[Tuple[int, int], int],
                        vocab: Dict[int, bytes]) -> Optional[Tuple[int, int]]:
        """Pop the highest-count pair whose count is still current (lazy deletion)."""
        while heap:
            neg_count, _tb, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if current == 0:
                continue
            if current == -neg_count:
                return pair
            # Count changed since push — re-insert with correct count
            tb = (vocab.get(pair[0], b""), vocab.get(pair[1], b""))
            heapq.heappush(heap, (-current, tb, pair))
        return None

    def _merge_all_occurrences(self, 
                               pretoken: List[int],
                               left_token: int, 
                               right_token: int, 
                               next_token_id: int) -> Tuple[List[int], bool]:
        """Apply merges to pretokens that have the pair with the max counts"""
        n = len(pretoken)

        if n < 2:
            return pretoken, False
        
        out = list()
        i = 0
        changed = False
        while i < n:
            if i + 1 < n and pretoken[i] == left_token and pretoken[i+1] == right_token:
                out.append(next_token_id)
                i += 2
                changed = True
            else:
                out.append(pretoken[i])
                i += 1

        return out, changed

    def _update_counts_and_index(self,
                                 old_pairs: List[Tuple[int, int]], 
                                 new_pairs: List[Tuple[int, int]],
                                 pretoken_idx: int,
                                 freq: int,
                                 pair_counts: Dict[Tuple[int, int], int],
                                 pair_index: Dict[Tuple[int, int], set]):
        """Incrementally update the pair counts (weighted by freq) and pair index; important for speed-up"""
        # Update global pair counts: remove all old pairs from this pretoken, add all new pairs
        if old_pairs:
            counts_old = Counter(old_pairs)
            for pair, count in counts_old.items():
                global_counts_left = pair_counts.get(pair, 0) - count * freq
                if global_counts_left > 0:
                    pair_counts[pair] = global_counts_left
                else:
                    pair_counts.pop(pair, None)
        if new_pairs:
            counts_new = Counter(new_pairs)
            for pair, count in counts_new.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + count * freq

        # Update global index: drop idx from pairs no longer present; add for new pairs
        old_set = set(old_pairs)
        new_set = set(new_pairs)

        for pair in old_set - new_set:
            s = pair_index.get(pair)
            if s is not None:
                s.discard(pretoken_idx)
                if not s:
                    pair_index.pop(pair, None)

        for pair in new_set:
            pair_index.setdefault(pair, set()).add(pretoken_idx)

        return pair_counts, pair_index
    
    def train(self,
              pretokens: List[bytes]) -> Tuple[Dict[int, bytes], List[bytes]]:
        """Run the training algorithm"""
        # Initialize vocabulary
        vocab = {i: bytes([i]) for i in range(256)}

        # Add special tokens to the vocabulary
        next_token_id = len(vocab)
        for special_token in self.special_tokens:
            vocab[next_token_id] = special_token.encode("utf-8")
            next_token_id += 1

        # Aggregate identical pretokens to reduce work
        pretoken_freqs = Counter(pretokens)
        pretokens_unique = list(pretoken_freqs.keys())   # bytes objects
        freqs = [pretoken_freqs[p] for p in pretokens_unique]

        # Get initial weighted byte pair counts and index
        pair_counts, pair_index = self._build_pair_index(pretokens_unique, freqs)

        # Build max-heap for O(log n) pair selection instead of O(n) scan
        heap = self._build_heap(pair_counts, vocab)

        merges: List[Tuple[bytes, bytes]] = list()

        for _ in tqdm(range(self.num_merges), desc="Merging"):

            if not pair_counts:
                break

            max_pair = self._heap_pop_valid(heap, pair_counts, vocab)
            if max_pair is None:
                break

            left_token, right_token = max_pair

            affected = list(pair_index.get(max_pair, ()))
            if not affected:
                pair_counts.pop(max_pair, None)
                pair_index.pop(max_pair, None)
                continue

            # Record merge and new token
            merges.append((vocab[left_token], vocab[right_token]))
            merged_token = vocab[left_token] + vocab[right_token]
            vocab[next_token_id] = merged_token

            # For each affected unique pretoken, remerge and update counts+index
            for pretoken_idx in affected:
                old_pretoken = pretokens_unique[pretoken_idx]
                old_pairs = self._pairs_in(old_pretoken)

                new_pretoken, changed = self._merge_all_occurrences(old_pretoken,
                                                                    left_token,
                                                                    right_token,
                                                                    next_token_id)
                if not changed:
                    continue

                pretokens_unique[pretoken_idx] = new_pretoken
                new_pairs = self._pairs_in(new_pretoken)

                pair_counts, pair_index = self._update_counts_and_index(old_pairs,
                                                                        new_pairs,
                                                                        pretoken_idx,
                                                                        freqs[pretoken_idx],
                                                                        pair_counts,
                                                                        pair_index)
                # Push new/updated pairs onto the heap
                for pair in set(new_pairs):
                    c = pair_counts.get(pair, 0)
                    if c > 0:
                        tb = (vocab.get(pair[0], b""), vocab.get(pair[1], b""))
                        heapq.heappush(heap, (-c, tb, pair))

            pair_counts.pop(max_pair, None)
            pair_index.pop(max_pair, None)

            next_token_id += 1

        return vocab, merges


def main(file_path: str,
         vocab_size: int,
         special_tokens_path: Optional[str] = None,
         special_tokens: Optional[List[str]] = ["<|endoftext|>"],
         max_workers: Optional[int] = 8,
         num_chunks: Optional[int] = 8):
    """Pretokenize in chunks and then train a BPE tokenizer"""

    with timer("Pretokenization time"):
        pretokens = pretokenize(
            file_path=file_path,
            max_workers=max_workers,
            num_chunks=num_chunks,
            special_tokens_path=special_tokens_path
        )
        print("Pretokenization complete.")

    # Read special tokens here for BPE vocab
    if special_tokens_path is not None:
        try:
            with open(special_tokens_path, "r") as special_tokens_file:
                special_tokens = [line.strip() for line in special_tokens_file]
        except Exception as e:
            print(f"Warning: Could not read special tokens from {special_tokens_path}: {e}")

    with timer("BPE training time"):
        bpe = Trainer(vocab_size, special_tokens)
        vocab, merges = bpe.train(pretokens)
    
    return vocab, merges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        required=True,
        help="Path to the input text file used for training"
    )
    parser.add_argument(
        "--vocab_size",
        "-v",
        type=int,
        required=True,
        help="Target vocabulary size (including bytes and special tokens)"
    )
    parser.add_argument(
        "--special_tokens_path",
        "-sp",
        type=str,
        required=False,
        help="Path to the text file with special tokens (one per line)"
    )
    parser.add_argument(
        "--max_workers", 
        "-w",
        type=int,
        required=False,
        help="Number of concurrent worker processes to use for pretokenization"
    )
    parser.add_argument(
        "--num_chunks", 
        "-n", 
        type=int,
        required=False,
        help="Number of chunks to split the input text into"
    )
    parser.add_argument(
        "--tokenizer_out_path",
        "-top",
        type=str,
        required=False,
        default="data/tokenizer.json",
        help="Output path for saving the single tokenizer JSON (vocab+merges)."
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Enable profiling and save stats to bpe_profile.prof"
    )
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        vocab, merges = main(
            file_path=args.file_path,
            vocab_size=args.vocab_size,
            special_tokens_path=args.special_tokens_path,
            max_workers=args.max_workers,
            num_chunks=args.num_chunks
        )
        profiler.disable()
        profiler.dump_stats('bpe_profile.prof')

        stats = pstats.Stats('bpe_profile.prof')
        print("\n=== TOP FUNCTIONS BY CUMULATIVE TIME ===")
        stats.sort_stats('cumulative').print_stats(15)
        print("\n=== TOP FUNCTIONS BY SELF TIME ===") 
        stats.sort_stats('tottime').print_stats(15)
    else:
        vocab, merges = main(
            file_path=args.file_path,
            vocab_size=args.vocab_size,
            special_tokens_path=args.special_tokens_path,
            max_workers=args.max_workers,
            num_chunks=args.num_chunks
        )

    # Prepare special tokens to persist
    special_tokens_out = None
    if args.special_tokens_path:
        try:
            with open(args.special_tokens_path, "r", encoding="utf-8") as stf:
                special_tokens_out = [line.strip() for line in stf if line.strip()]
        except Exception:
            special_tokens_out = None

    # Single robust JSON: base64-encoded bytes; includes vocab, merges, special tokens
    tokenizer_serializable = {
        "version": 1,
        "byte_encoding": "base64",
        "vocab": {base64.b64encode(v).decode("ascii"): int(k) for k, v in vocab.items()},
        "merges": [
            [base64.b64encode(l).decode("ascii"), base64.b64encode(r).decode("ascii")]
            for (l, r) in merges
        ],
        "special_tokens": special_tokens_out or [],
    }
    with open(args.tokenizer_out_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_serializable, f, indent=2, ensure_ascii=False)
