import collections
import heapq
import multiprocessing
import os
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class ReverseOrderedPairFreqs:
    def __init__(self, pair: tuple[bytes, bytes], freq: int):
        self.pair = pair
        self.freq = freq

    def __lt__(self, other: "ReverseOrderedPairFreqs") -> bool:
        if self.freq == other.freq:
            return self.pair > other.pair
        return self.freq > other.freq
    
    def __repr__(self) -> str:
        return f"ReverseOrderedPairFreqs(pair={self.pair}, freq={self.freq})"

def pre_tokenize_chunk(
    chunk: str,
    special_pattern: str,
) -> dict[tuple[bytes], int]:
    """
    Pre-tokenize a chunk of text and return a frequency dictionary of pre-tokens.
    """
    # Split the chunk into pre-tokens using the special pattern
    sub_chunks = re.split(special_pattern, chunk)
    
    # Create a frequency dictionary for the pre-tokens
    freq_dict = {}
    for chunk in sub_chunks:
        for match in re.finditer(PAT, chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freq_dict[match_bytes] = freq_dict.get(match_bytes, 0) + 1
            
    return freq_dict

def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> dict[tuple[bytes],int]:
    # divide corpus into chunks to parallelize pre-tokenization
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    results = []

    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            results.append(
                pool.apply_async(pre_tokenize_chunk, (chunk, special_pattern))
            )
    pool.close()
    pool.join()

    # combine results from all chunks
    freq_dict = {}
    for result in results:
        chunk_freq_dict = result.get()
        for pre_token, count in chunk_freq_dict.items():
            if pre_token in freq_dict:
                freq_dict[pre_token] += count
            else:
                freq_dict[pre_token] = count
    return freq_dict


def get_pair_freqs(
    freq_dict: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Get the frequency of pairs of tokens.
    """
    pair_freqs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = collections.defaultdict(set)

    for pre_token, count in freq_dict.items():
        if len(pre_token) < 2:
            continue
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pair_freqs[pair] += count
            pairs_to_keys[pair].add(pre_token)
    return pair_freqs, pairs_to_keys

def build_new_key(old_key: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    new_key = []
    i = 0
    while i < len(old_key):
        if i < len(old_key) - 1 and old_key[i] == pair[0] and old_key[i + 1] == pair[1]:
            new_key.append(old_key[i] + old_key[i + 1])
            i += 2
        else:
            new_key.append(old_key[i])
            i += 1
    return tuple(new_key)

def merge(
    pair: tuple[bytes, bytes],
    freq_dict: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
) -> set[tuple[bytes, bytes]]:
    changed_pairs = set()
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freq_dict.pop(old_key)
        new_key = build_new_key(old_key, pair)
        freq_dict[new_key] = freq_dict.get(new_key, 0) + old_freq

        # Decrement frequencies in pair_freqs for old_key's adjacencies
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[left, right] -= old_freq
            changed_pairs.add((left, right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        # Increment frequencies for new_key's adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[left, right] += old_freq
            changed_pairs.add((left, right))
            pairs_to_keys[left, right].add(new_key)

    pairs_to_keys[pair] = set()
    return changed_pairs

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocab and merges
    initial_tokens = [token.encode("utf-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    merges = []

    # pre-tokenization
    freq_dict = pre_tokenize(
        input_path=input_path,
        special_tokens=special_tokens,
    )
    # print(f"Pre-tokenization: {freq_dict=}")

    # get pair frequencies
    pair_freqs, pairs_to_keys = get_pair_freqs(freq_dict)
    # print(f"Pair frequencies: {pair_freqs=}")

    pair_heap = [
        ReverseOrderedPairFreqs(pair, freq) for pair, freq in pair_freqs.items()
    ]
    heapq.heapify(pair_heap)
    # print(f"Initial pair heap: {pair_heap=}")

    # start merging pairs
    for i in range(len(initial_tokens), vocab_size):
        # fetch the most frequent pair
        while pair_heap:
            pair_freq = heapq.heappop(pair_heap)
            top_pair, freq = pair_freq.pair, pair_freq.freq
            
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(pair_heap, ReverseOrderedPairFreqs(top_pair, pair_freqs[top_pair]))
        else:
            break

        vocab[i] = pair[0] + pair[1]
        merges.append(pair)

        changed_pairs = merge(pair, freq_dict, pair_freqs, pairs_to_keys)
        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, ReverseOrderedPairFreqs(cp, pair_freqs[cp]))

    # print(f"Final vocabulary: {vocab=}")
    # print(f"Final merges: {merges=}")

    return vocab, merges

if __name__ == "__main__":
    # Example usage
    input_path = "data/test.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )