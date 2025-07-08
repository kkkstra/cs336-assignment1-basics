import os
import pathlib
import pickle
import regex as re
from typing import Iterable, Iterator

from cs336_basics.train_bpe import PAT

class Tokenizer:
    '''
    Construct a tokenizer from a given vocabulary,
    list of merges, and (optionally) a list of special tokens.
    This function should accept the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
    '''
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}

        if special_tokens:
            # Prioritize special tokens by length
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

            # Note "(" and ")"
            self.special_pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"

            id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.token_to_id:
                    self.vocab[id] = token_bytes
                    self.token_to_id[token_bytes] = id
                    id += 1
        else:
            self.special_tokens = []
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # split the text using the special pattern
        if self.special_pattern:
            parts = re.split(self.special_pattern, text)
        else:
            parts = [text]
        # print(f"text: {text}, parts: {parts}")
        token_ids = []
        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.token_to_id[part.encode("UTF-8")])
            else:
                token_ids.extend(self._encode_chunk(part))
        return token_ids
    
    def _encode_chunk(self, text: str) -> list[int]:
        pre_tokens = self._pretokenize(text)
        ids = []

        for p in pre_tokens:
            merged = self._merge_subword(p)
            token_ids = [self.token_to_id[subword] for subword in merged]
            ids.extend(token_ids)

        return ids

    def _merge_subword(self, tokens: tuple[bytes, ...]) -> tuple[bytes, ...]:
        while True:
            best_rank = float("inf")
            best_idx = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            # no more merges
            if best_idx is None:
                return tokens

            # Merge the best pair
            merged = tokens[best_idx] + tokens[best_idx+1]
            tokens = tokens[:best_idx] + tuple([merged]) + tokens[best_idx+2:]

    def _pretokenize(self, text: str) -> list[tuple[bytes,...]]:
        pre_tokens = []

        for match in re.finditer(PAT, text):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            pre_tokens.append(match_bytes)

        return pre_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")
