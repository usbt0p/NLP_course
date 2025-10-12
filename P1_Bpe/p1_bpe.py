import sys
import pickle
from typing import Dict, Iterable, List, Optional, Tuple


class ByteLevelBPE:
    def __init__(self):
        self.merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self.vocab: Dict[Tuple[int, ...], int] = {}
        self.id2bytes: List[bytes] = []

    @staticmethod
    def _to_byte_tokens(s: str) -> List[Tuple[int, ...]]:
        b = s.encode("utf-8")
        return [(x,) for x in b]

    @staticmethod
    def _count_pairs(lines_tokens: List[List[Tuple[int, ...]]]) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        pair_counts = {}
        for line in lines_tokens:
            for i in range(len(line) - 1):
                pair = (line[i], line[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    @staticmethod
    def _merge_in_line(line: List[Tuple[int, ...]],
                       pair: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        merged = []
        i = 0
        while i < len(line):
            if i < len(line) - 1 and line[i] == pair[0] and line[i + 1] == pair[1]:
                merged.append(line[i] + line[i + 1])
                i += 2
            else:
                merged.append(line[i])
                i += 1
        return merged

    def train(self, lines: Iterable[str], vocab_size: int = 1000, max_merges: Optional[int] = None):
        lines_tokens = [self._to_byte_tokens(line) for line in lines]
        self.vocab = {(i,): i for i in range(256)}
        self.id2bytes = [bytes([i]) for i in range(256)]
        merges_done = 0
        while len(self.vocab) < vocab_size and (max_merges is None or merges_done < max_merges):
            pair_counts = self._count_pairs(lines_tokens)
            if not pair_counts:
                break
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(most_frequent_pair)
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            new_token_id = len(self.vocab)
            self.vocab[new_token] = new_token_id
            self.id2bytes.append(bytes(new_token))
            lines_tokens = [self._merge_in_line(line, most_frequent_pair) for line in lines_tokens]
            merges_done += 1
        print(f"Entrenamiento completo con {len(self.vocab)} tokens y {merges_done} merges.")

    def tokenize(self, text: str) -> List[str]:
        tokens = self._to_byte_tokens(text)
        for pair in self.merges:
            tokens = self._merge_in_line(tokens, pair)
        token_ids = [self.vocab.get(token, -1) for token in tokens]
        return [self.id2bytes[token_id].decode("utf-8", errors="ignore") for token_id in token_ids if token_id != -1]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "merges": self.merges,
                "vocab": self.vocab,
                "id2bytes": self.id2bytes,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.merges = data["merges"]
        self.vocab = data["vocab"]
        self.id2bytes = data["id2bytes"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python p1_bpe.py train <input_train_corpus> <output_model_file>")
        print("  python p1_bpe.py eval <input_model_file> <input_text>")
        sys.exit(1)

    mode = sys.argv[1]
    bpe = ByteLevelBPE()

    if mode == "train":
        if len(sys.argv) != 4:
            print("Uso: python p1_bpe.py train <input_train_corpus> <output_model_file>")
            sys.exit(1)
        input_corpus = sys.argv[2]
        output_model = sys.argv[3]
        with open(input_corpus, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bpe.train(lines)
        bpe.save(output_model)

    elif mode == "eval":
        if len(sys.argv) != 4:
            print("Uso: python p1_bpe.py eval <input_model_file> <input_text>")
            sys.exit(1)
        input_model = sys.argv[2]
        input_text = sys.argv[3]
        bpe.load(input_model)
        tokens = bpe.tokenize(input_text)
        print("Tokens:", tokens)

    else:
        print("Modo no reconocido. Usa `train` o `eval`.")
