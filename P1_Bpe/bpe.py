from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter
import sys
import pickle
from tqdm import tqdm

class ByteLevelBPE:
    """
    Implementación básica de BPE a nivel de bytes.
    - Los tokens iniciales son bytes individuales (0..255).
    - Durante el entrenamiento se obtienen los pares de tokens adyacentes más frecuentes y se fusionan, todo ello de forma iterativa.
    - La codificación (`encode`) aplica las fusiones aprendidas en orden.
    """

    def __init__(self):
        # esto se puede usar para hacer la tokenizaicon  una vez entrenado
        self.merges: List[Tuple[bytes, bytes]] = []  # lista de pares fusionados (tuplas de bytes)
        self.vocab : Dict = {} # mapea tokens (tuplas de bytes) a IDs para codificación
        self.id2bytes: List[bytes] = [] # mapea IDs a bytes (para decodificación)

    @staticmethod
    def _to_byte_tokens(s: str) -> List[Tuple[int, ...]]:
        """
        Devuelve una lista de tokens como tuplas de bytes individuales
        """
        b = s.encode("utf-8")
        return [(x,) for x in b]

    @staticmethod
    def _count_pairs(lines_tokens: List[List[Tuple[int, ...]]]) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        """
        Obtiene las frecuencias de pares de tokens adyacentes en todas las líneas
        """
        count = Counter()
        for line in lines_tokens:
            bigrams = [(line[i], line[i+1]) for i in range(len(line)-1)]
            count.update(bigrams) 
        return count

    @staticmethod
    def _merge_in_line(line: List[Tuple[int, ...]],
                       pair: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """
        Fusiona todas ocurrencias del par `pair` en una línea (sin solapamiento)
        """
        # Iterar hacia atrás para evitar problemas con índices al modificar la lista
        for i in range(len(line)-2, -1, -1):
            if (line[i], line[i+1]) == pair:
                # recotar por delante y por detrás, y añadir la fusión en medio
                line = line[:i] + [line[i] + line[i+1]] + line[i+2:]
        return line
    
    def _init_vocab(self, lines):
        """
        Inicializa el vocabulario con los bytes individuales.
        """
        self.vocab = {(i,): i for i in range(256)}
        self.id2bytes = [bytes([i]) for i in range(256)]


    def train(self, lines: Iterable[str], vocab_size: int = 1000,
              show_progress: bool = True):
        """
        Aprende las fusiones del BPE y construye los vocabularios.
        No se aplica max_merges, se para al alcanzar vocab_size.
        """
        # iicializar el vocabulario, que es el conjunto de bytes individuales
        self._init_vocab(lines)

        # contar frecuencias de pares de byte tokens
        lines_tokens = [self._to_byte_tokens(line) for line in lines]
        
        if show_progress:
            # configurar barra de progreso
            max_merges = vocab_size - len(self.vocab)
            pbar = tqdm(total=max_merges, desc="Entrenando BPE")
        
        # mientras el vocabulario no tenga el tamaño deseado:
        while len(self.vocab) < vocab_size:
            
            # contamos los pares de tokens adyacentes
            pair_counts = self._count_pairs(lines_tokens)
            if not pair_counts:
                break

            # obtenemos el par más frecuente y lo añadimos a las fusiones
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(most_frequent_pair)

            # creamos el nuevo token con su id y lo añadimos al vocabulario
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            new_token_id = len(self.vocab)
            self.vocab[new_token] = new_token_id
            self.id2bytes.append(bytes(new_token))

            # actualizamos las líneas tokenizadas aplicando la fusión
            lines_tokens = [self._merge_in_line(line, most_frequent_pair) for line in lines_tokens]
            
            # actualizar barra de progreso
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({
                    'vocab_size': len(self.vocab),
                    'merges': len(self.merges),
                    'most_freq_count': pair_counts[most_frequent_pair]
                })
        
        if show_progress: pbar.close()
        
        print(f"Entrenamiento completo con {len(self.vocab)} tokens y {len(self.merges)} merges.")  

    def encode(self, text: str) -> List[int]:
        """Convierte el texto de entrada en una lista de token IDs."""
        
        # Convertir a tokens de bytes y aplicar fusiones
        tokens = self._to_byte_tokens(text)
        for pair in self.merges:
            tokens = self._merge_in_line(tokens, pair)
        
        # convertimos tokens a ids usando el vocabulario
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # por si acaso
                print(f"Warning: Token {token} not found in vocabulary")
        
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """Convierte una lista de token IDs en texto."""
        # convertir ids a bytes usando id2bytes
        byte_sequences = []
        for token_id in ids:
            if 0 <= token_id < len(self.id2bytes):
                byte_sequences.append(self.id2bytes[token_id])
            else:
                print(f"Warning: Token ID {token_id} out of range")
        
        # juntamos todos los bytes y decodificamos
        all_bytes = b''.join(byte_sequences)
        try:
            return all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # mantener la longitud del texto sustituyendo bytes inválidos por un caracter de reemplazo
            return all_bytes.decode('utf-8', errors='replace') 
    
    def tokenize(self, text: str) -> List[str]:
        """Tokeniza un texto simplemente, sin convertir a IDs."""
        # Convertir a tokens de bytes y aplicar fusiones
        tokens = self._to_byte_tokens(text)
        for pair in self.merges:
            tokens = self._merge_in_line(tokens, pair)
        
        # Convertir tokens a strings usando el vocabulario
        token_ids = [self.vocab.get(token, -1) for token in tokens]
        return [self.id2bytes[token_id].decode("utf-8", errors="ignore") 
                for token_id in token_ids if token_id != -1] 
    
    def save(self, path: str):
        '''Just saves the merges and vocab into a pickle file'''
        with open(path, "wb") as f:
            pickle.dump({
                "merges": self.merges,
                "vocab": self.vocab,
                "id2bytes": self.id2bytes,
            }, f)

    def load(self, path: str):
        '''Loads the merges and vocab from a pickle file'''
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
        
        # Test de verificación después del entrenamiento
        test_text = "Hola mundo"
        print(f"\n--- Test de verificación ---")
        print(f"Texto de prueba: '{test_text}'")
        
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        tokens = bpe.tokenize(test_text)
        
        print(f"Encoded: {encoded}", end="\n\n")
        print(f"Tokens: {tokens}", end="\n\n")
        print(f"Decoded: '{decoded}'", end="\n\n")
        print(f"¿Codificación correcta?: {test_text == decoded}", end="\n\n")
        
        bpe.save(output_model)
        print(f"Modelo guardado en: {output_model}", end="\n\n")

    elif mode == "eval":
        if len(sys.argv) < 4 or len(sys.argv) > 5:
            print("Uso: python p1_bpe.py eval <input_model_file> <input_text> [-f]")
            print("  -f: indica que <input_text> es un archivo")
            sys.exit(1)
        
        input_model = sys.argv[2]
        input_text_or_file = sys.argv[3]
        
        # Check if -f flag is present
        if len(sys.argv) == 5 and sys.argv[4] == "-f":
            # Read from file
            with open(input_text_or_file, "r", encoding="utf-8") as f:
                input_text = f.read()
        else:
            # Use as direct text
            input_text = input_text_or_file
        
        bpe.load(input_model)
        
        # Mostrar diferentes representaciones
        token_ids = bpe.encode(input_text)
        tokens = bpe.tokenize(input_text)
        decoded_text = bpe.decode(token_ids)
        
        print("Texto original:", repr(input_text), end="\n\n")
        print("Token IDs:", token_ids, end="\n\n")
        print("Tokens (string):", tokens, end="\n\n")
        print("Texto decodificado:", repr(decoded_text), end="\n\n")
        print("¿Decodificación correcta?:", input_text == decoded_text, end="\n\n")

    else:
        print("Modo no reconocido. Usa `train` o `eval`.")



    