import numpy as np
from tqdm import tqdm
# TODO fix pythonpath issue when executing as module

def sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out
    

# TODO 1: Implementa un método de entrenamiento simple, esto es, con learning rate (LR) constante y ventana estática.
class Trainer:
    def _neg_sampling_fix(self):
        # 2: Inicializa `self.neg_prob`, que será usado como distribución de probabilidad 
            # a la hora de hacer el muestreo negativo, de modo que contenga las frecuencias 
            # relativas de cada token del vocabulario elevadas a 2/3.
        freq = self.token_counts / len(self.tokens)
        freq_power = np.power(freq, 2/3) # TODO por que??
        neg_prob = freq_power / np.sum(freq_power) # normalizar para que sume 1
        
        self.neg_prob = neg_prob
        return neg_prob

    def _subsample_data(self):
        # 3: Reduce la ocurrencia de los tokens más frecuentes usando la siguiente fórmula:
        # `p_keep = (np.sqrt(t / f) + t / f) if f > 0 else 1.0`
        # donde `t = 1e-5` y `f` es la frecuencia relativa del token.

        t = 1e-5
        f = self.token_counts / len(self.tokens)
        
        # Calcular probabilidad de mantener cada token
        # FIXME solve division by zero in a proper way
        p_keep = np.where(f > 0, np.sqrt(t / f) + t / f, 1.0)
        
        # *literalmente* eliminar tokens mas frecuentes del conjunto de tokens
        mask = self.rng.random(len(self.tokens)) < p_keep[self.tokens]
        self.tokens = [tok for i, tok in enumerate(self.tokens) if mask[i]]

    def __init__(self, encoded_tokens, rng, embedding_dim, window_size, epochs, lr, lr_min_factor, neg_samples):
        
        # Carga el corpus y tokenízalo usando el tokenizador BPE de la práctica anterior.
        self.tokens = encoded_tokens # the corpus must be tokenized with BPE
        assert len(self.tokens) > 0, "The corpus is empty."
        assert isinstance(self.tokens[0], int), "The corpus must be encoded as a sequence of token ids."

        self.rng = rng
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.lr_min_factor = lr_min_factor 
        self.neg_samples = neg_samples

        # precálculo de frecuencias de tokens ya que se usa en los 2 siguientes métodos
        self.token_counts = np.bincount(self.tokens)
        
        # vocab_size debe ser el tamaño del vocabulario ORIGINAL, no los tokens únicos después del subsampling
        # porque los IDs de los tokens siguen siendo del vocabulario original
        self.vocab_size = len(self.token_counts)
        
        # Aplica ajustes para evitar la sobreponderancia de tokens frecuentes
        self._neg_sampling_fix()
        self._subsample_data() # TODO esto tiene que ir antes o despues de neg sampling fix??

        del self.token_counts 

        # parámetros para activar / desactivar ventana dinámica y LR decreciente
        self.use_dynamic_window = True
        self.use_lr_decay = True

    def sample_neg(self, forbidden):
        # 1.2: Obtén una muestra negativa de tokens, evitando seleccionar aquellos en 
            # `forbidden`, que serán los que estén dentro de la ventana actual.

        neg_samples = []
        while len(neg_samples) < self.neg_samples:
            sampled_token = self.rng.choice(
                self.vocab_size, p=self.neg_prob
            )
            if sampled_token not in forbidden:
                neg_samples.append(sampled_token)

        return neg_samples

    def words_in_context(self, index):
        '''Devuelve los índices de los tokens en el contexto del token en `index`.'''
        start = max(0, index - self.window_size)
        end = min(len(self.tokens), index + self.window_size + 1)
        context_indices = list(range(start, index)) + list(range(index + 1, end))
        return context_indices

    def train(self):
        # TODO 1.3: Inicializa dos matrices de `self.vocab_size` x `self.embedding_dim` para tokens centrales y contexto.
        central_tok_matrix = self.rng.normal(
            loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float32)
        context_tok_matrix = self.rng.normal(
            loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float32)

        # TODO 1.4: Para cada `epoch` y para cada token en el corpus:
               
        for epoch in tqdm(range(self.epochs)):
            for i, central_tok in enumerate(self.tokens):
                
                # Para cada token en el contexto del token actual, es decir, para cada 
                #   token dentro de los `self.window_size` tokens a la derecha e izquieda del actual, 
                # sin contar este:
                
                if not self.use_dynamic_window:
                    context_indices = self.words_in_context(i)
                else:
                    # 4: Usa una ventana de contexto dinámica, con tamaños que varíen aleatoriamente 
                        # dentro del rango de la ventana estática original.
                    # no salir del rango de la ventana
                    dynamic_window_size = self.rng.integers(1, self.window_size + 1)
                    start = max(0, i - dynamic_window_size)
                    end = min(len(self.tokens), i + dynamic_window_size + 1)
                    context_indices = list(range(start, i)) + list(range(i + 1, end))
                
                for j in context_indices:

                    # Calcular el producto escalar entre las embeddings del token central y token de contexto.
                    # Pasar el resultado por la función `sigmoid`, obteniendo `pos_score`.
                    context_tok = self.tokens[j]

                    # FIXME maybe this will fail due to it being a list instead of an np.array
                    dot_product = np.dot(
                        central_tok_matrix[central_tok],
                        context_tok_matrix[context_tok]
                    )
                    pos_score = sigmoid(dot_product)
                    
                    # 5: Haz que el LR disminuya progresivamente durante el entrenamiento (linear decay).
                    if self.use_lr_decay:
                        normalized_epoch_progress = (epoch * len(self.tokens) + i
                                                    ) / (self.epochs * len(self.tokens)) * (1 - self.lr_min_factor)
                        lr = self.lr * max(1.0 - normalized_epoch_progress, self.lr_min_factor)
                    else:
                        lr = self.lr
                    
                    # Muestra positiva: actualizar las embeddings del token central y token contexto usando el LR, 
                    # `(1 - pos_score)` y la embedding (¡original!) del otro token.
                    # input enbedding update
                    central_tok_matrix[central_tok] += lr * (1 - pos_score) * context_tok_matrix[context_tok]
                    # output embedding update
                    context_tok_matrix[context_tok] += lr * (1 - pos_score) * central_tok_matrix[central_tok]
        
                    # Muestras negativas: obtener muestras negativas para el token central y, para cada una, 
                        # realizar un proceso similar al de la muestra positiva, con la salvedad de que ahora 
                        # `pos_score` es `neg_score` y se usa `-neg_score` para actualizar las embeddings.
                    
                    # Tokens que no se pueden muestrear porque ya están en el contexto 
                    # (no queremos que el modelo aprenda a predecirlos como negativos)
                    forbidden = set(context_indices + [central_tok]) 
                    for neg_tok in self.sample_neg(forbidden):
                        
                        dot_product_neg = np.dot(
                            central_tok_matrix[central_tok],
                            context_tok_matrix[neg_tok]
                        )
                        neg_score = sigmoid(dot_product_neg)

                        # input embedding update
                        central_tok_matrix[central_tok] += lr * (0 - neg_score) * context_tok_matrix[neg_tok]
                        # output embedding update
                        context_tok_matrix[neg_tok] += lr * (0 - neg_score) * central_tok_matrix[central_tok]

        return central_tok_matrix, context_tok_matrix


def dump_embeddings(
        # ...
        E
        ):
    # TODO 1.6: Escribe las embeddings en un fichero de texto donde, en la primera fila, 
        # aparezca el tamaño del vocabulario y el número de dimensiones de las embeddings y, 
        # en el resto de filas, cada token seguido de su correspondiente embedding, 
        # separando cada elemento con espacios simples. Ojo, los tokens pueden contener espacios.
    pass


def main():
    
    '''Execute with `python -m P2_Skipgram.p2_skipgram` from the root PLN directory
    until the pythonpath issue is solved.'''

    from P1_Bpe.bpe import encode_file

    # assuming we are executing from the root PLN directory
    bpe_model_path = "P1_Bpe/new_model.pkl"
    input_fpath = "P1_Bpe/tiny_cc_news.txt"

    trainer = Trainer(
        encoded_tokens=encode_file(bpe_model_path, input_fpath),
        rng=np.random.default_rng(42),
        embedding_dim=100,
        window_size=5,
        epochs=5,
        lr=0.05,
        lr_min_factor=0.0001,
        neg_samples=5,
    )

    T, C = trainer.train()
    E = (T + C) / 2.0  # Matriz final de embeddings
    dump_embeddings(
        # ...
        E
        )
    

if __name__ == "__main__":
    main()
