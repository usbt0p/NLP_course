import numpy as np
from tqdm import tqdm


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


# 1: Implementa un método de entrenamiento simple, esto es, con learning rate (LR) constante y ventana estática.
class Trainer:
    '''Clase para entrenar un modelo Skip-gram con Negative Sampling.
    Basado en un esqueleto proporcionado por la asignatura y el paper original:
    "Distributed Representations of Words and Phrases and their Compositionality"
    https://arxiv.org/pdf/1310.4546
    '''

    def _neg_sampling_fix(self):
        # 2: Inicializa `self.neg_prob`, que será usado como distribución de probabilidad
        # a la hora de hacer el muestreo negativo, de modo que contenga las frecuencias
        # absolutas de cada token del vocabulario elevadas a 3/4, y normalizadas,
        # de modo que las probabilidades resultantes sumen 1.
        freq = self.token_counts  # / len(self.tokens)
        freq_power = np.power(freq, 3 / 4)  # suaviza distribución
        neg_prob = freq_power / np.sum(freq_power)  # normalizar para que sume 1

        self.neg_prob = neg_prob
        return neg_prob

    def _subsample_data(self):
        # 3: Reduce la ocurrencia de los tokens más frecuentes usando la siguiente fórmula:
        # `p_keep = (np.sqrt(t / f) + t / f) if f > 0 else 1.0`
        # donde `t = 1e-5` y `f` es la frecuencia relativa del token.

        t = 1e-5
        f = self.token_counts / len(self.tokens)
        
        # Calcular probabilidad de mantener cada token, evitar división por cero usando un epsilon muy pequeño
        f_safe = np.clip(f, 1e-10, None)
        p_keep = np.sqrt(t / f_safe) + t / f_safe
        
        # *literalmente* eliminar tokens mas frecuentes del conjunto de tokens
        mask = self.rng.random(len(self.tokens)) < p_keep[self.tokens]
        self.tokens = [tok for i, tok in enumerate(self.tokens) if mask[i]]

    def __init__(
        self,
        encoded_tokens,
        rng=np.random.default_rng(42),
        embedding_dim=100,
        window_size=5,
        epochs=5,
        lr=0.05,
        lr_min_factor=0.0001,
        neg_samples=5,
    ):

        # Carga el corpus y tokenízalo usando el tokenizador BPE de la práctica anterior.
        self.tokens = encoded_tokens  # the corpus must be tokenized with BPE
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
        self._subsample_data() 

        del self.token_counts

        # parámetros para activar / desactivar ventana dinámica y LR decreciente
        self.use_dynamic_window = True
        self.use_lr_decay = True
        self.init = "uniform" # can be "uniform" or "xavier"

        self.loss_history = {"total_loss": [], "neg_loss": [], "pos_loss": []}

    def sample_neg(self, forbidden):
        # 1.2: Obtén una muestra negativa de tokens, evitando seleccionar aquellos en
        # `forbidden`, que serán los que estén dentro de la ventana actual.

        neg_samples = []
        while len(neg_samples) < self.neg_samples:
            sampled_token = self.rng.choice(self.vocab_size, p=self.neg_prob)
            if sampled_token not in forbidden:
                neg_samples.append(sampled_token)

        return neg_samples

    def words_in_context(self, index):
        """Devuelve los índices de los tokens en el contexto del token en `index`."""
        start = max(0, index - self.window_size)
        end = min(len(self.tokens), index + self.window_size + 1)
        context_indices = list(range(start, index)) + list(range(index + 1, end))
        return context_indices

    def train(self):
        # 1.3: Inicializa dos matrices de `self.vocab_size` x `self.embedding_dim` para tokens centrales y contexto.
        

        if self.init == "uniform":
            central_tok_matrix = self.rng.uniform(
                -0.5, 0.5, (self.vocab_size, self.embedding_dim)).astype(np.float32)
            context_tok_matrix = self.rng.uniform(
                -0.5, 0.5, (self.vocab_size, self.embedding_dim)).astype(np.float32)
        
        elif self.init == "xavier": # Inicialización Xavier/Glorot para función sigmoide
            # https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization
            # En este caso: fan_in = vocab_size, fan_out = embedding_dim
            limit = np.sqrt(6.0 / (self.vocab_size + self.embedding_dim))
            
            central_tok_matrix = self.rng.uniform(
                -limit, limit, (self.vocab_size, self.embedding_dim)).astype(np.float32)
            context_tok_matrix = self.rng.uniform(
                -limit, limit, (self.vocab_size, self.embedding_dim)).astype(np.float32)

        # 1.4: Para cada `epoch` y para cada token en el corpus hacer el bucle
        for epoch in tqdm(range(self.epochs)):
            # initialize losses for this epoch
            loss_pos_agg = 0.0
            loss_neg_agg = 0.0
            total_loss_epoch = 0.0
            num_pos_neg_pairs = 0

            # 5: Haz que el LR disminuya progresivamente durante el entrenamiento (linear decay).
            # TODO usar batch decay o epoch decay?? es decir; poner este decay dentro del bucle de batches o de epochs??
            if self.use_lr_decay:
                # decay linearly with epoch
                # TODO usar esto, o self.lr - (epoch / self.epochs) * (self.lr - self.lr_min_factor * self.lr)?
                lr = self.lr - (epoch / self.epochs) * (self.lr - self.lr_min_factor)

            else:
                lr = self.lr

            for i, central_tok in enumerate(self.tokens):

                # Para cada token en el contexto del token actual, es decir, para cada
                #   token dentro de los `self.window_size` tokens a la derecha e izquieda del actual,
                #   sin contar este:

                if not self.use_dynamic_window:
                    context_indices = self.words_in_context(i)
                else:
                    # 4: Usa una ventana de contexto dinámica, con tamaños que varíen aleatoriamente
                    # dentro del rango de la ventana estática original.
                    dynamic_window_size = self.rng.integers(1, self.window_size + 1)
                    start = max(0, i - dynamic_window_size)
                    end = min(len(self.tokens), i + dynamic_window_size + 1)
                    context_indices = list(range(start, i)) + list(range(i + 1, end))

                for j in context_indices:
                    # Calcular el producto escalar entre las embeddings del token central y token de contexto.
                    # Pasar el resultado por la función `sigmoid`, obteniendo `pos_score`.
                    context_tok = self.tokens[j]

                    dot_product = np.dot(
                        central_tok_matrix[central_tok], context_tok_matrix[context_tok]
                    )
                    pos_score = sigmoid(dot_product)

                    # Muestra positiva: actualizar las embeddings del token central y token contexto usando el LR,
                    # `(1 - pos_score)` y la embedding (¡original!) del otro token.
                    # Guardar embeddings originales antes de actualizar
                    central_emb_original = central_tok_matrix[central_tok].copy()
                    context_emb_original = context_tok_matrix[context_tok].copy()

                    # input enbedding update
                    central_tok_matrix[central_tok] += lr * (1 - pos_score) * context_emb_original
                    # output embedding update
                    context_tok_matrix[context_tok] += lr * (1 - pos_score) * central_emb_original
        
                    loss_pos = -np.log(pos_score + 1e-10)  # adding epsilon to avoid log(0)
                    
                    # Tokens que no se pueden muestrear porque ya están en el contexto 
                    # (no queremos que el modelo aprenda a predecirlos como negativos)
                    forbidden = set(context_indices + [central_tok])
                    
                    # Muestras negativas: obtener muestras negativas para el token central y, para cada una, 
                        # realizar un proceso similar al de la muestra positiva, con la salvedad de que ahora 
                        # `pos_score` es `neg_score` y se usa `-neg_score` para actualizar las embeddings.              
                    loss_neg = 0.0 # reset loss_neg for this positive sample
                    for neg_tok in self.sample_neg(forbidden):

                        dot_product_neg = np.dot(
                            central_tok_matrix[central_tok], context_tok_matrix[neg_tok]
                        )
                        neg_score = sigmoid(dot_product_neg)

                        # Guardar embeddings originales antes de actualizar
                        central_emb_neg_original = central_tok_matrix[central_tok].copy()
                        neg_emb_original = context_tok_matrix[neg_tok].copy()

                        # input embedding update
                        central_tok_matrix[central_tok] += lr * (0 - neg_score) * neg_emb_original
                        # output embedding update
                        context_tok_matrix[neg_tok] += lr * (0 - neg_score) * central_emb_neg_original

                        loss_neg += -np.log(1 - neg_score + 1e-10)  # adding epsilon to avoid log(0)

                    # accumulate losses for normalization later (purely for monitoring)
                    num_pos_neg_pairs += 1
                    total_loss_epoch += loss_pos + loss_neg
                    loss_pos_agg += loss_pos
                    loss_neg_agg += loss_neg

            # Normalizar correctamente
            avg_epoch_loss = (
                total_loss_epoch / num_pos_neg_pairs if num_pos_neg_pairs > 0 else 0
            )
            loss_pos_norm = loss_pos_agg / num_pos_neg_pairs
            loss_neg_norm = loss_neg_agg / num_pos_neg_pairs
            self.loss_history["total_loss"].append(avg_epoch_loss)
            self.loss_history["pos_loss"].append(loss_pos_norm)
            self.loss_history["neg_loss"].append(loss_neg_norm)

            # Imprimir loss para monitoreo
            print(
                f"\nEpoch {epoch+1}: Total Loss: {avg_epoch_loss:.4f}, Pos Loss: {loss_pos_norm:.4f}, Neg Loss: {loss_neg_norm:.4f}"
            )

        # we get the final embeddings as the average of input and output embeddings
        embeddings = (central_tok_matrix + context_tok_matrix) / 2.0
        return embeddings, central_tok_matrix, context_tok_matrix


def dump_embeddings(embeddings, output_file):
    """Guarda las embeddings en formato limpio con IDs de tokens"""
    vocab_size, embedding_dim = embeddings.shape
    
    # Usar encoding estándar UTF-8 sin BOM
    with open(output_file, "w", encoding="utf-8", newline='\n') as f:
        f.write(f"{vocab_size} {embedding_dim}\n")
        for token_id in range(vocab_size):
            embedding_str = " ".join(f"{x:.6f}" for x in embeddings[token_id])
            f.write(f"{token_id} {embedding_str}\n")


def main():
    """Execute with `python3 -m P2_Skipgram.p2_skipgram` from the root PLN directory
    until the pythonpath issue is solved."""

    from P1_Bpe.bpe import encode_file
    import matplotlib.pyplot as plt

    # assuming we are executing from the root PLN directory
    bpe_model_path = "P1_Bpe/bpe_model_1000.pkl"
    input_fpath = "P1_Bpe/tiny_cc_news.txt"
    input_precomputed = "P2_Skipgram/encoded_tokens_1000.txt"
    
    # Cargar tokens precomputados para evitar recodificar cada vez
    with open(input_precomputed, "r", encoding="utf-8") as f:
        encoded_tokens = list(map(int, f.read().strip().split()))
    
    print(f"Loaded {len(encoded_tokens)} precomputed tokens from {input_precomputed}")
    
    n_epochs = 5  # Número normal de épocas para entrenamiento real
    trainer = Trainer(
        encoded_tokens=encoded_tokens,
        epochs=n_epochs,
    )

    # Train the model
    E, _, _ = trainer.train()
    
    # Guardar embeddings en formato legible
    dump_embeddings(E, "./P2_Skipgram/embeddings_readable.txt")
    
    print("Embeddings guardados en embeddings_readable.txt")

    # Plot losses after training
    plt.figure(figsize=(10, 6))
    for loss_type, losses in trainer.loss_history.items():
        plt.plot(range(1, n_epochs + 1), losses, label=loss_type)
    # plot only total loss
    # plt.plot(range(n_epochs), trainer.loss_history['total_loss'], label='Total Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./P2_Skipgram/loss_plot.png")
    plt.show()
    print("Loss plot saved to ./P2_Skipgram/loss_plot.png")


if __name__ == "__main__":
    main()