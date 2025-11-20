import numpy as np


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
        # TODO 2: Inicializa `self.neg_prob`, que será usado como distribución de probabilidad a
        #  la hora de hacer el muestreo negativo, de modo que contenga las frecuencias absolutas 
        # de cada token del vocabulario elevadas a 3/4 y normalizadas, de modo que las 
        # probabilidades resultantes sumen 1.
        pass

    def _subsample_data(self):
        # TODO 3: Reduce la ocurrencia de los tokens más frecuentes usando la siguiente fórmula:
        # `p_keep = (np.sqrt(t / f) + t / f) if f > 0 else 1.0`
        # donde `t = 1e-5` y `f` es la frecuencia relativa del token.
        pass

    def __init__(self, corpus_fpath, rng, embedding_dim, window_size, epochs, lr, lr_min_factor, neg_samples):
        self.corpus_fpath = corpus_fpath
        self.rng = rng
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.lr_min_factor = lr_min_factor 
        self.neg_samples = neg_samples

        # TODO 1.1: Carga el corpus y tokenízalo usando el tokenizador BPE de la práctica anterior.
        # El corpus debería quedar codificado como una secuencia de ids de tokens.

        # Aplica ajustes para evitar la sobreponderancia de tokens frecuentes
        self._neg_sampling_fix()
        self._subsample_data()

    def sample_neg(self, forbidden):
        # TODO 1.2: Obtén una muestra negativa de tokens, evitando seleccionar aquellos en `forbidden`, que serán los que estén dentro de la ventana actual.
        pass

    def train(self):
        # TODO 1.3: Inicializa dos matrices de `self.vocab_size` x `self.embedding_dim` para tokens centrales y contexto.

        # TODO 1.4: Para cada `epoch` y para cada token en el corpus:
        # Para cada token en el contexto del token actual, es decir, para cada token dentro de los `self.window_size` tokens a la derecha e izquieda del actual, sin contar este:
        # Calcular el producto escalar entre las embeddings del token central y token de contexto.
        # Pasar el resultado por la función `sigmoid`, obteniendo `pos_score`.
        # Muestra positiva: actualizar las embeddings del token central y token contexto usando el LR, `(1 - pos_score)` y la embedding (¡original!) del otro token.
        # Muestras negativas: obtener muestras negativas para el token central y, para cada una, realizar un proceso similar al de la muestra positiva, con la salvedad de que ahora `pos_score` es `neg_score` y se usa `-neg_score` para actualizar las embeddings.

        # TODO 4: Usa una ventana de contexto dinámica, con tamaños que varíen aleatoriamente dentro del rango de la ventana estática original.
        # TODO 5: Haz que el LR disminuya progresivamente durante el entrenamiento (linear decay).

        # TODO 1.5: Devuelve las dos matrices de embeddings.
        pass


def dump_embeddings(
        # ...
        E
        ):
    # TODO 1.6: Escribe las embeddings en un fichero de texto donde, en la primera fila, aparezca el tamaño del vocabulario y el número de dimensiones de las embeddings y, en el resto de filas, cada token seguido de su correspondiente embedding, separando cada elemento con espacios simples. Ojo, los tokens pueden contener espacios.
    pass


def main():
    trainer = Trainer(
        corpus_fpath='./tiny_cc_news.txt',
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
