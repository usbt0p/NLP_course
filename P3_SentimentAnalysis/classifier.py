from os import getcwd
from sys import path
path.append(getcwd())

import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from P2_Skipgram import p2_skipgram
from P1_Bpe.bpe import ByteLevelBPE

# TODO 1: Implementa funciones para cargar token embeddings de un modelo entrenado y para obtener una 
# sola embedding por texto de entrada. Puedes usar la función de agregación que quieras. 


def load_embedding(path: str):
    # load embedding model and apply it to the input tex
    with open(path, 'r', encoding="utf-8") as e:
        header = e.readline()
        body = e.readlines()
        
    # Check if the first char of the first line is a digit to determine mode
    indexmode = body[0].split()[0].isdigit()

    if indexmode:
        data = {}
        for line in body:
            parts = line.strip().split()
            idx = int(parts[0])
            vec = [float(x) for x in parts[1:]]
            data[idx] = vec
        body = data
    else:
        # tab separation if the start is not a digit
        body = {line[:line.find("\t")] : 
                [float(d) for d in line[line.find("\t")+1:].split(" ")]
                 for line in body}
            
    return header, body

def embeddings_from_tokens(input_text : str, 
                           embeddings : dict, 
                           bpe_model: str, 
                           mode="idx"):
    # make tokens
    bpe = ByteLevelBPE()
    bpe.load(bpe_model)
    tokens = bpe.encode(input_text) if mode == "idx" else bpe.tokenize(input_text)
    
    for token in tokens:
        yield (token, embeddings[token])

# use the mean to compute a single embedding
# this will lose the order of the words but it's easier to do
# TODO other aggregations that are better /( loose less information)
def aggregate_single_embedding(embedding):
    # TODO maybe remove stopwords here, since we're already removing info,
    # in favour of simplicity, better to remove from the mean the non important stuff
    return np.average(embedding)

class LogisticRegression:

    # * Campos para almacenar los pesos, sesgo y learning rate. 
    def __init__(self, num_weights):
        # TODO comprobar que los pesos puedan ser cualquier random o si tienen que ser ceros
        self.W = np.random.uniform(-0.1, 0.1, num_weights).astype(np.float32)
        #np.zeros(num_weights, dtype=np.float32)
        self.b = 0.0
        self.lr = 0.01 # TODO mirar esto
    
    def sigmoid(self, z):
        assert isinstance(z, np.ndarray)
        return 1.0/(1.0 + np.exp(-z))

    # * Método `forward`, que implemente la combinación lineal de los pesos y sesgo con la entrada seguida de la función logística.
    def forward(self, input):
        # TODO esto igual tiene que ser self.W.T para la traspuesta 
        z = np.dot(self.W.T, input) + self.b
        return self.sigmoid(z)

    # * Método `backward` que, dada la entrada, la salida obtenida y la salida deseada, modifique los parámetros del modelo.
    def backward(self, predicted, real, inputs):        
        # https://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf
        # by the chain rule of ΔL/Δw = ΔL/Δsigmoid * Δsigmoid/Δ(w*x+b) * Δ(w*x+b)/Δw = (ŷ - y) @ X
        # and for biases: ΔL/Δb = ... = ŷ - y
        error = predicted - real

        # we do that and account for the number of elems
        new_weights = np.matmul(inputs, error.T) * (1 / float(real.shape[1]))
        new_bias = np.sum(error) * (1 / float(real.shape[1]))

        return new_weights, new_bias
        
    # * Método `compute_loss`, que implemente la función de entropía cruzada binaria.
    def compute_loss(predicted : list, real : list) -> float:
        # TODO hacer en numpy
        assert len(predicted) == len(real)
        loss = 0
        for ŷ, y, in zip(predicted, real):
            loss += y * math.log(ŷ) + (1 - y) * (1 - math.log(ŷ))
        return - loss / len(predicted)
        
    # * Método `fit`, que recibe los datos de entrenamiento y optimiza el modelo mediante descenso de gradiente.
    def fit(self, X, Y, epochs=100, interactive=False):
        
        for epoch in epochs:
            
            if self.live_plotting:
                plt.ion()
                fig, ax = plt.subplots(figsize=(10, 6))

            # fordward pass
            activations = self.forward(X)
            Ŷ = self.predict(activations)
            # calcular perdida
            self.compute_loss(Ŷ, Y)
            # backward pass, devolver nuevos W y b
            new_w, new_b = self.backward(Ŷ, Y, X)
            # actualizar pesos con descenso de gradiente
            self.w = self.w - self.lr * new_w
            self.b = self.b - self.lr * new_b

            # TODO plot loss, and save image
            
    # Método `predict`, que usa `forward` y obtiene la salida final de inferencia en `{0, 1}`.
    def predict(self, input):
        output = self.forward(input)
        y_hat = output > 0.5
        return y_hat, output

# TODO 3: Implementa una función principal que realice todos los pasos necesarios para entrenar y evaluar un modelo de regresión logística que usa agregados de token embeddings como características.

# NOTA: no es necesario almacenar el modelo de regresión.

if __name__ == "__main__":
    
    embeddings_file = "/home/usbt0p/Uni/PLN/PLN/P2_Skipgram/embeddings.txt"
    bpe_file = "/home/usbt0p/Uni/PLN/PLN/P1_Bpe/bpe_model_1000.pkl"
    h, embeddings = load_embedding(embeddings_file)
    
    text = """Kingsport Fire Department Senior Captain Terry Arnold 
    responded to Eastman Wednesday morning."""

    for emb in embeddings_from_tokens(text, embeddings, bpe_file):
        print(emb[0], emb[1][:3])