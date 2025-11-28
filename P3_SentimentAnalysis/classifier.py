import numpy as np
import pickle
from P2_Skipgram import p2_skipgram
from P1_Bpe import bpe

# TODO 1: Implementa funciones para cargar token embeddings de un modelo entrenado y para obtener una sola embedding por texto de entrada. Puedes usar la función de agregación que quieras. 

# TODO load embedding model and apply it to the input text

# TODO use the mean to compute a single embedding
# this will lose the order of the words but it's easier to do


# TODO 2: Implementa la clase LogisticRegression con los siguientes componentes:

class LogisticRegression:

    # * Campos para almacenar los pesos, sesgo y learning rate. 
    def __init__(self, num_weights):
        # TODO comprobar que los pesos puedan ser cualquier cosa
        self.W = np.random.uniform(-0.1, 0.1, num_weights).astype(np.float32)
        #np.zeros(num_weights, dtype=np.float32)
        self.b = 0.0
        self.lr = 0.01 # TODO mirar esto
    
    def sigmoid(self, z):
        assert isinstance(z, np.ndarray)
        return 1.0/(1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

    # * Método `forward`, que implemente la combinación lineal de los pesos y sesgo con la entrada seguida de la función logística.
    def forward(self, input):
        # TODO esto igual tiene que ser self.W.T para la traspuesta 
        z = np.dot(self.W.T, input) + self.b
        return self.sigmoid(z)

    # * Método `backward` que, dada la entrada, la salida obtenida y la salida deseada, modifique los parámetros del modelo.
    def backward(self, loss):
        #https://www.freecodecamp.org/news/demystifying-gradient-descent-and-backpropagation-via-logistic-regression-based-image-classification-9b5526c2ed46/
        ...
        
    # * Método `compute_loss`, que implemente la función de entropía cruzada binaria.
    def compute_loss():
        ...

    # * Método `fit`, que recibe los datos de entrenamiento y optimiza el modelo mediante descenso de gradiente.
    def fit():

        ...
        # por cada epoch:
            # fordward pass

            # calcular perdida (se puede hacer dentro del forward)
            
            # backward pass, devolver nuevos W y b

            # actualizar pesos con descenso de gradiente
            #self.w = self.w - self.lr * loss

            #mostrar perdida

    # Método `predict`, que usa `forward` y obtiene la salida final de inferencia en `{0, 1}`.
    def predict(self, input):
        output = self.forward(input)
        y_hat = output > 0.5 # TODO que pasa si es = a 0'5??
        return y_hat, output

# TODO 3: Implementa una función principal que realice todos los pasos necesarios para entrenar y evaluar un modelo de regresión logística que usa agregados de token embeddings como características.

# NOTA: no es necesario almacenar el modelo de regresión.