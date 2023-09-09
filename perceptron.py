import numpy as np
import random


class neurona():
    def __init__(self, pesos_iniciales: np.array, taza_de_aprendizaje: float) -> None:
        self.pesos = pesos_iniciales
        self.taza_de_aprendizaje = taza_de_aprendizaje
        self.error: float = None


    def entrenar_neurona(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array):
        while self.error != 0:
            for entrada, salida in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):
                if self.error == 0:
                    continue
                producto_punto = self.agregacion(entrada)
                salida_obtenida = self.activacion(producto_punto)
                self.error = salida - salida_obtenida
                self.pesos = self.pesos + self.taza_de_aprendizaje * self.error * entrada
    

    def agregacion(self, entrada):
        return np.dot(self.pesos, entrada)
    

    def activacion(self, resultado) -> int:
        return 1 if resultado > 0 else -1
    

    def predecir(self, entrada: np.array) -> int:
        producto_punto = self.agregacion(entrada)
        salida = self.activacion(producto_punto)
        return salida


    def __str__(self) -> str:
        return f"Neurona -> pesos: {self.pesos} || taza de aprendizaje: {self.taza_de_aprendizaje} || error: {self.error}"


class perceptron():
    def __init__(self, neurona_1: neurona, neurona_2: neurona) -> None:
        self.neuronas = [neurona_1, neurona_2]


    def entrenar(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array) -> None:
        salidas_de_entrenamiento = salidas_de_entrenamiento.T
        for neurona, salidas in zip(self.neuronas, salidas_de_entrenamiento):
            neurona.entrenar_neurona(entradas_de_entrenamiento, salidas)
    

    def entrenar_neurona(self, neurona: int, entradas_de_entrenamiento: np.array, salida_de_entrenamiento: np.array) -> None:
        self.neuronas[neurona].entrenar_neurona(entradas_de_entrenamiento, salida_de_entrenamiento)

    
    def predecir(self, entradas: np.array):
        for entrada in entradas:
            imprimir = f"{entrada} - ["
            for neurona in self.neuronas:
                imprimir += str(neurona.predecir(entrada)) + " "
            imprimir += "]"
            print(imprimir)
    

    def predecir_entradas_de_testeo(self, entradas: np.array):
        salidas_ok = np.array([[ 1, -1],
                               [ 1, -1],
                               [ 1,  1],
                               [-1,  1],
                               [ 1, -1],
                               [-1, -1],
                               [ 1,  1],
                               [ 1,  1],
                               [-1,  1]])
        salidas = list()
        imprimir = ""
        for entrada in entradas:
            imprimir += f"{entrada} - ["
            salidas_actuales = list()
            for neurona in self.neuronas:
                salida = neurona.predecir(entrada)
                salidas_actuales.append(salida)
                imprimir += str(salida) + " "
            imprimir += "]\n"
            salidas.append(salidas_actuales)
        salidas_matriz = np.vstack(salidas)
        if np.array_equal(salidas_matriz, salidas_ok):
            return imprimir
        else:
            return None
    

    def __str__(self) -> str:
        imprimir = "PERCEPTRON\n"
        for neurona in self.neuronas:
            imprimir += f"{neurona}\n"
        return imprimir
    
if __name__ == "__main__":
    entradas_de_entrenamiento = np.array([[ 1,  1,  1,  1], 
                                          [-1,  1,  1,  1], 
                                          [ 1,  1, -1, -1], 
                                          [-1, -1, -1, -1], 
                                          [ 1, -1,  1,  1], 
                                          [ 1,  1, -1,  1], 
                                          [ 1,  1,  1, -1]])
    
    salidas_de_entrenamiento = np.array([[-1, -1], 
                                         [-1,  1], 
                                         [ 1, -1], 
                                         [ 1,  1], 
                                         [ 1, -1], 
                                         [-1,  1], 
                                         [ 1, -1]])
    
    entradas_de_testeo =np.array([[ 1, -1, -1, -1],
                                  [ 1, -1,  1, -1],
                                  [-1, -1,  1,  1],
                                  [-1,  1, -1,  1],
                                  [-1, -1,  1, -1],
                                  [-1,  1,  1, -1],
                                  [ 1, -1, -1,  1],
                                  [-1, -1, -1,  1],
                                  [-1,  1, -1, -1]])
    
    neurona_1 = neurona(np.array([ 0.08174903, -0.8377704, 0.33671084, 0.57375835]),  0.03673239027963381)
    neurona_2 = neurona(np.array([-0.4094063,   0.53426245, -0.44470761,  0.98560924]), 0.05788280232040685)
    perceptron_actual = perceptron(neurona_1, neurona_2)
    
    perceptron_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
    print(perceptron_actual)

    perceptron_actual.predecir(entradas_de_testeo)

    # for _ in range(1000):
    #     pesos_random_neurona_1 = np.random.uniform(-1, 1, size=4)
    #     pesos_random_neurona_2 = np.random.uniform(-1, 1, size=4)
    #     taza_aprendizaje_1 = random.uniform(0.001, 0.1)
    #     taza_aprendizaje_2 = random.uniform(0.001, 0.1)

    #     neurona_1 = neurona(pesos_random_neurona_1, taza_aprendizaje_1)
    #     neurona_2 = neurona(pesos_random_neurona_2, taza_aprendizaje_2)
    #     perceptron_actual = perceptron(neurona_1, neurona_2)

    #     perceptron_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
    #     # print(perceptron_actual)

    #     sirve = perceptron_actual.predecir_entradas_de_testeo(entradas_de_testeo)

    #     if sirve is not None:
    #         print(f"Pesos iniciales neurona 1: {pesos_random_neurona_1}")
    #         print(f"Pesos iniciales neurona 2: {pesos_random_neurona_2}")
    #         print(perceptron_actual)
    #         print(sirve)
    #         break



"""
SALIDAS CORRECTAS
[ 1 -1 -1 -1] - [ 1 -1 ]
[ 1 -1  1 -1] - [ 1 -1 ]
[-1 -1  1  1] - [ 1  1 ]
[-1  1 -1  1] - [-1  1 ]
[-1 -1  1 -1] - [ 1 -1 ]
[-1  1  1 -1] - [-1 -1 ]
[ 1 -1 -1  1] - [ 1  1 ]
[-1 -1 -1  1] - [ 1  1 ]
[-1  1 -1 -1] - [-1  1 ]
"""
