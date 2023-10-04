import numpy as np


class neurona_bias():
    def __init__(self) -> None:
        self.valor_activacion = 1
    

    def to_dict(self):
        return {
            "pesos": "NaN",
            "taza_de_aprendizaje": "NaN",
            "error": "NaN",
        }


    def __str__(self) -> str:
        return f"Neurona BIAS"


class neurona():
    def __init__(self, pesos_iniciales: list, taza_de_aprendizaje: float) -> None:
        self.pesos = pesos_iniciales
        self.taza_de_aprendizaje = taza_de_aprendizaje
        self.valor_activacion = None
        self.error: float = None
        self.errores_retropropagados = np.empty(2)


    def to_dict(self):
        return {
            "pesos": list(self.pesos),
            "taza_de_aprendizaje": self.taza_de_aprendizaje,
            "error": self.error
        }


    def agregacion(self, entrada):
        return np.dot(self.pesos, entrada)
    

    def activacion(self, resultado) -> int:
        try:
            return (np.exp(resultado) - np.exp(-resultado))/(np.exp(resultado) + np.exp(-resultado))
        except RuntimeWarning as e:
            print(e)
            return 1
    

    def predecir(self, entrada: np.array) -> int:
        producto_punto = self.agregacion(entrada)
        salida = self.activacion(producto_punto)
        return salida


    def __str__(self) -> str:
        return f"Neurona -> pesos: {self.pesos} || taza de aprendizaje: {self.taza_de_aprendizaje} || error: {self.error}"


class backpropagation():
    def __init__(self, neuronas_capa_oculta, neuronas_capa_salida) -> None:
        self.capa_oculta = neuronas_capa_oculta
        self.capa_salida = neuronas_capa_salida


    def to_dict(self):
        neuronas = {"capa_oculta": dict(), "capa_salida": dict()}

        for i, neurona_oculta in enumerate(self.capa_oculta):
            if i == 0:
                neuronas["capa_oculta"][f"neurona_bias"] = neurona_oculta.to_dict()
            else:
                neuronas["capa_oculta"][f"neurona_{i}"] = neurona_oculta.to_dict()

        for i, neurona_salida in enumerate(self.capa_salida):
            neuronas["capa_salida"][f"neurona_{i+1}"] = neurona_salida.to_dict()

        return neuronas


    def entrenar(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array) -> None:
        contador = 0
        while contador < 300 or any(
            self.capa_salida[i].error is None or (
                self.capa_salida[i].error < -0.01 or self.capa_salida[i].error > 0.01
            )
            for i in (0, 1)
        ):
            
            if contador == 1000:
                return

            for entradas, salidas in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):

                # Listas donde se guardan los datos de la red
                activaciones_capa_oculta = np.empty(len(self.capa_oculta))

                # Paso hacia adelante capa oculta
                for i, neurona_oculta in enumerate(self.capa_oculta):
                    if i == 0:
                        activaciones_capa_oculta[i] = neurona_oculta.valor_activacion
                    else:
                        agregacion = neurona_oculta.agregacion(entradas)
                        activacion = neurona_oculta.activacion(agregacion)
                        activaciones_capa_oculta[i] = activacion
                        neurona_oculta.valor_activacion = activacion
                
                # Paso hacia adelante capa salida
                for neurona_salida, salida in zip(self.capa_salida, salidas):
                    agregacion = neurona_salida.agregacion(activaciones_capa_oculta)
                    activacion = neurona_salida.activacion(agregacion)
                    neurona_salida.valor_activacion = activacion
                    error_neurona_capa_salida = (1 - activacion) * (salida - activacion)
                    neurona_salida.error = error_neurona_capa_salida
                
                # Cálculo de los errores retropropagados
                for j, neurona_salida in enumerate(self.capa_salida):
                    for i, peso in enumerate(neurona_salida.pesos):
                        if i == 0:
                            continue
                        error_retropropagado = neurona_salida.error * peso
                        self.capa_oculta[i].errores_retropropagados[j] = error_retropropagado

                # Cálculo de los errores de la capa oculta
                for neurona_oculta in self.capa_oculta[1:]:
                    neurona_oculta.error = (1 - neurona_oculta.valor_activacion) * np.sum(neurona_oculta.errores_retropropagados)
                
                # Cálculo de los nuevos pesos para la capa de salida
                for neurona_salida in self.capa_salida:
                    neurona_salida.pesos = neurona_salida.pesos + neurona_salida.taza_de_aprendizaje * neurona_salida.error * activaciones_capa_oculta
                
                # Cálculo de los nuevos pesos para la capa oculta
                for neurona_oculta in self.capa_oculta[1:]:
                    neurona_oculta.pesos = neurona_oculta.pesos + neurona_oculta.taza_de_aprendizaje * neurona_oculta.error * entradas

            contador += 1
            

    def predecir(self, entrada_de_testeo: np.array) -> list:
        activaciones_capa_oculta = list()
        activaciones_capa_salida = list()

        # Paso hacia adelante capa oculta
        for i, neurona_oculta in enumerate(self.capa_oculta):
            if i == 0:
                activaciones_capa_oculta[i] = neurona_oculta.valor_activacion
            else:
                agregacion = neurona_oculta.agregacion(entrada_de_testeo)
                activacion = neurona_oculta.activacion(agregacion)
                activaciones_capa_oculta[i] = activacion
        
        # Paso hacia adelante capa salida
        for neurona_salida in self.capa_salida:
            agregacion = neurona_salida.agregacion(activaciones_capa_oculta)
            activacion = neurona_salida.activacion(agregacion)
            activaciones_capa_salida.append(activacion)
            
        return activaciones_capa_salida


    def __str__(self) -> str:
        imprimir = "BACKPROPAGATION\n"
        for neurona in self.capa_salida:
            imprimir += f"{neurona}\n"
        for neurona in self.capa_oculta:
            imprimir += f"{neurona}\n"
        return imprimir


def crear_backpropagation():
    entradas_de_entrenamiento = np.array([[1,  1,  1,  1,  1], 
                                          [1, -1,  1,  1,  1], 
                                          [1,  1,  1, -1, -1], 
                                          [1, -1, -1, -1, -1], 
                                          [1,  1, -1,  1,  1], 
                                          [1,  1,  1, -1,  1], 
                                          [1,  1,  1,  1, -1]])
    
    salidas_de_entrenamiento = np.array([[-1, -1], 
                                         [-1,  1], 
                                         [ 1, -1], 
                                         [ 1,  1], 
                                         [ 1, -1], 
                                         [-1,  1], 
                                         [ 1, -1]])
    
    nBias      = neurona_bias()
    neurona_1  = neurona([-0.0982822300177012, -0.08862935168246806, -0.035400257820270814, 0.007275970829338951, -0.08546643472557863], 0.13100980076001534)
    neurona_2  = neurona([0.0314991704535158, 0.048634512831941784, -0.04200547736317977, -0.05341259047563005, 0.0747707434135832], 0.12131514910500175)
    neurona_3  = neurona([0.08161709727783928, 0.07725450012747345, 0.05068593062766588, -0.03327020902353024, -0.011433649836421547], 0.172224098354397)
    neurona_4  = neurona([-0.01036691194653891, -0.05397761114767219, 0.07645260417937491, 0.06193566962667263, 0.05785543365092055], 0.1874809366943307)
    neurona_5  = neurona([0.08981304431371825, 0.0014049958305124982, -0.026404025831000474, 0.0016465234733926692, 0.03702023289168371], 0.17568536699370052)
    neurona_6  = neurona([-0.0771295366783926, 0.04527575367568443, -0.023783818665049306, 0.03637539905686671, 0.05732311625520897], 0.10645303409469609)
    neurona_7  = neurona([0.023747331311641487, -0.012590429462468272, -0.05569375495836602, 0.07417587908940881, 0.08530633325007356], 0.14132997282701865)
    neurona_8  = neurona([-0.05025041655180891, -0.0320908549792213, -0.009297980186042817, 0.04333881451893906, -0.06082510729340247], 0.13986041684500264)
    neurona_9  = neurona([0.05532068512254537, -0.06134187988943656, -0.003972071947865391, -0.027948934059037714, -0.03858843799103422, 0.07023626499618923, -0.09719152262778111, 0.06398837476717917, 0.09243878651357368], 0.10262979033276863)
    neurona_10 = neurona([-0.08615675212500387, 0.0065248107884566114, 0.0741236023325367, -0.08521917719647401, -0.00882338772139346, 0.04971677799081384, 0.049035246103505015, 0.03888226973017361, 0.09379512174389715], 0.1579060381954394)
    
    capa_oculta = [nBias, neurona_1, neurona_2, neurona_3, neurona_4, neurona_5, neurona_6, neurona_7, neurona_8]
    capa_salida = [neurona_9, neurona_10]

    backpropagation_actual = backpropagation(capa_oculta, capa_salida)
    backpropagation_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
    return backpropagation_actual
