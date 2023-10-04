from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests
import json
from perceptron_vectorizado import crear_perceptron as perceptron_vectorizado
from perceptron_no_vectorizado import crear_perceptron as perceptron_no_vectorizado
from red_backpropagation_no_vectorizado import crear_backpropagation as back_propagation_no_vectorizado
from red_backpropagation_vectorizado import crear_backpropagation as back_propagation_vectorizado

app = FastAPI()
red_neuronal = list()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"], 
)


@app.get('/')
def index():
    return {'message': 'hello world' }

@app.get('/entrenar_perceptron_vectorizado')
def entrenar_y_obtener_perceptron():
    red_neuronal.clear()
    red_neuronal.append(perceptron_vectorizado())
    return red_neuronal[0].to_dict()

@app.get('/entrenar_perceptron_no_vectorizado')
def entrenar_y_obtener_perceptron():
    red_neuronal.clear()
    red_neuronal.append(perceptron_no_vectorizado())
    return red_neuronal[0].to_dict()

@app.get('/entrenar_backpropagation_vectorizado')
def entrenar_y_obtener_backpropagation():
    red_neuronal.clear()
    red_neuronal.append(back_propagation_vectorizado())
    return red_neuronal[0].to_dict()

@app.get('/entrenar_backpropagation_no_vectorizado')
def entrenar_y_obtener_backpropagation_no_vectorizado():
    red_neuronal.clear()
    red_neuronal.append(back_propagation_no_vectorizado())
    return red_neuronal[0].to_dict()
    
@app.get('/listar_red_neuronal')
def listar_red_neuronal():
    return red_neuronal[0].to_dict()

@app.get('/predecir')
def predecir():
    url = "https://www.agurait.com/ubp/sia/2022/roberto/"
    respuesta = requests.get(url=url)
    if respuesta.status_code == 200:
        entrada = json.loads(respuesta.text)
        entrada_array = [int(item["Resp"][f"S{i}"]) for i in range(1,5) for item in entrada]
        prediccion =  red_neuronal[0].predecir(entrada_array)

        payload = entrada[0]["Resp"]
        payload.update(prediccion)
        payload_json = json.dumps([payload])
        headers = {"Content-Type": "application/json"}

        respuesta_post = requests.post(url=url, data=payload_json, headers=headers)
        if respuesta_post.status_code == 200:
            return {'message': f"Entrada: {entrada_array} -- Prediccion: {prediccion} -- Respuesta: {respuesta_post.text}"}
        else:
            return {'message': f"Error en la solicitud. Código de estado: {respuesta_post.status_code}"} 
    else:
        return {'message': f"Error en la solicitud. Código de estado: {respuesta.status_code}"}
