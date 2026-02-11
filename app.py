from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as genai
from google.generativeai import types
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import os 
 

load_dotenv()
app = Flask(__name__)
CORS(app)  # Initialize CORS for the entire application

model = 'models/gemini-embedding-exp-03-07'
modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl','rb'))
chave_secreta = os.getenv('API_KEY')
genai.configure(api_key=chave_secreta)
print(chave_secreta)
print("Chave secreta carregada:", repr(chave_secreta))


def gerarBuscarConsulta(consulta,dataset):
    embedding_consulta = genai.embed_content(model=model,
                                content=consulta,
                                task_type="retrieval_query",
                                )
    produtos_escalares = np.dot(np.stack(dataset["Embeddings"]), embedding_consulta['embedding']) # Calculo de distancia entre consulta e a base
    print(embedding_consulta)
    print(produtos_escalares)
    indice = np.argmax(produtos_escalares)
    print(produtos_escalares[indice])
    return dataset.iloc[indice]['Conteúdo']


def melhorarResposta(inputText):
    import google.generativeai as genai

    model = genai.GenerativeModel("gemini-1.5-flash")
    system_instruction = "Considere a consulta e resposta, reescreva as sentenças de resposta de uma forma alternativa, não apresente opções de reescrita"

    prompt = f"{system_instruction}\n\n{inputText}"

    response = model.generate_content(prompt, generation_config={
        "temperature": 1,
        "top_k": 32,
    })

    return response.text


@app.route("/")
def home():
    consulta = "oque é o Docker"
    resposta = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resposta}"
    response = melhorarResposta(prompt)
    return response


@app.route("/api", methods=["POST"])
def results():
    auth_key = request.headers.get("Authorization")
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    if not data or "consulta" not in data:
        return jsonify({"error": "Campo 'consulta' não fornecido"}), 400

    consulta = data["consulta"]
    resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resultado}"
    response = melhorarResposta(prompt)
    return jsonify({"mensagem": response})


