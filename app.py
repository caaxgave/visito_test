from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from utils import get_document_from_json, create_vector, process_chat

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

FILE_PATH = "data/paciente1.json"
chat_history = []

@app.route('/process', methods=['POST'])
def process_data():
    documents, patient_name = get_document_from_json(FILE_PATH)
    vector_store = create_vector(documents)

    data = request.json
    user_input = data['response']

    answer = process_chat(vector_store, user_input, chat_history, patient_name)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))
    return jsonify({'status': 'success', 'answer': answer})

if __name__ == '__main__':
    app.run(port=4000, debug=True)