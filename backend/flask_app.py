from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model.model import ChatbotAssistant

# Load delay model at startup
try:
    from train_delay_model import TrainDelayModel
    DELAY_MODEL = TrainDelayModel.load_model("train_delay_model")
    print("Delay prediction model loaded successfully!", flush=True)
except:
    DELAY_MODEL = None
    print("Delay prediction model not available", flush=True)

app = Flask(__name__)
CORS(app)

base = os.path.dirname(__file__)
intents_path = os.path.join(base, 'model/intents.json')
model_path = os.path.join(base, 'model.pth')
dimensions_path = os.path.join(base, 'dimensions.json')
stations_csv = os.path.join(base, 'data/stations.csv')

assistant = ChatbotAssistant(intents_path)
assistant.parse_intents()
assistant.load_model(model_path, dimensions_path)
assistant.load_stations(stations_csv)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print("Received data:", data, flush=True)
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message found"}), 400
    
    newChat = data.get("new_chat", False)
    print("new chat:", newChat, flush=True)
    if newChat:
        print("This is new chat", newChat, flush=True)
        #Starts a new chat 
        assistant.start_new_chat()
    
    reply = assistant.process_message(message)
    #Saves the response message to the SQL
    assistant.update_chat_messages(["chatbot",reply])
    return jsonify({"reply": reply})

@app.route('/get-chats', methods=['GET'])
def get_chats():
    chats = assistant.get_chats()
    return jsonify({"chats": chats})

@app.route('/get-messages/<int:chat_id>', methods=['GET'])
def get_messages(chat_id):
    messages = assistant.get_messages(chat_id)
    return jsonify({"messages": messages})

@app.route('/get-results', methods=['GET'])
def get_results():
    journeys = assistant.get_journeys()  
    return jsonify({"message": journeys})

if __name__ == "__main__":
    app.run(port=5000, debug=True)