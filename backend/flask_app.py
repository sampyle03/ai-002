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
railcards_txt = os.path.join(base, 'data/railcards.txt')

assistant = ChatbotAssistant(intents_path)
assistant.parse_intents()
assistant.load_model(model_path, dimensions_path)
assistant.load_stations(stations_csv)
assistant.load_railcards(railcards_txt)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message found"}), 400
    
    reply = assistant.process_message(message)
    return jsonify({"reply": reply})

@app.route('/get-results', methods=['GET'])
def get_results():
    journeys = assistant.get_journeys()  
    return jsonify({"message": journeys})

if __name__ == "__main__":
    app.run(port=5000, debug=True)