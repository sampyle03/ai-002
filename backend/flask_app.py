from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model.model import ChatbotAssistant, searchForCheapestTrain

app = Flask(__name__)
CORS(app)

base = os.path.dirname(__file__)
intents_path = os.path.join(base, 'model/intents.json')
model_path = os.path.join(base, 'model.pth')
dimensions_path = os.path.join(base, 'dimensions.json')
stations_csv = os.path.join(base, 'data/stations.csv')

assistant = ChatbotAssistant(intents_path, function_mappings={"get_from_x_to_y_date": searchForCheapestTrain})
assistant.parse_intents()
assistant.load_model(model_path, dimensions_path)
assistant.load_stations(stations_csv)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    print(message)
    if not message:
        return jsonify({"error": "No message found"}), 400
    
    reply = assistant.process_message(message)
    print(reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(port=5000, debug=True)