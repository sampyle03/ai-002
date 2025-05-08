import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nltk
import numpy as np
import os
import random
import json
import csv
import dateparser
from dateparser.search import search_dates
from datetime import datetime
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams

# nltk.download('punkt_tab')
LEMMATIZER = nltk.stem.WordNetLemmatizer()

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size): # Input size is the number of words in the vocab (number of features), output size is the number of intents
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # First layer - 128 neurons
        self.fc2 = nn.Linear(128, 64) # Second layer - 64 neurons
        self.fc3 = nn.Linear(64, output_size) # Output Layer
        self.relu = nn.ReLU() # Activation function
        self.dropout = nn.Dropout(0.5) # Dropout layer to prevent overfitting

    def forward(self, x):
        x = self.relu(self.fc1(x)) # First layer with ReLU activation
        x = self.dropout(x) # Apply dropout 
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x # Output layer

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = [] # List to hold the documents
        self.vocab = []
        self.intents = []
        self.intents_responses = {}
        self.previous_intents = []
        self.previous_responses = []

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

        self.required_slots = {"get_from_x_to_y_date": ["departure", "destination", "date"]}
        self.current_slots = {"departure": None, "destination": None, "date": None}

    @staticmethod
    def tokenize_and_lemmatize(text):
        words = nltk.word_tokenize(text)
        words = [LEMMATIZER.lemmatize(w.lower()) for w in words]

        return words
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocab]
    
    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as file:
                intents_data = json.load(file)

            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses[intent["tag"]] = intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocab.extend(pattern_words)
                    self.documents.append((pattern_words, intent["tag"]))

            self.vocab = sorted(set(self.vocab))

    def load_stations(self, path):
        regex_pattern = f"[{re.escape(string.punctuation)}]" # https://docs.vultr.com/python/examples/remove-punctuations-from-a-string
        self.stations = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["longname.name_alias"].lower() != "\\n":
                    self.stations.append(re.sub(regex_pattern, "", row["longname.name_alias"].replace(" Rail Station", "")))
                else:
                    self.stations.append(re.sub(regex_pattern, "", row["name"]))
        
        self.letter_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2,4)
        )
        stations_lower = [station.lower() for station in self.stations]
        self.letter_vectorizer.fit(stations_lower)
        self.stations_matrix = self.letter_vectorizer.transform(stations_lower)


    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):4f}")


    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
    
        with open(dimensions_path, "w") as f:
            json.dump({"input_size": self.X.shape[1], "output_size": len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, "r") as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions["input_size"], dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def extract_stations(self, message, predicted_intent):
        regex_pattern = f"[{re.escape(string.punctuation)}]" # https://docs.vultr.com/python/examples/remove-punctuations-from-a-string
        message = re.sub(regex_pattern, "", message.lower().replace(" rail station", "").replace("rail station", "").replace(" station", "").replace("station", "")).split()

        message_2grams = [(i, tuple(message[i:i+2])) for i in range(len(message)-1)]
        message_3grams = [(i, tuple(message[i:i+3])) for i in range(len(message)-2)]
        message_4grams = [(i, tuple(message[i:i+3])) for i in range(len(message)-3)]

        min_2gram_cosine_score = 0.81
        min_3gram_cosine_score = 0.71
        min_4gram_cosine_score = 0.6
        most_similar_stations = []
        while len(most_similar_stations) < 3 or min_2gram_cosine_score < 0.6:
            for start_idx, gram in message_2grams:
                vec = self.letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.stations_matrix).flatten()
                for station_idx, sim in enumerate(scores):
                    if sim > min_2gram_cosine_score:
                        most_similar_stations.append((
                            self.stations[station_idx],
                            sim,
                            start_idx,
                            1
                        ))
            for start_idx, gram in message_3grams:
                vec = self.letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.stations_matrix).flatten()
                for station_idx, sim in enumerate(scores):
                    if sim > min_3gram_cosine_score:
                        most_similar_stations.append((
                            self.stations[station_idx],
                            sim,
                            start_idx,
                            1
                        ))
            for start_idx, gram in message_4grams:
                vec = self.letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.stations_matrix).flatten()
                for station_idx, sim in enumerate(scores):
                    if sim > min_4gram_cosine_score:
                        most_similar_stations.append((
                            self.stations[station_idx],
                            sim,
                            start_idx,
                            1
                        ))
            min_2gram_cosine_score -= 0.01
            min_3gram_cosine_score -= 0.02
            min_4gram_cosine_score -= 0.03
        possible_stations = {}
        for name, sim, pos, w in most_similar_stations:
            sum_sim, sum_posw, sum_w = possible_stations.get(name, (0,0,0))
            possible_stations[name] = (sum_sim + sim, sum_posw + pos*w, sum_w + w)
        results = []
        for name, (sum_sim, sum_posw, sum_w) in possible_stations.items():
            avg_sim = sum_sim / sum_w
            avg_pos = sum_posw / sum_w
            results.append((name, avg_sim, avg_pos))
        first_station = results[0]
        second_station = results[-1]

        if len(possible_stations) < 2:
            return None
        if (abs(first_station[2]-second_station[2]) < 1):
            return False, [first_station[0], second_station[0]]

        if predicted_intent in ("get_from_x_to_y", "get_from_x_to_y_date"):
            departure, destination = first_station[0], second_station[0]
        else:  # reversed intent
            departure, destination = second_station[0], first_station[0]

        # get the position in the message of each statement, use the predicted intent to return which station is the departure and which is the destination
        if predicted_intent == "get_from_x_to_y_date" or predicted_intent == "get_from_x_to_y":
            return True, [departure, destination]
            




    def extract_date(self, text):
        text = text.replace("on", "this")
        dt = search_dates(text, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now(), "SKIP_TOKENS": ["to", "from", "rail", "station", "travel"]})
        return dt[0][1].date().isoformat() if dt else None

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        # Predict intent of message
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
        predicted_class_index = torch.argmax(predictions, dim=1).item() # Use argmax to get the index of the predicted class
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent in self.required_slots:
            print(self.extract_stations(input_message, predicted_intent))
            success, stations = self.extract_stations(input_message, predicted_intent)
            print(stations)
            if stations:
                if not success:
                    if not self.current_slots["departure"] and not self.current_slots["destination"]:
                        message = "It seems like you mentioned one of "
                        while len(stations) > 2:
                            message += stations.pop() + ", "
                        if len(stations) == 2:
                            message += stations.pop() + " and "
                        if len(stations) == 1:
                            message += stations.pop()
                        self.previous_responses.append("specify_which_station")
                        return(f"{message}. Please specify which of these you mean.")
                    elif not self.current_slots["departure"]:
                        self.current_slots["departure"] = stations[0]
                    elif not self.current_slots["destination"]:
                        self.current_slots["departure"] = stations[0]
                if not self.current_slots["departure"]:
                    self.current_slots["departure"] = stations[0]
                if not self.current_slots["destination"] and len(stations) > 1:
                    self.current_slots["destination"] = stations[1]
            else:
                return random.choice(["I'm not too sure what specific stations you mean! Could you clarify?", "Could you specify which stations you mean?"])
        
            date = self.extract_date(input_message)
            if date:
                self.current_slots["date"] = date
            
            unfilled_slots = []
            for slot in self.required_slots[predicted_intent]:
                if not self.current_slots[slot]:
                    unfilled_slots.append(slot)
            message = "Sure! Just tell me your "
            while len(unfilled_slots) > 2:
                message += unfilled_slots.pop() + ", "
            if len(unfilled_slots) == 2:
                message += unfilled_slots.pop() + " and "
            if len(unfilled_slots) == 1:
                return message + unfilled_slots.pop() + "."

            departure = self.current_slots["departure"]
            destination = self.current_slots["destination"]
            date = self.current_slots["date"]
            result = self.function_mappings[predicted_intent](departure, destination, date)

            self.current_slots = {s: None for s in self.current_slots}
            return result
        
        return random.choice(self.intents_responses[predicted_intent])

            
def searchForCheapestTrain(departureLoc, destinationLoc, time, railcard=None):
    """
    Function: Searches for the cheapest train from departureLoc to destinationLoc at a given time.
    Parameters: departureLoc (str), destinationLoc (str), time (str), railcard (str, optional)
    Returns: str - The cheapest train information.
    """

    return f"Searching from {departureLoc} to {destinationLoc} on {time}..."

if __name__ == "__main__":
    intents_path = os.path.join(os.path.dirname(__file__), "intents.json")
    assistant = ChatbotAssistant(intents_path, function_mappings={"get_from_x_to_y_date": searchForCheapestTrain})
    assistant.parse_intents()
    assistant.load_stations(os.path.join(os.path.dirname(__file__), "../data/stations.csv"))
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=300)
    assistant.save_model("model.pth", "dimensions.json")

    print("\n\n\n\nTime to chat! Type 'exit' to quit.\n")
    while True:
        message = input("You: ")
        if message.lower() == "exit":
            break

        print("Assistant:", assistant.process_message(message))