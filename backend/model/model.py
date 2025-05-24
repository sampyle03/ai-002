"""
Chatbot developed using tutorial of https://www.youtube.com/watch?v=a040VmmO-AY&t=688s&ab_channel=NeuralNine
"""
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nltk
import numpy as np
import os
import random
import json
import csv
from dateparser.search import search_dates
from datetime import datetime
from datetime import date
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from scraper import TicketFinder

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
    def __init__(self, intents_path):
        self.model = None
        self.intents_path = intents_path

        self.documents = [] # List to hold the documents
        self.vocab = []
        self.intents = []
        self.intents_responses = {}
        self.previous_intents = []
        self.previous_responses = []

        #self.function_mappings = function_mappings

        self.X = None
        self.y = None

        self.required_slots = {"departure", "destination", "date", "type"}
        self.required_slots_single = {"departure", "destination", "date", "type"}
        self.required_slots_return = {"departure", "destination", "date", "type", "return date"}
        self.current_slots = {"departure": None, "destination": None, "date": None, "type": None, "return date": None, "railcards": None, "adult passengers": None, "child passengers": None, "earliest inbound": None, "latest inbound": None, "earliest outbound": None, "latest outbound": None}

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
        with open(path) as f: # open stations.csv
            reader = csv.DictReader(f)
            for row in reader:
                if row["longname.name_alias"].lower() != "\\n":
                    self.stations.append(re.sub(regex_pattern, "", row["longname.name_alias"].replace(" Rail Station", "")))
                else:
                    self.stations.append(re.sub(regex_pattern, "", row["name"]))
        
        self.letter_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
        stations_lower = [station.lower() for station in self.stations]
        self.letter_vectorizer.fit(stations_lower)
        self.stations_matrix = self.letter_vectorizer.transform(stations_lower)
    
    def load_railcards(self, path):
        self.railcards = []
        with open(path) as f: # open railcards.txt
            for line in f:
                self.railcards.append(line.split(" : ")[0])

        railcards_lower = [railcard.lower() for railcard in self.railcards]
        self.letter_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
        self.letter_vectorizer.fit(railcards_lower)
        self.railcards_matrix = self.letter_vectorizer.transform(railcards_lower)


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

        message_tokens = [(i, word) for i, word in enumerate(message)]
        message_2grams = [(i, tuple(message[i:i+2])) for i in range(len(message)-1)]
        message_3grams = [(i, tuple(message[i:i+3])) for i in range(len(message)-2)]
        message_4grams = [(i, tuple(message[i:i+4])) for i in range(len(message)-3)]

        min_token_cosine_score = 0.92
        min_2gram_cosine_score = 0.9
        min_3gram_cosine_score = 0.76
        min_4gram_cosine_score = 0.7
        most_similar_stations = []
        unique_similar_stations = set()
        common_words = set(["i", "wanna", "want", "to", "go", "travel", "from", "to", "destination", "departure", "on", "form", "ot", "the", "a", "at", "must", "have", "and"])
        keep_going = True
        while keep_going:
            for start_idx, word in message_tokens:
                if word in common_words:
                    continue
                vec = self.letter_vectorizer.transform([word])
                scores = cosine_similarity(vec, self.stations_matrix).flatten()
                for station_idx, sim in enumerate(scores):
                    if sim > min_token_cosine_score:
                        most_similar_stations.append((
                            self.stations[station_idx],
                            sim,
                            start_idx,
                            1
                        ))
                        unique_similar_stations.add(self.stations[station_idx])
            for start_idx, gram in message_2grams:
                if gram[0] in common_words or gram[1] in common_words:
                    continue # skip iteration if one of the words found in the gram is a common word
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
                        unique_similar_stations.add(self.stations[station_idx])
            for start_idx, gram in message_3grams:
                if gram[0] in common_words or gram[1] in common_words or gram[2] in common_words:
                    continue # skip iteration if one of the words found in the gram is a common word
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
                        unique_similar_stations.add(self.stations[station_idx])
            for start_idx, gram in message_4grams:
                if gram[0] in common_words or gram[1] in common_words or gram[2] in common_words or gram[3] in common_words:
                    continue # skip iteration if one of the words found in the gram is a common word
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
                        unique_similar_stations.add(self.stations[station_idx])
            min_token_cosine_score -= 0.01
            min_2gram_cosine_score -= 0.01
            min_3gram_cosine_score -= 0.01
            min_4gram_cosine_score -= 0.01
            if min_2gram_cosine_score < 0.3 or len(unique_similar_stations) > 2:
                keep_going = False
        possible_stations = {}
        for name, sim, pos, w in most_similar_stations:
            sum_sim, sum_posw, sum_w = possible_stations.get(name, (0,0,0))
            possible_stations[name] = (sum_sim + sim, sum_posw + pos*w, sum_w + w)
        results = []
        for name, (total_similarity, total_position, total_weight) in possible_stations.items():
            avg_similarity = total_similarity / total_weight
            avg_position = total_position / total_weight
            results.append((name, avg_similarity, avg_position))
        found_stations = False
        removed = []
        results.sort(key=lambda x: (x[2], -x[1]))
        while not found_stations:
            two_stations = [("station0", 0, 0), ("station1", 0, 0)]
            for possible in results[:]:
                if two_stations[0] == ("station0", 0, 0):
                    two_stations[0] = possible
                elif two_stations[1] == ("station1", 0, 0):
                    two_stations[1] = possible
                elif possible[1] > two_stations[1][1]:
                    two_stations[1] = possible
                else:
                    results.remove(possible)
            if abs(two_stations[0][2] - two_stations[1][2]) >= 1:
                found_stations = True
            else:
                results.remove(two_stations[1])
                removed.append(two_stations[1])
                two_stations[0] = two_stations[1]
                two_stations[1] = ("station1", 0, 0)
            
        first_station = two_stations[0]
        second_station = two_stations[1]

        if len(possible_stations) < 2:
            return None, [None, None]
        if second_station[0] == "station1":
            removed.sort(key=lambda x: x[1])
            potentially_removed = removed.pop()
            print("Potenitally removed:")
            print(potentially_removed)
            if len(removed) > 0 and potentially_removed[2] != first_station[2]:
                second_station = potentially_removed
            else:
                return False, [first_station[0], None]

        if predicted_intent in ("get_from_x_to_y", "get_from_x_to_y_date", "get_from_x_to_y_date_single", "get_from_x_to_y_date_return"):
            departure, destination = first_station[0], second_station[0]
        else:  # reversed intent
            departure, destination = second_station[0], first_station[0]

        # get the position in the message of each statement, use the predicted intent to return which station is the departure and which is the destination
        if predicted_intent == "get_from_x_to_y_date" or predicted_intent == "get_from_x_to_y" or predicted_intent == "get_from_x_to_y_date_single" or predicted_intent == "get_from_x_to_y_date_return":
            return True, [departure, destination]
            
    def extract_railcard(self, message):
        regex_pattern = f"[{re.escape(string.punctuation)}]" # https://docs.vultr.com/python/examples/remove-punctuations-from-a-string
        message = re.sub(regex_pattern, "", message)

        message_tokens = [(i, word) for i, word in enumerate(message)]
        message_2grams = [(i, tuple(message[i:i+2])) for i in range(len(message)-1)]
        message_3grams = [(i, tuple(message[i:i+3])) for i in range(len(message)-2)]
        message_4grams = [(i, tuple(message[i:i+4])) for i in range(len(message)-3)]

        min_token_cosine_score = 0.92
        min_2gram_cosine_score = 0.9
        min_3gram_cosine_score = 0.76
        min_4gram_cosine_score = 0.7
        most_similar_railcards = []
        unique_similar_railcards = set()
        common_words = set(["i", "wanna", "want", "to", "go", "travel", "from", "to", "destination", "departure", "on", "form", "ot", "the", "a", "at", "must", "have", "and", "with", "this"])

        keep_going = True
        while keep_going:
            for start_idx, word in message_tokens:
                if word in common_words:
                    continue
                vec = self.letter_vectorizer.transform([word])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_token_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx], sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            for start_idx, gram in message_2grams:
                if gram[0] in common_words or gram[1] in common_words:
                    continue
                vec = self.letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_2gram_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx], sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            for start_idx, gram in message_3grams:
                if gram[0] in common_words or gram[1] in common_words or gram[2] in common_words:
                    continue
                vec = self.letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_3gram_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx], sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            for start_idx, gram in message_4grams:
                if gram[0] in common_words or gram[1] in common_words or gram[2] in common_words or gram[3] in common_words:
                    continue
                vec = self.letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_4gram_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx],sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            min_token_cosine_score -= 0.01
            min_2gram_cosine_score -= 0.01
            min_3gram_cosine_score -= 0.01
            min_4gram_cosine_score -= 0.01
            if min_2gram_cosine_score < 0.3 or len(unique_similar_railcards) > 2:
                keep_going = False
        possible_railcards = {}
        for name, sim in most_similar_railcards:
            if name in possible_railcards:
                possible_railcards[name] += sim
            else:
                possible_railcards[name] = sim
        highest = ("abc", -1)
        for name, sim in possible_railcards.items():
            if sim > highest[1]:
                highest = (name, sim)
        return highest[0]




    def extract_date(self, text):
        text = text.replace("on", "this")
        dt = search_dates(text, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now(), "SKIP_TOKENS": ["to", "from", "rail", "station", "travel"]})
        if not dt:
            return []
        #Sorts the dates by time
        dates = [d[1].date().isoformat() for d in dt]
        dates.sort()
        #Returns a list of dates found
        return dates
    
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
        print(f"Predicted intent: {predicted_intent}",flush=True)
        if predicted_intent == "get_from_x_to_y_date":
            result = self.process_get_from_x_to_y_date(input_message, predicted_intent)
            return result

        #If the user explicitly asks for a single/one way journey, set the type to single first and then call process_get_from_x_to_y_date
        if predicted_intent == "get_from_x_to_y_date_single":
            self.current_slots["type"] = "single"
            self.required_slots = self.required_slots_single
            result = self.process_get_from_x_to_y_date(input_message, predicted_intent)
            return result
        
        if predicted_intent == "get_from_x_to_y_date_return":
            result = self.process_get_from_x_to_y_return(input_message, predicted_intent)
            return result

        elif predicted_intent == "date":
            result = self.process_date(input_message)
            return result

        

        return random.choice(self.intents_responses[predicted_intent])
    
    def process_get_from_x_to_y_date(self, input_message, predicted_intent):
        print("stations extracted",self.extract_stations(input_message, predicted_intent),flush=True)
        success, stations = self.extract_stations(input_message, predicted_intent)
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
            #Sets the first date found to date
            self.current_slots["date"] = date[0]
        
        unfilled_slots = []
        for slot in self.required_slots:
            if not self.current_slots[slot]:
                unfilled_slots.append(slot)
        if len(unfilled_slots) == 0:
            #Gets the details from self.current_slots and sends them to searchForCheapestTrain
            departure = self.current_slots["departure"]
            destination = self.current_slots["destination"]
            date = self.current_slots["date"]
            result = self.searchForCheapestTrain(self.current_slots)
            return result
        
        #If the type hasnt already been set, return a message asking for the type
        if not self.current_slots["type"]:
            return "Ok! And did you want that to be a return journey or just a single?"
        try:
            #Removes return date from unfilled slots if it is not required
            if self.current_slots["type"] != "return":
                unfilled_slots.remove("return date")
        except:
            pass
        message = "Sure! Just tell me your "
        while len(unfilled_slots) > 2:
            message += unfilled_slots.pop() + ", "
        if len(unfilled_slots) == 2:
            message += unfilled_slots.pop() + " and "
        if len(unfilled_slots) == 1:
            return message + unfilled_slots.pop() + "."

        date = self.current_slots["date"]
        result = self.searchForCheapestTrain(self.current_slots)

        self.current_slots = {s: None for s in self.current_slots}
        return result

    
    def process_get_from_x_to_y_return(self, input_message, predicted_intent):
        self.current_slots["type"] = "return"
        self.required_slots = self.required_slots_return
        success, stations = self.extract_stations(input_message, predicted_intent)
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
        dates = self.extract_date(input_message)
        print("dates found", dates, flush=True)
        #If no dates were provided and no dates have already been set
        if len(dates) == 0 and not (self.current_slots["date"] or self.current_slots["return date"]):
            return "Sure! you want a return journey, when would you like your outboud journey to be and when would you like to come back?" 
        #If no dates were provided, but one has already been set
        elif len(dates) == 0 and (self.current_slots["date"] or self.current_slots["return date"]):
            return f"Sure! And what day would you like your other journey to be on?"
        #if one date was provided and no dates have already been set
        elif len(dates)== 1 and not (self.current_slots["date"] or self.current_slots["return date"]):
            self.current_slots["date"] = dates[0]
            return f"Sure! And what day would you like your other journey to be on?"
        #if one date was provided and one has already been set
        elif len(dates) == 1 and (bool(self.current_slots["date"]) ^ bool(self.current_slots["return date"])):
            if self.current_slots["date"]:
                self.set_outbound_and_return_dates(self.current_slots["date"], dates[0])
            elif self.current_slots["return date"]:
                self.set_outbound_and_return_dates(self.current_slots["return date"], dates[0])
        #if one date was provided and two have already been set, assume that the user wants to change one of them
        elif len(dates) == 1 and (self.current_slots["date"] and self.current_slots["return date"]):
            self.current_slots["date"] = dates[0]
            self.current_slots["return date"] = None
            return f"Sure! And what day would you like your other journey to be on?"
        elif len(dates) == 2:
            #Set the date to be the lower date and the return date to be the higher date
            self.current_slots["date"] = dates[0]
            self.current_slots["return date"] = dates[1]
        unfilled_slots = []
        for slot in self.required_slots:
            if not self.current_slots[slot]:
                unfilled_slots.append(slot)
        if len(unfilled_slots) == 0:
            #Gets the details from self.current_slots and sends them to searchForCheapestTrain
            result = self.searchForCheapestTrain(self.current_slots)
            return result

        try:
            #Removes return date from unfilled slots if it is not required
            if self.current_slots["type"] != "return":
                unfilled_slots.remove("return date")
        except:
            pass

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
        result = self.searchForCheapestTrain(self.current_slots)

        self.current_slots = {s: None for s in self.current_slots}
        return result
    
    def process_date(self, input_message):
        """Method that extracts date(s) from the message and sets the dates in self.current_slots appropriately"""
        dates = self.extract_date(input_message)
        #If one date was provided and no dates have already been set, set self.current_slots["date"] and ask the user for more details
        if len(dates) == 1 and not (self.current_slots["date"] or self.current_slots["return date"]):
            self.current_slots["date"] = dates[0]
            #return f"You want to travel on {dates[0]}! Where would you like to go?"
        #If one date was provided and one has already been set, call set_outbound_and_return_dates
        elif len(dates) == 1 and (bool(self.current_slots["date"]) ^ bool(self.current_slots["return date"])):
            if self.current_slots["date"]:
                self.set_outbound_and_return_dates(self.current_slots["date"], dates[0])
            elif self.current_slots["return date"]:
                self.set_outbound_and_return_dates(self.current_slots["return date"], dates[0])
        #if one date was provided and two have already been set, assume that the user wants to change one of them
        elif len(dates) == 1 and (self.current_slots["date"] and self.current_slots["return date"]):
            self.current_slots["date"] = dates[0]
            self.current_slots["return date"] = None
            return f"Sure! You now want to travel on {dates[0]}, please tell me again what day you would like your other journey to be on?"
        #If two dates were provided, call set_outbound_and_return_dates
        elif len(dates) == 2:
            self.set_outbound_and_return_dates(dates[0], dates[1])
        #If more than two dates were provided, ask the user to only provide two
        elif len(dates) > 2:
            return "Woah! you gave me alot of dates there, can you please give me no more than two?" 
        
        unfilled_slots = []
        for slot in self.required_slots:
            if not self.current_slots[slot]:
                unfilled_slots.append(slot)
        if len(unfilled_slots) == 0:
            #Gets the details from self.current_slots and sends them to searchForCheapestTrain
            departure = self.current_slots["departure"]
            destination = self.current_slots["destination"]
            date = self.current_slots["date"]
            result = self.searchForCheapestTrain(self.current_slots)
            return result
    
        try:
            #Removes return date from unfilled slots if it is not required
            if self.current_slots["type"] != "return":
                unfilled_slots.remove("return date")
        except:
            pass

        message = "Sure! Just tell me your "
        while len(unfilled_slots) > 2:
            message += unfilled_slots.pop() + ", "
        if len(unfilled_slots) == 2:
            message += unfilled_slots.pop() + " and "
        if len(unfilled_slots) == 1:
            return message + unfilled_slots.pop() + "."

        result = self.searchForCheapestTrain(self.current_slots)

        self.current_slots = {s: None for s in self.current_slots}
        return result

    def set_outbound_and_return_dates(self, date1, date2):
        """Method that takes two dates and sets self.current_slots["date"] to the earlier one and self.current_slots["return date"] to the later one"""
        d1 = date.fromisoformat(date1)
        d2 = date.fromisoformat(date2)
        if d1 < d2:
            self.current_slots["date"] = d1
            self.current_slots["return date"] = d2
        else:
            self.current_slots["date"] = d2
            self.current_slots["return date"] = d1

    def slotsFilled(self,intent):
        """Method that checks if the required slots are filled, returns True if they are, False if not"""
        for slot in self.required_slots[intent]:
            if not self.current_slots[slot]:
                return False
        return True
            
    def searchForCheapestTrain(self,Details):
        """
        Function: Searches for the cheapest train from departureLoc to destinationLoc at a given time.
        Parameters: departureLoc (str), destinationLoc (str), time (str), railcard (str, optional)
        Returns: str - The cheapest train information.
        """
        args = {"Origin": Details["departure"],
                "Destination": Details["destination"],
                "Type": Details["type"]
                }
        
        if Details["earliest outbound"]:
            args["Earliest_Outbound"] = Details["earliest outbound"]
        if Details["latest outbound"]:
            args["Latest_Outbound"] = Details["latest outbound"]
        if Details["earliest inbound"]:
            args["Earliest_Inbound"] = Details["earliest inbound"]
        if Details["railcards"]:
            args["Railcards"] = Details["railcards"]
        if Details["adult passengers"]:
            args["Adults"] = Details["adult passengers"]
        if Details["child passengers"]:
            args["Children"] = Details["child passengers"]

        Date = Details["date"]
        if Details["type"] == "return":
            Return_Date = Details["return date"]
        #Convert dates to DD/MM/YYYY
        if isinstance(Date, str):
            try:
                Date_obj = datetime.strptime(Date, "%Y-%m-%d")
                Date = Date_obj.strftime("%d/%m/%Y")
                args["Date"] = Date
            except ValueError:
                pass
        if Details["return date"]:
            if isinstance(Return_Date, str):
                try:
                    Return_Date_obj = datetime.strptime(Return_Date, "%Y-%m-%d")
                    returnDate = Return_Date_obj.strftime("%d/%m/%Y")
                    args["Return_Date"] = returnDate
                except ValueError:
                    pass
    
        self.Outbound_Journeys = []
        self.Inbound_Journeys = []
        #Run the scraper in a separate thread
        thread = threading.Thread(target=self.run_scraper, args=(args,), daemon=True)
        thread.start()

        #return "Outbound Journeys: " + str(Outbound_Journeys) + "\nInbound Journeys: " + str(Inbound_Journeys)
        #print(departureLoc, destinationLoc, date, railcard,flush=True)
        return "Searching for train tickets..."
        #return f"Searching for the cheapest train from {departureLoc} to {destinationLoc} on {date} with a {type} ticket. Return date: {returnDate}."

    def run_scraper(self,args):
        """Method that runs the WebScraper and sets Outbound_Journeys and Inbound_Journeys to the results"""
        WebScraper = TicketFinder(**args)
        Journies = WebScraper.Search()
        
        if Journies == [] or Journies[0] == [] or (Journies[1] == [] and self.current_slots["type"] == "return"):
            Output_Message = "Sorry, I couldn't find any journeys that match your criteria."
            self.Outbound_Journeys = Output_Message
            return
        
        #Gets the cheapest outbound single journey
        Cheapest_Outbound = Journies[0][0]
        for journey in Journies[0]:
            if journey["Price"] < Cheapest_Outbound["Price"]:
                Cheapest_Outbound = journey
            
        #Gets the fastest outbound single journey
        Fastest = Journies[0][0]
        for journey in Journies[0]:
            if journey["Duration"] < Fastest["Duration"]:
                Fastest = journey

        if self.current_slots["type"] == "single":
            #Output the cheapest and fasted single journeys
            Output_Message = "I found some journeys for you!\n\n"
            Output_Message += f"The cheapest journey you could get costs {self.convert_price(Cheapest_Outbound['Price'])}:\n" 
            Output_Message += f"Departure Time: {Cheapest_Outbound['Start_Time']}\n Arrival Time: {Cheapest_Outbound['Arrival_Time']}\n Duration: {Cheapest_Outbound['Duration']}\n\n"
            Output_Message += f"And the fastest journey you could get takes {Fastest['Duration']}:\n"
            Output_Message += f"Departure Time: {Fastest['Start_Time']}\n Arrival Time: {Fastest['Arrival_Time']}\n Price: {self.convert_price(Fastest['Price'])}\n\n"
            Output_Message += f"You can book the tickets here: {Cheapest_Outbound['Link']}"
            #display all the journeys
            # for journey in Journies[0]:
            #     Output_Message += f"\nPrice: {journey['Price']}, Departure Time: {journey['Start_Time']}, Arrival Time: {journey['Arrival_Time']}, Duration: {journey['Duration']}" 
        
        elif self.current_slots["type"] == "return":
            #Gets the cheapest inbound single journey
            Cheapest_Inbound = Journies[1][0]
            for journey in Journies[1]:
                if journey["Price"] < Cheapest_Inbound["Price"]:
                    Cheapest_Inbound = journey
            
            #Gets the fastest inbound single journey
            Fastest_Inbound = Journies[1][0]
            for journey in Journies[1]:
                if journey["Duration"] < Fastest_Inbound["Duration"]:
                    Fastest_Inbound = journey

            print("Cheapest Outbound", Cheapest_Outbound, flush=True)
            print("Cheapest Inbound", Cheapest_Inbound, flush=True)
            #Gets the cheapest return journey as two singles
            Cheapest_Singles = Cheapest_Outbound["Price"] + Cheapest_Inbound["Price"]

            #Find cheapest return price that exists in both outbound and inbound journeys
            Sorted_Outbounds = sorted(Journies[0], key=lambda journey: journey["Return_Price"])
            Inbound_Return_Prices = [journey["Return_Price"] for journey in Journies[1]]
            Cheapest_Return = Sorted_Outbounds[0]
            for journey in Sorted_Outbounds:
                if journey["Return_Price"] in Inbound_Return_Prices:
                    print("Found a return price",journey["Return_Price"],flush=True)
                    Cheapest_Return = journey
                    break
            
            #Whichever is lower is the price of the cheapest return journey possible, wether thats by using two singles or a just a return
            print("Cheapest Singles", Cheapest_Singles, "Cheapest Return", Cheapest_Return["Return_Price"],flush=True)
            if Cheapest_Singles < Cheapest_Return["Return_Price"]:
                #Get the times of all of the return journeys that have a single price of Cheapest_Inbound
                Cheapest_Returns = []
                for journey in Journies[1]:
                    if journey["Price"] == Cheapest_Inbound["Price"]:
                        print("Appending cheapest price",journey["Price"],flush=True)
                        Cheapest_Returns.append(journey)
                Cheapest_Return_Singles = True
                Cheapest_Return = Cheapest_Outbound

            else:
                Cheapest_Returns = []
                for journey in Journies[1]:
                    if journey["Return_Price"] == Cheapest_Return["Return_Price"]:
                        print("Appending cheapest return price",journey["Return_Price"],flush=True)
                        Cheapest_Returns.append(journey)


            Output_Message = "I found some journeys for you!\n\n"
            if Cheapest_Return_Singles:
                Output_Message += f"The cheapest journey you could get costs {self.convert_price(Cheapest_Singles)}: (this price includes the return journey)\n" 
            else:
                Output_Message += f"The cheapest journey you could get costs {self.convert_price(Cheapest_Return['Return_Price'])}: (this price includes the return journey)\n" 
            Output_Message += f"Departure Time: {Cheapest_Return['Start_Time']}\n Arrival Time: {Cheapest_Return['Arrival_Time']}\n Duration: {Cheapest_Return['Duration']}\n\n"
            Output_Message += f"This would allow you to return at any of these times:\n"
            for journey in Cheapest_Returns:
                Output_Message += f"{journey['Start_Time']}\n"
            Output_Message += f"\nAnd the fastest journey you could get takes {Fastest['Duration']} outbound:\n"
            Output_Message += f"Departure Time: {Fastest['Start_Time']}\n Arrival Time: {Fastest['Arrival_Time']}\n Price: {self.convert_price(Fastest['Return_Price'])} (this price includes the return journey)\n\n"
            Output_Message += f"With the fastest return journey taking {Fastest_Inbound['Duration']}:\n"
            Output_Message += f"Departure Time: {Fastest_Inbound['Start_Time']}\n Arrival Time: {Fastest_Inbound['Arrival_Time']}\n\n"
            Output_Message += f"You can book the tickets for any of these journeys here: {Cheapest_Outbound['Link']}"
        self.Outbound_Journeys = Output_Message
        self.Inbound_Journeys = Journies[1]

    def convert_price(self, price):
        """Method that converts the price from P to £x.xx"""
        Pounds = str(price)[:-2]
        Pence = str(price)[-2:]
        return f"£{Pounds}.{Pence}"
    
    def get_journeys(self):
        """Method that checks if the results returned by the WebScraper have been found, returns False if not, returns self.Journeys if they have"""
        if not self.Outbound_Journeys:
            return ""
        else:
            return self.Outbound_Journeys

def getDelay(currentStation, destination, originalArrivalTime, currentDelay):
    return f"Calculating delay"

if __name__ == "__main__":
    intents_path = os.path.join(os.path.dirname(__file__), "intents.json")
    assistant = ChatbotAssistant(intents_path)
    assistant.parse_intents()
    assistant.load_stations(os.path.join(os.path.dirname(__file__), "../data/stations.csv"))
    assistant.load_railcards(os.path.join(os.path.dirname(__file__), "../data/railcards.txt"))
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=150)
    assistant.save_model("model.pth", "dimensions.json")