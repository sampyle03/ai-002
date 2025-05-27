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
import psycopg2
try:
    from scraper import TicketFinder
except:
    from not_web_scraper import TicketFinder

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
        self.previous_response = None
        self.current_chat = False #Variable to keep track of the current chat
        #self.function_mappings = function_mappings

        self.X = None
        self.y = None

        self.current_task = None
        self.required_slots = {"departure", "destination", "date", "type"}
        self.required_slots_delay = {"current station", "destination", "original arrival time", "current delay"}
        self.required_slots_single = {"departure", "destination", "date", "type"}
        self.required_slots_return = {"departure", "destination", "date", "type", "return date"}
        self.current_slots = {"departure": None, "destination": None, "date": None, "type": None, "return date": None, "railcards": None, "adult passengers": 1, "child passengers": None, "earliest inbound": None, "latest inbound": None, "earliest outbound": None, "latest outbound": None}
        self.current_slots_delay = {"current station": None, "destination": None, "original arrival time": None, "current delay": None}

        self.temp = None

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
        
        self.station_letter_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
        stations_lower = [station.lower() for station in self.stations]
        self.station_letter_vectorizer.fit(stations_lower)
        self.stations_matrix = self.station_letter_vectorizer.transform(stations_lower)
    
    def load_railcards(self, path):
        self.railcards = []
        with open(path) as f: # open railcards.txt
            for line in f:
                self.railcards.append(line.split(" : ")[0])

        railcards_lower = [railcard.lower() for railcard in self.railcards]
        self.railcard_letter_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
        self.railcard_letter_vectorizer.fit(railcards_lower)
        self.railcards_matrix = self.railcard_letter_vectorizer.transform(railcards_lower)


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

    def get_most_similar_stations(self, message):
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
        common_words = set(["i", "wanna", "want", "to", "go", "travel", "from", "to", "destination", "departure", "on", "form", "ot", "the", "a", "at", "must", "have", "and", "going", "delayed", "delay", "go", "gone", "my", "me", "meant", "supposed", "but", "however", "when"])
        keep_going = True
        while keep_going:
            for start_idx, word in message_tokens:
                if word in common_words:
                    continue
                vec = self.station_letter_vectorizer.transform([word])
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
                vec = self.station_letter_vectorizer.transform([" ".join(gram)])
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
                vec = self.station_letter_vectorizer.transform([" ".join(gram)])
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
                vec = self.station_letter_vectorizer.transform([" ".join(gram)])
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
            if min_2gram_cosine_score < 0.7 or len(unique_similar_stations) > 2:
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
        results.sort(key=lambda x: (x[2], -x[1]))
        return results, possible_stations

    def extract_one_station(self, message):
        results, possible_stations = self.get_most_similar_stations(message)
        if len(results) > 0:
            return True, results[0][0]
        else:
            return False, None

    def extract_stations(self, message, predicted_intent):
        results, possible_stations = self.get_most_similar_stations(message)
        found_stations = False
        removed = []
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
                try:
                    results.remove(two_stations[1])
                except: # called when no stations are found usually
                    return False, [None, None]
                removed.append(two_stations[1])
                two_stations[0] = two_stations[1]
                two_stations[1] = ("station1", 0, 0)
            
        first_station = two_stations[0]
        second_station = two_stations[1]

        if len(possible_stations) < 2:
            return False, [list(possible_stations.keys())[0], None]
        if second_station[0] == "station1":
            removed.sort(key=lambda x: x[1])
            potentially_removed = removed.pop()
            if len(removed) > 0 and potentially_removed[2] != first_station[2]:
                second_station = potentially_removed
            else:
                return False, [first_station[0], None]

        if predicted_intent in ("get_from_x_to_y", "get_from_x_to_y_date", "get_from_x_to_y_date_single", "get_from_x_to_y_date_return", "get_delay_from_x_to_y"):
            departure, destination = first_station[0], second_station[0]
        else:  # reversed intent
            departure, destination = second_station[0], first_station[0]

        # get the position in the message of each statement, use the predicted intent to return which station is the departure and which is the destination
        #if predicted_intent == "get_from_x_to_y_date" or predicted_intent == "get_from_x_to_y" or predicted_intent == "get_from_x_to_y_date_single" or predicted_intent == "get_from_x_to_y_date_return":
        return True, [departure, destination]
            
    def extract_railcard(self, message, useCommonWords = True):
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
        common_words = set(["i", "wanna", "want", "to", "go", "travel", "from", "to", "destination", "departure", "on", "form", "ot", "the", "a", "at", "must", "have", "and", "going", "delayed", "delay", "go", "gone", "my", "me", "meant", "supposed", "but", "however", "when"])

        keep_going = True
        while keep_going:
            for start_idx, word in message_tokens:
                if word in common_words and useCommonWords:
                    continue
                vec = self.railcard_letter_vectorizer.transform([word])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_token_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx], sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            for start_idx, gram in message_2grams:
                if (gram[0] in common_words or gram[1] in common_words) and useCommonWords:
                    continue
                vec = self.railcard_letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_2gram_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx], sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            for start_idx, gram in message_3grams:
                if (gram[0] in common_words or gram[1] in common_words or gram[2] in common_words) and useCommonWords:
                    continue
                vec = self.railcard_letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_3gram_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx], sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            for start_idx, gram in message_4grams:
                if (gram[0] in common_words or gram[1] in common_words or gram[2] in common_words or gram[3] in common_words) and useCommonWords:
                    continue
                vec = self.railcard_letter_vectorizer.transform([" ".join(gram)])
                scores = cosine_similarity(vec, self.railcards_matrix).flatten()
                for railcard_idx, sim in enumerate(scores):
                    if sim > min_4gram_cosine_score:
                        most_similar_railcards.append((self.railcards[railcard_idx],sim))
                        unique_similar_railcards.add(self.railcards[railcard_idx])
            min_token_cosine_score -= 0.01
            min_2gram_cosine_score -= 0.01
            min_3gram_cosine_score -= 0.01
            min_4gram_cosine_score -= 0.01
            if min_2gram_cosine_score < 0.4 or len(unique_similar_railcards) > 2:
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
        if highest[0] == "abc":
            return self.extract_railcard(message, False)
        return highest[0]


    def preprocess_date_text(self, text):
        text = text.lower()
        text = text.replace("on", "this")
        text = text.replace("next week", "in 7 days")
        return text

    def extract_date(self, text):
        text = self.preprocess_date_text(text)
        dt = search_dates(text, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now(), "SKIP_TOKENS": ["to", "from", "rail", "station", "travel"], "DATE_ORDER": "DMY"})
        print(f"dt = {dt}", flush=True)
        if not dt:
            day = re.search(r"\b(\d{1,2})(st|nd|rd|th)\b", text.lower()) # https://stackoverflow.com/questions/2118825/modify-regex-to-match-dates-with-ordinals-st-nd-rd-th
            print(f"day = {day}", flush=True)
            if day:
                day = int(day.group(1))
                now = datetime.now()
                try:
                    candidate = datetime(now.year, now.month, day)
                    print(f"candidate = {candidate}", flush=True)
                    if candidate.date() <= now.date():
                        # If candidate day has passed this month, use next month
                        if now.month == 12:
                            candidate = datetime(now.year + 1, 1, day)
                        else:
                            candidate = datetime(now.year, now.month + 1, day)
                    return [candidate.date().isoformat()]
                except:
                    candidate = datetime(now.year, now.month, day)
                    print(f"candidate = {candidate}", flush=True)
                    return []
        if not dt:
            return []
        #Sorts the dates by time
        dates = [d[1].date().isoformat() for d in dt]
        dates.sort()
        #Returns a list of dates found
        return dates
    
    def extract_passengers(self, text):
        self.current_slots["adult passengers"] = None
        # "2 adults and 3 children", "one adult"
        numbers = set(["1","2","3","4","5"])
        wordsForAdultTicket = set(["adult", "adults", "adult's", "adults'", "adult's", "adults'", "adult's", "adults'", "people", "person", "people's", "person's", "people's", "person's"])
        wordsForChildTicket = set(["child", "children", "child's", "children's", "child's", "children's", "child's", "children's", "kids", "kid", "kids'", "kid's", "kids'", "kid's", "kids'", "boy", "girl", "boys", "girls"])
        
        text = text.replace("one", "1").replace("two", "2").replace("three", "3").replace("four", "4").replace("five", "5")
        text = nltk.word_tokenize(text.lower())
        numberOfAdultsOrChildren = None
        for word in text:
            if word in numbers:
                numberOfAdultsOrChildren = word
            elif word.lower() in wordsForAdultTicket:
                try:
                    self.current_slots["adult passengers"] = numberOfAdultsOrChildren
                except:
                    pass
            elif word.lower() in wordsForChildTicket:
                try:
                    self.current_slots["child passengers"] = numberOfAdultsOrChildren
                except:
                    pass


    def extract_time(self, text):
        cleaned_text = re.sub(r"[^\w\s:]", "", text)
        time_regex = "^(?:[01]?[0-9]|2[0-3]):[0-5]?[0-9](?::[0-5]?[0-9])?$" # https://www.geeksforgeeks.org/validating-traditional-time-formats-using-regular-expression/
        message_tokens = nltk.word_tokenize(cleaned_text)
        for token in message_tokens:
            if re.match(time_regex, token):
                return token
        return None
    
    def extract_current_delay(self, text):
        message_number_corrections = text.replace("one", "1").replace("two", "2").replace("three", "3").replace("four", "4").replace("five", "5")
        message_tokens_number_corrections = nltk.word_tokenize(message_number_corrections)
        words_for_hour = set(["hour", "hours", "hour's", "hours'", "hour's", "hours'", "hour's", "hours'", "hour's", "hours'", "hour's", "hours'", "hr", "hrs", "hr's"])
        words_for_minute = set(["minute", "minutes", "mins", "min", "minute's", "minutes'", "min's", "min's", "minute's", "minutes'", "min's", "minute's", "minutes'", "min's"])
        words_for_second = set(["second", "seconds", "sec", "secs", "second's", "seconds'", "sec's", "secs'", "second's", "seconds'", "sec's", "secs'", "second's", "seconds'", "sec's", "secs'"])
        for token in message_tokens_number_corrections:
            if token.isnumeric():
                number = token
            elif token.lower() in words_for_hour:
                try:
                    return int(number) * 60
                except:
                    pass
            elif token.lower() in words_for_minute:
                try:
                    return int(number)
                except:
                    pass 
            elif token.lower() in words_for_second:
                try:
                    return int(number) / 60
                except:
                    pass
        return None
    
    def extract_info_for_delay_calculation(self, input_message, predicted_intent):
        success, stations = self.extract_stations(input_message, predicted_intent)
        return_needed = False
        if stations[0] and stations[1]:
            if not success:
                if not self.current_slots_delay["current station"] and not self.current_slots_delay["destination"]:
                    message = "It seems like you mentioned one of "
                    while len(stations) > 2:
                        message += stations.pop() + ", "
                    if len(stations) == 2:
                        message += stations.pop() + " and "
                    if len(stations) == 1:
                        message += stations.pop()
                    self.previous_response = "specify_which_station"
                    return(f"{message}. Please specify which of these you mean.")
                elif not self.current_slots_delay["current station"]:
                    self.current_slots_delay["current station"] = stations[0]
                elif not self.current_slots_delay["destination"]:
                    self.current_slots_delay["destination"] = stations[1]
            if not self.current_slots_delay["current station"]:
                self.current_slots_delay["current station"] = stations[0]
            if not self.current_slots_delay["destination"] and len(stations) > 1:
                self.current_slots_delay["destination"] = stations[1]
        elif stations[0] is not None and stations[1] is None:
            self.previous_response = "is_station_current"
            self.temp = stations[0]
            return_needed = True
        
        possible_original_arrival_time = self.extract_time(input_message)
        if possible_original_arrival_time:
            self.current_slots_delay["original arrival time"] = possible_original_arrival_time

        possible_current_delay = self.extract_current_delay(input_message)
        if possible_current_delay:
            self.current_slots_delay["current delay"] = possible_current_delay
        
        if return_needed:
            return f"Ok! {stations[0]} is your current station - correct?"
        else:
            return None


    def get_next_slot(self):
        if self.current_task == "get_delay":
            return self.get_next_slot_delay()
        else:
            return self.get_next_slot_ticket()

    def get_next_slot_ticket(self):
        """Fixed get_next_slot logic for ticket booking"""
        print(f"Current slots: {self.current_slots}", flush=True)
        
        # Ensure adult passengers is never None
        if self.current_slots["adult passengers"] is None:
            self.current_slots["adult passengers"] = 1
        
        # Check required slots
        missing_slots = []
        for slot in self.required_slots:
            if self.current_slots[slot] is None:
                missing_slots.append(slot)
        print(f"missing_slots: {missing_slots}", flush=True)
        
        if not missing_slots:
            # All required slots are filled
            self.previous_response = "required_details_entered_any_other_details"
            return "Ok! Do you want to enter any other details?"
        
        # Ask for the first missing slot
        slot = missing_slots[0]
        if slot == "type":
            return "Ok! Will you require a single or return ticket?"
        elif slot == "date":
            self.previous_response = "when_departure_journey"
            return "Ok! And what day will be your departure date?"
        elif slot == "return date":
            self.previous_response = "when_return_journey"
            return "Ok! And what day will be your return date?"
        elif slot == "departure":
            self.previous_response = "where_departure_station"
            return "Ok! And what is your departure station?"
        elif slot == "destination":
            self.previous_response = "where_destination_station"
            return "Ok! And what is your destination station?"
        else:
            return f"Ok! And what is your {slot}?"

    def get_next_slot_delay(self):
        """Gets the next required slot for delay calculation"""
        print("Delay slots:", self.current_slots_delay, flush=True)
        
        missing_slots = []
        for slot in self.required_slots_delay:
            if self.current_slots_delay[slot] is None:
                missing_slots.append(slot)
        
        if not missing_slots:
            # All slots filled, calculate delay
            return getDelay(
                self.current_slots_delay["current station"],
                self.current_slots_delay["destination"], 
                self.current_slots_delay["original arrival time"],
                self.current_slots_delay["current delay"]
            )
        
        # Ask for the next missing slot
        slot = missing_slots[0]
        if slot == "current station":
            self.previous_response = "where_current_station"
            return "What station are you currently at?"
        elif slot == "destination":
            self.previous_response = "where_destination_station"
            return "What's your destination station?"
        elif slot == "original arrival time":
            self.previous_response = "when_original_arrival"
            return "What time were you originally scheduled to arrive? (Please use HH:MM format, e.g., 14:30)"
        elif slot == "current delay":
            self.previous_response = "how_long_delayed"
            return "How many minutes is your train currently delayed?"
        
        return f"I need your {slot}. Could you please provide that information?"
                
    
    def process_message(self, input_message):
        #Checks if a chat is ongoing, if not, create a new chat
        if not self.current_chat:
            self.current_chat = self.create_new_chat(["chatbot","Welcome to Train Talk! How can I help?"]) #This will create a new chat in the DB and return the new chat id
            self.update_chat_messages(["user", input_message])
        else:
            self.update_chat_messages(["user", input_message])
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        # Predict intent of message
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        boosted_predictions = self.boost_intent_likelihood(predictions)
        predicted_class_index = torch.argmax(boosted_predictions, dim=1).item() # Use argmax to get the index of the predicted class
        predicted_intent = self.intents[predicted_class_index]
        print(f"Predicted intent: {predicted_intent}",flush=True)

        

        # Handle task switching and delay intents
        if predicted_intent == "get_delay_from_x_to_y" or predicted_intent == "delay_general":
            self.current_task = "get_delay"
            if predicted_intent == "get_delay_from_x_to_y":
                message = self.extract_info_for_delay_calculation(input_message, predicted_intent)
                if message is not None:
                    return message
            return self.get_next_slot()
        elif self.current_task == "get_delay" and predicted_intent == "new_search":
            self.current_task = "get_ticket"
            self.current_slots = {"departure": None, "destination": None, "date": None, "type": None, "return date": None, "railcards": None, "adult passengers": 1, "child passengers": None, "earliest inbound": None, "latest inbound": None, "earliest outbound": None, "latest outbound": None}
            self.current_slots_delay = {"current station": None, "destination": None, "original arrival time": None, "current delay": None}
            self.previous_response = None
            return random.choice(self.intents_responses[predicted_intent])
        elif predicted_intent != "get_delay_from_x_to_y" and predicted_intent != "delay_general":
            if self.current_task != "get_delay":
                self.current_task = "get_ticket"

        # Handle delay-specific responses when in delay mode
        if self.current_task == "get_delay":
            # Handle time input for original arrival time
            if self.previous_response == "when_original_arrival":
                possible_time = self.extract_time(input_message)
                if possible_time:
                    self.current_slots_delay["original arrival time"] = possible_time
                    return self.get_next_slot()
                else:
                    return "I didn't understand that time format. Please use HH:MM format (e.g., 14:30 for 2:30 PM)."
            
            # Handle delay amount input
            elif self.previous_response == "how_long_delayed":
                possible_delay = self.extract_current_delay(input_message)
                if possible_delay is not None:
                    self.current_slots_delay["current delay"] = possible_delay
                    return self.get_next_slot()
                else:
                    return "I didn't catch how long the delay is. Please tell me in minutes (e.g., '15 minutes' or '1 hour')."
            
            # Handle station confirmation for delay
            elif predicted_intent == "yes" and self.previous_response == "is_station_current":
                self.current_slots_delay["current station"] = self.temp
                station_name = self.temp
                self.temp = None
                return f"Ok! Current station is {station_name}!\n" + self.get_next_slot()
            
            elif predicted_intent == "no" and self.previous_response == "is_station_current":
                self.current_slots_delay["destination"] = self.temp
                station_name = self.temp
                self.temp = None
                return f"Ok! Destination is {station_name}!\n" + self.get_next_slot()
            
            # Handle station names when in delay mode
            elif predicted_intent == "noanswer":
                success, possible_station = self.extract_one_station(input_message)
                if success:
                    # Check what we're waiting for
                    missing_slots = [slot for slot in self.required_slots_delay if self.current_slots_delay[slot] is None]
                    if missing_slots:
                        first_missing = missing_slots[0]
                        if first_missing == "current station":
                            self.current_slots_delay["current station"] = possible_station
                            return self.get_next_slot()
                        elif first_missing == "destination":
                            self.current_slots_delay["destination"] = possible_station
                            return self.get_next_slot()
                # If no station found, ask for clarification
                return "I didn't recognize that station name. Could you try again?"
            
            # Continue with delay flow for other intents
            return self.get_next_slot()

        if predicted_intent == "date":
            if self.current_task == "get_delay":
                if self.previous_response == "when_arrival_time":
                    possible_original_arrival_time = self.extract_date(input_message)
                    if possible_original_arrival_time:
                        self.current_slots_delay["original arrival time"] = possible_original_arrival_time
            elif self.current_task == "get_ticket":
                result = self.process_date(input_message)
                if result:
                    return result
                return self.get_next_slot()
        elif predicted_intent == "new_search":
            self.current_slots = {"departure": None, "destination": None, "date": None, "type": None, "return date": None, "railcards": None, "adult passengers": None, "child passengers": None, "earliest inbound": None, "latest inbound": None, "earliest outbound": None, "latest outbound": None}
            self.current_slots_delay = {"current station": None, "destination": None, "original arrival time": None, "current delay": None}
            self.previous_response = None
            return random.choice(self.intents_responses[predicted_intent])
        elif predicted_intent == "adult_passengers" or predicted_intent == "child_passengers":
            self.extract_passengers(input_message)
            return self.get_next_slot()
        elif predicted_intent == "single":
            self.current_slots["type"] = "single"
            self.required_slots = self.required_slots_single
            return self.get_next_slot()
        # if predicted intent is "return", "return please" etc 
        elif predicted_intent == "return":
            self.current_slots["type"] = "return"
            self.required_slots = self.required_slots_return
            return self.get_next_slot()
        # if user has said "16-25 Railcard", "I've not got a railcard", "North Lincolnshire Concessionary 34%" etc.
        elif predicted_intent == "railcard":
            potential_railcard = self.extract_railcard(input_message)
            # If the user does have a railcard
            if potential_railcard == "Yes yeah ye, I do indeed have a railcard":
                return "Ok! Which type of railcard do you have?"
            # If the user does not have a railcard
            elif potential_railcard == "No, nah I've don't not got a Railcard":
                self.current_slots["railcard"] = None
                return self.get_next_slot()
            # If the user has entered a specific railcard
            self.current_slots["railcard"] = potential_railcard
            return self.get_next_slot()
        elif predicted_intent == "yes":
            if self.previous_response == "is_station_current":
                self.current_slots["current station"] = self.temp
                self.temp = None
                return f"Ok! Current station is {self.current_slots['current station']}!\n"+self.get_next_slot()
            elif self.previous_response == "is_station_departure":
                self.current_slots["departure"] = self.temp
                self.temp = None
                return f"Ok! Departure station is {self.current_slots['departure']}!\n"+self.get_next_slot()

        # if predicted intent is "no", "nah thanks", "nope" etc AND they've been asked whether they'd like to enter any other details because they have enetered all required details
        elif predicted_intent == "no" and self.previous_response == "required_details_entered_any_other_details":
            return self.searchForCheapestTrain(self.current_slots)
        elif predicted_intent == "no" and self.previous_response == "is_station_current":
            self.current_slots["destination"] = self.temp
            self.temp = None
            return f"Ok! Destination is {self.current_slots['destination']}!\n"+self.get_next_slot()
        # if predicted intent is "no", "nah thanks", "nope" etc AND they've NOT been asked whether they'd like to enter any other details because they have enetered all required details
        elif predicted_intent == "no" and self.previous_response != "required_details_entered_any_other_details":
            return self.get_next_slot()
        # if predicted intent is "I wanna travel from Norwich to Shenfield on Friday", "I would like a ticket from Blackpool North to Blackpool South on 23/06/2025"
        elif predicted_intent == "get_from_x_to_y_date":
            result = self.process_get_from_x_to_y_date(input_message, predicted_intent)
            return result
        # if predicted intent is "I wanna travel one-way from Norwich to Shenfield on Friday", "I would like a single ticket from Blackpool North to Blackpool South on 23/06/2025"
        elif predicted_intent == "get_from_x_to_y_date_single":
            self.current_slots["type"] = "single"
            self.required_slots = self.required_slots_single
            result = self.process_get_from_x_to_y_date(input_message, predicted_intent)
            return result
        # if predicted intent is "I wanna travel to Shenfield and back from Norwich on Friday", "I would like a return ticket from Blackpool North to Blackpool South on 23/06/2025"
        elif predicted_intent == "get_from_x_to_y_date_return":
            self.required_slots = self.required_slots_return
            result = self.process_get_from_x_to_y_return(input_message, predicted_intent)
            return result
        elif predicted_intent == "earliest_inbound" or predicted_intent == "latest_inbound" or predicted_intent == "earliest_outbound" or predicted_intent == "latest_outbound":
            if self.current_task == "get_ticket":
                time = self.extract_time(input_message)
                if time:
                    if predicted_intent == "earliest_inbound":
                        self.current_slots["earliest inbound"] = time
                    elif predicted_intent == "latest_inbound":
                        self.current_slots["latest inbound"] = time
                    elif predicted_intent == "earliest_outbound":
                        self.current_slots["earliest outbound"] = time
                    elif predicted_intent == "latest_outbound":
                        self.current_slots["latest outbound"] = time
                    return self.get_next_slot()
        elif predicted_intent == "noanswer":
            if self.current_task == "get_ticket":
                # Handle station input for ticket booking (existing functionality)
                success, possible_station = self.extract_one_station(input_message)
                if success:
                    if self.previous_response == "where_current_station":
                        self.current_slots["current station"] = possible_station
                        return self.get_next_slot()
                    elif self.previous_response == "where_destination_station":
                        self.current_slots["destination"] = possible_station
                        return self.get_next_slot()
            elif self.current_task == "get_ticket":
                success, possible_station = self.extract_one_station(input_message)
                if success:
                    if self.previous_response == "where_departure_station":
                        self.current_slots["departure"] = possible_station
                        return self.get_next_slot()
                    elif self.previous_response == "where_destination_station":
                        self.current_slots["destination"] = possible_station
                        return self.get_next_slot()
        elif predicted_intent == "thanks":
            self.current_slots = {"departure": None, "destination": None, "date": None, "type": None, "return date": None, "railcards": None, "adult passengers": None, "child passengers": None, "earliest inbound": None, "latest inbound": None, "earliest outbound": None, "latest outbound": None}
            self.current_slots_delay = {"current station": None, "destination": None, "original arrival time": None, "current delay": None}
            self.previous_response = None
            return random.choice(self.intents_responses[predicted_intent])
        

        return random.choice(self.intents_responses[predicted_intent])

    def boost_intent_likelihood(self, predictions):
        context_boost_map = {
            "where_departure_station": {"noanswer": 5},
            "where_destination_station": {"noanswer": 5},
            "where_current_station": {"noanswer": 5},
            "when_departure_journey": {"date": 3.0},
            "when_return_journey": {"date": 3.0},
            "is_station_current": {"yes": 5, "no": 5}
        }

        if self.previous_response in context_boost_map:
            boosts = context_boost_map[self.previous_response]
            for intent, boost in boosts.items():
                if intent in self.intents:
                    index = self.intents.index(intent)
                    predictions[0][index] += boost

        return predictions

    
    def process_get_from_x_to_y_date(self, input_message, predicted_intent):
        success, stations = self.extract_stations(input_message, predicted_intent)
        print(success, stations)
        if stations[0] and stations[1]:
            if not success:
                if not self.current_slots["departure"] and not self.current_slots["destination"]:
                    message = "It seems like you mentioned one of "
                    while len(stations) > 2:
                        message += stations.pop() + ", "
                    if len(stations) == 2:
                        message += stations.pop() + " and "
                    if len(stations) == 1:
                        message += stations.pop()
                    self.previous_response = "specify_which_station"
                    return(f"{message}. Please specify which of these you mean.")
                elif not self.current_slots["departure"]:
                    self.current_slots["departure"] = stations[0]
                elif not self.current_slots["destination"]:
                    self.current_slots["departure"] = stations[0]
            if not self.current_slots["departure"]:
                self.current_slots["departure"] = stations[0]
            if not self.current_slots["destination"] and len(stations) > 1:
                self.current_slots["destination"] = stations[1]
        elif stations[0] is not None and stations[1] is None:
            self.previous_response = "is_station_departure"
            self.temp = stations[0]
            return f"Ok! {stations[0]} is your departure station - correct?"
    
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
            # result = self.searchForCheapestTrain(self.current_slots)
            # return result
            return self.get_next_slot()
        
        #If the type hasnt already been set, return a message asking for the type
        if not self.current_slots["type"]:
            return "Ok! And did you want that to be a return journey or just a single?"
        try:
            #Removes return date from unfilled slots if it is not required
            if self.current_slots["type"] != "return":
                unfilled_slots.remove("return date")
        except:
            pass
        return self.get_next_slot()

    
    def process_get_from_x_to_y_return(self, input_message, predicted_intent):
        self.current_slots["type"] = "return"
        self.required_slots = self.required_slots_return
        success, stations = self.extract_stations(input_message, predicted_intent)
        if stations[0] and stations[1]:
            if not success:
                if not self.current_slots["departure"] and not self.current_slots["destination"]:
                    message = "It seems like you mentioned one of "
                    while len(stations) > 2:
                        message += stations.pop() + ", "
                    if len(stations) == 2:
                        message += stations.pop() + " and "
                    if len(stations) == 1:
                        message += stations.pop()
                    self.previous_response = "specify_which_station"
                    return(f"{message}. Please specify which of these you mean.")
                elif not self.current_slots["departure"]:
                    self.current_slots["departure"] = stations[0]
                elif not self.current_slots["destination"]:
                    self.current_slots["departure"] = stations[0]
            if not self.current_slots["departure"]:
                self.current_slots["departure"] = stations[0]
            if not self.current_slots["destination"] and len(stations) > 1:
                self.current_slots["destination"] = stations[1]
        elif stations[0] is not None and stations[1] is None:
            self.previous_response = "is_station_departure"
            self.temp = stations[0]
            return f"Ok! {stations[0]} is your departure station - correct?"
        else:
            return random.choice(["I'm not too sure what specific stations you mean! Could you clarify?", "Could you specify which stations you mean?"])
        dates = self.extract_date(input_message)
        #If no dates were provided and no dates have already been set
        if len(dates) == 0 and not (self.current_slots["date"] or self.current_slots["return date"]):
            self.previous_response = "when_departure_journey"
            return "Sure! You want a return journey. What day will be your departure journey?" 
        #If no dates were provided, but one has already been set
        elif len(dates) == 0 and (self.current_slots["date"] or self.current_slots["return date"]):
            self.previous_response = "when_return_journey"
            return f"Sure! And what day would you like your other journey to be on?"
        #if one date was provided and no dates have already been set
        elif len(dates)== 1 and not (self.current_slots["date"] or self.current_slots["return date"]):
            self.current_slots["date"] = dates[0]
            self.previous_response = "when_return_journey"
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

        self.get_next_slot()
    
    def process_date(self, input_message):
        """Method that extracts date(s) from the message and sets the dates in self.current_slots appropriately"""
        dates = self.extract_date(input_message)
        print("date = ", dates, flush = True)
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
        #If two dates were provided, call set_outbound_and_return_dates
        elif len(dates) == 2:
            self.set_outbound_and_return_dates(dates[0], dates[1])
        #If more than two dates were provided, ask the user to only provide two
        elif len(dates) > 2:
            return "Woah! you gave me alot of dates there, can you please give me no more than two?"
        return None
        

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
                "Type": Details["type"],
                "Date": Details["date"],
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

    def start_new_chat(self):
        """Method that gets triggered when user clicks the New Chat button, this only needs to reset the current and required slots"""
        self.current_chat = False #
        self.current_task = None
        self.previous_response = None ##
        self.temp = None
        self.required_slots = {"departure", "destination", "date", "type"} ##
        self.current_slots = {"departure": None, "destination": None, "date": None, "type": None, "return date": None, "railcards": None, "adult passengers": 1, "child passengers": None, "earliest inbound": None, "latest inbound": None, "earliest outbound": None, "latest outbound": None}##
        self.current_slots_delay = {"current station": None, "destination": None, "original arrival time": None, "current delay": None}

    def create_new_chat(self, input_message):
        """Method that creates a new chat"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        messages = [input_message]
        current_slots = self.current_slots
        required_slots = list(self.required_slots)
        current_slots_delay = self.current_slots_delay
        current_task = self.current_task
        temp = self.temp
    
        cursor.execute(
            """
            INSERT INTO chats (messages, current_slots, required_slots, current_slots_delay, current_task, temp)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
        (json.dumps(messages), json.dumps(current_slots), json.dumps(required_slots), json.dumps(current_slots_delay), current_task, temp))
        

        #Gets the newly created chat id that was returned by the database
        chat_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
        connection.close()
        return chat_id

    def get_chats(self):
        """Method that retrieves every chats id and title (first message) from the database"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, messages[1] FROM chats
            ORDER BY id DESC;
            """
        )
        chats = cursor.fetchall()
        cursor.close()
        connection.close()
        return chats

    def get_messages(self, chat_id):
        """Method that retrieves the messages for a given chat_id"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT messages FROM chats
            WHERE id = %s;
            """,
            (chat_id,)
        )
        messages = cursor.fetchone()[0]
        cursor.close()
        connection.close()

        #Updates the current and required slots for the new chat
        self.current_slots = self.get_current_slots(chat_id)
        self.required_slots = set(self.get_required_slots(chat_id))
        self.current_slots_delay = self.get_current_slots_delay(chat_id)
        self.current_task = self.get_current_task(chat_id)
        self.temp = self.get_current_temp(chat_id)
        #Sets self.previous_response to the last message in the chat that has the type "chatbot"
        for msg in reversed(messages):
            if isinstance(msg, list) and msg[0] == "chatbot":
                self.previous_response = msg[1]
                break
        #Sets self.current_chat to the chat_id
        self.current_chat = chat_id
        return messages
    
    def get_current_slots(self, chat_id):
        """Method that retrieves the current slots for a given chat_id"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT current_slots FROM chats
            WHERE id = %s;
            """,
            (chat_id,)
        )
        current_slots = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return current_slots
    
    def get_required_slots(self, chat_id):
        """Method that retrieves the required slots for a given chat_id"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT required_slots FROM chats
            WHERE id = %s;
            """,
            (chat_id,)
        )
        required_slots = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return required_slots

    def get_current_temp(self, chat_id):
        """Method that retrieves the temporary variable for a given chat_id"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT temp FROM chats
            WHERE id = %s;
            """,
            (chat_id,)
        )
        temp = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return temp

    def get_current_task(self, chat_id):
        """Method that retrieves the current task for a given chat_id"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT current_task FROM chats
            WHERE id = %s;
            """,
            (chat_id,)
        )
        current_task = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return current_task

    def get_current_slots_delay(self, chat_id):
        """Method that retrieves the current slots for a given chat_id"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT current_slots_delay FROM chats
            WHERE id = %s;
            """,
            (chat_id,)
        )
        current_slots_delay = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        return current_slots_delay

    def update_current_slots(self, chat_id):
        """Method that updates the current slots in the database"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        #Updates the current slots for the current chat
        temp_current_slots = self.current_slots.copy()
        # convert date objects to strings for JSON serialization
        for key, value in temp_current_slots.items():
            if isinstance(value, date):
                temp_current_slots[key] = value.isoformat()
            elif isinstance(value, datetime):
                temp_current_slots[key] = value.isoformat()
        cursor.execute(
            """
            UPDATE chats
            SET current_slots = %s
            WHERE id = %s;
            """,
            (json.dumps(temp_current_slots), chat_id)
        )
        connection.commit()
        cursor.close()
        connection.close()

    def update_required_slots(self, chat_id):
        """Method that updates the required slots in the database"""
        connection = self.get_db_connection()
        cursor = connection.cursor()

        #Updates the required slots for the current chat
        cursor.execute(
            """
            UPDATE chats
            SET required_slots = %s
            WHERE id = %s;
            """,
            (json.dumps(list(self.required_slots)), chat_id)
        )
        connection.commit()
        cursor.close()
        connection.close()

    def update_chat_messages(self,input_message):
        """Method that updates the chat messages in the database"""
        chat_id = self.current_chat
        connection = self.get_db_connection()
        cursor = connection.cursor()

        #Adds the new message to the messages for the current chat
        cursor.execute(
            """
            UPDATE chats
            SET messages = messages || %s::jsonb
            WHERE id = %s;
            """,
            (json.dumps([input_message]), chat_id)
        )
        connection.commit()
        cursor.close()
        connection.close()

        #Checks if either current_slots or required_slots have changed, and updates them if they have
        if self.current_slots != self.get_current_slots(chat_id):
            self.update_current_slots(chat_id)
        if self.required_slots != self.get_required_slots(chat_id):
            self.update_required_slots(chat_id)


    def get_db_connection(self):
        """Method that returns a connection to the database"""
        return psycopg2.connect(
            dbname="traintalk",
            user="postgres",
            password="password",
            host="localhost"
        )



def getDelay(currentStation, destination, originalArrivalTime, currentDelay):
    # Import the global delay model from flask_app
    import flask_app
    
    if flask_app.DELAY_MODEL is None:
        return "I'm sorry, but delay predictions are not available right now. The delay prediction model needs to be trained first."
    
    try:
        from datetime import datetime
        now = datetime.now()
        
        # Make the prediction using the pre-loaded model
        prediction = flask_app.DELAY_MODEL.predict_journey(
            current_station=currentStation,
            destination=destination,
            scheduled_arrival_time=originalArrivalTime,
            current_delay=currentDelay,
            day_of_week=now.weekday(),
            hour_of_day=now.hour,
            month=now.month
        )
        
        if 'error' in prediction:
            return f"I'm sorry, but I couldn't calculate the delay: {prediction['error']}"
        
        # Extract values for cleaner formatting
        current_station = prediction['current_status']['station']
        current_delay_mins = prediction['current_status']['current_delay']
        destination_station = prediction['journey']['to']
        new_eta = prediction['prediction']['expected_arrival_time']
        total_delay = prediction['prediction']['total_delay_at_destination']
        
        # Format the response nicely
        response = f"Based on your current delay of {current_delay_mins} minutes at {current_station}, your new expected arrival is {new_eta}, with a total delay of {total_delay:.0f} minutes when you arrive at {destination_station}."
        
        return response
        
    except Exception as e:
        return f"Error calculating delay: {str(e)}"

if __name__ == "__main__":
    intents_path = os.path.join(os.path.dirname(__file__), "intents.json")
    assistant = ChatbotAssistant(intents_path)
    assistant.parse_intents()
    assistant.load_stations(os.path.join(os.path.dirname(__file__), "../data/stations.csv"))
    assistant.load_railcards(os.path.join(os.path.dirname(__file__), "../data/railcards.txt"))
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=300)
    assistant.save_model(os.path.join(os.path.dirname(__file__), "../model.pth"), os.path.join(os.path.dirname(__file__), "../dimensions.json"))