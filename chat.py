import random
import json
import speech_recognition as sr
import pyttsx3

import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the voice of the text-to-speech engine
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # Change the index to select a different voice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # Use the microphone as the audio source
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        # Listen for audio input
        audio = r.listen(source)
        try:
            # Recognize speech using Google Speech Recognition
            sentence = r.recognize_google(audio)
            print(f"You: {sentence}")
            if sentence == "quit":
                break

            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            probs = torch.softmax(output, dim=1)
            prob, predicted = torch.max(probs, dim=1)

            tag = tags[predicted.item()]
            max_prob_intent = None
            max_prob_intent_pattern_index = None
            max_prob = 0
            for intent in intents['intents']:
                if tag == intent["tag"] and prob.item() > max_prob:
                    max_prob = prob.item()
                    max_prob_intent = intent
                    max_prob_intent_pattern_index = tags.index(max_prob_intent["patterns"][predicted.item()])

            if max_prob_intent:
                response = max_prob_intent['responses'][max_prob_intent_pattern_index]
                print(f"{bot_name}: {response}")
                # Speak the bot's response
                engine.say(response)
                engine.runAndWait()
            else:
                response = "I do not understand..."
                print(f"{bot_name}: {response}")
                # Speak the bot's response
                engine.say(response)
                engine.runAndWait()
        except sr.UnknownValueError:
            print("Sorry, I could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
