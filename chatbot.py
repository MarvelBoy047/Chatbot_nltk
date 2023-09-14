# Import necessary libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Initialize the Lancaster Stemmer
stemmer = LancasterStemmer()

# Import required libraries
import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle

# Load the intents data from a JSON file
with open("intents.json") as file:
    data = json.load(file)

# Try to load preprocessed data if available, otherwise preprocess and save it
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Process the intents and their patterns
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # Stem and preprocess words
    words = [stemmer.stem(w.lower()) for w in words if w not in ["!", "?"]]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []  # Initialize as a list
    output = []  # Initialize as a list
    out_empty = [0 for _ in range(len(labels))]

    # Create training data and one-hot encode labels
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)  # Append to the list
        output.append(output_row)  # Append to the list

    # Convert lists to numpy arrays
    training = numpy.array(training)
    output = numpy.array(output)

    # Save the preprocessed data
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Reset the TensorFlow graph
tf.compat.v1.reset_default_graph()

# Define the neural network architecture
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Create a Deep Neural Network model
model = tflearn.DNN(net)

# Try to load a pre-trained model, otherwise train and save the model
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Define a function to convert user input to a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

# Define a function for user interaction with the chatbot
def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predict the intent and generate responses
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    print(random.choice(responses))
        else:
            print("I didn't get that, try again.")

# Start the chatbot interaction
chat()
