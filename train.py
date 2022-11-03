import json
import nltk
import numpy as np
import pandas as pd
import random
import os.path
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json
from nltk.stem.lancaster import LancasterStemmer


# 聊天机器人
class ChatRobot:
    stemmer = LancasterStemmer()
    intents = []
    documents = []
    tags = []
    words = []
    model = []
    ignoreWords = []
    errorThreshold = 0.25
    debug = False

    def __init__(self, debug=False):
        self.debug = debug

    def preHandleData(self):
        if not os.path.exists("train.json"):
            raise BaseException("不存在训练数据")
        else:
            with open('train.json', "r") as file:
                self.intents = json.load(file)

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.tags:
                    self.tags.append(intent['tag'])

        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in self.ignoreWords]
        self.words = sorted(list(set(self.words)))
        self.tags = sorted(list(set(self.tags)))

        if self.debug:
            print(len(self.tags), "语境", self.tags)
            print(len(self.words), "词数", self.words)
            print(len(self.documents), "文档", self.documents)

    def cleanUpSentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words):
        sentenceWords = self.cleanUpSentence(sentence)
        bag = [0] * len(words)
        for s in sentenceWords:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def trainModel(self):
        training = []
        outputEmpty = [0] * len(self.tags)
        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]

            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            outputRow = list(outputEmpty)
            outputRow[self.tags.index(doc[1])] = 1

            training.append([bag, outputRow])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    def saveModel(self):
        json_file = self.model.to_json()
        with open('weight.json', "w") as file:
            file.write(json_file)
            self.model.save_weights('./weight.h5f')

    def loadModel(self):
        self.preHandleData()
        if not os.path.exists("./weight.json"):
            self.trainModel()
            self.saveModel()
        else:
            with open('./weight.json', "r") as file:
                self.model = model_from_json(file.read())
                self.model.load_weights("./weight.h5f")

    def reply(self, sentence):
        inputData = pd.DataFrame([self.bow(sentence, self.words)], dtype=float, index=['input'])
        predicts = self.model.predict([inputData], verbose=0)[0]
        predicts = [[k, v] for k, v in enumerate(predicts) if v > self.errorThreshold]
        predicts.sort(key=lambda x: x[1], reverse=True)
        results = []
        for r in predicts:
            results.append((self.tags[r[0]], str(r[1])))
        result = ""
        for intent in self.intents['intents']:
            if intent['tag'] == results[0][0]:
                result = random.choice(intent['responses'])
                break
        return result


if __name__ == '__main__':
    chatRobot = ChatRobot(debug=True)
    chatRobot.preHandleData()
    chatRobot.trainModel()
    chatRobot.saveModel()
