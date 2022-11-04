import json
import random
import numpy as np

def makeMultiData(data,maxSize):
    multiData = [data,]
    for x in range(maxSize):
        tempData = data
        tempData = tempData.replace(random.choice(tempData),'?')
        multiData.append(tempData)
    return multiData

def readMinData():
    doc = open('data/minTrain.txt').read()
    #print(doc)
    doc = doc.split('\n')
    utterance = []
    response = []
    for k,v in enumerate(doc):
        conversation = json.loads(v)["conversation"]
        for con in conversation:
            utterance.append(makeMultiData(con["utterance"],2))
            response.append([])
            for res in con["response_candidates"]:
                response[len(response)-1].append(res)
    print(utterance)
    print(response)
    return utterance,response

def makeMinTrain():
    utterance,response = readMinData()
    content = {"intents":[]}
    for k in range(len(utterance)):
        content["intents"].append({"patterns":[], "responses":[]})
        for utt in utterance[k]:
            content["intents"][len(content["intents"]) - 1]["patterns"].append(utt)
        for res in response[k]:
            content["intents"][len(content["intents"]) - 1]["responses"].append(res)
    print(content)
    file = open("minTrain.json", 'w')
    file.write(json.dumps(content))
    file.close()

makeMinTrain()
