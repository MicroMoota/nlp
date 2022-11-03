from train import ChatRobot

if __name__ == "__main__":
    chatRobot = ChatRobot()
    chatRobot.loadModel()
    while True:
        print(chatRobot.reply(input()))
