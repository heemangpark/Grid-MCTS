from agent import *

if __name__ == "__main__":
    ag = Agent()
    ag.play(rounds=1000)
    print(ag.show_values())
