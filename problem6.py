from mcts import MCTS
from contextlib import closing
import gymnasium as gym
from gymnasium import utils
from io import StringIO
import math


gym.envs.register(
     id='FrozenLakeNoRender-v0',
     entry_point='frozen_lake:FrozenLakeEnv',
     max_episode_steps=1000,
)

map_name = "4x4"
env = gym.make('FrozenLakeNoRender-v0',render_mode=None,is_slippery=False,map_name=map_name)
state = env.reset()
print(state)

maps = {"4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ],
        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]}

def simple_render(map_name,s):
    m = maps[map_name]
    outfile = StringIO()
    row,col = s//len(m), s % len(m)
    desc = [[c for c in line] for line in m]
    desc[row][col] = utils.colorize(desc[row][col], "cyan", highlight=True)
    outfile.write("\n")
    outfile.write("\n".join("".join(line) for line in desc) + "\n")

    with closing(outfile):
        return outfile.getvalue()


print(simple_render(map_name,state[0]))

# c = 1/math.sqrt(2)
c = 1.44
for _ in range(1000):
    mcts = MCTS(env,state=state,d=50,m=1000,c=c,gamma=0.9)
    action = mcts.search()
    state = env.step(action)
    print(state)
    print(simple_render(map_name,state[0]))
    done = state[2]
    if done:
        print("Done")
        break