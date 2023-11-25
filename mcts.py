import numpy as np
from copy import deepcopy
import gymnasium as gym
# from mctsNode import Node
from collections import defaultdict,namedtuple



class Node:
    """
    This is a node of the MCTS search tree. 
    Each node will capture a particular sate. The edges are actions. 
    """
    def __init__(self,parent,state,action) -> None:
        
        if len(state) == 2: # Is this the stating state of the environment env.reset() only returns obs and info ... 
            self.state = defaultdict(bool,{k:v for k,v in zip(["obs","info"],state)})  #crea
        else: self.state = defaultdict(dict,{k:v for k,v in zip(["obs","reward","terminated","truncated","info"],state)})

        self.action = action #what was the action taken to get here? The "incoming action"
        self.parent = parent #what is the parent node 
        #self.children = {} #children coresponding to the action taken ... #BUG: list ?
        self.children = []
        self.N = 0 #N this the number of time this node has been visited
        self.Q = 0 # What is the curretn value of this node ... initialized to zero. 
        
    def is_leaf(self) -> bool:
        """
        Is the node terminal?
        """
        return self.children == []
    
    def is_fully_expanded(self,env) -> bool:
        """
        Is the node fuly expanded

        """
        return len(self.children) == env.action_space.n
    
    def is_terminal(self):
        """
        Check if the current state of OpenAI gym is a terminal node ... 

        Returns: bool or empty dict is starting state ... 
        """
        return self.state["terminated"]

class ChanceNode(Node):
    """
    Here I am going to add 
    
    """
    def __init__():
        super().__init__()
    # TODO: Figure out how to implement this... 


class MCTS:

    """
    This is my own implementation of MCTS. 

    The algoirhtm procedes as follows

    1. Given some root node we need to find the best action.
    2. We do this by statring at the root node, 
    3. From the root node we select some candidate state (by taking an action)
    4. Fromt the candidate state we simulate.
        4.1 moves are made during simluation according to some default policy (in the simplist case a unifrom distribution of moves. )
    5. The search tree is then reccursivly updated using backprop. 
    6. The selected node is then added to the search tree.

    There are four steps applied per serach iteration

    1. Selection: starting from the root node a child selection policiy is recursively applied to descend through the tree until the most "urgent" expandable node is reached. 
        1.1: A node is expandable if it is non-terminal and un-expanded children. 
    2. Expansion: One or more child nodes are added to expand the tree (according to available actions)
    3. Simulation: A simulation is run form the new nodes according to default polict to produce an outcome
    4. Backpropagation: The simlulation result it backpropagets through the selectd nodse to update their statistics


    Selection and expansion are combined into the "treepolicy method"
    The rollout/simluation is the "default" policy. 


    NOTE: This implemention will only support discrete action spaces. 
    """
    def __init__(self,env:gym.Env,state,d,m,c,gamma) -> None:
        """
        INPUTS:

        gym.Env env : openAi gym environment
        tuple state : state of environment (observation, reward,terminated,info)
        int m : number of iterations to run mcts
        float c : UTC search parameter. 
        float gamma : discount factor
        """
        self.env = env # This is the current state of the mdp
        self.d = d # depth #TODO icorportae this into simulation depth
        self.m = m # number of simulations
        self.c = c # exploration constant
        #self.U = U # value funciton estimate
        self.v0 = Node(parent=None,state=state,action=None)  #root node of search tree. 
        self.env = env
        self.gamma = gamma        

    def search(self):
        """
        Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action that maximizes the estimate of Q(s,a)
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deep copy of of the og env at the root nod 
            vl = self._tree_policy(self.v0) #vl is the last node visitied by the tree search
            R = self._default_policy(vl)
            self._backpropagation(R,vl)
            # if k == self.m//2:
            #     print(vl.Q)

        # for k,child in self.v0.children.items():
        #     print(f"Q:{child.Q}")
        #     print(f"N : {child.N}")

        action_values = [(child.Q/child.N) for child in self.v0.children]

        # action_values = {k:(child.Q/child.N) for k,child in self.v0.children.items()} #BUG? could be unpackign weird
        # print(action_values)
        # best_action = max(action_values,key=action_values.get)
        best_action = self.v0.children[np.argmax(action_values)].action
        return best_action



    def _tree_policy(self,v:Node):
        while not v.is_terminal(): #TODO: Figure out how the root node would behave here. 
            if not v.is_fully_expanded(self.sim_env):
                return self._expand(v)
            else:
                v = self._selection(v) # Traverse the search tree using UTC to select the next node. 
        return v 


    def _default_policy(self,v:Node):
        """
        Simulate/Playout step 

        While state is non-terminal choose  an action uniformly at random, transition to new state.
        Return the reward for final  state. 

        """
        # sim_env = deepcopy(v.env)
        if v.is_terminal():
            return v.state['reward']
        tot_reward = 0
        # terminated = v.is_terminal()
        terminated = False
        #TODO: Include discount factor here 
        depth = 0
        while not terminated and depth < self.d:
            action = self.sim_env.action_space.sample() #randomly sample from environments action space
            observation,reward,terminated,truncated,info = self.sim_env.step(action)
            tot_reward += reward
            depth+=1
        #print(f"T reward:{tot_reward}")

        tot_reward = tot_reward*self.gamma**depth #
        return tot_reward

    def _selection(self,v:Node):
        """
        Pick the next node to go down in the search tree based on UTC
        """
        # child_nodes = list(v.children.values()) #dictionary k = action, v = node
        child_nodes = v.children

        # best_child_ind = np.argmax([child.Q/child.N + self.c * np.sqrt(2*np.log(v.N)/(child.N)) for child in child_nodes])        
        best_child_ind = np.argmax([child.Q/child.N + self.c * np.sqrt(2*np.log(v.N)/(child.N)) for child in child_nodes])        
        best_child = child_nodes[best_child_ind]
        # best_child = child_nodes[best_child_ind] 
        # best_chile = np.argmax(bes)
        best_action = best_child.action
        self.sim_env.step(best_action)
        return best_child
    
    def _expand(self,v:Node):
        """
        The idea hear is to choose an untired action in state s then 

        """
        # Select an action that we have not taken yet. 
        all_actions = {i for i in range(self.sim_env.action_space.n)} #grab all actions in action space
        old_actions = set(child.action for child in v.children)#look at all the actions that we have already taken. 
        new_actions = list(all_actions.difference(old_actions)) #get a list of action we have not taken
        action = new_actions[np.random.randint(len(new_actions))] #sample a random untaken action
        new_state = self.sim_env.step(action) #take random action and grab state


        v_prime = Node(parent=v,state=new_state,action=action) #save new state and action taken in search tree node. 
        v.children.append(v_prime)

        return v_prime


    def _backpropagation(self,R,v:Node):
        """
        Backtrack to update the number of times a node has beenm visited and the value of a node untill we reach the root node. 
        """
        #TODO: discount factor here

        while v:
            v.N+= 1
            v.Q = v.Q + R
            R = R*self.gamma
            v = v.parent






if __name__ == "__main__":
  pass




