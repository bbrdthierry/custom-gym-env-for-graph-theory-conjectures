import sys
from gym import Env
from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


class GraphEnv(Env):
    """This environment consists of generating graphs and evaluate their construction by the help of a reward function.
    It can be used to find counter examples of conjectures, or to represent the existing ones.

    A "state" is a description of the actual environment. When we want to interact with the latter, we can make a "step".
    A step is an action that has an impact to the environment. So, a new state is returned when a step is made.
    In this environment, the state describes all the connections between nodes (binary vector). When the environment is 
    firstly initialized, the nodes are not connected : the state is only a vector filled by zeros.

    The environment should be closed after utilization, using "env.close()" function. 
    When training a reinforcement learning algorithm, after each episodes, the environment should be reset.
    """
    def __init__(self, N_vertices=10):
        """The environment initializes the number of vertices, a state which is a binary vector that
        indicates the connection between nodes (but initialized with a zeros vector), and the index position
        within the latter. 
        
        Args:
            N_vertices (int, optional): Number of graph vertices. Defaults to 10.
        """
        self.N = N_vertices
        self.decisions = int(N_vertices * (N_vertices - 1) / 2)
        self.state = np.zeros(self.decisions, dtype=np.uint8)
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(1, self.decisions), dtype=np.uint8)
        self.INF = 1000000
        self.position = 0

    def graph_construction(self):
        """This function will generate a graph of N vertices. 

        The nodes are connected using a binary decision vector of length "N * (N - 1) / 2", which is similar to an adjacency matrix.
        The binary decision vector is a state of the environment.
        For example, if two nodes are connected, we put 1 in the decision vector, else 0.

        Returns:
            (G) networkx.classes.graph.Graph: A networkx graph.
        """
        # Generates the networkx graph instance.
        G = nx.Graph()
        # Adds nodes of length N vertices.
        G.add_nodes_from(list(range(self.N)))
        count = 0
        # Connections of all nodes, by using the binary decision vector (i.e the state of the environment).
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.state[count] == 1:
                    G.add_edge(i, j)
                count += 1
        return G

    def custom_reward(self, G):
        """This function assesses the networkx graph construction by giving a reward.

        The score returned by the evaluation depends on the conjecture specification.
        By default, we would like the function to find a graph that contains more nodes than edges 
        (i.e the graph contains at least one connected component which is tree-like).

        Feel free to change the custom reward function/the conjecture specification.

        Args:
            G (networkx.classes.graph.Graph): The networkx graph construction.

        Returns:
            (reward) int: The score of the graph construction.
            (done) bool: The done variable indicates if the episode ended (i.e. if all decisions in the binary decision vector are taken or if a conjecture is found).
        """
        # Done is equal to False while the conjecture is not found.
        done = False

        # The assess of the construction is made by calculating the difference between the number of nodes and the number of edges.
        # You can change the way to assess a construction/to find a new conjecture below.
        reward = G.number_of_nodes() - G.number_of_edges()
        # While the graph is not connected, reward the very negative, in order to penalize the reinforcement learning model.
        if not nx.is_connected(G):
            reward = - self.INF
        # If the reward is positive, done is True and the found construction is showed.
        if reward > 0:
            print(self.state)
            done = True
            nx.draw_kamada_kawai(G)
            plt.show()
            sys.exit()
        return reward, done

    def step(self, action):
        """The step function allows to make a decision in the binary decision vector (i.e if we decide to connect a pair of nodes or not).

        Once we made a decision in the input state, the index position inside it is incremented. A new state and a reward are given in return.

        Args:
            action (int): Action must be binary (equals to 0 or 1).

        Raises:
            ValueError: None

        Returns:
            (state) numpy.ndarray: A numpy vector which contains element that are binary.
            (reward) int: The score of the graph construction.
            (done) bool: The done variable indicates if the episode ended.
            (info) dict: A dictionnary that contains complementary info about the state (not necessary for modeling).
        """
        if action < 0 or action >= 2:
            raise ValueError("La valeur de l'action doit être égale à 0 ou 1.")

        if self.position <= len(self.state):
            try:
                # Decision to put 0 or 1 in the binary vector.
                self.state[self.position] = action
                # Position starts from 0, then the index is incremented to make the next decision in the binary vector.
                self.position += 1
                # Generation of the graph construction.
                G = self.graph_construction()
                # Assesses the construction.
                reward, done = self.custom_reward(G)
            except:
                # Only trace a graph and gives the reward, without incrementing the decision vector index.
                G = self.graph_construction()
                reward, done = self.custom_reward(G)
                # Done is equals to True, since we already assessed the whole decision vector.
                done = True
        else:
            done = True
        # Informations about the index position in the binary decision vector.
        info = {"position": self.position}
        return self.state, reward, done, info

    def render(self):
        """None
        """
        pass

    def reset(self):
        """Reset the environment state, by returning a zeros vector with a length of N vertices.

        The environment should be reset after the end of each episode (i.e when done is equal to True).

        Returns:
            (state) numpy.ndarray: A numpy zeros vector.
        """
        # Generates a new state, which is a zeros numpy vector.
        self.state = np.zeros(self.decisions, dtype=np.uint8)
        # The index position is back to 0.
        self.position = 0
        return self.state
