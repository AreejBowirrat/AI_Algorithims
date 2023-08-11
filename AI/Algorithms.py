#submitters: amir masarweh 211863659 , areej bowirat 324122845

import numpy as np
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict



class Node:
    def __init__(self, parent, state, action, cost, terminated):
        self.parent = parent
        self.state = state
        self.action = action
        self.cost = cost
        self.terminated = terminated

class BFSAgent():
    def __init__(self):
         self.total_expanded = 0

    def if_open(self, state, open_list):
        for node in open_list:
            if node.state == state:
                return True
        return False

    def if_close(self, state, close_list):
        for node in close_list:
            if node.state == state:
                return True
        return False

    def backtrackted_path(self, node):
        action_path = []
        reversed_action_path = []

        total_cost = 0
        while (node.parent != None):
            total_cost += node.cost
            action_path.append(node.action)
            node = node.parent

        for item in action_path[::-1]:
            reversed_action_path.append(item)

        return (reversed_action_path, float(total_cost), self.total_expanded)

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        initial_state = env.get_initial_state()
        node = Node(None, initial_state, None,0 , False)
        self.total_expanded = 0
        open_list = []
        close_list = set()
        if env.is_final_state(node.state):
            return self.backtrackted_path(node)
        open_list.append(node)
        while open_list:
            node = open_list.pop(0)
            close_list.add(node)
            self.total_expanded += 1
            if node.terminated:
                continue
            dictionary = env.succ(node.state)
            for key, value in dictionary.items():
                child = Node(node, value[0],key, value[1],value[2])
                if not self.if_close(child.state, close_list) and not self.if_open(child.state, open_list):
                    if env.is_final_state(child.state):
                        return self.backtrackted_path(child)
                    open_list.append(child)

        return (None,None,None)

class DFSAgent(BFSAgent):
    def __init__(self) -> None:
        BFSAgent.__init__(self)

    def RecursiveDFS(self, env, open_list, closed_list):

        node = open_list.pop()
        closed_list.add(node)

        if env.is_final_state(node.state):
            return self.backtrackted_path(node)
        self.total_expanded += 1

        if (node.terminated):
            return ([], None, None)

        dictionary = env.succ(node.state)
        for key, value in dictionary.items():
            child = Node(node, value[0], key, value[1], value[2])

            if (self.if_close(child.state,closed_list) == False ) and (self.if_open(child.state, open_list) == False):
                open_list.append(child)
                result = self.RecursiveDFS(env, open_list, closed_list)
            else:
                continue
            if (result != ([], None, None)):
                return result
        return ([], None, None)


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        initial_state = env.get_initial_state()
        node = Node(None, initial_state, None,0 , False)
        open_list = []
        closed_list = set()
        self.total_expanded = 0
        open_list.append(node)
        return self.RecursiveDFS(env, open_list, closed_list)

class NodeUCS(Node):
    def __init__(self, parent, state, action, cost, terminated, g):
        Node.__init__(self, parent, state, action, cost, terminated)
        self.g = g


class UCSAgent(BFSAgent):
    def __init__(self) -> None:
        self.total_expanded = 0

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:

        initial_state = env.get_initial_state()
        node = NodeUCS(None, initial_state, None, 0, False, 0)
        self.total_expanded = 0
        open_heapdict = heapdict.heapdict()
        open_heapdict[node] = (node.g, node.state)
        close_list = set()

        while len(open_heapdict) != 0:
            (node, val_tuple) = open_heapdict.popitem()
            close_list.add(node.state)

            if env.is_final_state(node.state):
                return self.backtrackted_path(node)

            self.total_expanded += 1

            if node.terminated:
                continue

            dictionary = env.succ(node.state)
            for key, value in dictionary.items():
                child = NodeUCS(node, value[0], key, value[1], value[2], node.g + value[1])

                if child.state not in close_list and not self.if_open(child.state, open_heapdict):
                    open_heapdict[child] = (child.g, child.state)

                elif self.if_open(child.state, open_heapdict):
                    for node_key in open_heapdict.keys():
                        if node_key.g > child.g and node_key.state == child.state:
                            del open_heapdict[key]
                            open_heapdict[child] = (child.g, child.state)

        return (None,None,None)


class G_Node(Node):
    def __init__(self, parent, state, action, cost, terminated, h):
        Node.__init__(self, parent, state, action, cost, terminated)
        self.heuristic = h

class GreedyAgent(BFSAgent):
    def __init__(self) -> None:
        BFSAgent.__init__(self)

    def get_h_value(self, env, state):
        x , y = env.to_row_col(state)
        board_rows, board_cols = env.nrow, env.ncol
        h_manhatten = (board_rows - 1 - x) + (board_cols - 1 - y)
        return min(h_manhatten,100)

    def swap(self, state, child, open_list):
        for key in open_list.keys():
            if key.state == state:
                if (key.heuristic > child.heuristic):
                    del open_list[key]
                    open_list[child] = (child.heuristic, child.state)

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.total_expanded = 0
        h_val = self.get_h_value(env, env.get_initial_state())
        initial_state = env.get_initial_state()
        node = G_Node(None, initial_state, None, 0, False, h_val)

        open_heapdict = heapdict.heapdict()
        open_heapdict[node] = (node.heuristic, node.state)
        closed_list = set()

        while open_heapdict:
            min_node = open_heapdict.popitem()
            node = min_node[0]
            closed_list.add(node)
            if env.is_final_state(node.state):
                return self.backtrackted_path(node)
            self.total_expanded += 1
            if node.terminated:
                continue

            successors = env.succ(node.state)
            for key, value in successors.items():
                h_val = self.get_h_value(env, value[0])
                child = G_Node(node, value[0], key, value[1], value[2],h_val)

                if (self.if_close(child.state,closed_list) == False ) and (self.if_open(child.state, open_heapdict) == False):
                    open_heapdict[child] = (child.heuristic, child.state)
                elif (self.if_open(child.state, open_heapdict) == True):
                    self.swap(child.state, child, open_heapdict)

        return (None, None, None)


class WASNode(Node):
    def __init__(self, state, cost, action, parent, terminated, h, g, f):
        Node.__init__(self, parent, state, action, cost, terminated)
        self.h = h
        self.g = g
        self.f = f


class WeightedAStarAgent():

    def __init__(self):
        self.total_expanded = 0

    def check_in_open(self,state, open_heapdict):
        for node in open_heapdict.keys():
            if node.state == state:
                return True
        return False

    def if_close(self, state, close_list):
        for node in close_list:
            if node.state == state:
                return True
        return False

    def backtrackted_path(self, node):
        action_path = []
        reversed_action_path = []

        total_cost = 0.0
        while node.parent is not None:
            total_cost += node.cost
            action_path.append(node.action)
            node = node.parent

        for item in action_path[::-1]:
            reversed_action_path.append(item)

        return (reversed_action_path, float(total_cost), self.total_expanded)

    def calculate_f_value(self, h, g, h_weight):
         return (((1 - h_weight) * g) + (h_weight * h))


    def MinManhatenFromAllGoals(self, env, state):
        min_manhaten = float('inf')
        agent_row, agent_col = env.to_row_col(state)

        goals_list = env.get_goal_states()
        for goal in goals_list:
            goal_row, goal_col = env.to_row_col(goal)
            menhaten = (goal_col - agent_col) + (goal_row - agent_row)
            min_manhaten = min(menhaten, min_manhaten)
        return min_manhaten

    def calculate_MSAP(self, env, state):
        manhaten = self.MinManhatenFromAllGoals(env, state)
        return manhaten if manhaten < 100 else 100

    def state_to_node_close_list(self, state, close_list):
        for node in close_list:
            if node.state == state:
                return node
        return None

    def state_to_node_open_heap(self, state, open_heapdict):
        for node in open_heapdict.keys():
            if node.state == state:
                return node
        return None

    def search(self, env: FrozenLakeEnv, h_weight: float) -> Tuple[List[int], float, int]:

        h = self.calculate_MSAP(env, env.get_initial_state())
        f = self.calculate_f_value(h, 0, h_weight)
        node = WASNode(env.get_initial_state(), 0, None, None, False, h, 0, f)
        self.total_expanded = 0

        close_list = set()
        open_heapdict = heapdict.heapdict()
        open_heapdict[node] = (node.f, node.state)

        while open_heapdict:
            node = open_heapdict.popitem()[0]
            close_list.add(node)

            if env.is_final_state(node.state):
                return self.backtrackted_path(node)

            self.total_expanded += 1

            if node.terminated:
                continue

            dictionary = env.succ(node.state)
            for action, (state, cost, terminated) in dictionary.items():
                h = self.calculate_MSAP(env, state)
                f = self.calculate_f_value(h, node.g + cost, h_weight)
                child = WASNode(state, cost, action, node, terminated, h, node.g + cost, f)

                if not self.if_close(state, close_list) and not self.check_in_open(state, open_heapdict):
                    open_heapdict[child] = (child.f, child.state)

                elif self.if_close(state, close_list):
                    current_node = self.state_to_node_close_list(state, close_list)
                    if f < current_node.f:
                        close_list.remove(current_node)
                        open_heapdict[child] = (child.f, child.state)

                elif self.check_in_open(state, open_heapdict):
                    current_node = self.state_to_node_open_heap(state, open_heapdict)
                    if f < current_node.f:
                        del open_heapdict[current_node]
                        open_heapdict[child] = (child.f, child.state)

        return (None, None, None)




class IDAStarNode:
    def __init__(self,parent, state, action, cost, terminated, g, h, f):
        self.parent = parent
        self.state = state
        self.action = action
        self.cost = cost
        self.terminated = terminated
        self.g = g
        self.h = h
        self.f = f


class IDAStarAgent:
    def __init__(self) -> None:
        self.total_expanded = 0

    def MinManhatenFromAllGoals(self, env, state):
        min_manhaten = float('inf')
        agent_row, agent_col = env.to_row_col(state)

        goals_list = env.get_goal_states()
        for goal in goals_list:
            goal_row, goal_col = env.to_row_col(goal)
            menhaten = (goal_col - agent_col) + (goal_row - agent_row)
            min_manhaten = min(menhaten, min_manhaten)
        return min_manhaten

    def MSAPValue(self, env, state):
        manhaten = self.MinManhatenFromAllGoals(env, state)
        return manhaten if manhaten < 100 else 100

    def GetSolution(self, path):
        action_path = []
        total_cost = 0
        path.pop(0)
        for node in path:
            total_cost += node.cost
            action_path.append(node.action)
        return (action_path, total_cost, self.total_expanded)

    def IfInPath(self, env, path,state):
        for node in path:
            if node.state == state:
                return True
        return False

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        manhaten = self.MinManhatenFromAllGoals(env, env.get_initial_state())
        manhaten = manhaten if manhaten < 100 else 100
        new_limit = [manhaten]
        node_h = manhaten
        node = IDAStarNode(None, env.get_initial_state(), None, 0, False, 0, node_h, node_h)

        while 1:
            path = [node]
            f_limit = new_limit[0]
            new_limit = [float('inf')]
            result = self.DFS_f(node, 0, path, f_limit, env, new_limit)
            new_limit = result[1]
            if result[0] != ([], None, None):
                return result

        return ([], None, None)

    def DFS_f(self, node, g, path, f_limit, env, new_limit) -> Tuple[Tuple[List[int], int, float], List[int]]:
        state = node.state
        new_f = g + node.h

        if new_f > f_limit:
            new_limit = [min(new_limit[0], new_f)]
            return (([], None, None), new_limit)
        if env.is_final_state(state):
            return self.GetSolution(path)

        returned_dict = env.succ(state)
        self.total_expanded += 1


        for action, value_tuple in returned_dict.items():

            if(value_tuple[0] == node.state):
                continue
            if (value_tuple[2] == True) and  (not env.is_final_state(value_tuple[0])):
                continue
            if self.IfInPath(env, path,value_tuple[0]):
                continue
            child_h_value = self.MSAPValue(env, value_tuple[0])
            child_g_value = g+value_tuple[1]
            child_f_value = child_h_value + child_g_value
            child = IDAStarNode(node, value_tuple[0], action, value_tuple[1], value_tuple[2], child_g_value,
                                child_h_value, child_f_value)
            path.append(child)
            result= self.DFS_f(child, child.g,path , f_limit, env, new_limit)
            new_limit = result[1]
            if result[0] != ([], None, None):
                return result
            path.pop()

        return (([], None, None), new_limit)


