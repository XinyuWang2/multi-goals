from queue import PriorityQueue
from math import pi
from wall_goalenv_continuous import PointEnv
import numpy as np
import time
import gym

# A* parameter
PI_2 = 2 * pi
CONNECTED = 4 # 4: "4-connected" neighbors; 8: "8-connected" neighbors
DIS_HEURISTIC = "euclidean"  # 'euclidean' or 'manhattan'
STEP = 0.25

# Env setting
# ENV_NAME = "Spiral7x7"
# ENV_NAME = "Spiral5x5"
ENV_NAME = "Cross"
RESIZE = 1


#defines a basic node class
class Node:
    def __init__(self, config_in:tuple, gn_in, fn_in, parent_in):
        # self.x = x_in
        # self.y = y_in
        # self.theta = theta_in
        self.config = config_in # tuple of (x, y, theta)
        self.gn = gn_in
        self.fn = fn_in
        self.parent = parent_in

    def __lt__(self, other):
        return self.fn < other.fn

    def __eq__(self, other):
        return self.fn == other.fn

    def printme(self):
        print("\tNode id", self.id,":", "x =", self.x, "y =",self.y, "theta =", self.theta, "parentid:", self.parentid)



# compute the cost between two config (tuple of x, y, theta)
def dist_cost(config_from, config_to, dist_heuristic):
    """heuristic: 'euclidean' or 'manhattan'."""
    dx = config_from[0] - config_to[0]
    dy = config_from[1] - config_to[1]

    if dist_heuristic == "euclidean":
        dist = np.sqrt(dx**2 + dy**2)
    if dist_heuristic == "manhattan":
        dist = abs(dx) + abs(dy)
    return dist


def heuristic(config_from, config_to):
    """Get the distance heuristic."""
    return dist_cost(config_from, config_to, DIS_HEURISTIC)


def get_neighbors(cur_config):
    """get neighbors based on 4-connected or 8-connected."""
    x = [-STEP, STEP]
    y = [-STEP, STEP]

    neighbor_configs = []
    if CONNECTED == 4:
        for dx in x:
            neighbor_configs.append((cur_config[0] + dx, cur_config[1]))
        for dy in y:
            neighbor_configs.append((cur_config[0], cur_config[1] + dy))

    
    elif CONNECTED == 8:
        for l in [x, y]:
            l.append(0)
        for dx in x:
            for dy in y:
                if dx == 0 and dy == 0: 
                    continue
                neighbor_configs.append((cur_config[0] + dx, cur_config[1] + dy))
    else:
        raise ValueError("Only accept connected_num to be 4 or 8.")

    return neighbor_configs


def goal_reached(cur_config, goal_config):
    # return cur_config == goal_config
    return dist_cost(cur_config, goal_config, "euclidean") < 0.15


def trace_back(cur_node, searched_list):
    while True:
        print(cur_node.config)
        if cur_node.parent != None:
            cur_node = cur_node.parent
        else:
            return


def retrieve_path(node):
    """Retrieve the whole trajectory from the last config."""
    path = []
    n = node
    while n:
        path.insert(0, n.config)
        n = n.parent
    return path

def visualize(env, paths):
    """
    map: 2d np.ndarray
    paths: list of points in continuous space
    """
    from walls_cfg import resize_walls
    import cv2

    resize = 100
    map = resize_walls(env.walls, resize).astype(float)

    for x, y in paths:
        map[int(y * resize), int(x * resize)] = 1

    cv2.imshow("navigation", map)
    cv2.waitKey(0)


def main():
    env = PointEnv(ENV_NAME, resize_factor = RESIZE)
    env.reset() 

    start_config = tuple(env._state)
    goal_config = tuple(env._goal)
    path = []
    open_list = PriorityQueue()

    # hash from searched config(x,y,theta) to its Node obj
    searched_list =  {}

    # set of config(x,y, theta) of all expanded nodes 
    closed_list = set()
    start_node =  Node(start_config, 0, heuristic(start_config, goal_config), None)
    searched_list[start_node.config] = start_node
    open_list.put(start_node)

    search_time = 0
    search_start = time.time()
    while not open_list.empty():
        cur_node = open_list.get()
        if goal_reached(cur_node.config, goal_config):
            print("cost = ", cur_node.gn)
            path = retrieve_path(cur_node)
            break
        neighbor_configs = get_neighbors(cur_node.config)

        for nb_config in neighbor_configs:
            if not env._is_collide(nb_config):
                if nb_config not in closed_list:
                    # try to find it this config has been visited, if not, temp_node is None 
                    temp_node = searched_list.get(nb_config)
                    gn = cur_node.gn + dist_cost(cur_node.config, nb_config, "euclidean")

                    if temp_node and temp_node.gn <= gn:
                        # Visited this config before and got lower cost then
                        continue

                    hn = heuristic(nb_config, goal_config)
                    neighbor_node = Node(nb_config, gn, gn + hn, cur_node)
                    open_list.put(neighbor_node)
                    searched_list[nb_config] = neighbor_node

            #         draw_helper(nb_config, BLUE)
            # else:
            #     draw_helper(nb_config, RED)
        
        closed_list.add(cur_node.config)
    
    search_time = time.time() - search_start
    print("search_time",search_time)

    # Draw the computed path in black 
    visualize(env, path)

    # Execute planned path



if __name__ == '__main__':
    main()
