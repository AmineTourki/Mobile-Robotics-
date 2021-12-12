import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors


def _get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]


def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately
                     preceding it on the cheapest path from start to n
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list
        total_path.insert(0, cameFrom[current])
        current = cameFrom[current]
    return total_path


def A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N", ):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    :param h: h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param coords: List of all coordinates in the grid
    :param occupancy_grid: the grid map
    :param movement_type: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return path: The array containing the optimal path from start to goal
    :return vistedNodes: The array of explored nodes by the A* algorithm
    """

    # -----------------------------------------
    # DO NOT EDIT THIS PORTION OF CODE
    # -----------------------------------------

    # Check if the start and goal are within the boundaries of the map
    assert start[0] in range(occupancy_grid.shape[1]) and start[1] in range(
        occupancy_grid.shape[0]), "Start location not contained in the map"
    assert goal[0] in range(occupancy_grid.shape[1]) and goal[1] in range(
        occupancy_grid.shape[0]), "Goal location not contained in the map"

    # check if start and goal nodes correspond to free spaces
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')

    # get the possible movements corresponding to the selected connectivity
    if movement_type == '4N':
        movements = _get_movements_4n()
    elif movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')

    # --------------------------------------------------------------------------------------------
    # A* Algorithm implementation - feel free to change the structure / use another pseudo-code
    # --------------------------------------------------------------------------------------------

    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]

    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:

        # the node in openSet having the lowest fScore[] value
        fScore_openSet = {key: val for (key, val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet

        # If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)

        # for each neighbor of current:
        for dx, dy, deltacost in movements:

            neighbor = (current[0] + dx, current[1] + dy)

            # if the node is not in the map, skip
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (
                    neighbor[0] < 0) or (neighbor[1] < 0):
                continue

            # if the node is occupied or has already been visited, skip
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet):
                continue

            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost

            if neighbor not in openSet:
                openSet.append(neighbor)

            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet


def run_A_Star(occupancy_grid, start, goal):
    """
    Run the A* algorith with this function
    :param occupancy_grid: array grid of the map. (1=obstacle, 0=empty)
    :param start: the start node (x,y)
    :param goal: the goal node (x,y)
    :return path: The array containing the optimal path from start to goal
    :return vistedNodes: The array of explored nodes by the A* algorithm
    """
    start = [int(math.floor(start[0])), int(math.floor(start[1]))]
    # List of all coordinates in the grid
    x, y = np.mgrid[0:occupancy_grid.shape[1]:1, 0:occupancy_grid.shape[0]:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0] * x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic, here = distance to goal ignoring obstacles
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Run the A* algorithm with movement 8N
    path, visitedNodes = A_Star(tuple(start), tuple(goal), h, coords, occupancy_grid, movement_type="8N")
    path = np.array(path)
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
    plot_path(occupancy_grid, visitedNodes, start, goal, path)

    return path, visitedNodes


def plot_path(occupancy_grid,visited_nodes,start,goal,path):
    '''
     save the path returned by the A* for later display
    :param occupancy_grid: array grid of the map. (1=obstacle, 0=empty)
    :param start: the start node (x,y)
    :param goal: the goal node (x,y)
    :return path: The array containing the optimal path from start to goal
    :param visted_nodes: The array of explored nodes by the A* algorithm
    '''


    fig, ax = plt.subplots(figsize=(8, 8))

    major_ticks = np.arange(0, 51, 5)
    minor_ticks = np.arange(0, 51, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([50, -1])
    ax.set_xlim([-1, 50])
    ax.grid(True)

    path_plot = np.array(path).reshape(-1, 2).transpose()

    plt.imshow(occupancy_grid.transpose(), cmap=colors.ListedColormap(['white', 'black']));
    plt.scatter(visited_nodes[0], visited_nodes[1], marker="o", color='orange', s=5)
    plt.plot(path_plot[0], path_plot[1], color='blue');
    plt.scatter(goal[0], goal[1], marker="X", color='red', s=200);
    plt.scatter(start[0], start[1], marker="o", color='green', s=200)
    # plt.title("A* path planning with a green\n checkpoint and a red goal");
    plt.savefig('A_star_plot.png')

