import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Grid & Q-table Configuration
n = 6
start = (0, 0)
goal = (5, 5)
barriers = [(1, 1), (1, 2), (2, 2), (3, 3), (4, 1), (4, 2)]
Q = np.zeros((n, n, 4))
alpha, gamma, eps, episodes = 0.1, 0.9, 0.2, 1200

def step(s, a):
    x, y = s
    # Standard Grid Actions: 0=Up, 1=Down, 2=Left, 3=Right
    if a == 0 and x > 0: x -= 1
    if a == 1 and x < n - 1: x += 1
    if a == 2 and y > 0: y -= 1
    if a == 3 and y < n - 1: y += 1
    
    ns = (x, y)
    
    # Check for barriers or goal
    if ns in barriers: 
        return s, -50
    if ns == goal: 
        return ns, 100
    return ns, -1

# Training the Q-Learning Agent
for _ in range(episodes):
    s = start
    while s != goal:
        x, y = s
        
        # Epsilon-greedy action selection
        if random.random() < eps:
            a = random.randint(0, 3)
        else:
            a = np.argmax(Q[x, y])
            
        ns, r = step(s, a)
        nx, ny = ns
        
        # Bellman Equation Update
        Q[x, y, a] += alpha * (r + gamma * np.max(Q[nx, ny]) - Q[x, y, a])
        s = ns

# Path extraction based on learned Q-table
s = start
path = [start]
while s != goal:
    s, _ = step(s, np.argmax(Q[s[0], s[1]]))
    
    # Prevent infinite loops if policy is suboptimal
    if s in path: 
        break 
    path.append(s)

# Visualization Setup
grid = np.zeros((n, n))
for b in barriers: grid[b] = 1
for p in path: grid[p] = 2
grid[start], grid[goal] = 3, 4

cmap = ListedColormap(['white', 'black', 'yellow', 'green', 'red'])
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap=cmap)
plt.grid(True)
plt.title("Q-Learning Agent Navigation", fontsize=15)

# Adding Text Labels to the Grid
for i in range(n):
    for j in range(n):
        if (i, j) == start: 
            plt.text(j, i, 'S', ha='center', va='center', fontsize=13, weight='bold')
        elif (i, j) == goal: 
            plt.text(j, i, 'G', ha='center', va='center', fontsize=13, weight='bold')
        elif (i, j) in barriers: 
            plt.text(j, i, 'X', ha='center', va='center', fontsize=13, color='white')
        elif (i, j) in path: 
            plt.text(j, i, '*', ha='center', va='center', fontsize=13)

# Configuring the Legend
legend = [
    Patch(fc='green', label='Start'),
    Patch(fc='red', label='Goal'),
    Patch(fc='black', label='Barrier'),
    Patch(fc='yellow', label='Path')
]
plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.05, 1))

# Show Plot
plt.tight_layout()
plt.show()
