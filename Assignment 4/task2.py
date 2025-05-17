"""
=================================================================================
    DSCI-553: Assignment 4 (Spring 2025)
    Task 2: Community Detection Using the Girvan-Newman Algorithm
---------------------------------------------------------------------------------
    OVERVIEW of the steps:
      1) Read input CSV using Spark RDD.
      2) Build undirected graph G: user -> set of neighbors.
         - Edges formed if two users have >= threshold overlap of businesses.
         - We do NOT include isolated users (no edges).
      3) Compute betweenness for every edge in G using the BFS-based approach.
      4) Output the betweenness in the specified format & order:
         ( 'u1', 'u2' ), betweenness_value
         * Sort by betweenness DESC, tie-break by (u1 < u2) lexicographically.
      5) Iteratively remove edges with the highest betweenness, 
         re-compute betweenness, track communities, and compute modularity.
         - Stop once all edges are removed (or as soon as you prefer).
         - Keep track of the best (highest) modularity partition.
      6) Output the best partition's communities:
         Each line: 'user1', 'user2', ...
         * Sorted by community size ascending, then lexicographically by 
           the first user in the community. 
         * Each community’s user IDs sorted lexicographically as well.
---------------------------------------------------------------------------------
    NOTE: 
      - Strictly Spark RDD + standard Python (no GraphFrames, no DataFrames).
      - The BFS-based betweenness approach is from 
        "Mining of Massive Datasets", Chapter 10.
---------------------------------------------------------------------------------
    Example Run:
      spark-submit task2.py 7 /path/ub_sample_data.csv betweenness.txt community.txt
=================================================================================
"""

import sys
import time
from collections import defaultdict, deque
from copy import deepcopy

from pyspark import SparkContext

def compute_betweenness(graph, vertices):
    """
    Compute the betweenness for each edge in 'graph' using 
    the BFS-based Girvan-Newman approach.
    
    :param graph: dict { user: set(neighbors) } representing the adjacency list
    :param vertices: list of all vertex IDs (users)
    :return: dict { (u, v): betweenness_value } for each edge (u < v)
    
    Steps:
     1. For each vertex 'root' in 'vertices':
        - Perform a BFS to get: 
           a) level dict (distance from root)
           b) parent info (which nodes lead to a shortest path)
           c) # of shortest paths (sigma)
        - Then do a bottom-up pass to compute partial edge credits
          and add them to a global betweenness dictionary.
     2. Because each edge is counted from both directions, 
        we typically sum partial credits and/or divide by 2 once at the end.
    """
    
    # We'll store betweenness in a dictionary, keyed by an edge (smaller, larger).
    edge_bet = defaultdict(float)
    
    # BFS-based approach for each root node
    for root in vertices:
        # -----------------------------
        # 1) BFS to build tree layers
        # -----------------------------
        parent_map = defaultdict(set)  # parent_map[v] = set of predecessors of v
        level_map  = dict()           # level_map[v] = distance from root
        num_shortest_paths = defaultdict(float)  # sigma[v] = # of shortest paths from root to v
        
        # Use a queue for BFS
        queue = deque()
        queue.append(root)
        
        level_map[root] = 0
        num_shortest_paths[root] = 1.0  # There's exactly 1 path from root to itself
        
        visited = set([root])
        bfs_order = []  # We'll store the order in which nodes are popped from queue
        
        while queue:
            current = queue.popleft()
            bfs_order.append(current)
            
            # For each neighbor of current
            for nbr in graph[current]:
                if nbr not in visited:
                    # First time we see this neighbor
                    visited.add(nbr)
                    queue.append(nbr)
                    level_map[nbr] = level_map[current] + 1
                    # Initialize sigma for neighbor
                    num_shortest_paths[nbr] = num_shortest_paths[current]
                    parent_map[nbr].add(current)
                else:
                    # If we've seen neighbor in BFS, check if it's in the next level
                    # If so, it means there's another equally short path
                    if level_map[nbr] == level_map[current] + 1:
                        parent_map[nbr].add(current)
                        # Add the # of paths leading to current
                        num_shortest_paths[nbr] += num_shortest_paths[current]
        
        # -----------------------------
        # 2) Bottom-up edge credit
        # -----------------------------
        # Start from the deepest nodes in BFS order, go backwards
        node_credits = defaultdict(float)
        for v in bfs_order:
            node_credits[v] = 1.0
        
        # We'll process in reverse BFS order
        for v in reversed(bfs_order):
            for p in parent_map[v]:
                # fraction = (sigma[p] / sigma[v]) ??? Actually it's:
                #   fraction = (#paths from root->p) / (#paths root->v)
                # But we are distributing v's node credit to p
                share = (node_credits[v] * (num_shortest_paths[p] / num_shortest_paths[v]))
                node_credits[p] += share
                
                # Edge is (p, v) but we store as (smaller, bigger) for consistency
                e = tuple(sorted([p, v]))
                edge_bet[e] += share
        
    # The BFS method above effectively counts each edge in BOTH directions 
    # from each pair of root perspectives, so we typically do final /2
    for e in edge_bet:
        edge_bet[e] /= 2.0
    
    return edge_bet


def get_connected_components(subgraph, vertices):
    """
    Given a 'subgraph' adjacency (possibly with some edges removed), 
    find all connected components using BFS or DFS.
    
    :param subgraph: dict { user: set(neighbors in subgraph) }
    :param vertices: list of all user IDs
    :return: list of components (each component is a sorted list of users)
    """
    visited = set()
    components = []
    for v in vertices:
        if v not in visited:
            # BFS (or DFS) from v
            queue = deque([v])
            visited.add(v)
            comp_nodes = [v]
            
            while queue:
                curr = queue.popleft()
                for nbr in subgraph[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
                        comp_nodes.append(nbr)
            
            # Sort the nodes in this component and add to components
            comp_nodes.sort()
            components.append(comp_nodes)
    
    return components


def compute_modularity(components, original_graph, original_degs, m):
    """
    Compute the modularity of the current partition (components).
    
    :param components: list of lists, each list is one community
    :param original_graph: the adjacency from the *original* graph
    :param original_degs: dict { node: degree_in_original_graph }
    :param m: number of edges in the original graph
    :return: a float, the modularity score
    
    From the PDF:
       modularity = (1 / (2m)) * sum_{u,v in same community} [ A_uv - (k_u * k_v) / (2m) ]
       Here, A_uv=1 if there's an edge in *current subgraph*, 0 otherwise
       but the degrees k_u, k_v are from the original graph and m is the 
       total # of edges in the original graph.
       
    However, the assignment's clarifications:
       - "m", "k_i", "k_j" should not be changed (always from original)
       - "A" is from the *updated graph* (where edges are removed).
         Actually, the PDF's "Hint" #5 says "For A, using current graph; 
         for k_i*k_j, use original graph".
         
    So let's interpret:
       - We'll check if edge (u,v) is present in the *current subgraph* 
         or not (that subgraph must be passed in).
       - We'll rely on the original adjacency for degrees 
         and the total # of edges for the second term.
    """
    # We’ll do a sum over all pairs (u,v) in each community
    # but that can be large. We’ll do a simpler approach:
    # for each community, for each pair within that community, 
    # compute the contribution to modularity.
    # We'll create a set for quick membership check in subgraph 
    # if needed. But let's do the adjacency check carefully.
    
    # Build a quick adjacency check from the *current subgraph*
    # so we know if an edge exists or not.
    # However, we can also just check original_graph. 
    # But we must reflect the "current" edges, not the original edges!
    # => We'll build a set from the adjacency as it stands:
    A_set = set()
    for u in original_graph:
        # original_graph[u] is the neighbors in the *current subgraph*, 
        # since we presumably keep removing edges from the adjacency we pass in.
        for v in original_graph[u]:
            if u < v:
                A_set.add((u, v))
            else:
                A_set.add((v, u))
    
    mod_sum = 0.0
    for c in components:
        # for each community c, consider pairs (u,v)
        # c might have length n, so we do combinations or double loop
        length_c = len(c)
        for i in range(length_c):
            for j in range(i+1, length_c):
                u = c[i]
                v = c[j]
                # A_uv = 1 if (u,v) in subgraph, else 0
                A_uv = 1.0 if (u, v) in A_set else 0.0
                mod_sum += ( A_uv - ( (original_degs[u] * original_degs[v]) / (2.0*m) ) )
    
    return mod_sum / (2.0 * m)


def main():
    """
    Main driver for Task 2:
      1) Parse input
      2) Build the graph with threshold
      3) Compute and write betweenness
      4) Iteratively remove edges with highest betweenness, 
         track best (max) modularity partition
      5) Write final communities
    """
    if len(sys.argv) != 5:
        print("Usage: task2.py <filter_threshold> <input_file_path> <betweenness_output_path> <community_output_path>")
        sys.exit(-1)
    
    filter_threshold = int(sys.argv[1])
    input_file_path  = sys.argv[2]
    betweenness_file = sys.argv[3]
    community_file   = sys.argv[4]
    
    start_time = time.time()
    
    # ------------------------------------------------------------------------------
    # 1) SparkContext + load data
    # ------------------------------------------------------------------------------
    sc = SparkContext(appName="task2")
    sc.setLogLevel("ERROR")
    
    # We read the CSV file, skip the header, split by comma => (user_id, business_id)
    lines = sc.textFile(input_file_path)
    header = lines.first()
    data_rdd = lines.filter(lambda x: x != header).map(lambda x: x.split(","))
    
    # user_business_map: dict(user -> set(businesses))
    user_business_map = data_rdd \
        .groupByKey() \
        .mapValues(lambda biz_list: set(biz_list)) \
        .collectAsMap()
    
    # Distinct list of users
    all_users = list(user_business_map.keys())
    
    # ------------------------------------------------------------------------------
    # 2) Build the graph adjacency (undirected)
    # ------------------------------------------------------------------------------
    # We want an edge between userA, userB if 
    # len(user_business_map[A] ∩ user_business_map[B]) >= filter_threshold
    # We'll store adjacency in a dict: graph[u] = set of neighbors
    
    # We'll do combinations, not permutations (since it's an undirected edge).
    # But for large data, we might broadcast sets. For the assignment, let's do a direct approach.
    from itertools import combinations
    edges = []
    adjacency = defaultdict(set)
    
    # We can do a naive approach because the data isn't huge:
    for u, v in combinations(all_users, 2):
        busA = user_business_map[u]
        busB = user_business_map[v]
        common_count = len(busA.intersection(busB))
        if common_count >= filter_threshold:
            # add edge both ways
            adjacency[u].add(v)
            adjacency[v].add(u)
    
    # The "vertices" are all users that appear in adjacency keys (or adjacency values)
    vertices = set(adjacency.keys())  # all nodes that have edges
    # If a user never appears in adjacency, it means no edges for that user => exclude them.
    vertices = list(vertices)
    
    # We'll also store the original adjacency in a separate structure so we can 
    # revert or do modularity with original degrees. But according to the PDF, we keep 'k_i' from the original graph, 
    # i.e. the initial adjacency right after building. We do not remove edges from that. 
    # We'll call it "original_graph" = adjacency_for_mod. But let's do a deep copy:
    original_graph = deepcopy(adjacency)
    
    # Count how many edges in original. Because it's undirected, 
    # each edge is stored twice in adjacency. So let's do a quick sum:
    #   m = (1/2) * sum of degrees
    # We'll also build a dictionary of degrees for each node from the original graph.
    original_degs = dict()
    sum_deg = 0
    for u in original_graph:
        deg = len(original_graph[u])
        original_degs[u] = deg
        sum_deg += deg
    m = sum_deg / 2.0  # total edges in the original
    
    # ------------------------------------------------------------------------------
    # 3) Compute Betweenness & Write it out
    # ------------------------------------------------------------------------------
    edge_bet_dict = compute_betweenness(adjacency, vertices)
    
    # Sort edges by:
    # 1) betweenness descending
    # 2) then lexicographically by user1, user2
    # Edge is (u, v) with u < v guaranteed from compute_betweenness
    bet_sorted = sorted(edge_bet_dict.items(), 
                        key=lambda x: (-x[1], x[0][0], x[0][1]))
    
    # Write betweenness
    # format: ( 'u1', 'u2' ), betweenness
    # with 5 decimals for betweenness
    with open(betweenness_file, 'w') as fout:
        for edge, val in bet_sorted:
            u1, u2 = edge
            # Tuple should be ( 'u1', 'u2' )
            # Then a comma, then betweenness value
            # e.g.  ('abc', 'zzz'), 1.12345
            fout.write(f"('{u1}', '{u2}'), {round(val,5)}\n")
    
    # ------------------------------------------------------------------------------
    # 4) Girvan-Newman Edge Removal to Find Maximum Modularity
    # ------------------------------------------------------------------------------
    # We'll remove edges with highest betweenness in each iteration, 
    # re-compute betweenness on the updated subgraph, 
    # compute communities, modularity, keep track of best partition.
    
    best_modularity = float('-inf')
    best_communities = []
    
    # We'll keep removing edges until adjacency is empty or betweenness is empty
    current_graph = adjacency  # We'll remove edges in 'current_graph'
    
    while True:
        # 4a) Find connected components in the current subgraph
        current_components = get_connected_components(current_graph, vertices)
        
        # 4b) Compute the modularity of this partition
        # "A" is from the current subgraph, "k_u" from original_degs, "m" from original
        current_mod = compute_modularity(current_components, current_graph, original_degs, m)
        
        if current_mod > best_modularity:
            best_modularity = current_mod
            best_communities = current_components
        
        # 4c) Compute new betweenness for the current subgraph
        new_betweenness = compute_betweenness(current_graph, vertices)
        if not new_betweenness:
            # No edges left, break
            break
        
        # Sort to find the max betweenness
        bet_list = sorted(new_betweenness.items(), key=lambda x: x[1], reverse=True)
        if not bet_list:
            break
        
        highest_bet = bet_list[0][1]  # the top betweenness value
        # 4d) Remove all edges with betweenness >= highest_bet
        # (like the PDF says: if multiple edges have the same highest bet, remove them all)
        edges_to_remove = [e for e, val in bet_list if abs(val - highest_bet) < 1e-12]
        
        for (u, v) in edges_to_remove:
            # Remove the edge from both sides
            current_graph[u].discard(v)
            current_graph[v].discard(u)
            
        # If we removed all edges, we might do another iteration to compute 
        # the partition with no edges at all (which might be all singletons).
        # We'll see in the next iteration of while.
        
        # If at this point the graph is empty or no edges remain, 
        # next loop will just compute betweenness as empty => break anyway.
        
    # ------------------------------------------------------------------------------
    # 5) Write the best partition
    # ------------------------------------------------------------------------------
    # The PDF says:
    #   1) sort communities by ascending size,
    #      then by the lexicographically smallest user
    #   2) each community line => 'userA', 'userB', ...
    #   3) user IDs inside community sorted lex
    #
    # We already sorted each community internally in BFS. Let's just confirm.
    final_communities = sorted(best_communities, key=lambda c: (len(c), c[0]))
    
    with open(community_file, 'w') as fout:
        for comm in final_communities:
            # Format: 'u1', 'u2', 'u3'
            line = ", ".join(f"'{u}'" for u in comm)
            fout.write(line + "\n")
    
    end_time = time.time()
    print("Duration:", end_time - start_time)
    
    sc.stop()


# Standard "if main" block
if __name__ == "__main__":
    main()
