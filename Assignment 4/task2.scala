import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.{Map, mutable}
import scala.collection.mutable.{ArrayBuffer, HashMap, HashSet, Queue}
import java.io.{File, PrintWriter}
import scala.math.{abs, min, max}
import scala.util.control.Breaks._

/**
 * DSCI-553: Assignment 4 (Spring 2025)
 * Task 2: Community Detection Using the Girvan-Newman Algorithm
 *
 * This implementation follows the steps outlined in the assignment:
 * 1. Construct a graph based on user-business shared reviews
 * 2. Calculate betweenness for all edges using BFS-based approach
 * 3. Iteratively remove edges with highest betweenness
 * 4. Compute modularity and identify communities with maximum modularity
 * 5. Output formatted results according to specifications
 */
object task2 {
  type UserID = String
  type BusinessID = String
  type Edge = (UserID, UserID)
  
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println("Usage: Task2 <filter_threshold> <input_file_path> <betweenness_output_path> <community_output_path>")
      System.exit(1)
    }

    val startTime = System.nanoTime()

    // Parse command line arguments
    val filterThreshold = args(0).toInt
    val inputFilePath = args(1)
    val betweennessOutputPath = args(2)
    val communityOutputPath = args(3)

    // Initialize Spark context
    val conf = new SparkConf().setAppName("Task2-GirvanNewman")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR") // Reduce log verbosity

    // Read input CSV, skip header, and parse rows
    val rawData = sc.textFile(inputFilePath)
    val header = rawData.first()
    val data = rawData.filter(_ != header).map(_.split(","))

    // Build user -> businesses map
    val userBusinessMap = data.map(row => (row(0), row(1)))
      .groupByKey()
      .mapValues(_.toSet)
      .collectAsMap()

    // Get all distinct users
    val allUsers = userBusinessMap.keys.toArray

    // Build undirected graph (adjacency list)
    val adjacency = mutable.HashMap.empty[UserID, mutable.HashSet[UserID]]
    
    // Create edges between users who share >= threshold businesses
    for (i <- allUsers.indices; j <- (i + 1) until allUsers.length) {
      val user1 = allUsers(i)
      val user2 = allUsers(j)
      
      val businesses1 = userBusinessMap.getOrElse(user1, Set.empty)
      val businesses2 = userBusinessMap.getOrElse(user2, Set.empty)
      
      val commonCount = businesses1.intersect(businesses2).size
      
      if (commonCount >= filterThreshold) {
        // Add edge in both directions for undirected graph
        adjacency.getOrElseUpdate(user1, mutable.HashSet.empty) += user2
        adjacency.getOrElseUpdate(user2, mutable.HashSet.empty) += user1
      }
    }

    // Get vertices that have at least one edge
    val vertices = adjacency.keys.toArray
    
    // Create deep copy of original graph for modularity calculations
    val originalGraph = adjacency.map { 
      case (user, neighbors) => user -> neighbors.toSet 
    }.toMap
    
    // Calculate original degree for each node and total edges
    val originalDegrees = originalGraph.map { case (user, neighbors) => user -> neighbors.size }
    val m = originalDegrees.values.sum / 2.0 // Total number of edges (divide by 2 because undirected)
    
    // Calculate initial betweenness and write to file
    val edgeBetweenness = computeBetweenness(adjacency, vertices)
    
    // Sort betweenness results by value (desc) then by userIDs
    val sortedBetweenness = edgeBetweenness.toSeq.sortWith { 
      case ((e1, v1), (e2, v2)) => 
        if (math.abs(v1 - v2) < 1e-10) {
          // If betweenness values are equal (with tolerance), sort by edge lexicographically
          if (e1._1 == e2._1) e1._2 < e2._2 else e1._1 < e2._1
        } else {
          // Sort by betweenness descending
          v1 > v2
        }
    }
    
    // Write betweenness to file
    writeBetweennessToFile(betweennessOutputPath, sortedBetweenness)
    
    // Run Girvan-Newman algorithm to find communities
    val (bestModularity, bestCommunities) = findCommunitiesWithMaxModularity(
      adjacency, vertices, originalGraph, originalDegrees, m
    )
    
    // Sort communities by size (asc) then by first user lexicographically
    val sortedCommunities = bestCommunities.sortWith { 
      case (c1, c2) => 
        if (c1.length == c2.length) c1.head < c2.head 
        else c1.length < c2.length
    }
    
    // Write communities to file
    writeCommunitiesToFile(communityOutputPath, sortedCommunities)
    
    val endTime = System.nanoTime()
    println(s"Duration: ${(endTime - startTime) / 1e9} seconds")
    
    sc.stop()
  }

  /**
   * Computes betweenness for each edge in the graph using the BFS-based approach.
   * 
   * @param graph Adjacency list representation of the graph
   * @param vertices Array of all vertex IDs
   * @return Map of edges to betweenness values
   */
  def computeBetweenness(
    graph: mutable.Map[UserID, mutable.HashSet[UserID]], 
    vertices: Array[UserID]
  ): mutable.Map[Edge, Double] = {
    val edgeBetweenness = mutable.HashMap.empty[Edge, Double].withDefaultValue(0.0)
    
    // Calculate betweenness from each root node perspective
    for (root <- vertices) {
      // Maps for BFS traversal
      val parentMap = mutable.HashMap.empty[UserID, mutable.Set[UserID]]
      val levelMap = mutable.HashMap.empty[UserID, Int]
      val numShortestPaths = mutable.HashMap.empty[UserID, Double]
      
      // Initialize for root node
      levelMap(root) = 0
      numShortestPaths(root) = 1.0
      
      // BFS queue and tracking
      val queue = mutable.Queue(root)
      val visited = mutable.HashSet(root)
      val bfsOrder = ArrayBuffer[UserID]()
      
      // BFS traversal to build shortest path tree
      while (queue.nonEmpty) {
        val current = queue.dequeue()
        bfsOrder += current
        
        // For each neighbor of current node
        val neighbors = graph.getOrElse(current, mutable.HashSet.empty)
        for (nbr <- neighbors) {
          if (!visited.contains(nbr)) {
            // First time seeing this neighbor
            visited += nbr
            queue.enqueue(nbr)
            levelMap(nbr) = levelMap(current) + 1
            numShortestPaths(nbr) = numShortestPaths(current)
            parentMap.getOrElseUpdate(nbr, mutable.HashSet.empty) += current
          } else if (levelMap.getOrElse(nbr, Int.MaxValue) == levelMap(current) + 1) {
            // Another equally short path to this neighbor
            parentMap.getOrElseUpdate(nbr, mutable.HashSet.empty) += current
            numShortestPaths(nbr) = numShortestPaths.getOrElse(nbr, 0.0) + numShortestPaths(current)
          }
        }
      }
      
      // Bottom-up traversal to calculate edge credits
      val nodeCredits = mutable.HashMap.empty[UserID, Double].withDefaultValue(1.0)
      
      // Process in reverse BFS order
      for (v <- bfsOrder.reverse) {
        val parents = parentMap.getOrElse(v, mutable.Set.empty)
        for (p <- parents) {
          // Share of credit from v to p
          val share = nodeCredits(v) * (numShortestPaths(p) / numShortestPaths(v))
          nodeCredits(p) = nodeCredits(p) + share
          
          // Store edge with smaller node ID first (for consistency)
          val edge = if (p < v) (p, v) else (v, p)
          edgeBetweenness(edge) = edgeBetweenness(edge) + share
        }
      }
    }
    
    // Divide by 2 since each edge is counted from both directions
    for (edge <- edgeBetweenness.keys.toList) {
      edgeBetweenness(edge) = edgeBetweenness(edge) / 2.0
    }
    
    edgeBetweenness
  }

  /**
   * Finds all connected components in the given graph using BFS.
   * 
   * @param graph Adjacency list representation of the graph
   * @param vertices Array of all vertex IDs
   * @return Sequence of components (each component is a sorted array of vertices)
   */
  def getConnectedComponents(
    graph: mutable.Map[UserID, mutable.HashSet[UserID]], 
    vertices: Array[UserID]
  ): Seq[Seq[UserID]] = {
    val visited = mutable.HashSet.empty[UserID]
    val components = ArrayBuffer.empty[Seq[UserID]]
    
    for (v <- vertices if !visited.contains(v)) {
      // BFS from v
      val queue = mutable.Queue(v)
      visited += v
      val componentNodes = ArrayBuffer(v)
      
      while (queue.nonEmpty) {
        val current = queue.dequeue()
        val neighbors = graph.getOrElse(current, mutable.HashSet.empty)
        
        for (nbr <- neighbors if !visited.contains(nbr)) {
          visited += nbr
          queue.enqueue(nbr)
          componentNodes += nbr
        }
      }
      
      // Sort nodes in this component and add to components list
      components += componentNodes.sorted
    }
    
    components
  }

  /**
   * Computes modularity for the given partitioning of the graph.
   * 
   * @param components List of communities (each a sorted list of users)
   * @param currentGraph Current graph adjacency
   * @param originalDegrees Map of node to original degree
   * @param m Total number of edges in the original graph
   * @return Modularity value
   */
  def computeModularity(
    components: Seq[Seq[UserID]], 
    currentGraph: mutable.Map[UserID, mutable.HashSet[UserID]],
    originalDegrees: Map[UserID, Int], 
    m: Double
  ): Double = {
    // Build set of existing edges for quick lookup
    val edgeSet = mutable.HashSet.empty[Edge]
    for {
      (u, neighbors) <- currentGraph
      v <- neighbors
    } {
      val edge = if (u < v) (u, v) else (v, u)
      edgeSet += edge
    }
    
    var modularitySum = 0.0
    
    // For each community
    for (community <- components) {
      // For each pair of nodes in the community
      for (i <- community.indices; j <- (i + 1) until community.length) {
        val u = community(i)
        val v = community(j)
        
        // A_uv is 1 if edge exists in current graph, 0 otherwise
        val edge = if (u < v) (u, v) else (v, u)
        val aUV = if (edgeSet.contains(edge)) 1.0 else 0.0
        
        // k_u * k_v / (2m) - expected number of edges
        val expectedEdges = (originalDegrees.getOrElse(u, 0) * originalDegrees.getOrElse(v, 0)) / (2.0 * m)
        
        // Add this pair's contribution to modularity
        modularitySum += (aUV - expectedEdges)
      }
    }
    
    // Final normalization
    modularitySum / (2.0 * m)
  }

  /**
   * Finds communities with maximum modularity using the Girvan-Newman algorithm.
   * 
   * @param adjacency Initial graph adjacency
   * @param vertices Array of vertex IDs
   * @param originalGraph Original graph (for modularity calculation)
   * @param originalDegrees Map of node to original degree
   * @param m Total edges in original graph
   * @return Tuple of (best modularity, best communities)
   */
  def findCommunitiesWithMaxModularity(
    adjacency: mutable.Map[UserID, mutable.HashSet[UserID]],
    vertices: Array[UserID],
    originalGraph: Map[UserID, Set[UserID]],
    originalDegrees: Map[UserID, Int],
    m: Double
  ): (Double, Seq[Seq[UserID]]) = {
    var bestModularity = Double.NegativeInfinity
    var bestCommunities = Seq.empty[Seq[UserID]]
    
    // Make a working copy of the graph that we'll modify
    val currentGraph = mutable.HashMap.empty[UserID, mutable.HashSet[UserID]]
    adjacency.foreach { case (k, v) => 
      currentGraph(k) = mutable.HashSet.empty[UserID] ++ v
    }
    
    // Iteratively remove edges until no edges remain
    breakable {
      while (true) {
        // Find connected components in current graph
        val currentComponents = getConnectedComponents(currentGraph, vertices)
        
        // Calculate modularity of current partition
        val currentModularity = computeModularity(currentComponents, currentGraph, originalDegrees, m)
        
        // Update best modularity if current is better
        if (currentModularity > bestModularity) {
          bestModularity = currentModularity
          bestCommunities = currentComponents
        }
        
        // Calculate betweenness on current graph
        val newBetweenness = computeBetweenness(currentGraph, vertices)
        if (newBetweenness.isEmpty) {
          break // No edges left
        }
        
        // Find highest betweenness value
        val maxBetweenness = newBetweenness.values.max
        
        // Find all edges with highest betweenness (within tolerance)
        val edgesToRemove = newBetweenness.filter {
          case (_, value) => math.abs(value - maxBetweenness) < 1e-10
        }.keys.toList
        
        // Remove these edges from the graph
        for ((u, v) <- edgesToRemove) {
          currentGraph.getOrElseUpdate(u, mutable.HashSet.empty) -= v
          currentGraph.getOrElseUpdate(v, mutable.HashSet.empty) -= u
        }
        
        // If no edges were removed or all edges are gone, we're done
        if (edgesToRemove.isEmpty || currentGraph.forall(_._2.isEmpty)) {
          break
        }
      }
    }
    
    (bestModularity, bestCommunities)
  }

  /**
   * Writes betweenness results to file in required format.
   * 
   * @param filePath Output file path
   * @param betweenness Sorted sequence of (edge, betweenness value)
   */
  def writeBetweennessToFile(
    filePath: String, 
    betweenness: Seq[(Edge, Double)]
  ): Unit = {
    val writer = new PrintWriter(new File(filePath))
    try {
      for (((u, v), value) <- betweenness) {
        // Format: ('user1', 'user2'), betweenness_value (5 decimal places)
        writer.println(s"('$u', '$v'), ${value.round(5)}") 
      }
    } finally {
      writer.close()
    }
  }

  /**
   * Writes community results to file in required format.
   * 
   * @param filePath Output file path
   * @param communities Sorted sequence of communities
   */
  def writeCommunitiesToFile(
    filePath: String, 
    communities: Seq[Seq[UserID]]
  ): Unit = {
    val writer = new PrintWriter(new File(filePath))
    try {
      for (community <- communities) {
        // Format: 'user1', 'user2', 'user3', ...
        val line = community.map(u => s"'$u'").mkString(", ")
        writer.println(line)
      }
    } finally {
      writer.close()
    }
  }
  
  // Helper extension method to round doubles to specified decimal places
  implicit class DoubleOps(val d: Double) extends AnyVal {
    def round(places: Int): Double = {
      val factor = math.pow(10, places)
      math.round(d * factor) / factor
    }
  }
}
