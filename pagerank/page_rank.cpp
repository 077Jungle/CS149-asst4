#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  // initialization: see example code above
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  
  // all nodes v in graph with no outgoing edges
  Vertex* no_outgoing = new int[numNodes];
  int no_outgoing_num = 0;
  for (int i = 0; i < numNodes; i++) {
    if (outgoing_size(g, i) == 0) {
        no_outgoing[no_outgoing_num++] = i;
    }
  }

  bool converged = false;
  double global_diff;
  double rest1 = (1.0 - damping) / numNodes;
  double* score_new = new double[numNodes];
  while (!converged) {
    // sum over all nodes v in graph with no outgoing edges
    double rest2 = 0.;
    #pragma omp parallel for reduction (+:rest2)
    for (int i = 0; i < no_outgoing_num; i++) {
        rest2 += solution[no_outgoing[i]];
    }
    rest2 *= damping;
    rest2 /= numNodes;

    global_diff = 0.;
    #pragma omp parallel for
    for (int i=0; i<numNodes; i++) {
      // init
      score_new[i] = 0.;
      // compute score_new[vi] for all nodes vi:
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      // sum over all nodes vj reachable from incoming edges
      for (const Vertex* v=start; v!=end; v++)
        score_new[i] += solution[*v] / outgoing_size(g, *v);
      // plus rest
      score_new[i] = (damping * score_new[i]) + rest1 + rest2;
    }
    #pragma omp parallel for reduction (+:global_diff)
    for (int i=0; i<numNodes; i++) {
      // compute global_diff
      global_diff += abs(score_new[i] - solution[i]);
      // update
      solution[i] = score_new[i];
    }
    converged = (global_diff < convergence);
  }
  delete [] score_new;
  delete [] no_outgoing;
  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
