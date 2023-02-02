#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define THRESHOLD 750000
// #define VERBOSE

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
    int iter) {
    #pragma omp parallel
    {
        int local_count = 0;
        int* local_frontier = (int*)malloc(sizeof(int) * (g->num_nodes)); 
        #pragma omp for
        for (int i=0; i<frontier->count; i++) {

            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                
                if (distances[outgoing] == NOT_VISITED_MARKER && 
                    __sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, iter + 1)) {
                    local_frontier[local_count++] = outgoing;
                }
            }
        }
        if (local_count) {
            int start_idx = 0;
            #pragma omp atomic capture 
            {
                start_idx = new_frontier->count;
                new_frontier->count += local_count;
            }
            memcpy(new_frontier->vertices + start_idx, local_frontier, local_count * sizeof(int));
        }
        free(local_frontier);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for 
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int iter = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, iter);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        iter++;
    }

    free(list1.vertices);
    free(list2.vertices);
}

int bottom_up_step(
    Graph g, 
    bool* frontier, 
    bool* new_frontier, 
    int* distances, 
    int iter
) {
    int count = 0;
    #pragma omp parallel for reduction(+:count) schedule(dynamic, 256)
    for (int i=0; i<g->num_nodes; i++) {

        if (distances[i] != NOT_VISITED_MARKER) continue;

        int start_edge = g->incoming_starts[i];
        int end_edge = (i == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[i + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            if (frontier[incoming]) {
                distances[i] = iter + 1;
                new_frontier[i] = true;
                count++;
                break;
            }
        }
    }
    return count;
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    bool* frontier = new bool[graph->num_nodes];
    bool* new_frontier = new bool[graph->num_nodes];
    // initialize all nodes to NOT_VISITED and all frontier
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++) {
        frontier[i] = false;
        new_frontier[i] = false;
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;
    int count = 1;
    int iter = 0;

    while (count) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        int tmp_count = count;
        #pragma omp parallel for
        for (int i=0; i<graph->num_nodes; i++) {
            new_frontier[i] = false;
        }

        count = bottom_up_step(graph, frontier, new_frontier, sol->distances, iter);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", tmp_count, end_time - start_time);
#endif

        // swap pointers
        bool* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        iter++;
    }

    delete [] frontier;
    delete [] new_frontier;
}

void bool_convert_to_vertex(bool* frontier_bool, vertex_set* frontier_vertex) {
    const int num_nodes = frontier_vertex->max_vertices;
    #pragma omp parallel
    {
        int local_count = 0;
        int* local_frontier = (int*)malloc(sizeof(int) * (num_nodes)); 
        #pragma omp for
        for (int i=0; i<num_nodes; i++) {
            if (frontier_bool[i]) {
                local_frontier[local_count++] = i;
            }
        }
        if (local_count) {
            int start_idx;
            #pragma omp atomic capture 
            {
                start_idx = frontier_vertex->count;
                frontier_vertex->count += local_count;
            }
            memcpy(frontier_vertex->vertices + start_idx, local_frontier, local_count * sizeof(int));
        }
        free(local_frontier);
    }
}

int vertex_convert_to_bool(vertex_set* frontier_vertex, bool* frontier_bool) {
    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (int i=0; i<frontier_vertex->count; i++) {
        frontier_bool[frontier_vertex->vertices[i]] = true;
        count++;
    }
    return count;
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    bool* frontier_bool = new bool[graph->num_nodes];
    bool* new_frontier_bool = new bool[graph->num_nodes];

    // initialize all nodes to NOT_VISITED and bool*
    #pragma omp parallel for 
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier_bool[i] = false;
        new_frontier_bool[i] = false;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int count = 1;
    bool top_downing = true;
    int iter = 0;

    while (count) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        int tmp_count = count;
        if (frontier->count < THRESHOLD) {
            if (!top_downing) {
                top_downing = true;
                vertex_set_clear(frontier);
                bool_convert_to_vertex(frontier_bool, frontier);
            }
            vertex_set_clear(new_frontier);
            top_down_step(graph, frontier, new_frontier, sol->distances, iter);
            count = new_frontier->count;

            // swap pointers
            vertex_set* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        } else {
            if (top_downing) {
                top_downing = false;
                #pragma omp parallel for
                for (int i=0; i<graph->num_nodes; i++) {
                    frontier_bool[i] = false;
                    new_frontier_bool[i] = false;
                }
                vertex_convert_to_bool(frontier, frontier_bool);
            } else {
                #pragma omp parallel for
                for (int i=0; i<graph->num_nodes; i++) {
                    new_frontier_bool[i] = false;
                }
            }
            count = bottom_up_step(graph, frontier_bool, new_frontier_bool, sol->distances, iter);

            // swap pointers
            bool* tmp = frontier_bool;
            frontier_bool = new_frontier_bool;
            new_frontier_bool = tmp;
        }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", tmp_count, end_time - start_time);
#endif

        iter++;
    }

    free(list1.vertices);
    free(list2.vertices);
    delete [] frontier_bool;
    delete [] new_frontier_bool;
}
