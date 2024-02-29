from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import matplotlib.figure
import time
import random
import matplotlib.pyplot as plt
import threading
import math

#Graph class to store vertices and edges
class Graph:
    #Graph constructor
    def __init__(self):
        self.vertices = []
        self.edges = []

    #method to add vertex to vertices list
    def add_vertex(self, x, y):
        self.vertices.append((x,y))

    #method to add edge to edges list
    def add_edge(self, v1, v2):
        self.edges.append([v1, v2])

#Function that reads the .tsp file and stores the x and y values of the vertices to the graph.vertices and graph.edges lists
def read_input_file(file_name, graph):
    with open(file_name, "r") as file:
        parsing_coordinates = False
        parsing_edges = False
        #iterate through lines in .tsp file
        for line in file:
            tokens = line.strip().split()
            if parsing_coordinates:
                if len(tokens) == 3:
                    x = float(tokens[1])
                    y = float(tokens[2])
                    graph.add_vertex(x, y)
                #Switches from parsing vertices to parsing edges
                elif tokens and tokens[0] == "EDGES":
                    parsing_edges = True
                    parsing_coordinates = False
            #Parsing Edges
            elif parsing_edges:
                if len(tokens) == 3:
                    v1 = int(tokens[0]) + 1
                    v2 = int(tokens[1]) + 1
                    graph.add_edge(v1, v2)
            #Start parsing vertices after "NODE_COORD_SECTION" is found
            elif tokens and tokens[0] == "NODE_COORD_SECTION":
                parsing_coordinates = True

                
#GUI popup for genetic algorithm and wisdom of crowds settings
def genetic_alg_popup(canvas, ax):
    #Creates window
    lp_window = tk.Toplevel()
    lp_window.geometry("600x400")
    lp_window.title("Longest Path Settings")

    #Instructions of what input values mean/do
    label_instructions1 = tk.Label(lp_window, text="Input starting vertex and ending vertex to find longest path between.")
    label_instructions2 = tk.Label(lp_window, text="Then input the mutation rate and number of experts for WOC.")
    label_instructions3 = tk.Label(lp_window, text="Finally, the length of the path is the amount of vertices that should be in the final path.\n If a path with that many vertices is present, true along with the path will be output.\n If not, then false will be output at the end.")

    label_instructions1.pack()
    label_instructions2.pack()
    label_instructions3.pack()

    #Mutation rate input
    label_mutation_rate = tk.Label(lp_window, text="Mutation Rate: ")
    mutation_rate = tk.Entry(lp_window)

    #Number of GA runs input
    label_expert_amt = tk.Label(lp_window, text="# GA Runs (4 experts per run): ")
    expert_amt = tk.Entry(lp_window)

    #Length of path to search for input
    label_path_length = tk.Label(lp_window, text="Path Length: ")
    path_length = tk.Entry(lp_window)

    label_mutation_rate.pack(pady=5)
    mutation_rate.pack(pady=5)
    label_expert_amt.pack(pady=5)
    expert_amt.pack(pady=5)
    label_path_length.pack(pady=5)
    path_length.pack(pady=5)

    #Button to start running GA
    run_genetic_alg_button = tk.Button(lp_window, text = "Run GA for Longest Path", command=lambda: start_genetic_algorithm(graph, canvas, ax, int(mutation_rate.get()), run_genetic_alg_button, int(expert_amt.get()), int(path_length.get())), bg = 'blue')
    run_genetic_alg_button.pack()

#Creates initial population based on population size variable
def create_initial_pop(graph, population_size):
    population = []

    for _ in range(population_size):
        path = create_path(graph)
        population.append(path)

    return population

#Creates a random simple path through the graph along edges
def create_path(graph):
    start = random.choice(range(len(graph.vertices)))
    end = random.choice(range(len(graph.vertices)))
    
    #Ensures start and end are not the same vertex
    while start == end:
        end = random.choice(range(len(graph.vertices)))

    #Starts path with random starting vertex
    path = [start]
    current_vertex = start

    #Iterates until path from start to end is found
    while current_vertex != end:
        #Adds all vertices connected via one edge to neighbors list
        neighbors = [v2 for v1, v2 in graph.edges if v1 == current_vertex and v2 not in path]
        neighbors += [v1 for v1, v2 in graph.edges if v2 == current_vertex and v1 not in path]

        #Break if dead end before current_vertex == end
        if not neighbors:
            break
        #Picks a random neighbor to be the next vertex in path
        else:
            next_vertex = random.choice(neighbors)
            path.append(next_vertex)
            current_vertex = next_vertex

    return path


#Starts genetic algorithm on a thread
def start_genetic_algorithm(graph, canvas, ax, mutation_rate, run_genetic_alg_button, num_runs, path_length):
    #Function used to run algorithm in thread
    def run_algorithm():
        genetic_algorithm(graph, canvas, ax, mutation_rate, num_runs, path_length)
        run_genetic_alg_button.config(state=tk.NORMAL)
    #disables button when program is running
    run_genetic_alg_button.config(state=tk.DISABLED)
    #Start genetic algorithm on a thread
    ga_thread = threading.Thread(target=run_algorithm)
    ga_thread.start()

#Main GA function
def genetic_algorithm(graph, canvas, ax, mutation_rate, num_runs, path_length):
    experts_paths = []
    experts_fitness = []
    #Gets starting time for GA
    start_time = time.time()
    #Iterates for the number of runs specified by user
    for i in range(num_runs):
        #Change population and generation variables here
        population_size = 500
        max_generations = 100
    
        

        #Creates an intial random population of simple paths
        current_pop = create_initial_pop(graph, population_size)
        #Selects four parents with the best fitness (highest length)
        parents, fitness = select_parents(graph, current_pop)

        #Iterates through generations in GA
        for generation in range(max_generations):
            #Crosses over two parents at a time to create a new population
            population = crossover(parents, population_size, current_pop)
            #Mutates population based on user input mutation rate
            current_pop = mutation(population, mutation_rate, graph)
            #Elitism so fitness never decreases
            current_pop.append(parents[0])
            current_pop.append(parents[1])
            current_pop.append(parents[2])
            current_pop.append(parents[3])
            #Selects new parents based on current population
            parents, fitness = select_parents(graph, current_pop)
            #Selects best parent to be representative
            best_individual = parents[0]
            #Outputs run number, generation number, best fitness, and best path of generation
            print(f"Run {i + 1}, Generation {generation + 1}, Best Fitness: {fitness[0]:.2f}")
            print(f"Best Path: {best_individual}")
            #Graphs path of best individual for dynamic representation
            plot(graph, ax, canvas, best_individual)

        #Assigns final four parents of current run as experts of run for WOC
        experts_paths.append(parents[0])
        experts_paths.append(parents[1])
        experts_paths.append(parents[2])
        experts_paths.append(parents[3])
        experts_fitness.append(fitness[0])
        experts_fitness.append(fitness[1])
        experts_fitness.append(fitness[2])
        experts_fitness.append(fitness[3])

    #Saves end time for GA
    end_time = time.time()
    #Calculates total time in seconds for GA
    GA_time = end_time - start_time
    #Output experts for WOC
    GA_results_output(graph, GA_time, experts_paths, experts_fitness, path_length, ax, canvas,)

def GA_results_output(graph, time, experts_paths, experts_fitness, path_length, ax, canvas,):
    #Creates popup window
    results_window = tk.Toplevel()
    results_window.geometry("800x800")
    results_window.title("Time Elapsed, Fitness, and Path(s)")

    #Canvas used to store returned information
    canvas_window = tk.Canvas(results_window)
    canvas_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    #Scrollbar implementation for use in GUI
    scroll_bar = tk.Scrollbar(results_window, orient=tk.VERTICAL, command = canvas_window.yview)
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas_window.configure(yscrollcommand=scroll_bar.set)

    result_frame = tk.Frame(canvas_window)
    canvas_window.create_window((0, 0), window=result_frame, anchor=tk.NW)

    #Creates time output in window
    label_time = tk.Label(result_frame, text=f"Time Elapsed: {time:.2f} seconds")
    label_time.pack()

    #Iterates through experts to display paths and fitness
    for i in range(len(experts_paths)):
        path = experts_paths[i]
        fitness = experts_fitness[i]
        #Text box to hold path
        text_widget = tk.Text(result_frame, wrap=tk.WORD, height = 5, width = 80)
        text_widget.insert(tk.END, "Best Path:\n")
        for vertex in path[:-1]:
            text_widget.insert(tk.END, f"{vertex} -> ")
        text_widget.insert(tk.END, f"{path[-1]}")
        #Text box cannot be edited
        text_widget.config(state=tk.DISABLED)
        text_widget.pack()
        #Output fitness value of expert
        label_fitness = tk.Label(result_frame, text=f"Fitness: {fitness:.2f}")
        label_fitness.pack()

    result_frame.update_idletasks()
    canvas_window.config(scrollregion=canvas_window.bbox("all"))

    #Button to run Wisdom of Crowds on GA data
    WOC_button = tk.Button(results_window, text="Run WOC on dataset", command=lambda: WOC(experts_paths, time, path_length, graph, ax, canvas,))
    WOC_button.pack()

#WOC function
def WOC(expert_paths, elapsed_time, path_length, graph, ax, canvas,):
    #Saves starting time for WOC
    start_time = time.time()
    #Dictionary used to store edges and how many times they appear in experts
    edges_occurances = {}
    #Iterates through paths of experts
    for path in expert_paths:
        #Iterates through vertices in expert paths
        for i in range(len(path) - 1):
            v1 = path[i]
            v2 = path[i+1]
            #Saves edge as tuple
            edge = (v1, v2) if v1 < v2 else (v2, v1)
            #Increases number of occurances in dictionary if already present in dictionary
            if edge in edges_occurances:
                edges_occurances[edge] += 1
            #Adds edge to dictionary
            else:
                edges_occurances[edge] = 1
    common_edges = []
    #Saves all edges that appear twice or more as common edges
    for edge, count in edges_occurances.items():
        if count >= 2:
            common_edges.append(edge)
    #failures variable used in create_woc_path function
    failures = 0
    #Creates a path from common edges of length requirement input by user, or false if path unable to be created
    path_found, path = create_woc_path(common_edges, graph, path_length, expert_paths, failures)
    #Saves ending time of WOC
    end_time = time.time()
    #Calculates total time for WOC
    WOC_time = end_time - start_time
    #Calculates total time for GA and WOC
    total_time = elapsed_time + WOC_time
    #If a path was found, calculates fitness, plots path in GUI, and outputs results
    if path_found:
        fitness = fitness_calc(graph, path)
        plot(graph, ax, canvas, path)
        WOC_results_output(total_time, path, fitness, path_found, path_length)
    #If no path found, fitness is 0, path is blank, and false is output in GUI
    else:
        path = []
        fitness = 0
        WOC_results_output(total_time, path, fitness, path_found, path_length)

#Attempts to create a path of length indicated by user via WOC
def create_woc_path(common_edges, graph, length, expert_paths, failures):
        #If user inputs length of 1, output a random vertex as "path"
        if length < 2:
            return True, random.choice(range(len(graph.vertices))) + 1
        
        #Checks to see if any experts meet length requirement, if so path is output
        for expert_path in expert_paths:
            if len(expert_path) == length:
                return True, expert_path

        #List used to store vertices able to be used in path
        available_vertices = []
        #Dictionary used to store how many times a vertex appears in common edges
        edge_occurrences = {}

        #Iterates through common edges
        for edge in common_edges:
            v1, v2 = edge
            #Adds edge or increases value if edge already present in dictionary
            edge_occurrences[edge] = edge_occurrences.get(edge, 0) + 1
            #Adds vertices of common edges to set
            available_vertices.append(v1)
            available_vertices.append(v2)

        starting_vertex = None
        #Iterates through possible vertices
        for vertex in available_vertices:
            #Calculates total number of occurances vertex has in common edges, if only appears once use as start
            if sum(1 for edge in common_edges if vertex in edge) == 1:
                starting_vertex = vertex
                break

        #If all vertices appear more than once, pick a random starting vertex
        if starting_vertex is None:
            starting_vertex = random.choice(available_vertices)

        #Starts path with vertex chosen and saves as current vertex
        path = [starting_vertex]
        current_vertex = starting_vertex

        #Iterates until length of path is met
        while len(path) < length:
            #Creates a list of possible edges from current vertex
            viable_edges = [
                edge for edge in common_edges if current_vertex in edge and edge_occurrences[edge] > 0
            ]

            #Recursively call function if dead end and it has not failed at least 100 times
            if not viable_edges and failures < 100:
                #Increase number of failures
                failures += 1
                return create_woc_path(common_edges, graph, length, expert_paths, failures)
            #No path found in 100 attempts
            elif failures == 100:
                path = []
                return False, path

            #Randomly choose next edge to use
            next_edge = random.choice(viable_edges)
            v1, v2 = next_edge
            #Sets neighboring vertex to current vertex as next vertex
            next_vertex = v1 if v2 == current_vertex else v2
            
            #Ensures vertex is not already in path
            if next_vertex not in path:
                path.append(next_vertex)
                #Decrease value in dictionary
                edge_occurrences[next_edge] -= 1
                #Continue with next vertex
                current_vertex = next_vertex
            else:
                #Decrease value in dictionary
                edge_occurrences[next_edge] -= 1

        #Path found, while loop quit
        return True, path
        

def WOC_results_output(time, path, fitness, path_found, length):
    woc_results_window = tk.Toplevel()
    woc_results_window.geometry("800x800")
    woc_results_window.title("Wisdom of Crowds Results")

    #Output time, if path was found, and fitness of path found
    label_tot_time = tk.Label(woc_results_window, text =f"Total Time Elapsed: {time:.2f} seconds")
    label_path_found = tk.Label(woc_results_window, text =f"Path of length {length} found: {path_found}")
    label_fitness = tk.Label(woc_results_window, text = f"Fitness: {fitness:.2f}")


    label_tot_time.pack()
    label_path_found.pack()
    label_fitness.pack()

    #Output for WOC path that fits length requirements
    label_path = tk.Label(woc_results_window, text = "Final WOC path: ")
    label_path.pack()

    #output of WOC calculated path
    if path_found == True:
        text_widget = tk.Text(woc_results_window, wrap=tk.WORD, height = 5, width = 80)
        for i in range(len(path) - 1):
            text_widget.insert(tk.END, f"{path[i]} -> ")
        text_widget.insert(tk.END, f"{path[-1]}")
        text_widget.config(state=tk.DISABLED)
        text_widget.pack()

#Crossover function for GA
def crossover(parents, population_size, current_pop):
    children = []
    #Iterates to create population based on population size variable minus four because four parents carry over
    for _ in range(population_size - 4):
        #Selects two parents to crossover randomly out of four parents
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        #Ensures parent1 is not the same as parent2
        while parent1 == parent2:
            parent1 = random.choice(parents)
        #Creates a set of vertices present in both parents
        common_vertices = set(parent1) & set(parent2)
        #If there are no common vertices reselect parent2
        if not common_vertices:
            parent2 = random.choice(current_pop)
            common_vertices = set(parent1) & set(parent2)
        child = []
        visited_vertices = set()
        #Selects a crossover point randomly from list of common vertices
        common_vertex = random.choice(list(common_vertices))
        
        #Gets index of vertex from each parent path
        index1 = parent1.index(common_vertex)
        index2 = parent2.index(common_vertex)

        #Ensures common vertex chosen is not the starting vertex of a parent
        while index1 == 0 or index2 == 0:
            common_vertex = random.choice(list(common_vertices))
            index1 = parent1.index(common_vertex)
            index2 = parent2.index(common_vertex)

        #Iterates through parent1 to index1, adding each unvisited vertex at a time to child
        for i in range(index1 + 1):
            vertex = parent1[i]
            #Add vertex to child if it is not present already
            if vertex not in visited_vertices:
                child.append(vertex)
                visited_vertices.add(vertex)

        #Iterates through parent2
        for i in range(len(parent2)):
            vertex = parent2[i]
            #adds vertex if it is not already present in child and if it forms a valid edge
            if vertex not in visited_vertices and is_valid_edge(child[-1], vertex, graph):
                child.append(vertex)
                visited_vertices.add(vertex)
        children.append(child)
    return children

#Checks if edge is within graph.edges in some form
def is_valid_edge(v1, v2, graph):
    if [v1, v2] in graph.edges or [v2, v1] in graph.edges:
        return True
    else:
        return False
        
#Mutation function to call two different mutation types
def mutation(population, rate, graph):
    #Stores mutated population
    mutated_population = []
    #iterates through paths
    for path in population:
        #if random value is less than the rate then mutated
        if random.random() <= rate/100:
            #If path constains all vertices, swap mutation
            if len(path) == len(graph.vertices):
                mutated_population.append(swap_mutation(path, graph))
            #if path does not contain all vertices, insert mutation
            else:
                mutated_population.append(insert_mutation(path, graph))
        #Adds unmutated path to mutated population
        else:
            mutated_population.append(path)
    return mutated_population

#Inserts a random vertex only if viable
def insert_mutation(path, graph):
    #new path starts at same vertex
    new_path = [path[0]]

    #Iterates through vertices of path
    for j in range(len(path) - 1):
        v1, v2 = path[j], path[j + 1]
        common_vertex = None

        #Iterates through all vertices
        for vertex in range(len(graph.vertices) + 1):
            #If vertex is not currently in the path, and vertex forms a valid edges between v1 itself and v2 then it is a common_vertex
            if vertex != v1 and vertex != v2 and ([v1, vertex] in graph.edges or [vertex, v1] in graph.edges) and ([v2, vertex] in graph.edges or [vertex, v2] in graph.edges) and vertex not in path:
                common_vertex = vertex
                break

        #Appends common vertex between v1 and v2 if one was found
        if common_vertex is not None:
            new_path.append(common_vertex)
        new_path.append(v2)

    #Iterates through all vertices
    for v in range(len(graph.vertices)):
        #If vertex not in path and forms a valid edge with the start then add vertex before start
        if v not in new_path and ([v, new_path[0]] in graph.edges or [new_path[0], v] in graph.edges):
            new_path = [v] + new_path
        #Else if vertex not in path and forms a valid edge with the end then add vertex to end
        elif v not in new_path and ([v, new_path[-1]] in graph.edges or [new_path[-1], v] in graph.edges):
            new_path = new_path + [v]
    #If no vertex was found to insert then do swap mutation
    if path == new_path:
        new_path = swap_mutation(path, graph)

    return new_path

#Splits the path in two, reverses the order of one of the subgraphs, and adds it to front of a new path
def swap_mutation(path, graph):
    mutated_path = []
    #Sets v2 to be starting vertex
    v2 = path[0]
    #Creates a list of possible vertices the starting vertex forms a valid edge with v2
    valid_neighbors = [v for v in range(len(graph.vertices)) if is_valid_edge(v2, v, graph)]

    #If no valid neighbors, return unmutated path
    if not valid_neighbors:
        return path
    
    #Picks a random vertex out of valid neighbors
    v1 = random.choice(valid_neighbors)

    #Sets index v1 to the index of v1 in path
    index_v1 = path.index(v1)

    #Creates path from v1 to end
    mutated_path = path[index_v1:]
    #Reverses path, now end to v1
    mutated_path.reverse()
    #Adds rest of path unchanged from v2 to end
    mutated_path += path[:index_v1]

    return mutated_path

#Function to calculate fitness (distance of path)
def fitness_calc(graph, path):
    fitness = 0
    #iterates through path and calculates distance between each variable 
    for i in range(len(path)):
        start = path[i]
        end = path[(i+1)%len(path)]
        x1, y1 = graph.vertices[start - 1]
        x2, y2 = graph.vertices[end - 1]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        fitness += dist
    return fitness
    
#Selects parents for crossover
def select_parents(graph, population):
    #Sorts the population in a list by fitness
    sorted_population = sorted(population, key=lambda path: fitness_calc(graph, path), reverse=True)
    parents = []
    fitness = []
    #picks out 4 parents (highest fitness)
    for path in sorted_population:
        if len(parents) == 4:
            break
        if path not in parents:
            parents.append(path)
            fitness.append(fitness_calc(graph, path))
    return parents, fitness

def plot(graph, ax, canvas, best_path):
    #Plots vertices
    ax.clear()
    x = [coord[0] for coord in graph.vertices]
    y = [coord[1] for coord in graph.vertices]
    ax.scatter(x, y)
    
    #Adds labels to vertices
    for i, (x_val, y_val) in enumerate(graph.vertices):
        ax.annotate(str(i + 1), (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='center', size=10)

    #Plots edges between vertices in blue
    for edge in graph.edges:
        v1 = graph.vertices[edge[0] - 1]
        v2 = graph.vertices[edge[1] - 1]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'b')

    #Plots path in red
    if best_path:
        best_x = [graph.vertices[i - 1][0] for i in best_path]
        best_y = [graph.vertices[i - 1][1] for i in best_path]
        ax.plot(best_x, best_y, 'r')
    
    canvas.draw()

def main():
    #Create graph instance
    global graph
    graph = Graph()

    #Open file and store vertices' coordinates
    file_name = "LongestPath16.txt"                 #Change filename here
    read_input_file(file_name, graph)

    #Initialize GUI window and graph
    root = tk.Tk()
    fig = matplotlib.figure.Figure(figsize=(10, 8))
    ax = fig.add_subplot()

    frame = tk.Frame(root)
    label = tk.Label(text = "Longest Path GUI")
    label.config(font=("Courier", 32))
    label.pack()


    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar = False)
    toolbar.update()
    toolbar.pack(anchor="w", fill = tk.X)

    frame.pack(fill=tk.BOTH, expand=True)

    #Plots the edges and vertices read from input file
    tk.Button(frame, text = "Plot graph", command=lambda: plot(graph, ax, canvas, best_path=False)).pack(pady = 10)
    #Opens popup for user inputs
    tk.Button(frame, text = "Prepare Graph Coloring Settings and Run", command=lambda: genetic_alg_popup(canvas, ax)).pack(pady = 10)

    root.mainloop()

    #Initializes and runs thread for GUI to run on.
    gui_thread = threading.Thread(target=root.mainloop)
    gui_thread.start()

if __name__ == "__main__":
    main()