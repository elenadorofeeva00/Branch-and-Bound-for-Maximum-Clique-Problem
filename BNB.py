from docplex.mp.model import Model
import numpy as np
import networkx as nx
import random
import math
import time
from collections import Counter
import csv

def ReadGraphFile(filename):
    with open(filename, 'r') as file:
        vertices = 0
        edges_num = 0
        edges = []
        for line in file:
            if line.startswith('c'):
                continue

            elif line.startswith('p'):
                _, name, vertices, edges_num = line.split()
                vertices = int(vertices)
                edges_num = int(edges_num)
            
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                v1 = int(v1)
                v2 = int(v2)
                edges.append((v1, v2))
        
        return nx.Graph(edges) 
    
class TabuSearch:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.c_border = 0
        self.q_border = 0
        self.qco = [0 for i in range(self.graph.number_of_nodes())]
        self.neighbour_sets = [set() for i in range(self.graph.number_of_nodes())]
        self.non_neighbours = [set() for i in range(self.graph.number_of_nodes())]
        self.index = [0 for i in range(self.graph.number_of_nodes())]
        self.best_clique = []
        self.tightness = [0 for i in range(self.graph.number_of_nodes())]
        self.add_tabu = []
        self.rem_tabu = []
        
    def RunSearch(self, starts):
        for node in range(self.graph.number_of_nodes()):
            for n in self.graph.neighbors(node + 1):
                self.neighbour_sets[node].add(n-1)
                
        for i in range(self.graph.number_of_nodes()):
            for j in range(self.graph.number_of_nodes()):
                if j not in self.neighbour_sets[i] and i != j:
                    self.non_neighbours[i].add(j)
                    
        for i in range(starts):
            self.ClearClique()
            for i in range(len(self.neighbour_sets)):
                self.qco[i] = i
                self.index[i] = i
            
            self.RunInitialHeuristic()
        
            self.c_border = self.q_border
            swaps = 0
        
            while swaps < len(self.neighbour_sets) * 100:
                if self.Move() == False:
                    if self.Swap1To1() == False:
                        break
                    else:
                        swaps += 1
                    
            if self.q_border > len(self.best_clique):
                self.best_clique.clear()
                for i in range(self.q_border):
                    self.best_clique.append(self.qco[i])
                
    def SwapVertices(self, vertex, border):
        vertex_at_border = self.qco[border]
        self.qco[self.index[vertex]], self.qco[border] = self.qco[border], self.qco[self.index[vertex]]
        self.index[vertex], self.index[vertex_at_border] = self.index[vertex_at_border], self.index[vertex]
            
    def InsertToClique(self, i):
        for j in self.non_neighbours[i]:
            if self.tightness[j] == 0:
                self.c_border -= 1
                self.SwapVertices(j, self.c_border)
            self.tightness[j] += 1
    
        self.SwapVertices(i, self.q_border)
        self.q_border += 1
            
    def RemoveFromClique(self, k):
        for j in self.non_neighbours[k]:
            if self.tightness[j] == 1:
                self.SwapVertices(j, self.c_border)
                self.c_border += 1

            self.tightness[j] -= 1
            
        self.q_border -= 1
        self.SwapVertices(k, self.q_border)
    
    def Swap1To1(self):
        for counter in range(self.q_border):
            vertex = self.qco[counter]
            if vertex in self.add_tabu:
                continue
            for i in self.non_neighbours[vertex]:
                if self.tightness[i] == 1 and i not in self.rem_tabu:
                    self.RemoveFromClique(vertex)
                    self.rem_tabu.append(vertex)
                    self.add_tabu.append(i)
                    self.InsertToClique(i)
                    return True
            
        return False

    def Move(self):
        if self.c_border == self.q_border:
            return False
    
        vertex = self.qco[self.q_border]
        self.InsertToClique(vertex)
        return True

    def RunInitialHeuristic(self):
    
        graph = self.graph
    
        vertices = sorted(graph, key=lambda x: len(graph[x]), reverse=True)
    
        best_clique = []
    
        for i in range(len(vertices)):
    
            clique = []
        
            #выбираем рандомную вершину из 20% первых вершин, добавляем ее в клику
            index = random.randint(0, int(len(vertices)/5)) 
            vertex = vertices[index]
            clique.append(vertex)
            
            candidates = [n for n in graph.neighbors(vertex)]
                    
            while len(candidates) != 0:
                #выбираем рандомного кандидата
                index = random.randint(0, int(len(candidates))-1)
                vertex = candidates[index]
                clique.append(vertex)
                
                #ищем кандидатов
                list_of_candidates = []
                neighbours_of_vertex = [n for n in graph.neighbors(vertex)] 
                for ver in candidates:
                    if ver in neighbours_of_vertex:
                        list_of_candidates.append(ver)

                candidates = list_of_candidates
                    
            #проверка на лучшую клику
            if len(clique) > len(best_clique):
                best_clique = clique
            
                
        for vertex in best_clique:
            #print(self.q_border)
            self.SwapVertices(vertex-1, self.q_border)
            self.q_border += 1
        
    def Check(self):
        for i in self.best_clique:
            for j in self.best_clique:
                if i != j and j not in self.neighbour_sets[i]:
                    print("Returned subgraph is not clique\n")
                    return False
            
        return True

    def ClearClique(self):
        self.q_border = 0
        self.c_border = 0

class MaxClique:
    def __init__(self, graph: nx.Graph, randomization):
        self.model = Model("model")
        self.constraints = []
        self.eps = 10e-6
        self.graph = graph
        self.randomization = randomization
        self.best_found_solution = []
        self.time_h = 0
        self.best_f = len(self.best_found_solution)
        self.start_bnb = 0
        self.variables = []
        self.variables_names = []
        self.used_variables = [0 for i in range(graph.number_of_nodes())]
        self.count = 0
        self.branch_num = [0 for i in range(graph.number_of_nodes())]
        
    def neighbours_list(self, vertice):
        return [n for n in self.graph.neighbors(vertice)] 
    
    def Check(self, solution):
        counter = Counter(solution)
    
        if sum(counter.values()) > len(counter):
            print("Duplicated vertices in the clique\n")
            return False
    
        for i in solution:
        
            for j in solution:
            
                if i != j and j not in self.neighbours_list(i):
                    print("Returned subgraph is not a clique\n")
                    return False
            
        return True
        
    def heuristic(self): 
        start = time.time()
        
        graph = self.graph
    
        vertices = sorted(graph, key=lambda x: len(graph[x]), reverse=True)
    
        best_clique = []
    
        for i in range(self.randomization):
    
            clique = []
        
            #выбираем рандомную вершину из 20% первых вершин, добавляем ее в клику
            index = random.randint(0, int(len(vertices)/5)) 
            vertex = vertices[index]
            clique.append(vertex)
            
            candidates = self.neighbours_list(vertex)
                    
            while len(candidates) != 0:
                #выбираем рандомного кандидата
                index = random.randint(0, int(len(candidates))-1)
                vertex = candidates[index]
                clique.append(vertex)
                
                #ищем кандидатов
                list_of_candidates = []
                neighbours_of_vertex = self.neighbours_list(vertex)
                for ver in candidates:
                    if ver in neighbours_of_vertex:
                        list_of_candidates.append(ver)

                candidates = list_of_candidates
                    
            if self.Check(clique) is True:
                #проверка на лучшую клику
                if len(clique) > len(best_clique):
                    best_clique = clique
            
        time_h = time.time() - start
        print("found heuristic")
        return best_clique, time_h
      
 
    def biggest_ind_sets(self):
        graph = nx.complement(self.graph)
    
        vertices = sorted(graph, key=lambda x: len(graph[x]), reverse=True)
    
        best_clique = []
    
        for i in range(len(vertices) // 5):
    
            clique = []
        
            #выбираем рандомную вершину из 20% первых вершин, добавляем ее в клику
            index = random.randint(0, len(vertices) // 5) 
            vertex = vertices[index]
            clique.append(vertex)
            
            candidates = self.neighbours_list(vertex)
                    
            while len(candidates) != 0:
                #выбираем рандомного кандидата
                index = random.randint(0, int(len(candidates))-1)
                vertex = candidates[index]
                clique.append(vertex)
                
                #ищем кандидатов
                list_of_candidates = []
                neighbours_of_vertex = self.neighbours_list(vertex)
                for ver in candidates:
                    if ver in neighbours_of_vertex:
                        list_of_candidates.append(ver)

                candidates = list_of_candidates
                    
            if self.Check(clique) is True:
                #проверка на лучшую клику
                if len(clique) > len(best_clique):
                    best_clique = clique
            
        return best_clique
    
    def find_ind_sets(self):
        graph = self.graph
        
        ind_sets = []
    
        strategies = [nx.coloring.strategy_largest_first,
                      nx.coloring.strategy_random_sequential,
                      nx.coloring.strategy_independent_set,
                      nx.coloring.strategy_connected_sequential_bfs,
                      nx.coloring.strategy_connected_sequential_dfs,
                      nx.coloring.strategy_saturation_largest_first
                      ]     
    
        for strategy in strategies:
            colored_set = nx.greedy_color(graph, strategy = strategy)
            for color in set(color for node, color in colored_set.items()):
                ind_sets.append([node for node, value in colored_set.items() if value == color])
                
        return ind_sets
    
    def coloring(self):
        graph = self.graph
        
        color_map = {} #словарь с цветами, ключ - вершина, значение - цвет
 
        #проходим по каждой вершине в списке, отсортированном по убыванию степеней вершин
        for node in sorted(graph, key=lambda x: len(graph[x]), reverse=True):
            neighbour_colors = set(color_map.get(neigh) for neigh in self.neighbours_list(node)) #множество цветов соседей вершины node
            for color in range(1, graph.number_of_nodes()): #присваиваем вершине первый свободный цвет
                if color not in neighbour_colors:
                    color_map[node] = color
                    break
        
        colors = []
        
        for color in set(color for node, color in color_map.items()):
            colors.append([node for node, value in color_map.items() if value == color]) 
            
        return colors
        
        
    def create_model(self):
        graph = self.graph
        num_nodes = graph.number_of_nodes()
        self.variables = self.model.continuous_var_list(range(num_nodes), lb=0, ub=1)
        #self.variables = [vars for vars in self.model.iter_variables()]
        self.model.maximize(self.model.sum(self.variables))
        
        #adding basic constraints
        non_edges = list(nx.non_edges(graph))
        for edge in non_edges:
            self.model.add_constraint(self.variables[edge[0]-1] + self.variables[edge[1]-1] <= 1)
            
        #adding stronger constraints
    
        #добавляю встроенную раскраску
        ind_sets = self.find_ind_sets()
        for ind_set in ind_sets:
            self.model.add_constraint(np.sum([self.variables[i-1] for i in ind_set]) <= 1)
        
        ind_sets = self.coloring()
        for ind_set in ind_sets:
            self.model.add_constraint(np.sum([self.variables[i-1] for i in ind_set]) <= 1)
        
        for i in range(50):
            ind_set = self.biggest_ind_sets()
            self.model.add_constraint(np.sum([self.variables[i-1] for i in ind_set]) <= 1)
            
        print("create model")
    
    def is_integer(self, solution):
        for value in solution:
            if abs(value - np.round(value)) >= self.eps:
                return False
        return True
            
    def branching(self, solution):
        #choosing the branching variable by finding the closest to the integer
        min_dist = 1
        closest_to_int_node = None
        for node, value in enumerate(solution):
            if abs(value - np.round(value)) >= self.eps and self.used_variables[node] == 0:
                curr_min_dist = abs(value - np.round(value))
                if curr_min_dist < min_dist:
                    closest_to_int_node = node
                    min_dist = abs(value - 1)
        return closest_to_int_node
    
    def branches(self, branching_name, solution):
        #choosing the branch with the greatest value
        
        if solution[branching_name] < 0.5:
            return [0, 1]
        else:
            return [1, 0]
        
    def add_constraint(self, var_name, var_value, sign, label):
        if sign == "less":
            self.model.add_constraint(self.variables[var_name] <= var_value, str(label))
            
        elif sign == "more":
            self.model.add_constraint(self.variables[var_name] >= var_value, str(label))
            
    def delete_constraint(self, label):
        self.model.remove_constraint(str(label))
        
    def BnB(self):
        end = time.time()
        
        if end - self.start_bnb > 7200:
            print("IT'S TAKING TOO LONG")
            return
        
        solve = self.model.solve(log_output=False)
        
        if solve is None:
            print("ERROR")
            return
        
        #if the value is not in the solution, it will return 0
        solution = [solve.get_value(i) for i in self.variables]
        f = solve.objective_value
        
        upper_bound = math.floor(f)
        
        if self.best_f >= upper_bound:
            return
            
        if self.is_integer(solution) is True:
            if self.Check([i + 1 for i in range(len(solution)) if solution[i] != 0]) is False:
                return
            
            if self.best_f < int(f):
                solution_name = [i for i in range(len(solution)) if solution[i] != 0]
                self.best_found_solution = solution_name
                self.best_f = int(f)
                return
        
        branching_name = self.branching(solution)
        
        if branching_name is None:
            return
        
        for branch in self.branches(branching_name, solution):
            if branch == 0:
                self.count += 1
                count = self.count
                self.used_variables[branching_name] = 1
                self.add_constraint(branching_name, branch, "less", count)
                self.BnB()
                self.used_variables[branching_name] = 0
                self.delete_constraint(count)
                
            elif branch == 1:
                self.count += 1
                count = self.count
                self.add_constraint(branching_name, branch, "more", count)
                self.used_variables[branching_name] = 1
                self.BnB()
                self.used_variables[branching_name] = 0
                self.delete_constraint(count)
                
        return
    
    def get_solution(self):
        return self.best_found_solution, self.best_f
    
def main():
    EASY = ["c-fat200-1.clq",
        "c-fat200-2.clq",
        "c-fat200-5.clq",
        "c-fat500-1.clq",
        "c-fat500-10.clq",
        "c-fat500-2.clq",
        "c-fat500-5.clq",
        "johnson8-2-4.clq",
        "johnson8-4-4.clq",
        "hamming6-2.clq",
        "hamming6-4.clq",
        "hamming8-2.clq",
        "MANN_a9.clq",
        "san200_0.7_1.clq",
        "san200_0.9_1.clq"]
    MODERATE = ["gen200_p0.9_55.clq",
        "johnson16-2-4.clq",
        "hamming8-4.clq", 
        "san200_0.9_2.clq"]
    HARD = ["brock200_1.clq", 
        "brock200_2.clq", 
        "brock200_3.clq", 
        "brock200_4.clq",
        "C125.9.clq", 
        "gen200_p0.9_44.clq", 
        "keller4.clq", 
        "MANN_a27.clq", 
        #"MANN_a45.clq", #не посчитал
        "p_hat300-1.clq",
        "p_hat300-2.clq",
        "p_hat300-3.clq",
        "san200_0.7_2.clq",
        "san200_0.9_3.clq",
        "sanr200_0.7.clq"
        ]
 
    with open("BnB_EASY_final.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter = ";", lineterminator="\r")
        file_writer.writerow(["Instance", "Heuristic time (sec)", "BnB time (sec)", "Clique size", "Clique vertices"])
    
        print("Instance", "Heuristic time (sec)", "BnB time (sec)", "Clique size", "Clique vertices")
        
        for file in EASY:
            graph = ReadGraphFile(file)
            
            tabu = TabuSearch(graph)
            start_h = time.time()
            tabu.RunSearch(100)
            time_h = time.time() - start_h
            print(tabu.best_clique, len(tabu.best_clique))
        
            bnb = MaxClique(graph, graph.number_of_nodes() // 2)    
            bnb.time_h = time_h
            bnb.best_found_solution = tabu.best_clique
            bnb.best_f = len(bnb.best_found_solution)            
            bnb.create_model()
            start_bnb = time.time()
            bnb.start_bnb = start_bnb
            bnb.BnB()
            time_bnb = time.time() - start_bnb
        
            solution, f = bnb.get_solution()
            time_h = bnb.time_h
            file_writer.writerow([file, time_h, time_bnb, int(f), solution, "\n"])
            print("Instance: ", file, "; Heuristic time (sec): ", time_h, "; BnB time (sec): ", time_bnb, "; Clique size: ", int(f), "; Clique vertices: ", solution)
            
    with open("BnB_MEDIUM_final.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter = ";", lineterminator="\r")
        file_writer.writerow(["Instance", "Heuristic time (sec)", "BnB time (sec)", "Clique size", "Clique vertices"])
    
        print("Instance", "Heuristic time (sec)", "BnB time (sec)", "Clique size", "Clique vertices")
            
        for file in MODERATE:
            graph = ReadGraphFile(file)
            
            tabu = TabuSearch(graph)
            start_h = time.time()
            tabu.RunSearch(100)
            time_h = time.time() - start_h
            
        
            bnb = MaxClique(graph, graph.number_of_nodes() * 20)    
            bnb.time_h = time_h
            bnb.best_found_solution = tabu.best_clique
            bnb.best_f = len(bnb.best_found_solution)            
            bnb.create_model()
            start_bnb = time.time()
            bnb.start_bnb = start_bnb
            bnb.BnB()
            time_bnb = time.time() - start_bnb
        
            solution, f = bnb.get_solution()
            time_h = bnb.time_h
            file_writer.writerow([file, time_h, time_bnb, int(f), solution, "\n"])
            print("Instance: ", file, "; Heuristic time (sec): ", time_h, "; BnB time (sec): ", time_bnb, "; Clique size: ", int(f), "; Clique vertices: ", solution)
           
    with open("BnB_HARD_final.csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter = ";", lineterminator="\r")
        file_writer.writerow(["Instance", "Heuristic time (sec)", "BnB time (sec)", "Clique size", "Clique vertices"])
    
        print("Instance", "Heuristic time (sec)", "BnB time (sec)", "Clique size", "Clique vertices")
            
        for file in HARD:
            graph = ReadGraphFile(file)
            
            tabu = TabuSearch(graph)
            start_h = time.time()
            tabu.RunSearch(100)
            time_h = time.time() - start_h
            print(tabu.best_clique, len(tabu.best_clique))
        
            bnb = MaxClique(graph, graph.number_of_nodes() * 70)
            bnb.time_h = time_h
            bnb.best_found_solution = tabu.best_clique
            bnb.best_f = len(bnb.best_found_solution)
            bnb.create_model()
            start_bnb = time.time()
            bnb.start_bnb = start_bnb
            bnb.BnB()
            time_bnb = time.time() - start_bnb
        
            solution, f = bnb.get_solution()
            time_h = bnb.time_h
            file_writer.writerow([file, time_h, time_bnb, int(f), solution, "\n"])
            print("Instance: ", file, "; Heuristic time (sec): ", time_h, "; BnB time (sec): ", time_bnb, "; Clique size: ", int(f), "; Clique vertices: ", solution)
            
        
        
if __name__ == '__main__':
    main()
        
        
    