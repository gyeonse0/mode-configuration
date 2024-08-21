from RouteGenerator import *
from FileReader import *
from MultiModalState import *
from sklearn.ensemble import RandomForestRegressor
import random
import itertools
import config

file_reader = FileReader()

vrp_file_path = config.vrp_file_path


data = file_reader.read_vrp_file(vrp_file_path)

globals
IDLE = 0 # 해당 노드에 드론이 트럭에 업힌 상태의 경우
FLY = 1 # 해당 노드에서 트럭이 드론의 임무를 위해 드론을 날려주는 경우
ONLY_DRONE = 2 # 해당 노드에 드론만이 임무를 수행하는 서비스 노드인 경우
CATCH = 3 # 해당 노드에서 트럭이 임무를 마친 드론을 받는 경우
ONLY_TRUCK = 4 # 해당 노드에서 트럭만이 임무를 수행하는 경우 (드론이 업혀있지 않음)
NULL = None # 해당 노드가 전체 ROUTES에 할당이 안되어 있는 경우 (SOLUTION에서 UNVISITED 한 노드)

class Repair():


    def random_repair(self, state, rnd_state):
        random_repair = MultiModalState(state.routes, state.unassigned)
        routes =  random_repair.routes
        unassigned =  random_repair.unassigned
        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route_t, best_idx_t = self.truck_best_insert(customer, routes)
            best_route_d, best_idx_d = self.drone_best_insert(customer, routes)

            if best_route_t is not None and best_idx_t is not None and best_route_d is not None and best_idx_d is not None:
                chosen_route = random.choice([best_route_t, best_route_d])
                if chosen_route == best_route_d:
                    for i, route in enumerate(routes):
                        if route == best_route_d:
                            routes[i] = route[:best_idx_d] + [customer] + route[best_idx_d:]
                            self.drone_repair_visit_type_update(routes)
                else:
                    for i, route in enumerate(routes):
                        if route == best_route_t:
                            routes[i] = route[:best_idx_t] + [customer] + route[best_idx_t:]
                            self.truck_repair_visit_type_update(routes)

            elif best_route_t is None and best_idx_t is None and best_route_d is not None and best_idx_d is not None:
                for i, route in enumerate(routes):
                        if route == best_route_d:
                            routes[i] = route[:best_idx_d] + [customer] + route[best_idx_d:]
                            self.drone_repair_visit_type_update(routes)

            elif best_route_t is not None and best_idx_t is not None and best_route_d is None and best_idx_d is None:
                for i, route in enumerate(routes):
                        if route == best_route_t:
                            routes[i] = route[:best_idx_t] + [customer] + route[best_idx_t:]
                            self.truck_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

            self.drone_repair_visit_type_update(routes) #최종적으로 visit_type 검사
            self.truck_repair_visit_type_update(routes)
            self.route_duplication_check(routes)

            state_after_random_repair = MultiModalState(routes, unassigned)
        
            if len(state_after_random_repair.unassigned) > 0:
                return self.greedy_truck_repair(state_after_random_repair, rnd_state)
            else:
                return state_after_random_repair


    def light_drone_repair(self, state, rnd_state):
        light_first_repair = MultiModalState(state.routes, state.unassigned)
        routes =  light_first_repair.routes
        unassigned =  light_first_repair.unassigned
        condition = lambda x: data["logistic_load"][x[0]] < data["cargo_limit_drone"] 
        
        unassigned_light, unassigned = self.extract_and_remain(unassigned, condition)

        while len(unassigned_light) > 0:
            customer_light = random.choice(unassigned_light)
            unassigned_light.remove(customer_light)
            best_route, best_idx = self.drone_best_insert(customer_light, routes)
                
            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer_light] + route[best_idx:]
                        self.drone_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

        self.drone_repair_visit_type_update(routes) #최종적으로 visit_type 검사
        self.route_duplication_check(routes)

        state_after_light_repair = MultiModalState(routes, unassigned)
        
        if len(state_after_light_repair.unassigned) > 0:
            return self.greedy_truck_repair(state_after_light_repair,rnd_state)
        else:
            return state_after_light_repair


    def heavy_truck_repair(self, state, rnd_state):
        heavy_first_repair = MultiModalState(state.routes, state.unassigned)
        routes = heavy_first_repair.routes
        unassigned = heavy_first_repair.unassigned
        condition = lambda x: data["logistic_load"][x[0]] > data["cargo_limit_drone"] 
        
        unassigned_heavy, unassigned = self.extract_and_remain(unassigned, condition)

        while len(unassigned_heavy) > 0:
            customer_heavy = random.choice(unassigned_heavy)
            unassigned_heavy.remove(customer_heavy)
            best_route, best_idx = self.truck_best_insert(customer_heavy, routes)
                
            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer_heavy] + route[best_idx:]
                        self.truck_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

        self.truck_repair_visit_type_update(routes) #최종적으로 visit_type 검사
        self.route_duplication_check(routes)

        state_after_heavy_repair = MultiModalState(routes, unassigned)
        
        if len(state_after_heavy_repair.unassigned) > 0:
            return self.greedy_truck_repair(state_after_heavy_repair, rnd_state)
        else:
            return state_after_heavy_repair
    

    def drone_first_truck_second(self, state, rnd_state):

        state_after_drone_repair = self.greedy_drone_repair(MultiModalState(state.routes, state.unassigned),rnd_state)

        if len(state_after_drone_repair.unassigned) > 0:
            return self.greedy_truck_repair(state_after_drone_repair,rnd_state)
        else:
            return state_after_drone_repair
    

    def genetic_feasibility(self, data, route):
        new_route = route
        
        # max system duration 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['eVTOL'][0] > data["maximum_system_duration"]:
            return False
        
        # max system duration 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['eTruck'][0] > data["maximum_system_duration"]:
            return False

        #max waiting time 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['Over Waiting Time']:
           return False
        
        drone_soc = self.soc_calculate(new_route)[1]
        for soc in drone_soc:
            for value in soc:
                if value < data["min_soc_d"]:
                    return False
                
        truck_soc = self.soc_calculate(new_route)[0]
        for soc in truck_soc:
            for value in soc:
                if value < data["min_soc_t"]:
                    return False
                
        return True


    def initialize_population(self, elements, population_size):
        # 초기 집단 생성: 랜덤으로 생성된 경로들
        population = []
        for _ in range(population_size):
            individual = self.random_route(elements)
            population.append(individual)
        return population
    
    def random_route(self, elements):
        # 랜덤한 경로를 생성하는 방법 (예: 초기 해에서 셔플)
        route = list(elements)  # 요소를 랜덤하게 변형
        random.shuffle(route)
        return route

    def evaluate_fitness_truck(self, individual, start_node, end_node):
        # 트럭의 적합도 평가 함수: 경로의 총 거리를 기반으로 평가
        route = [start_node] + individual + [end_node]
        distance = 0
        for i in range(len(route) - 1):
            edge_weight = (data["edge_km_t"][route[i]][route[i+1]])*data["curvy_road"]
            distance += edge_weight
        return 1000 / distance
    
    def evaluate_fitness_drone(self, individual, start_node, end_node):
        # 트럭의 적합도 평가 함수: 경로의 총 거리를 기반으로 평가
        route = [start_node] + individual + [end_node]
        distance = 0
        for i in range(len(route) - 1):
            edge_weight = (data["edge_km_d"][route[i]][route[i+1]])*data["curvy_air"]
            distance += edge_weight
        return 1000 / distance
    
    def select_top_two(self, population, fitness_scores):
        # Combine population and fitness_scores into a list of tuples
        population_with_fitness = list(zip(population, fitness_scores))
        
        # Sort the list by fitness_scores in descending order
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top two individuals with the highest fitness
        top_two_individuals = [individual for individual, fitness in population_with_fitness[:2]]
        
        return top_two_individuals
    
    def roulette_wheel_selection(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(population, k=len(population))
        
        selection_probs = [fitness / total_fitness for fitness in fitness_scores]
        selected_individuals = random.choices(population, weights=selection_probs, k=len(population))
        return selected_individuals
    
    def crossover_and_mutate(self, selected_individuals, mutation_rate):
    # 교차 및 변이 연산을 통해 자손 생성
        offspring = []
        
        # 선택된 개체의 수가 2 미만이면 교차를 할 수 없으므로 예외 처리
        if len(selected_individuals) < 2:
            offspring.extend(selected_individuals)  # 선택된 개체들을 offspring 리스트에 추가
            return offspring  # offspring 리스트 반환
        
        while len(offspring) < len(selected_individuals):
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = self.crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = self.mutate(child)
            offspring.append(child)
        
        # 만약 홀수 개의 offspring이 필요한 경우, 마지막에 하나를 더 추가 (복제 또는 변이)
        if len(offspring) < len(selected_individuals):
            last_child = offspring[-1]
            if random.random() < mutation_rate:
                last_child = self.mutate(last_child)
            offspring.append(last_child)
        
        return offspring

    def crossover(self, parent1, parent2):
        size = len(parent1)
        # 두 부모의 임의의 부분을 교환하여 자식 생성
        if size < 2:
            # 크기가 2 미만일 경우, 교차하지 않고 부모 중 하나를 자식으로 사용
            return parent1 if random.random() < 0.5 else parent2
        else:
            start, end = sorted(random.sample(range(size), 2))
            
            child = [None] * size
            child[start:end+1] = parent1[start:end+1]
            
            current_pos = (end + 1) % size
            for i in range(size):
                if parent2[i] not in child:
                    child[current_pos] = parent2[i]
                    current_pos = (current_pos + 1) % size
        
        return child
    
    def mutate(self, individual):
        # 변이 연산: 랜덤으로 두 노드를 교환
        if len(individual) < 2:
            return individual
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def run_ga_truck(self, elements, start_node, end_node, population_size, generations, mutation_rate):
        # 유전 알고리즘 실행
        population = self.initialize_population(elements, population_size)
        #print(population)

        best_individual = min(population, key=lambda x: self.evaluate_fitness_truck(x, start_node, end_node))
        best_fitness = self.evaluate_fitness_truck(best_individual, start_node, end_node)

        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness_truck(individual, start_node, end_node) for individual in population]
            selected_individuals = self.select_top_two(population, fitness_scores)
            offspring = self.crossover_and_mutate(selected_individuals, mutation_rate)
            population = offspring
            
            current_best_individual = min(population, key=lambda x: self.evaluate_fitness_truck(x, start_node, end_node))
            current_best_fitness = self.evaluate_fitness_truck(current_best_individual, start_node, end_node)
            
            if current_best_fitness > best_fitness:
                best_individual = current_best_individual
                best_fitness = current_best_fitness
            
            #print(f"Truck GA Generation {generation}: Best fitness = {self.evaluate_fitness_truck(best_individual, start_node, end_node)}")

        return best_individual
    
    def run_ga_drone(self, elements, start_node, end_node, population_size, generations, mutation_rate):
        population = self.initialize_population(elements, population_size)
        #print(population)

        best_individual = min(population, key=lambda x: self.evaluate_fitness_drone(x, start_node, end_node))
        best_fitness = self.evaluate_fitness_drone(best_individual, start_node, end_node)

        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness_drone(individual, start_node, end_node) for individual in population]
            selected_individuals = self.select_top_two(population, fitness_scores)
            offspring = self.crossover_and_mutate(selected_individuals, mutation_rate)
            population = offspring
            
            current_best_individual = min(population, key=lambda x: self.evaluate_fitness_drone(x, start_node, end_node))
            current_best_fitness = self.evaluate_fitness_drone(current_best_individual, start_node, end_node)
            
            if current_best_fitness > best_fitness:
                best_individual = current_best_individual
                best_fitness = current_best_fitness
            
            #print(f"Truck GA Generation {generation}: Best fitness = {self.evaluate_fitness_truck(best_individual, start_node, end_node)}")

        return best_individual

    def genetic_algorithm(self, state, population_size, generations, mutation_rate):
        ga = MultiModalState(state.routes, state.unassigned)
        routes = ga.routes
        unassigned = ga.unassigned
        current_ofv = MultiModalState(routes, unassigned).cost_objective()

        for i, route in enumerate(routes):

            start_index =0
            end_index = len(route)-1

            start_node = route[start_index][0]
            end_node = route[end_index][0]

            only_idles=[]
            for j in range(start_index + 1, end_index):
                only_idles.append(route[j][0])
            
            best_idles = self.run_ga_truck(only_idles, start_node, end_node, population_size, generations, mutation_rate)
                        
            first_route = route[:start_index + 1] + [(truck, IDLE) for truck in best_idles] + route[end_index:]

            if self.genetic_feasibility(data, first_route):
                if MultiModalState(routes, unassigned).cost_objective() < current_ofv:
                    routes[i] = first_route
                    current_ofv = MultiModalState(routes, unassigned).cost_objective()
                else:
                    routes[i] = route
            else:
                routes[i] = route

        return MultiModalState(routes, unassigned)

    
    def truck_first_drone_second(self, state, rnd_state): #수정완료
        
        truck_first_drone_second_repairs = MultiModalState(state.routes, state.unassigned)
        routes = truck_first_drone_second_repairs.routes
        unassigned = truck_first_drone_second_repairs.unassigned

        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route, best_idx = self.truck_best_insert(customer, routes)

            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer] + route[best_idx:]
                        self.truck_repair_visit_type_update(routes)
            
            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

        self.truck_repair_visit_type_update(routes)
        self.route_duplication_check(routes)

        state_after_truck_repairs = MultiModalState(routes,unassigned) #unassigned 여러개일수도

        if len(state_after_truck_repairs.unassigned)>0:
            return self.drone_first_truck_second(state_after_truck_repairs,rnd_state)
        else:
            return state_after_truck_repairs


    def greedy_truck_repair(self, state, rnd_state): #수정완료
        truck_repair = MultiModalState(state.routes, state.unassigned)
        routes = truck_repair.routes
        unassigned = truck_repair.unassigned

        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route, best_idx = self.truck_best_insert(customer, routes)

            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer] + route[best_idx:]
                        self.truck_repair_visit_type_update(routes)

            else: 
                if not any(route == [(0, 0), (0, 0)] for route in routes):
                    # routes 뒤에 새로운 route 추가
                    routes.append([(0, 0), customer, (0, 0)])
                else:
                    for i, route in enumerate(routes):
                        if route == [(0, 0), (0, 0)]:
                            # 빈 route에 고객 추가
                            routes[i] = [(0, 0), (customer[0],0), (0, 0)]

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]


        self.truck_repair_visit_type_update(routes)
        self.route_duplication_check(routes)
        
        return MultiModalState(routes,unassigned)

    
    def greedy_drone_repair(self, state, rnd_state):
        drone_repair = MultiModalState(state.routes, state.unassigned)
        routes = drone_repair.routes
        unassigned = drone_repair.unassigned

        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route, best_idx = self.drone_best_insert(customer, routes)    
            
            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer] + route[best_idx:]
                        self.drone_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]
                                        
        self.drone_repair_visit_type_update(routes) #최종적으로 visit_type 검사
        self.route_duplication_check(routes)

        return MultiModalState(routes, unassigned)
    

    def drone_best_insert(self, customer, routes):
        best_cost, best_route, best_idx = None, None, None

        for route in routes:   
            for idx in range(1, len(route)):  # depot을 제외한 index를 고려해줘야함
                if self.drone_can_insert(data, customer, route, idx): # 삽입 가능한 경우
                    cost = self.drone_insert_cost(customer, route, idx, routes)
                    if best_cost is None or cost < best_cost:
                        best_cost, best_route, best_idx = cost, route, idx

        return best_route, best_idx


    def drone_randomize_greedy_insert(self, customer, routes):
        best_costs = [None, None, None]
        best_routes = [None, None, None]
        best_idxs = [None, None, None]

        for route in routes:
            for idx in range(1, len(route)):
                if self.drone_can_insert(data, customer, route, idx):  # 삽입 가능한 경우
                    cost = self.drone_insert_cost(customer, route, idx, routes)

                    for i in range(3):
                        if best_costs[i] is None or cost < best_costs[i]:
                            best_costs.insert(i, cost)
                            best_routes.insert(i, route)
                            best_idxs.insert(i, idx)
                            break

        # Randomly select one of the top three costs
        selected_index = random.randint(0, 2)

        return best_routes[selected_index], best_idxs[selected_index]


    def truck_randomize_greedy_insert(self, customer, routes):
        best_costs = [None, None, None]
        best_routes = [None, None, None]
        best_idxs = [None, None, None]

        for route in routes:
            for idx in range(1, len(route)):
                if self.truck_can_insert(data, customer, route, idx):  # 삽입 가능한 경우
                    cost = self.truck_insert_cost(customer, route, idx, routes)

                    for i in range(3):
                        if best_costs[i] is None or cost < best_costs[i]:
                            best_costs.insert(i, cost)
                            best_routes.insert(i, route)
                            best_idxs.insert(i, idx)
                            break

        # Randomly select one of the top three costs
        selected_index = random.randint(0, 2)

        return best_routes[selected_index], best_idxs[selected_index]
        
    def drone_can_insert(self, data, customer, route, idx): 
        
        # 해당 노드의 Landing spot의 availability가 1이 아닐 때 False
        if data["availability_landing_spot"][customer[0]] == 0:
            return False

        # 해당 고객의 Drone delivery preference가 1이 아닐 때 False
        if data["customer_drone_preference"][customer[0]] == 0:
            return False
        
        new_route = route[:idx] + [customer] + route[idx:]
        self.drone_repair_visit_type_update_route(new_route)
        
        drone_path = [value for value in new_route if value[1] != ONLY_TRUCK]  
        truck_path = [value for value in new_route if value[1] != ONLY_DRONE]

        # 해당 고객들의 logistic_load가 cargo_limit_drone을 넘으면 False
        flag = 0
        total_load = 0
        for i in range(1,len(drone_path)):
            if drone_path[i-1][1]==FLY and drone_path[i][1] == ONLY_DRONE:
                start_index = i
                flag = 1 #fly를 만났어
                j = i + 1
                while flag == 1 and j < len(drone_path):
                    if drone_path[j][1] == ONLY_DRONE:
                        flag = 1
                        j += 1
                    elif drone_path[j][1] != ONLY_DRONE:
                        flag = 0
                        end_index = j
                    
                for k in range(start_index, end_index):
                    total_load += data["logistic_load"][drone_path[k][0]]
                if total_load > data["cargo_limit_drone"]:
                    return False
                total_load = 0

        # max system duration 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['eVTOL'][0] > data["maximum_system_duration"]:
            return False
        
        #max waiting time 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['Over Waiting Time']:
           return False
        
        drone_soc = self.soc_calculate(new_route)[1]
        for soc in drone_soc:
            for value in soc:
                if value < data["min_soc_d"]:
                    return False
                
        truck_soc = self.soc_calculate(new_route)[0]
        for soc in truck_soc:
            for value in soc:
                if value < data["min_soc_t"]:
                    return False
        
        return True
    
    def current_logistic_load(self, route):
        d_route = [value for value in route if value[1] != ONLY_TRUCK]
        drone_route = [value for value in d_route if value[1] != IDLE]

        flag = 0
        current_logistic_load = 0
        current_logistic_loads = []
        start_index = None  # 초기화

        j = 0
        while j < len(drone_route):
            if flag == 0 and (drone_route[j][1] == FLY):
                start_index = j
                flag = 1

            elif (drone_route[j][1] == CATCH or drone_route[j][1] == FLY) and flag == 1:
                for k in range(start_index, j):
                    if drone_route[k][1] == ONLY_DRONE:
                        current_logistic_load += data["logistic_load"][drone_route[k][0]]

                current_logistic_loads.append((drone_route[start_index][0], current_logistic_load))

                for l in range(start_index+1, j):
                    if drone_route[l+1][1] == ONLY_DRONE:
                        current_logistic_load -= data["logistic_load"][drone_route[l+1][0]]
                        current_logistic_loads.append((drone_route[l][0], current_logistic_load))

                    elif drone_route[l+1][1] == CATCH:
                        current_logistic_load = 0
                        current_logistic_loads.append((drone_route[l][0], current_logistic_load))
                        flag = 0

                    elif drone_route[l+1][1] == FLY:
                        current_logistic_load = 0
                        current_logistic_loads.append((drone_route[l][0], current_logistic_load))
                        flag = 0
                        j = l  # 현재 위치로 이동하여 다시 시작
                        break

                if flag == 0:
                    continue
            j += 1

        for node in route:
                if node[0] not in [item[0] for item in current_logistic_loads]:
                    current_logistic_loads.append((node[0], 0))

        return current_logistic_loads

    def detail_drone_modeling(self, current_logistic_load, edge_time):
        drone_consumption = ((((data["mass_d"] + current_logistic_load) * data["speed_d"] * 60) / (370 * data["lift_to_drag"] * data["power_motor_prop"])) + data["power_elec"]) * edge_time
        return drone_consumption

    def soc_calculate(self, route):
        truck_soc = []
        drone_soc = []
        truck_energy_consumption = 0
        drone_energy_consumption = 0
        truck_kwh = data["battery_kwh_t"]
        drone_kwh = data["battery_kwh_d"]
        truck_ofvs = [truck_kwh]
        drone_ofvs = [drone_kwh]
        jump = 0 #트럭 
        flee = 0 #드론
        fly=0
        current_logistic_loads = self.current_logistic_load(route)
        
        for idx in range(1, len(route)):
            if route[idx][1] == CATCH or ((route[idx][1] == FLY) and (route[idx-1][1] in [ONLY_TRUCK, ONLY_DRONE])):
                
                truck_energy_consumption = (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) * data["energy_kwh/km_t"]
                
                drone_time = (((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"])))/((data["speed_d"]))/60
                drone_distance = data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]
                    
                for n, w in current_logistic_loads:
                    if n == route[idx-flee-1][0]:
                        weight = w
                        break

                drone_energy_consumption_per_meter = self.detail_drone_modeling(weight,drone_time) / (drone_distance*1000)
                drone_energy_consumption = self.detail_drone_modeling(weight,drone_time)+ drone_energy_consumption_per_meter * 0.96 * data["altitude"]
                
                flee = 0
                jump = 0
                truck_kwh -= truck_energy_consumption
                drone_kwh -= drone_energy_consumption
                truck_ofvs.append(truck_kwh)
                drone_ofvs.append(drone_kwh)

            elif route[idx][1] == ONLY_DRONE:
                drone_time = (((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"])))/((data["speed_d"]))/60
                drone_distance = data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]
                    
                for n, w in current_logistic_loads:
                    if n == route[idx-flee-1][0]:
                        weight = w
                        break

                if fly==1:
    
                    drone_energy_consumption_per_meter = self.detail_drone_modeling(weight,drone_time) / (drone_distance*1000)
                    drone_energy_consumption = drone_energy_consumption_per_meter * 1.18 * data["altitude"] + self.detail_drone_modeling(weight,drone_time)

                    flee = 0
                    drone_kwh -= drone_energy_consumption
                    truck_ofvs.append(truck_kwh)
                    drone_ofvs.append(drone_kwh)
                    jump += 1
                    fly = 0
                
                elif fly==0:
                    drone_energy_consumption_per_meter = self.detail_drone_modeling(weight,drone_time) / (drone_distance*1000)
                    drone_energy_consumption = self.detail_drone_modeling(weight,drone_time)+drone_energy_consumption_per_meter * 1.18 * data["altitude"]+ drone_energy_consumption_per_meter * 0.96 * data["altitude"]
                    
                    flee = 0
                    drone_kwh -= drone_energy_consumption
                    truck_ofvs.append(truck_kwh)
                    drone_ofvs.append(drone_kwh)
                    jump += 1

            elif route[idx][1] == ONLY_TRUCK:
                truck_energy_consumption = (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) * data["energy_kwh/km_t"]
                
                jump = 0
                truck_kwh -= truck_energy_consumption
                truck_ofvs.append(truck_kwh)
                drone_ofvs.append(drone_kwh)
                flee += 1

            elif route[idx][1] in [IDLE, FLY]:
                
                if route[idx][1]== FLY:
                    fly=1

                truck_energy_consumption = (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) *  data["energy_kwh/km_t"]
               
                jump = 0
                truck_kwh -= truck_energy_consumption

                truck_ofvs.append(truck_kwh)
                drone_ofvs.append(drone_kwh)

        soc_t = [(x / data["battery_kwh_t"]) * 100 for x in truck_ofvs]
        soc_d = [(x / data["battery_kwh_d"]) * 100 for x in drone_ofvs]
        truck_soc.append(soc_t)
        drone_soc.append(soc_d)

        return truck_soc, drone_soc
    

    def drone_insert_cost(self, customer, route, idx, routes): #드론 경로도 고려되어있으므로 분할 후, MM State.object() 적용
        """
        Computes the insertion cost for inserting customer in route at idx.
        """
        
        new_route = route[:idx] + [customer] + route[idx:]
        routes = [new_route] + [r for r in routes if r != route]  # 선택된 route만을 새로운 route로 대체하여 리스트에 수정
        
        self.drone_repair_visit_type_update(routes)
        self.route_duplication_check(routes)

        return MultiModalState(routes).cost_objective()


    def truck_best_insert(self, customer, routes):
        """
        Finds the best feasible route and insertion idx for the customer.
        Return (None, None) if no feasible route insertions are found.
        """
        best_cost, best_route, best_idx = None, None, None

        for route in routes:   
            for idx in range(1,len(route)): #depot을 제외한 index를 고려해줘야함
                if self.truck_can_insert(data, customer, route, idx): #트럭이 insert 될 수 있는 조건
                    cost = self.truck_insert_cost(customer, route, idx, routes)

                    if best_cost is None or cost < best_cost:
                        best_cost, best_route, best_idx = cost, route, idx

        return best_route, best_idx
    
    
    def truck_can_insert(self, data, customer, route, idx): 
        new_route = route[:idx] + [customer] + route[idx:]
        self.truck_repir_visit_type_update_route(new_route)

        truck_path = [value for value in new_route if value[1] != ONLY_DRONE]

        total_logistic_load = 0
        for customer in new_route:
            total_logistic_load += data["logistic_load"][customer[0]]

        if total_logistic_load > data["cargo_limit_truck"]:
            return False

        # max system duration 넘으면 false 수정버전
        # Multimodalstate의 calculate_time_per_route()는 단일 route를 인풋으로 받아서 그 라우트의 total_time_list와 waiting_time_list를 반환
        #if MultiModalState.calculate_time_per_route(self, new_route)[0][-1] > data["maximum_system_duration"]:
        #    return False
        
        if MultiModalState.new_time_arrival_per_route(self, new_route)['eTruck'][0] > data["maximum_system_duration"]:
            return False
        
        
        #max waiting time 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['Over Waiting Time']:
           return False
           
    
        truck_soc = self.soc_calculate(new_route)[0]
        for soc in truck_soc:
            for value in soc:
                if value < data["min_soc_t"]:
                    return False
                
        drone_soc = self.soc_calculate(new_route)[1]
        for soc in drone_soc:
            for value in soc:
                if value < data["min_soc_d"]:
                    return False
        
        return True
                
        
    def truck_insert_cost(self, customer, route, idx, routes): #드론 경로도 고려되어있으므로 분할 후, MM State.object() 적용
        """
        Computes the insertion cost for inserting customer in route at idx.
        """
        
        new_route = route[:idx] + [customer] + route[idx:]
        routes = [new_route] + [r for r in routes if r != route]  # 선택된 route만을 새로운 route로 대체하여 리스트에 수정
        
        self.truck_repair_visit_type_update(routes)
        self.route_duplication_check(routes)

        return MultiModalState(routes).cost_objective()
    

    def drone_repair_visit_type_update(self,routes):
        """
        visit_type update 함수
        """

        for route in routes:
            for j in range(1,len(route)):
                if route[j][1] is None:
                    route[j] = (route[j][0], ONLY_DRONE)
                    k = j - 1  # 현재 노드의 이전 노드부터 시작
                    while k >= 0:
                        if route[k][1] is not None and route[k][1] is not ONLY_DRONE and route[k][1] is not ONLY_TRUCK:  # 이전 노드가 None이 아닌 경우
                            if route[k][1] == IDLE:
                                route[k] = (route[k][0], FLY)
                            elif route[k][1] == FLY:
                                route[k] = (route[k][0], FLY)
                            elif route[k][1] == CATCH:
                                route[k] = (route[k][0], FLY)   
                            break  # 노드의 상태를 설정했으므로 루프를 종료합니다.
                        k -= 1  # k를 감소하여 이전 노드로 이동
                        
                    l = j + 1
                    while l < len(route):
                        if route[l][1] is not None and route[l][1] is not ONLY_DRONE and route[l][1] is not ONLY_TRUCK:
                            if route[l][1] == IDLE:
                                route[l] = (route[l][0], CATCH)
                            elif route[l][1] == FLY:
                                route[l] = (route[l][0], FLY)
                            elif route[l][1] == CATCH:
                                route[l] = (route[l][0], CATCH)   
                            break  # 노드의 상태를 설정했으므로 루프를 종료합니다.
                        l += 1  # k를 감소하여 이전 노드로 이동
                        
        return routes
    

    def drone_repair_visit_type_update_route(self,route):
        for j in range(1,len(route)):
                if route[j][1] is None:
                    route[j] = (route[j][0], ONLY_DRONE)
                    k = j - 1  # 현재 노드의 이전 노드부터 시작
                    while k >= 0:
                        if route[k][1] is not None and route[k][1] is not ONLY_DRONE and route[k][1] is not ONLY_TRUCK:  # 이전 노드가 None이 아닌 경우
                            if route[k][1] == IDLE:
                                route[k] = (route[k][0], FLY)
                            elif route[k][1] == FLY:
                                route[k] = (route[k][0], FLY)
                            elif route[k][1] == CATCH:
                                route[k] = (route[k][0], FLY)   
                            break  # 노드의 상태를 설정했으므로 루프를 종료합니다.
                        k -= 1  # k를 감소하여 이전 노드로 이동
                        
                    l = j + 1
                    while l < len(route):
                        if route[l][1] is not None and route[l][1] is not ONLY_DRONE and route[l][1] is not ONLY_TRUCK:
                            if route[l][1] == IDLE:
                                route[l] = (route[l][0], CATCH)
                            elif route[l][1] == FLY:
                                route[l] = (route[l][0], FLY)
                            elif route[l][1] == CATCH:
                                route[l] = (route[l][0], CATCH)   
                            break  # 노드의 상태를 설정했으므로 루프를 종료합니다.
                        l += 1  # k를 감소하여 이전 노드로 이동
                        
        return route
        
    
    def truck_repair_visit_type_update(self,routes):
        """
        visit_type update 함수
        """
        for route in routes:
            for j in range(1,len(route)-1):
                if route[j][1] is None and route[j-1][1] == IDLE:
                    route[j] = (route[j][0], IDLE)
                elif route[j][1] is None and route[j-1][1] == FLY:
                    route[j] = (route[j][0], ONLY_TRUCK)
                elif route[j][1] is None and route[j-1][1] == ONLY_DRONE:
                    route[j] = (route[j][0], ONLY_TRUCK)
                elif route[j][1] is None and route[j-1][1] == CATCH:
                    route[j] = (route[j][0], IDLE)
                elif route[j][1] is None and route[j-1][1] == ONLY_TRUCK:
                    route[j] = (route[j][0], ONLY_TRUCK)
                                
        return routes
    
    def truck_repir_visit_type_update_route(self,route):
        for j in range(1,len(route)-1):
            if route[j][1] is None and route[j-1][1] == IDLE:
                route[j] = (route[j][0], IDLE)
            elif route[j][1] is None and route[j-1][1] == FLY:
                route[j] = (route[j][0], ONLY_TRUCK)
            elif route[j][1] is None and route[j-1][1] == ONLY_DRONE:
                route[j] = (route[j][0], ONLY_TRUCK)
            elif route[j][1] is None and route[j-1][1] == CATCH:
                route[j] = (route[j][0], IDLE)
            elif route[j][1] is None and route[j-1][1] == ONLY_TRUCK:
                route[j] = (route[j][0], ONLY_TRUCK)
                            
        return route
    
    def unassigned_check(self, routes, unassigned):
            
        for node_id in range(1, data["dimension"]):
            is_in_routes = any(node_id == node[0] for route in routes for node in route)
            if not is_in_routes and (node_id, None) not in unassigned:
                unassigned.append((node_id, None))
                
        return unassigned
    
    def route_duplication_check(self, routes):
        for i, route in enumerate(routes):
            for j in range(i + 1, len(routes)):
                if route == routes[j]:
                    routes[j] = [(0, 0), (0, 0)]
    
        return routes
    
    def extract_and_remain(self, original_list, condition):
        # 새로운 리스트 추출
        new_list = [x for x in original_list if condition(x)]
        
        # 추출한 요소를 제외한 리스트 유지
        remaining_list = [x for x in original_list if x not in new_list]
        
        # 결과 반환
        return new_list, remaining_list
    

    def regret_truck_repair(self, state, rnd_state): #수정완료
        truck_repair = MultiModalState(state.routes, state.unassigned)
        routes = truck_repair.routes
        unassigned = truck_repair.unassigned

        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route, best_idx = self.truck_randomize_greedy_insert(customer, routes)

            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer] + route[best_idx:]
                        self.truck_repair_visit_type_update(routes)

            else: 
                if not any(route == [(0, 0), (0, 0)] for route in routes):
                    # routes 뒤에 새로운 route 추가
                    routes.append([(0, 0), customer, (0, 0)])
                else:
                    for i, route in enumerate(routes):
                        if route == [(0, 0), (0, 0)]:
                            # 빈 route에 고객 추가
                            routes[i] = [(0, 0), (customer[0],0), (0, 0)]

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]


        self.truck_repair_visit_type_update(routes)
        self.route_duplication_check(routes)
        
        return MultiModalState(routes,unassigned)

    
    def regret_drone_repair(self, state, rnd_state):
        drone_repair = MultiModalState(state.routes, state.unassigned)
        routes = drone_repair.routes
        unassigned = drone_repair.unassigned

        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route, best_idx = self.drone_randomize_greedy_insert(customer, routes)    
            
            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer] + route[best_idx:]
                        self.drone_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]
                                        
        self.drone_repair_visit_type_update(routes) #최종적으로 visit_type 검사
        self.route_duplication_check(routes)

        return MultiModalState(routes, unassigned)


    def regret_random_repair(self, state, rnd_state):
        random_repair = MultiModalState(state.routes, state.unassigned)
        routes =  random_repair.routes
        unassigned =  random_repair.unassigned
        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route_t, best_idx_t = self.truck_randomize_greedy_insert(customer, routes)
            best_route_d, best_idx_d = self.drone_randomize_greedy_insert(customer, routes)

            if best_route_t is not None and best_idx_t is not None and best_route_d is not None and best_idx_d is not None:
                chosen_route = random.choice([best_route_t, best_route_d])
                if chosen_route == best_route_d:
                    for i, route in enumerate(routes):
                        if route == best_route_d:
                            routes[i] = route[:best_idx_d] + [customer] + route[best_idx_d:]
                            self.drone_repair_visit_type_update(routes)
                else:
                    for i, route in enumerate(routes):
                        if route == best_route_t:
                            routes[i] = route[:best_idx_t] + [customer] + route[best_idx_t:]
                            self.truck_repair_visit_type_update(routes)

            elif best_route_t is None and best_idx_t is None and best_route_d is not None and best_idx_d is not None:
                for i, route in enumerate(routes):
                        if route == best_route_d:
                            routes[i] = route[:best_idx_d] + [customer] + route[best_idx_d:]
                            self.drone_repair_visit_type_update(routes)

            elif best_route_t is not None and best_idx_t is not None and best_route_d is None and best_idx_d is None:
                for i, route in enumerate(routes):
                        if route == best_route_t:
                            routes[i] = route[:best_idx_t] + [customer] + route[best_idx_t:]
                            self.truck_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

            self.drone_repair_visit_type_update(routes) #최종적으로 visit_type 검사
            self.truck_repair_visit_type_update(routes)
            self.route_duplication_check(routes)

            state_after_random_repair = MultiModalState(routes, unassigned)
        
            if len(state_after_random_repair.unassigned) > 0:
                return self.greedy_truck_repair(state_after_random_repair, rnd_state)
            else:
                return state_after_random_repair


    def regret_light_drone_repair(self, state, rnd_state):
        light_first_repair = MultiModalState(state.routes, state.unassigned)
        routes =  light_first_repair.routes
        unassigned =  light_first_repair.unassigned
        condition = lambda x: data["logistic_load"][x[0]] < data["cargo_limit_drone"] 
        
        unassigned_light, unassigned = self.extract_and_remain(unassigned, condition)

        while len(unassigned_light) > 0:
            customer_light = random.choice(unassigned_light)
            unassigned_light.remove(customer_light)
            best_route, best_idx = self.drone_randomize_greedy_insert(customer_light, routes)
                
            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer_light] + route[best_idx:]
                        self.drone_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

        self.drone_repair_visit_type_update(routes) #최종적으로 visit_type 검사
        self.route_duplication_check(routes)

        state_after_light_repair = MultiModalState(routes, unassigned)
        
        if len(state_after_light_repair.unassigned) > 0:
            return self.greedy_truck_repair(state_after_light_repair,rnd_state)
        else:
            return state_after_light_repair


    def regret_heavy_truck_repair(self, state, rnd_state):
        heavy_first_repair = MultiModalState(state.routes, state.unassigned)
        routes = heavy_first_repair.routes
        unassigned = heavy_first_repair.unassigned
        condition = lambda x: data["logistic_load"][x[0]] > data["cargo_limit_drone"] 
        
        unassigned_heavy, unassigned = self.extract_and_remain(unassigned, condition)

        while len(unassigned_heavy) > 0:
            customer_heavy = random.choice(unassigned_heavy)
            unassigned_heavy.remove(customer_heavy)
            best_route, best_idx = self.truck_randomize_greedy_insert(customer_heavy, routes)
                
            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer_heavy] + route[best_idx:]
                        self.truck_repair_visit_type_update(routes)

            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

        self.truck_repair_visit_type_update(routes) #최종적으로 visit_type 검사
        self.route_duplication_check(routes)

        state_after_heavy_repair = MultiModalState(routes, unassigned)
        
        if len(state_after_heavy_repair.unassigned) > 0:
            return self.greedy_truck_repair(state_after_heavy_repair, rnd_state)
        else:
            return state_after_heavy_repair
        

    def regret_drone_first_truck_second(self, state, rnd_state):

        state_after_drone_repair = self.regret_drone_repair(MultiModalState(state.routes, state.unassigned),rnd_state)

        if len(state_after_drone_repair.unassigned) > 0:
            return self.regret_truck_repair(state_after_drone_repair,rnd_state)
        else:
            return state_after_drone_repair


    def regret_truck_first_drone_second(self, state, rnd_state): #수정완료
    
        truck_first_drone_second_repairs = MultiModalState(state.routes, state.unassigned)
        routes = truck_first_drone_second_repairs.routes
        unassigned = truck_first_drone_second_repairs.unassigned

        while len(unassigned) > 0:
            customer = random.choice(unassigned)
            unassigned.remove(customer)
            best_route, best_idx = self.truck_randomize_greedy_insert(customer, routes)

            if best_route is not None and best_idx is not None:
                for i, route in enumerate(routes):
                    if route == best_route:
                        routes[i] = route[:best_idx] + [customer] + route[best_idx:]
                        self.truck_repair_visit_type_update(routes)
            
            routes = [route for route in routes if route != [(0, 0), (0, 0)]]

        self.truck_repair_visit_type_update(routes)
        self.route_duplication_check(routes)

        state_after_truck_repairs = MultiModalState(routes,unassigned) #unassigned 여러개일수도

        if len(state_after_truck_repairs.unassigned)>0:
            return self.regret_drone_first_truck_second(state_after_truck_repairs,rnd_state)
        else:
            return state_after_truck_repairs
        

    
    def drone_k_opt_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            edge_weight = data["edge_km_d"][route[i][0]][route[i+1][0]]*data["curvy_road"]
            distance += edge_weight
        return distance
    
    def truck_k_opt_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            edge_weight = data["edge_km_t"][route[i][0]][route[i+1][0]]*data["curvy_air"]
            distance += edge_weight
        return distance
    
    def k_opt_algorithm(self, state, rnd_state):
        drone_k_opt = MultiModalState(state.routes, state.unassigned)
        routes = drone_k_opt.routes
        unassigned = drone_k_opt.unassigned

        current_ofv = MultiModalState(routes, unassigned).cost_objective()


        for i, route in enumerate(routes):
            j = 0
            while j < len(route):
                if route[j][1] == FLY:
                    only_drones = []  # 순열을 생성할 ONLY_DRONE 집합
                    only_trucks = []
                    start_index = j
                    end_index = None  # 초기에는 end_index가 설정되지 않았음을 표시
                    for k in range(start_index + 1, len(route)):
                        if route[k][1] == CATCH or route[k][1] == FLY:  # CATCH 또는 FLY일 때
                            end_index = k
                            break  # end_index를 찾았으므로 루프 중지
                    if end_index is not None:  # end_index가 설정된 경우에만 아래 작업을 수행
                        # FLY와 CATCH 또는 FLY 사이에 있는 ONLY_DRONE들을 k_opt에 추가(순열고려 대상)
                        for l in range(start_index + 1, end_index):
                            if route[l][1] == ONLY_DRONE:
                                only_drones.append(route[l][0])
                            elif route[l][1] == ONLY_TRUCK:
                                only_trucks.append(route[l][0])

                        best_distance_d = float('inf')
                        best_perm_d = None
                        
                        perms_d = [list(perm_d) for perm_d in itertools.permutations(only_drones)]

                        best_distance_t = float('inf')
                        best_perm_t = None
                        
                        perms_t = [list(perm_t) for perm_t in itertools.permutations(only_trucks)]

                        for perm_d in perms_d:
                            path = [(route[start_index][0], route[start_index][1])] + [(drone, ONLY_DRONE) for drone in perm_d] + [(route[end_index][0], route[end_index][1])]
                            distance = self.drone_k_opt_distance(path)
                            
                            if distance < best_distance_d:
                                best_perm_d = perm_d
                                best_distance_d = distance

                        for perm_t in perms_t:
                            path = [(route[start_index][0], route[start_index][1])] + [(truck, ONLY_TRUCK) for truck in perm_t] + [(route[end_index][0], route[end_index][1])]
                            distance = self.truck_k_opt_distance(path)
                            
                            if distance < best_distance_t:
                                best_perm_t = perm_t
                                best_distance_t = distance

                        if best_perm_d is None and best_perm_t is None:
                            routes[i] = route

                        elif best_perm_d is not None and best_perm_t is None:
                            new_route = route[:start_index+1] + [(truck, ONLY_TRUCK) for truck in only_trucks] + [(drone, ONLY_DRONE) for drone in best_perm_d] + route[end_index:]
                            if self.k_opt_feasibility(data, new_route):
                                routes[i] = new_route
                                if current_ofv > MultiModalState(routes, unassigned).cost_objective():
                                    current_ofv = MultiModalState(routes, unassigned).cost_objective()
                                    break  # 적절한 경로를 찾았으므로 반복 종료
                                else:
                                    routes[i] = route

                        elif best_perm_d is None and best_perm_t is not None:
                            new_route = route[:start_index+1] + [(truck, ONLY_TRUCK) for truck in best_perm_t] + [(drone, ONLY_DRONE) for drone in only_drones] + route[end_index:]
                            if self.k_opt_feasibility(data, new_route):
                                routes[i] = new_route
                                if current_ofv > MultiModalState(routes, unassigned).cost_objective():
                                    current_ofv = MultiModalState(routes, unassigned).cost_objective()
                                    break  # 적절한 경로를 찾았으므로 반복 종료
                                else:
                                    routes[i] = route
                        j = end_index + 1  # Move j to end_index + 1 to avoid re-processing the same section
                        continue
                    else:
                        j += 1  # Increment j if end_index is not found
                else:
                    j += 1

            j = 0
            while j < len(route):
                catch = 0
                if route[j][1] == IDLE or route[j][1] == CATCH:
                    if route[j][1] == CATCH:
                        catch = 1
                    only_idles = [] 
                    start_index = j
                    end_index = None

                    for k in range(start_index + 1, len(route)):
                        if route[k][1] == FLY:
                            end_index = k
                            
                        elif catch == 1 and route[k][1] == IDLE:
                            end_index = k
                            
                    if end_index is not None:
                        for l in range(start_index + 1, end_index):
                            if route[l][1] == IDLE:
                                only_idles.append(route[l][0])

                        if len(only_idles)>999999:
                            best_distance_i = float('inf')
                            best_perm_i = None
                            perms_i = [list(perm_i) for perm_i in itertools.permutations(only_idles)]

                            for perm_i in perms_i:
                                path = [(route[start_index][0], route[start_index][1])] + [(truck, IDLE) for truck in perm_i] + [(route[end_index][0], route[end_index][1])]
                                distance = self.truck_k_opt_distance(path)
                                
                                if distance < best_distance_i:
                                    best_perm_i = perm_i
                                    best_distance_i = distance

                            if best_perm_i is None:
                                routes[i] = route

                            elif best_perm_i is not None:
                                new_route = route[:start_index+1] + [(truck, IDLE) for truck in best_perm_i] + route[end_index:]
                                if self.k_opt_feasibility(data, new_route):
                                    routes[i] = new_route
                                    if current_ofv > MultiModalState(routes, unassigned).cost_objective():
                                        current_ofv = MultiModalState(routes, unassigned).cost_objective()
                                        break  # 적절한 경로를 찾았으므로 반복 종료
                                    else:
                                        routes[i] = route
                        else:
                            routes[i] = route

                        j = end_index + 1  # Move j to end_index + 1 to avoid re-processing the same section
                        continue
                    else:
                        j += 1  # Increment j if end_index is not found
                else:
                    j += 1      
           
        return MultiModalState(routes, unassigned)
    

    def k_opt_feasibility(self, data, route):
        new_route = route
        
        # max system duration 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['eVTOL'][0] > data["maximum_system_duration"]:
            return False
        
        # max system duration 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['eTruck'][0] > data["maximum_system_duration"]:
            return False

        #max waiting time 넘으면 false 수정버전
        if MultiModalState.new_time_arrival_per_route(self, new_route)['Over Waiting Time']:
           return False
        
        drone_soc = self.soc_calculate(new_route)[1]
        for soc in drone_soc:
            for value in soc:
                if value < data["min_soc_d"]:
                    return False
                
        truck_soc = self.soc_calculate(new_route)[0]
        for soc in truck_soc:
            for value in soc:
                if value < data["min_soc_t"]:
                    return False
        return True