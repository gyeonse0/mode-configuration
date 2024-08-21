import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import atexit
from FileReader import *
from RouteInitializer import *
from RouteGenerator import *
import config

SEED = 1234
rnd_state = np.random.RandomState(None)

vrp_file_path = config.vrp_file_path

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
time_matrix = np.array(data["edge_km_t"])/data["speed_t"]

#max value 보다 1작은게 최대값으로 설정되고 있음
data["logistic_load"] = file_reader.randomize_vrp_data(data["logistic_load"], max_value=8, key_seed=78)
data["availability_landing_spot"] = file_reader.randomize_vrp_data(data["availability_landing_spot"], key_seed=12)
data["customer_drone_preference"] = file_reader.randomize_vrp_data(data["customer_drone_preference"], key_seed=34)

# price는 무게에 의존
logistic_load = np.array(list(data['logistic_load'].values()))
price = np.zeros_like(logistic_load)
price[logistic_load >= 5] = 6
price[(logistic_load <= 5) & (logistic_load > 3)] = 5
price[(logistic_load <= 3) & (logistic_load > 1)] = 4
price[logistic_load <= 1] = 3
data["price"] = {key: value for key, value in zip(data["price"].keys(), price)}

revenue=0
for i in range(1, data["dimension"]):
    revenue = revenue + data["price"][i]

file_reader.update_vrp_file(vrp_file_path, data)

from SolutionPlotter import *
from MultiModalState import *
from Destroy import *
from Repair import *

destroyer = Destroy()
Rep = Repair()
plotter = SolutionPlotter(data)

initializer = RouteInitializer(data)
initial_truck = initializer.init_truck()
ga_count=0
k_opt_count=0

# 초기 설정
iteration_num=15000

sa_num= 5000
start_temperature = 1000
end_temperature = 0.1
step = (end_temperature / start_temperature) ** (1 / sa_num)

temperature = start_temperature
current_num= 1
outcome_counts = {1: 0, 2: 0, 3: 0, 4: 0}
start_time = time.time()
min_objective_time = start_time

current_states = []  # 상태를 저장할 리스트
objectives = []  # 목적 함수 값을 저장할 리스트
time_based_objectives = [0] # 시간 OFV 값을 저장할 리스트

current_states.append(initial_truck)
objective_value = MultiModalState(initial_truck).cost_objective()
objectives.append(objective_value)

destroy_operators = [destroyer.random_removal]

repair_operators = [Rep.greedy_truck_repair]
                    #Rep.regret_random_repair, Rep.regret_drone_first_truck_second, Rep.regret_truck_first_drone_second, Rep.regret_heavy_truck_repair, Rep.regret_light_drone_repair]
destroy_counts = {destroyer.__name__: 0 for destroyer in destroy_operators}
repair_counts = {repairer.__name__: 0 for repairer in repair_operators}  

best_flag=0
notbest_count=0

class RouletteWheel:
    def __init__(self):
        self.scores = [20, 5, 1, 0.5] 
        self.decay = 0.8
        self.operators = None
        self.weights = None

    def set_operators(self, operators):
        self.operators = operators
        self.weights = [1.0] * len(operators)

    def select_operators(self):
        total_weight = sum(self.weights)
        probabilities = [weight / total_weight for weight in self.weights]
        selected_index = np.random.choice(len(self.operators), p=probabilities)
        selected_operator = self.operators[selected_index]
        return selected_operator

    def update_weights(self, outcome_rank, selected_operator_idx):
        idx = selected_operator_idx
        score = self.scores[outcome_rank-1]
        self.weights[idx] = self.decay * self.weights[idx] + (1 - self.decay) * score                  

destroy_selector = RouletteWheel()
repair_selector = RouletteWheel()

destroy_selector.set_operators(destroy_operators)
repair_selector.set_operators(repair_operators)

while current_num < iteration_num:
    if current_num==1:
        selected_destroy_operators = destroy_selector.select_operators()
        selected_repair_operators = repair_selector.select_operators()

        destroyed_state = selected_destroy_operators(initial_truck, rnd_state)
        repaired_state = selected_repair_operators(destroyed_state, rnd_state)

        current_states.append(repaired_state)
        objective_value = MultiModalState(repaired_state).cost_objective()
        objectives.append(objective_value)
        time_based_objectives.append(time.time() - start_time)

        d_idx = destroy_operators.index(selected_destroy_operators)
        r_idx = repair_operators.index(selected_repair_operators)

        destroy_counts[destroy_operators[d_idx].__name__] += 1
        repair_counts[repair_operators[r_idx].__name__] += 1
        current_num+=1
        # 이때 새로 갱신했으면 이때 min_objective_time을 기억하도록 하기    
    
    elif (best_flag == 1 or current_num %50 == 0):
        genetic_state = Rep.genetic_algorithm(current_states[-1],population_size=50, generations=300, mutation_rate=0.2)
        current_states.append(genetic_state)
        objective_value = MultiModalState(genetic_state).cost_objective()
        objectives.append(objective_value)
        time_based_objectives.append(time.time() - start_time)
        if objective_value == min(objectives):
            min_objective_time = time.time()

        ga_count +=1
        current_num +=1
        best_flag = 0
        
    else:
        selected_destroy_operators = destroy_selector.select_operators()
        selected_repair_operators = repair_selector.select_operators()
    
        destroyed_state = selected_destroy_operators(current_states[-1].copy(), rnd_state)
        repaired_state = selected_repair_operators(destroyed_state, rnd_state)

        # 이전 objective 값과 비교하여 수락 여부 결정(accept)
        if np.exp((MultiModalState(current_states[-1]).cost_objective() - MultiModalState(repaired_state).cost_objective()) / temperature) >= rnd.random():
            current_states.append(repaired_state)
            objective_value = MultiModalState(repaired_state).cost_objective()
            objectives.append(objective_value)
            time_based_objectives.append(time.time() - start_time)
            if objective_value == min(objectives):
                outcome = 1
                min_objective_time = time.time()
                best_flag = 1

            elif objective_value <= MultiModalState(current_states[-1]).cost_objective():
                outcome = 2
                notbest_count +=1
            else: 
                outcome = 3
                notbest_count +=1

        else:
            # 이전 상태를 그대로 유지(reject)
            current_states.append(current_states[-1])
            objectives.append(MultiModalState(current_states[-1]).cost_objective())
            time_based_objectives.append(time.time() - start_time)
            outcome = 4
            notbest_count +=1

        outcome_counts[outcome] += 1

        d_idx = destroy_operators.index(selected_destroy_operators)
        r_idx = repair_operators.index(selected_repair_operators)

        destroy_selector.update_weights(outcome, d_idx)
        repair_selector.update_weights(outcome, r_idx)

        # 온도 갱신
        temperature = max(end_temperature, temperature * step)

        destroy_counts[destroy_operators[d_idx].__name__] += 1
        repair_counts[repair_operators[r_idx].__name__] += 1
        current_num+=1

min_objective = min(objectives)
min_objective_per_min = min(time_based_objectives)
min_index = objectives.index(min_objective)
min_index_per_min = time_based_objectives.index(min_objective_per_min)
end_time = time.time()
execution_time = end_time - start_time
min_objective_time = min_objective_time - start_time
drone_count = check_drone_num_of_route(MultiModalState(current_states[min_index]))[0]
except_drone = check_drone_num_of_route(MultiModalState(current_states[min_index]))[1]


truck_soc, drone_soc = MultiModalState(current_states[min_index]).soc()[:2]
truck_time_arrival, drone_time_arrival = MultiModalState(current_states[min_index]).renew_time_arrival()
total_routes = MultiModalState(current_states[min_index]).routes
truck_current_kwh = data["battery_kwh_t"]
drone_current_kwh = data["battery_kwh_d"]


solution_printer = SolutionPrinter(
    current_states=current_states,
    initial_truck=initial_truck,
    min_index=min_index,
    objectives=objectives,
    outcome_counts=outcome_counts,
    destroy_counts=destroy_counts,
    repair_counts=repair_counts,
    ga_count=ga_count,
    k_opt_count=k_opt_count,
    execution_time=execution_time,
    min_objective_time=min_objective_time,
    revenue=revenue
)

saver = SolutionSaver(
    filename=config.folder_name,
    data=data,
    current_states=current_states,
    min_index=min_index,
    initial_truck=initial_truck,
    drone_count=drone_count,
    except_drone=except_drone,
    ga_count=ga_count,
    k_opt_count=k_opt_count,
    execution_time=execution_time,
    min_objective_time=min_objective_time,
    revenue = revenue,
    outcome_counts=outcome_counts,
    destroy_counts=destroy_counts,
    repair_counts=repair_counts
)

#plotter.plot_current_solution(current_states[min_index])
#solution_printer.print_solution()
#solution_printer.plot_objectives()

# SOC 플롯 생성
#soc_plotter = SOCPlotter(data, truck_soc, drone_soc, truck_current_kwh, drone_current_kwh)
#soc_plotter.plot(total_routes)

# 경과 시간 플롯 생성
#elapsed_time_plotter = ElapsedTimePlotter(data, truck_time_arrival, drone_time_arrival)
#elapsed_time_plotter.plot(total_routes)
# 솔루션 텍스트 파일/사진 저장
interval_min = [round(i / 60, 2) for i in time_based_objectives]

save_path = 'time_and_ofv.npz'

np.savez(save_path, time=interval_min, ofv=objectives)

saver.save_solution()
saver.save_current_solution(current_states[min_index], name=config.folder_name+" Solution")
saver.save_convergence_graph(objectives, name=config.folder_name+" Convergence graph")
saver.save_convergence_graph_by_time(interval_min, objectives, name=config.folder_name+" Convergence graph by time")