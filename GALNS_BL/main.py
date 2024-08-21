import copy
import time
import numpy as np
import numpy.random as rnd
import atexit
import pandas as pd
import seaborn as sns
from FileReader import *
from RouteInitializer import *
from RouteGenerator import *
import fileinput
import subprocess
import os
import concurrent.futures
import matplotlib
import matplotlib.pyplot as plt
import queue

matplotlib.use('Agg')

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

SEED = 1234
rnd_state = np.random.RandomState(None)

# 파일이 이미 열려 있는지 확인
save_file_path = r'C:\Users\User\OneDrive\바탕 화면\sensitivity_analysis_results.xlsx'
if os.path.isfile(save_file_path):
    try:
        os.rename(save_file_path, save_file_path)  # 파일을 잠시 다른 이름으로 변경 시도
    except OSError:
        raise PermissionError(f"파일이 이미 열려있습니다. 닫고 실행하세요.")

def check_file_permissions(filepath):
    # 파일에 대한 쓰기 권한이 있는지 확인
    if not os.access(filepath, os.W_OK):
        raise PermissionError(f"Write permission denied for the file {filepath}")
    
file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)

def initialize_data(data, speed_t, drone_charging):
    
    # 변수 업데이트
    data["speed_t"] = speed_t
    data["charging_kwh_d"] = drone_charging
    #data["cargo_limit_drone"] = drone_payload
    #data["service_time"] = service_time
    #data["temperature"] = temperature
    
    #time_matrix = np.array(data["edge_km_t"]) / data["speed_t"]
    #data["priority_delivery_time"] = file_reader.randomize_vrp_data(data["priority_delivery_time"], time=time_matrix, key_seed=56, p_first=p_first)
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

    """
    # priority_delivery_time에 따른 가격 조정 함수
    def adjust_price_by_priority(price, priority_delivery_time):
        if priority_delivery_time == 30:
            return price * 2
        elif priority_delivery_time == 60:
            return price * 1.8
        elif priority_delivery_time == 90:
            return price * 1.6
        elif priority_delivery_time == 120:
            return price * 1.4
        elif priority_delivery_time == 150:
            return price * 1.2
        else:
            return price

    # 가격 조정 적용
    adjusted_price = np.array([adjust_price_by_priority(p, t) for p, t in zip(price, list(data["priority_delivery_time"].values()))])

    # 소수점 첫째 자리까지 반올림
    adjusted_price = np.round(adjusted_price, 1)

    data["price"] = {key: value for key, value in zip(range(50), adjusted_price)}
    """

    return data
    

from MultiModalState import *
from SolutionPlotter import *
from Destroy import *
from Repair import *

def run_simulation(data):
    destroyer = Destroy()
    Rep = Repair()
    initializer = RouteInitializer(data)
    initial_truck = initializer.init_truck()

    destroy_operators = [destroyer.random_removal, destroyer.can_drone_removal, destroyer.high_cost_removal, destroyer.priority_customer_removal]

    repair_operators = [Rep.random_repair, Rep.drone_first_truck_second, Rep.truck_first_drone_second, Rep.heavy_truck_repair, Rep.light_drone_repair]
                        #Rep.regret_random_repair, Rep.regret_drone_first_truck_second, Rep.regret_truck_first_drone_second, Rep.regret_heavy_truck_repair, Rep.regret_light_drone_repair]
                        
    destroy_selector = RouletteWheel()
    repair_selector = RouletteWheel()

    destroy_selector.set_operators(destroy_operators)
    repair_selector.set_operators(repair_operators)

    destroy_counts = {destroyer.__name__: 0 for destroyer in destroy_operators}
    repair_counts = {repairer.__name__: 0 for repairer in repair_operators}
    k_opt_count=0
    ga_count=0

    # 초기 설정
    iteration_num= 30000

    start_temperature = 100
    end_temperature = 0.01
    step = 0.1

    temperature = start_temperature
    current_num=1
    outcome_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    start_time = time.time()
    min_objective_time = None

    current_states = []  # 상태를 저장할 리스트
    objectives = []  # 목적 함수 값을 저장할 리스트

    current_states.append(initial_truck)
    objective_value = MultiModalState(initial_truck).cost_objective()
    objectives.append(objective_value)
    while current_num <= iteration_num:
        if current_num==1:
            selected_destroy_operators = destroy_selector.select_operators()
            selected_repair_operators = repair_selector.select_operators()

            destroyed_state = selected_destroy_operators(initial_truck, rnd_state)
            repaired_state = selected_repair_operators(destroyed_state, rnd_state)

            current_states.append(repaired_state)
            objective_value = MultiModalState(repaired_state).cost_objective()
            objectives.append(objective_value)

            d_idx = destroy_operators.index(selected_destroy_operators)
            r_idx = repair_operators.index(selected_repair_operators)

            destroy_counts[destroy_operators[d_idx].__name__] += 1
            repair_counts[repair_operators[r_idx].__name__] += 1
            current_num+=1
            # 이때 새로 갱신했으면 이때 min_objective_time을 기억하도록 하기    
            
        elif current_num % 100 == 0 and current_num != 0:
            if current_num % 500 == 0:
                genetic_state = Rep.genetic_algorithm(current_states[-1],population_size=5, generations=100, mutation_rate=0.2)

                current_states.append(genetic_state)
                objective_value = MultiModalState(genetic_state).cost_objective()
                objectives.append(objective_value)
                if objective_value == min(objectives):
                    min_objective_time = time.time()
                temperature = max(end_temperature, temperature - step)
                ga_count+=1
                current_num +=1

            else:
                k_opt_state = Rep.drone_k_opt(current_states[-1],rnd_state)
                current_states.append(k_opt_state)
                objective_value = MultiModalState(k_opt_state).cost_objective()
                objectives.append(objective_value)
                if objective_value == min(objectives):
                    min_objective_time = time.time()
                temperature = max(end_temperature, temperature - step)
                k_opt_count+=1
                current_num +=1
            

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
                if objective_value == min(objectives):
                    outcome = 1
                    min_objective_time = time.time()

                elif objective_value <= MultiModalState(current_states[-1]).cost_objective():
                    outcome = 2
                else: 
                    outcome = 3

            else:
                # 이전 상태를 그대로 유지(reject)
                current_states.append(current_states[-1])
                objectives.append(MultiModalState(current_states[-1]).cost_objective())
                outcome = 4

            outcome_counts[outcome] += 1

            d_idx = destroy_operators.index(selected_destroy_operators)
            r_idx = repair_operators.index(selected_repair_operators)

            destroy_selector.update_weights(outcome, d_idx)
            repair_selector.update_weights(outcome, r_idx)

            # 온도 갱신
            temperature = max(end_temperature, temperature - step)

            destroy_counts[destroy_operators[d_idx].__name__] += 1
            repair_counts[repair_operators[r_idx].__name__] += 1
            current_num+=1

    min_objective = min(objectives)
    min_index = objectives.index(min_objective)
    end_time = time.time()
    execution_time = end_time - start_time
    min_objective_time = min_objective_time - start_time

    return objectives, min_objective, current_states[min_index], min_objective_time, execution_time

def print_current_parameters(data, params, ofv):
    parameters = {
        'speed_t': params.get('speed_t', 'N/A'),
        #'p_first': params.get('p_first', 'N/A'),
        'drone_charging': params.get('drone_charging', 'N/A'),
        #'drone_payload': params.get('drone_payload', 'N/A'),
        #'service_time': params.get('service_time', 'N/A'),
        #'temperature' : params.get('temperature', 'N/A')
    }
    ofv_formatted = f'{ofv:.2f} USD'
    print(', '.join(f'{key}: {value}' for key, value in parameters.items()) + f', OFV: {ofv_formatted}')

def save_to_excel(filename, sheetname, results):
    check_file_permissions(filename)
    df = pd.DataFrame(results)
    with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheetname, index=False)


# 베이스라인 값 설정
baseline_values = {
    'speed_t': 0.5,
    #'p_first': 0.6,
    'drone_charging': 1.0,
    #'drone_payload': 6,
    #'service_time': 5,
    #'temperature': 10.0
}

variable_test_values = {
    'speed_t': [0.2, 0.5, 0.8, 1.1],
    #'p_first': [0.5, 0.6, 0.7, 0.8],
    'drone_charging': [0.5, 1.0, 1.5, 2.0],
    #'drone_payload': [2, 6, 10, 14],
    #'service_time': [2, 5, 7, 11],
    #'temperature': [0.0, 10.0, 20.0, 30.0]
}
# 각 변수별 objective values를 저장할 딕셔너리
variable_objective_values = {
    'speed_t': [],
    #'p_first': [],
    'drone_charging': [],
    #'drone_payload': [],
    #'service_time': [],
    #'temperature': []
}

vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\realrealrealreal-main\realrealrealreal-main\GALNS_FLP\data\multi_modal_data.vrp'
backup_file_path = r'C:\Users\User\OneDrive\바탕 화면\realrealrealreal-main\realrealrealreal-main\GALNS_FLP\data\multi_modal_backup.vrp'

def check_drone_num_of_route(solution):
    routes = solution.routes
    result = {}
    for i, route in enumerate(routes):
        num_drone_routes = sum(1 for idx in route if idx[1] == ONLY_DRONE)
        key = f'# Drone of Route{i + 1}'
        result[key] = num_drone_routes
    return result
plotter = SolutionPlotter({})

# 파일 백업
file_reader = FileReader()
file_reader.create_backup(vrp_file_path, backup_file_path)

# 민감도 분석 결과 저장용 리스트 초기화
results = []


# 베이스라인 결과 한 번만 출력
data = initialize_data(data, **baseline_values)

# 수익 계산
revenue = sum(data["price"][i] for i in range(1, data["dimension"]))
baseline_objectives, baseline_best_objective, best_solution, min_objective_time, execution_time = run_simulation(data)
best_solution.cost_objective()
results.append({
        'Population(#)': 50,
        'Speed_t(km/min)': baseline_values['speed_t'],
        #'Fast Demand(%)': baseline_values['p_first'] * 100,
        'Drone charging(kW)': baseline_values['drone_charging'], 
        #'Drone Payload(kg)': baseline_values['drone_payload'],
        #'Service Time(min)': baseline_values['service_time'],
        #'temperature(Celsius)': baseline_values['temperature'],
        'Carrier Cost': best_solution.get_carrier_cost(), 
        'Energy Cost': best_solution.get_energy_cost(), 
        'Truck Cost' : best_solution.get_truck_cost(),
        'Drone Cost' : best_solution.get_drone_cost(), 
        'Total Cost': baseline_best_objective, 
        'Revenue': revenue, 
        'Net Profit': revenue - baseline_best_objective, 
        'Min Objective Time': min_objective_time, 
        'Execution_time': execution_time, 
        'Number of Pair': best_solution.get_route_count()
    })

results[-1].update(check_drone_num_of_route(best_solution))
print_current_parameters(data, baseline_values, baseline_best_objective)
plotter.data = data  # plotter의 데이터를 현재 데이터로 업데이트
plotter.save_current_solution(best_solution, name="GALNS_FC - Baseline")
plotter.save_convergence_graph(baseline_objectives, name="Convergence Graph - Baseline")

# Create a queue to communicate with the main thread
result_queue = queue.Queue()

def analyze_sensitivity(data, variable_name, results):
    variable_values = variable_test_values[variable_name]
    objective_values = variable_objective_values[variable_name]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_sensitivity_single_run, data, value, variable_name, results, objective_values) for value in variable_values]
        concurrent.futures.wait(futures)
    process_result_queue()

def analyze_sensitivity_single_run(data, value, variable_name, results, objective_values):
    if value == baseline_values[variable_name]:
        return

    params = baseline_values.copy()
    params[variable_name] = value
    data = initialize_data(data, **params)
    
    objectives, best_objective, best_solution, min_objective_time, execution_time = run_simulation(data)
    
    revenue = sum(data["price"][i] for i in range(1, data["dimension"]))
    
    objective_values.append(best_objective)
    
    best_solution.cost_objective()
    result = {
        'Population(#)': 50,
        'speed_t(km/min)': params['speed_t'],
        #'Fast Demand(%)': params['p_first'] * 100,
        'Drone Charging(kWh)': params['drone_charging'],
        #'Drone Payload(kg)': params['drone_payload'],
        #'Service Time(min)': params['service_time'],
        #'temperature(Celsius)': baseline_values['temperature'],
        'Carrier Cost': best_solution.get_carrier_cost(),
        'Energy Cost': best_solution.get_energy_cost(),
        'Truck Cost' : best_solution.get_truck_cost(),
        'Drone Cost' : best_solution.get_drone_cost(),
        'Total Cost': best_objective,
        'Revenue': revenue,
        'Net Profit': revenue - best_objective,
        'Min Objective Time': min_objective_time,
        'Execution_time': execution_time,
        'Number of Pair': best_solution.get_route_count()
    }
    result.update(check_drone_num_of_route(best_solution))
    results.append(result)
    
    print_current_parameters(data, params, best_objective)
    
    result_queue.put((data, best_solution, variable_name, value, objectives))

def process_result_queue():
    while not result_queue.empty():
        data, best_solution, variable_name, value, objectives = result_queue.get()
        plotter.data = data
        plotter.save_current_solution(best_solution, name=f"GALNS_FC - {variable_name} - {value}")
        plotter.save_convergence_graph(objectives, name=f"Convergence Graph - {variable_name} - {value}")
        print("HI")

# plot 코드
def plot_results(variable_name):
    plt.figure(figsize=(10, 6))
    full_objectives = []
    baseline_value = baseline_values[variable_name]
    variable_values = variable_test_values[variable_name]
    objective_values = variable_objective_values[variable_name]
    
    index = 0
    error_occurred = False
    
    for value in variable_values:
        if value == baseline_value:
            full_objectives.append(baseline_best_objective)
        else:
            try:
                full_objectives.append(objective_values[index])
                index += 1
            except IndexError:
                print(f"Index {index} is out of range for {variable_name}_objective_values")
                error_occurred = True
                break
    
    if not error_occurred:
        sns.barplot(x=variable_values, y=full_objectives, hue=variable_values, palette="Blues_d", dodge=False, legend=False)
        plt.title(f'Objective Function Value vs {variable_name}')
        plt.xlabel(f'{variable_name}')
        plt.ylabel('Objective Function Value (USD)')
        plt.grid(True)
        plt.savefig(f'results/Objective Function Value vs {variable_name}.png')
        plt.close()
    else:
        print("Plotting skipped due to an error during data collection.")

# 민감도 분석 수행
results = []
for variable_name in variable_test_values.keys():
    analyze_sensitivity(data, variable_name, results)
    plot_results(variable_name)

save_to_excel(r'C:\Users\User\OneDrive\바탕 화면\sensitivity_analysis_results.xlsx', 'result_flp', results)

atexit.register(file_reader.restore_on_exit, backup_file_path, vrp_file_path)
