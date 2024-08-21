import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from MultiModalState import *


def check_drone_num_of_route(solution):
        routes = solution.routes
        drone = {}
        rest = {}
        total_drone_routes = 0
        
        for i, route in enumerate(routes):
            num_drone_routes = sum(1 for idx in route if idx[1] == ONLY_DRONE)
            key = f'Route {i + 1}'
            drone[key] = num_drone_routes
            rest[key] = len(route) - 2 - num_drone_routes
            total_drone_routes += num_drone_routes
        
        drone['Total number of Drones'] = total_drone_routes
        return drone, rest


class SolutionPlotter:
    """
    특정 route를 기반으로 location 및 path, cost 정보등을 시각화 해주는 클래스
    """
    def __init__(self, data):
        self.data = data
        self.drone_colors = ['red', 'blue', 'green']
        self.truck_colors = ['cyan', 'magenta', 'pink']
        self.drone_color_index = 0
        self.truck_color_index = 0

    def get_next_drone_color(self):
        color = self.drone_colors[self.drone_color_index]
        self.drone_color_index = (self.drone_color_index + 1) % len(self.drone_colors)
        return color

    def get_next_truck_color(self):
        color = self.truck_colors[self.truck_color_index]
        self.truck_color_index = (self.truck_color_index + 1) % len(self.truck_colors)
        return color

    def plot_current_solution(self, state, name="Multi_Modal Solution"):
        """
        우리가 뽑아낸 routes 딕셔너리 집합과 solution class를 통해서 현재의 cost와 path를 plot 해주는 함수
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        new_state = MultiModalState(state.routes, state.unassigned)
        routes = new_state.routes
        unassigned = new_state.unassigned

        divided_routes = apply_dividing_route_to_routes(routes)

        uni_drone = []

        for route_info in divided_routes:
            vtype = route_info['vtype']
            vid = route_info['vid']
            path = route_info['path']

            if vtype == 'drone':
                color = self.get_next_drone_color()
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '--'
                offset = 0.0001 * (self.drone_color_index + 1) 
                linewidth = 2
                uni_drone.extend([customer for customer, visit_type in path if visit_type == ONLY_DRONE])

            elif vtype == 'truck':
                color = self.get_next_truck_color()  # 겹치지 않는 색상 생성
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '-'
                offset = 0
                linewidth = 1

            # 경로 그리기
            ax.plot(
                [self.data['node_coord'][loc_getter(loc)][0] for loc in path],
                [self.data['node_coord'][loc_getter(loc)][1] + offset for loc in path],
                color=color,
                linestyle=linestyle, 
                linewidth=linewidth,
                marker='.',
                label=f'{vtype} {vid}'
            )

            # 방향 화살표 그리기
            for i in range(len(path) - 1):
                start = self.data['node_coord'][loc_getter(path[i])]
                end = self.data['node_coord'][loc_getter(path[i + 1])]
                ax.annotate("", xy=(end[0], end[1] + offset), xytext=(start[0], start[1] + offset), arrowprops=dict(arrowstyle="->", color=color))
        
        kwargs = dict(label="Depot", zorder=3, marker="s", s=80)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        
        for node, (x, y) in self.data["node_coord"].items():
            # 주석의 색상을 조건에 따라 설정
            if node == self.data["depot"]:
                annotation_color = 'red'
            elif node in uni_drone:
                annotation_color = 'red'
            else:
                annotation_color = 'black'
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center', color=annotation_color)

        new_state.cost_objective()
        total_carrier_cost = new_state.get_carrier_cost()
        total_energy_cost = new_state.get_energy_cost()
        truck_cost = new_state.get_truck_cost()
        drone_cost = new_state.get_drone_cost()

        ax.set_title(f"{name}\nTotal Carrier Cost: {total_carrier_cost:.2f} USD\n"
            f"Total Energy Cost: {total_energy_cost:.2f} USD"
            f"/  Truck Energy Cost: {truck_cost:.2f} USD"
            f"/  Drone Energy Cost: {drone_cost:.2f} USD\n"
            f"Total Cost: {total_energy_cost + total_carrier_cost:.2f} USD")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")

        existing_handles = ax.get_legend_handles_labels()[0]
        ax.legend(handles=(existing_handles), frameon=False, ncol=3)
        
        plt.show()

    #여기서 부터 추가
    def save_current_solution(self, state, name="Multi_Modal Solution"):
        """
        우리가 뽑아낸 routes 딕셔너리 집합과 solution class를 통해서 현재의 cost와 path를 plot 해주는 함수
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        new_state = MultiModalState(state.routes,state.unassigned)
        routes = new_state.routes
        unassigned = new_state.unassigned

        divided_routes = apply_dividing_route_to_routes(routes)

        for route_info in divided_routes:
            vtype = route_info['vtype']
            vid = route_info['vid']
            path = route_info['path']

            if vtype == 'drone':
                color = self.get_next_drone_color()
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '--'
                offset = 0.0001 * (self.drone_color_index + 1) 
                linewidth=2

            elif vtype == 'truck':
                color = self.get_next_truck_color()  # 겹치지 않는 색상 생성
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '-'
                offset = 0
                linewidth=1

            # 경로 그리기
            ax.plot(
                [self.data['node_coord'][loc_getter(loc)][0] for loc in path],
                [self.data['node_coord'][loc_getter(loc)][1]+ offset for loc in path],
                color=color,
                linestyle=linestyle, 
                linewidth=linewidth,
                marker='.',
                label=f'{vtype} {vid}'
            )

            # 방향 화살표 그리기
            for i in range(len(path)-1):
                start = self.data['node_coord'][loc_getter(path[i])]
                end = self.data['node_coord'][loc_getter(path[i+1])]
                ax.annotate("", xy=(end[0], end[1] + offset), xytext=(start[0], start[1] + offset), arrowprops=dict(arrowstyle="->", color=color))


        kwargs = dict(label="Depot", zorder=3, marker="s", s=80)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        for node, (x, y) in self.data["node_coord"].items():
            # 주석의 색상을 조건에 따라 설정
            annotation_color = 'red' if self.data["priority_delivery_time"][node] != 0 else 'black'
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center', color=annotation_color)
        new_state.cost_objective()
        total_carrier_cost = new_state.get_carrier_cost()
        total_energy_cost = new_state.get_energy_cost()
        total_refund_cost = new_state.get_refund_cost()
        truck_cost = new_state.get_truck_cost()
        drone_cost = new_state.get_drone_cost()

        ax.set_title(f"{name}\nTotal Carrier Cost: {total_carrier_cost:.2f} USD\n"
            f"Total Energy Cost: {total_energy_cost:.2f} USD"
            f"/  Truck Energy Cost: {truck_cost:.2f} USD"
            f"/  Drone Energy Cost: {drone_cost:.2f} USD\n"
            #f"Total Refund Cost: {total_refund_cost:.2f} USD\n"
            f"Total Cost: {total_energy_cost + total_carrier_cost:.2f} USD") #total refund 제외함
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        existing_handles = ax.get_legend_handles_labels()[0]
        ax.legend(handles=(existing_handles), frameon=False, ncol=3)
        plt.savefig(f'results/{name}.png')
        plt.close()

    def save_convergence_graph(self, my_objectives, name):
        plt.figure(figsize=(10, 6))
        plt.plot(my_objectives, label='Current Objective')
        plt.plot(np.minimum.accumulate(my_objectives), color='orange', linestyle='-', label='Best Objective')

        plt.title(name)
        plt.xlabel('Iteration(#)')
        plt.ylabel('Objective Value(USD)')
        plt.grid(True)
        plt.legend()
        # 그래프를 동적으로 설정된 파일명으로 저장
        plt.savefig(f'results/{name}.png')
        plt.close()


class SOCPlotter:
    def __init__(self, data, truck_soc, drone_soc, truck_current_kwh, drone_current_kwh):
        self.data = data
        self.truck_soc = truck_soc
        self.drone_soc = drone_soc
        self.truck_current_kwh = truck_current_kwh
        self.drone_current_kwh = drone_current_kwh

    def plot(self, total_routes):
        for i, route in enumerate(total_routes):
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # TRUCK_PATH
            truck_path = [x if route[idx][1] != ONLY_DRONE else None for idx, x in enumerate(self.truck_soc[i])]
            excluded_indices_truck = [i for i, value in enumerate(truck_path) if value is None]
            for j in range(1, len(truck_path) - 1):
                if truck_path[j] is None:
                    left_index = j - 1
                    right_index = j + 1
                    left_value = None
                    right_value = None

                    while left_index >= 0 and truck_path[left_index] is None:
                        left_index -= 1
                    if left_index >= 0:
                        left_value = truck_path[left_index]

                    while right_index < len(truck_path) and truck_path[right_index] is None:
                        right_index += 1
                    if right_index < len(truck_path):
                        right_value = truck_path[right_index]

                    if left_value is not None and right_value is not None:
                        truck_path[j] = (left_value + right_value) / 2

            ax1.plot(range(len(route)), truck_path, marker='', linestyle='-', label='eTruck', color='blue')
            for iter in range(len(truck_path)):
                if iter in excluded_indices_truck:
                    ax1.plot(iter, truck_path[iter], marker='', linestyle='', color='blue')
                else:
                    ax1.plot(iter, truck_path[iter], marker='.', color='blue')

            # DRONE_PATH
            drone_path = [x if route[idx][1] != ONLY_TRUCK else None for idx, x in enumerate(self.drone_soc[i])]
            excluded_indices_drone = [i for i, value in enumerate(drone_path) if value is None]
            for j in range(1, len(drone_path) - 1):
                if drone_path[j] is None:
                    left_index = j - 1
                    right_index = j + 1
                    left_value = None
                    right_value = None

                    while left_index >= 0 and drone_path[left_index] is None:
                        left_index -= 1
                    if left_index >= 0:
                        left_value = drone_path[left_index]

                    while right_index < len(drone_path) and drone_path[right_index] is None:
                        right_index += 1
                    if right_index < len(drone_path):
                        right_value = drone_path[right_index]

                    if left_value is not None and right_value is not None:
                        drone_path[j] = (left_value + right_value) / 2

            ax1.plot(range(len(route)), drone_path, marker='', linestyle='--', label='eVTOL', color='red')
            for iter in range(len(drone_path)):
                if iter in excluded_indices_drone:
                    ax1.plot(iter, drone_path[iter], marker='', linestyle='', color='red')
                else:
                    ax1.plot(iter, drone_path[iter], marker='.', color='red')

            ax1.set_xlabel('Customer', fontsize=13, labelpad=10)
            ax1.set_ylabel('State of Charge (%)', fontsize=13, labelpad=10)
            ax1.legend(loc='upper right', fontsize='large')

            ax1.set_ylim(0, 105)
            plt.title(f"Progress of State of Charge", fontsize=18, pad=20)
            plt.grid(True)
            plt.xticks(range(len(route)), [customer[0] for customer in route])
            ax1.tick_params(axis='x', which='major', labelsize=10)
            ax1.tick_params(axis='y', which='major', labelsize=10)
            plt.subplots_adjust(bottom=0.15, left=0.15)
            plt.show()

class ElapsedTimePlotter:
    def __init__(self, data, truck_time_arrival, drone_time_arrival):
        self.data = data
        self.truck_time_arrival = truck_time_arrival
        self.drone_time_arrival = drone_time_arrival

    def plot(self, total_routes):
        for i, route in enumerate(total_routes):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(list(self.truck_time_arrival[i].keys()), list(self.truck_time_arrival[i].values()), marker='.', linestyle='-', label='eTruck', color='green')
            ax.plot(list(self.drone_time_arrival[i].keys()), list(self.drone_time_arrival[i].values()), marker='.', linestyle='--', label='eVTOL', color='orange')

            def fill_none_values(arr):
                filled_arr = arr.copy()
                for i in range(1, len(filled_arr)):
                    if filled_arr[i] is None:
                        left_index = i - 1
                        right_index = i + 1
                        while left_index >= 0 and filled_arr[left_index] is None:
                            left_index -= 1
                        while right_index < len(filled_arr) and filled_arr[right_index] is None:
                            right_index += 1
                        if left_index >= 0 and right_index < len(filled_arr):
                            filled_arr[i] = (filled_arr[left_index] + filled_arr[right_index]) / 2
                return filled_arr

            filled_truck_time = fill_none_values(list(self.truck_time_arrival[i].values()))
            filled_drone_time = fill_none_values(list(self.drone_time_arrival[i].values()))

            ax.plot(range(len(route)), filled_truck_time, linestyle='-', color='green')
            ax.plot(range(len(route)), filled_drone_time, linestyle='--', color='orange')

            ax.set_xlabel('Customer', fontsize=13, labelpad=10)
            ax.set_ylabel('Elapsed Time (min)', fontsize=13, labelpad=10)
            ax.legend(loc='upper left', fontsize='large')
            plt.title(f"Progress of Elapsed Time", fontsize=18, pad=20)
            plt.grid(True)
            plt.xticks(range(len(route)), [customer[0] for customer in route])
            ax.tick_params(axis='x', which='major', labelsize=10)
            ax.tick_params(axis='y', which='major', labelsize=10)
            plt.subplots_adjust(bottom=0.15, left=0.15)
            plt.show()

class SolutionPrinter:
    
    def __init__(self, current_states, initial_truck, min_index, objectives, outcome_counts, destroy_counts, repair_counts, ga_count, k_opt_count, execution_time, min_objective_time, revenue):
        self.current_states = current_states
        self.initial_truck = initial_truck
        self.min_index = min_index
        self.objectives = objectives
        self.outcome_counts = outcome_counts
        self.destroy_counts = destroy_counts
        self.repair_counts = repair_counts
        self.ga_count = ga_count
        self.k_opt_count = k_opt_count
        self.execution_time = execution_time
        self.min_objective_time = min_objective_time
        self.revenue = revenue

    def print_solution(self):
        best_state = MultiModalState(self.current_states[self.min_index])
        initial_state = MultiModalState(self.initial_truck)

        best_objective_value = best_state.cost_objective()
        initial_objective_value = initial_state.cost_objective()
        pct_diff = 100 * (best_objective_value - initial_objective_value) / initial_objective_value

       
        # 메서드를 호출하여 값을 가져오고 출력하기
        carrier_cost = best_state.get_carrier_cost()
        energy_cost = best_state.get_energy_cost()
        truck_cost = best_state.get_truck_cost()
        drone_cost = best_state.get_drone_cost()

        print("\nBest Objective Value:", best_objective_value)
        print("\nBest Solution:", best_state.routes)
        print("\nIteration #:", self.min_index)
        print(f"\nThis is {-(pct_diff):.1f}% better than the initial solution, which is {initial_objective_value}.")
        print("\nGA Counts(#):", self.ga_count)
        print("\nK OPT Counts(#):", self.k_opt_count)
        print("\nExecution time:", self.execution_time, "seconds")
        print("\nMin objective time:", self.min_objective_time, "seconds")
        print("\nCost:", best_objective_value)
        print(f"\nCarrier Cost: {carrier_cost}")
        print(f"\nEnergy Cost: {energy_cost}")
        print(f"\neTruck Cost: {truck_cost}")
        print(f"\nDrone Cost: {drone_cost}")
        print("\nRevenue: ${:.2f}".format(self.revenue))
        print("\nNet profit: ${:.2f}\n".format(self.revenue - best_objective_value))

        outcome_messages = {
            1: "The candidate solution is a new global best.(1)",
            2: "The candidate solution is better than the current solution, but not a global best.(2)",
            3: "The candidate solution is accepted.(3)",
            4: "The candidate solution is rejected.(4)"
        }
        for outcome, count in self.outcome_counts.items():
            print(f"{outcome_messages[outcome]}: {count} times")

        print("\nDestroy Operator Counts(#):")
        for name, count in self.destroy_counts.items():
            print(f"{name}: {count}")
        print("\nRepair Operator Counts(#):")
        for name, count in self.repair_counts.items():
            print(f"{name}: {count}")

    def plot_objectives(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.objectives, label='Current Objective')
        plt.plot(np.minimum.accumulate(self.objectives), color='orange', linestyle='-', label='Best Objective')
        plt.title('Progress of Objective Value')
        plt.xlabel('Iteration(#)')
        plt.ylabel('Objective Value(USD)')
        plt.grid(True)
        plt.legend()
        plt.show()

class SolutionSaver:
    def __init__(self, data, filename, current_states, min_index, initial_truck, drone_count,except_drone, ga_count, k_opt_count, execution_time, min_objective_time, revenue, outcome_counts, destroy_counts, repair_counts):
        self.filename = filename
        self.current_states = current_states
        self.min_index = min_index
        self.initial_truck = initial_truck
        self.drone_count = drone_count
        self.ga_count = ga_count
        self.except_drone= except_drone
        self.k_opt_count = k_opt_count
        self.execution_time = execution_time
        self.min_objective_time = min_objective_time
        self.revenue = revenue
        self.outcome_counts = outcome_counts
        self.destroy_counts = destroy_counts
        self.repair_counts = repair_counts
        self.folder_path = self.save_to_desktop(filename)
        self.data = data
        self.drone_colors = ['red', 'blue', 'green']
        self.truck_colors = ['cyan', 'magenta', 'pink']
        self.drone_color_index = 0
        self.truck_color_index = 0

    def save_to_desktop(self, filename):
        # Windows 경로를 직접 사용하여 바탕화면 경로 설정
        desktop = r"C:\Users\User\OneDrive\바탕 화면\Results_integrated"
        folder_name = filename
        folder_path = os.path.join(desktop, folder_name)
        counter = 1

        while os.path.exists(folder_path):
            folder_path = os.path.join(desktop, f"{folder_name}_{counter}")
            counter += 1

        os.makedirs(folder_path)
        return folder_path

    def save_solution(self):
        save_path = os.path.join(self.folder_path, 'solution.txt')
        with open(save_path, 'w') as f:
            best_state = MultiModalState(self.current_states[self.min_index])
            initial_state = MultiModalState(self.initial_truck)

            best_objective_value = best_state.cost_objective()
            initial_objective_value = initial_state.cost_objective()
            pct_diff = 100 * (best_objective_value - initial_objective_value) / initial_objective_value
            carrier_cost = best_state.get_carrier_cost()
            energy_cost = best_state.get_energy_cost()
            truck_cost = best_state.get_truck_cost()
            drone_cost = best_state.get_drone_cost()

            f.write("\nBest Objective Value: {}\n".format(best_objective_value))
            f.write("\nBest Solution: {}\n".format(best_state.routes))
            f.write("\nIteration #: {}\n".format(self.min_index))
            f.write(f"\nThis is {-(pct_diff):.1f}% better than the initial solution, which is {initial_objective_value}.\n")
            f.write("\nGA Counts(#): {}\n".format(self.ga_count))
            f.write("\nK OPT Counts(#): {}\n".format(self.k_opt_count))
            f.write("\nExecution time: {} seconds\n".format(self.execution_time))
            f.write("\nMin objective time: {} seconds\n".format(self.min_objective_time))
            f.write("\nCost: {}\n".format(best_objective_value))
            f.write("\nCarrier Cost: {}\n".format(carrier_cost))
            f.write("\nEnergy Cost: {}\n".format(energy_cost))
            f.write("\neTruck Cost: {}\n".format(truck_cost))
            f.write("\nDrone Cost: {}\n".format(drone_cost))
            f.write("\nRevenue: ${:.2f} dollar($)\n".format(self.revenue))
            f.write("\nNet profit: ${:.2f} dollar($)\n".format(self.revenue - best_objective_value))
           
            outcome_messages = {
                1: "The candidate solution is a new global best.(1)",
                2: "The candidate solution is better than the current solution, but not a global best.(2)",
                3: "The candidate solution is accepted.(3)",
                4: "The candidate solution is rejected.(4)"
            }
            for outcome, count in self.outcome_counts.items():
                f.write(f"{outcome_messages[outcome]}: {count} times\n")

            f.write("\nDestroy Operator Counts(#):\n")
            for name, count in self.destroy_counts.items():
                f.write(f"{name}: {count}\n")
            f.write("\nRepair Operator Counts(#):\n")
            for name, count in self.repair_counts.items():
                f.write(f"{name}: {count}\n")

    def save_plot(self, plot_function, filename, file_format='png'):
        save_path = os.path.join(self.folder_path, f"{filename}.{file_format}")
        plt.figure()
        plot_function()
        plt.savefig(save_path, format=file_format, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_multiple_plots(self, plot_function, filenames, file_format='png'):
        for idx, filename in enumerate(filenames):
            save_path = os.path.join(self.folder_path, f"{filename}.{file_format}")
            plt.figure()
            plot_function(idx)
            plt.savefig(save_path, format=file_format, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    def get_next_drone_color(self):
        color = self.drone_colors[self.drone_color_index]
        self.drone_color_index = (self.drone_color_index + 1) % len(self.drone_colors)
        return color

    def get_next_truck_color(self):
        color = self.truck_colors[self.truck_color_index]
        self.truck_color_index = (self.truck_color_index + 1) % len(self.truck_colors)
        return color
    
    def save_current_solution(self, state, name="Multi_Modal Solution"):
        """
        우리가 뽑아낸 routes 딕셔너리 집합과 solution class를 통해서 현재의 cost와 path를 plot 해주는 함수
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        new_state = MultiModalState(state.routes, state.unassigned)
        routes = new_state.routes
        unassigned = new_state.unassigned

        divided_routes = apply_dividing_route_to_routes(routes)

        uni_drone = []

        for route_info in divided_routes:
            vtype = route_info['vtype']
            vid = route_info['vid']
            path = route_info['path']

            if vtype == 'drone':
                color = self.get_next_drone_color()
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '--'
                offset = 0.0001 * (self.drone_color_index + 1) 
                linewidth = 2
                uni_drone.extend([customer for customer, visit_type in path if visit_type == ONLY_DRONE])

            elif vtype == 'truck':
                color = self.get_next_truck_color()  # 겹치지 않는 색상 생성
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '-'
                offset = 0
                linewidth = 1

            # 경로 그리기
            ax.plot(
                [self.data['node_coord'][loc_getter(loc)][0] for loc in path],
                [self.data['node_coord'][loc_getter(loc)][1] + offset for loc in path],
                color=color,
                linestyle=linestyle, 
                linewidth=linewidth,
                marker='.',
                label=f'{vtype} {vid}'
            )

            # 방향 화살표 그리기
            for i in range(len(path) - 1):
                start = self.data['node_coord'][loc_getter(path[i])]
                end = self.data['node_coord'][loc_getter(path[i + 1])]
                ax.annotate("", xy=(end[0], end[1] + offset), xytext=(start[0], start[1] + offset), arrowprops=dict(arrowstyle="->", color=color))
        
        kwargs = dict(label="Depot", zorder=3, marker="s", s=80)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        
        for node, (x, y) in self.data["node_coord"].items():
            # 주석의 색상을 조건에 따라 설정
            if node == self.data["depot"]:
                annotation_color = 'red'
            elif node in uni_drone:
                annotation_color = 'red'
            else:
                annotation_color = 'black'
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center', color=annotation_color)

        new_state.cost_objective()
        total_carrier_cost = new_state.get_carrier_cost()
        total_energy_cost = new_state.get_energy_cost()
        truck_cost = new_state.get_truck_cost()
        drone_cost = new_state.get_drone_cost()

        ax.set_title(f"{name}\nTotal Carrier Cost: {total_carrier_cost:.2f} USD\n"
            f"Total Energy Cost: {total_energy_cost:.2f} USD"
            f"/  Truck Energy Cost: {truck_cost:.2f} USD"
            f"/  Drone Energy Cost: {drone_cost:.2f} USD\n"
            f"Total Cost: {total_energy_cost + total_carrier_cost:.2f} USD")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")

        existing_handles = ax.get_legend_handles_labels()[0]
        ax.legend(handles=(existing_handles), frameon=False, ncol=3)
        save_path = os.path.join(self.folder_path, f'{name}.png')
        plt.savefig(save_path)
        plt.close()


    def save_convergence_graph(self, my_objectives, name):
        plt.figure(figsize=(10, 6))
        plt.plot(my_objectives, label='Current Objective')
        plt.plot(np.minimum.accumulate(my_objectives), color='orange', linestyle='-', label='Best Objective')

        plt.title(name)
        plt.xlabel('Iteration(#)')
        plt.ylabel('Objective Value(USD)')
        plt.grid(True)
        plt.legend()
        
        save_path = os.path.join(self.folder_path, f'{name}.png')
        plt.savefig(save_path)
        plt.close()

    def save_convergence_graph_by_time(self, interval, my_objectives, name):
        plt.figure(figsize=(10, 6))
        plt.plot(interval, my_objectives, label='Current Objective by time')
        plt.plot(interval, np.minimum.accumulate(my_objectives), color='red', linestyle='-', label='Best Objective')

        plt.title(name)
        plt.xlabel('Time(min)')
        plt.ylabel('Objective Value(USD)')
        plt.grid(True)
        plt.legend()
        
        save_path = os.path.join(self.folder_path, f'{name}.png')
        plt.savefig(save_path)
        plt.close()