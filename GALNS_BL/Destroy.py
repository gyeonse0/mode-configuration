import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from RouteGenerator import *
from RouteInitializer import *
from FileReader import *
from Repair import *
import config

vrp_file_path = config.vrp_file_path

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)

globals
IDLE = 0 # 해당 노드에 드론이 트럭에 업힌 상태의 경우
FLY = 1 # 해당 노드에서 트럭이 드론의 임무를 위해 드론을 날려주는 경우
ONLY_DRONE = 2 # 해당 노드에 드론만이 임무를 수행하는 서비스 노드인 경우
CATCH = 3 # 해당 노드에서 트럭이 임무를 마친 드론을 받는 경우
ONLY_TRUCK = 4 # 해당 노드에서 트럭만이 임무를 수행하는 경우 (드론이 업혀있지 않음)
NULL = None # 해당 노드가 전체 ROUTES에 할당이 안되어 있는 경우 (SOLUTION에서 UNVISITED 한 노드)

degree_of_destruction = 0.2
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)

class Destroy:
    def random_removal(self, state, rnd_state):
        """
        내가 설정한 파괴 상수에 따라 파괴할 고객의 수가 결정되고, 그에 따라 랜덤으로 고객노드를 제거한다.
        one_path 읽어와서 visit type update 후, 분할하면 훨씬 간단 -> 2, 4 는 1, 3 사이에 올 수 밖에 없음을 이용
        """
        destroyer = MultiModalState(state.routes, state.unassigned)
        routes = destroyer.routes
        unassigned = destroyer.unassigned
        
        repair_instance = Repair() #repair에서 넘어올때 visit_type_update 한번 더 검사
        repair_instance.truck_repair_visit_type_update(routes)
        repair_instance.drone_repair_visit_type_update(routes)
        
        self.unassigned_check(routes, unassigned)

        for customer in rnd_state.choice(range(1, data["dimension"]), customers_to_remove, replace=False):
            self.removal_visit_type_update(customer,routes,unassigned) #visit_type_update
            routes = [[point for point in route if point[1] is not None] for route in routes]

        return MultiModalState(routes, unassigned)
    

    def priority_customer_removal(self, state, rnd_state):
        """
        내가 설정한 파괴 상수에 따라 파괴할 고객의 수가 결정되고, 그에 따라 랜덤으로 고객노드를 제거한다.
        one_path 읽어와서 visit type update 후, 분할하면 훨씬 간단 -> 2, 4 는 1, 3 사이에 올 수 밖에 없음을 이용
        """
        destroyer = MultiModalState(state.routes, state.unassigned)
        routes = destroyer.routes
        unassigned = destroyer.unassigned
        
        repair_instance = Repair() #repair에서 넘어올때 visit_type_update 한번 더 검사
        repair_instance.truck_repair_visit_type_update(routes)
        repair_instance.drone_repair_visit_type_update(routes)
        
        self.unassigned_check(routes, unassigned)

        eligible_customers = [i for i in range(1, data["dimension"]) if data["priority_delivery_time"][i] != 0]
        if len(eligible_customers) == 0:
            for customer in rnd_state.choice(range(1, data["dimension"]), customers_to_remove, replace=False):
                self.removal_visit_type_update(customer,routes,unassigned) #visit_type_update
                routes = [[point for point in route if point[1] is not None] for route in routes]

        else:
            if len(eligible_customers) < customers_to_remove:
                # eligible_customers의 모든 고객을 제거
                for customer in eligible_customers:
                    self.removal_visit_type_update(customer, routes, unassigned) # visit_type_update
                    routes = [[point for point in route if point[1] is not None] for route in routes]

            else:
                for customer in rnd_state.choice(
                    eligible_customers, customers_to_remove, replace=False):
                    
                    self.removal_visit_type_update(customer,routes,unassigned) #visit_type_update
                    routes = [[point for point in route if point[1] is not None] for route in routes]

        return MultiModalState(routes, unassigned)


    def can_drone_removal(self, state, rnd_state):
        destroyer = MultiModalState(state.routes, state.unassigned)
        routes = destroyer.routes
        unassigned = destroyer.unassigned

        repair_instance = Repair() #repair에서 넘어올때 visit_type_update 한번 더 검사
        repair_instance.truck_repair_visit_type_update(routes)
        repair_instance.drone_repair_visit_type_update(routes)
        
        self.unassigned_check(routes, unassigned)
        eligible_customers = [customer for customer in range(1, data["dimension"]) if data["logistic_load"][customer] < data["cargo_limit_drone"] or data["availability_landing_spot"][customer] == 1 or data["customer_drone_preference"][customer] == 1]
        finally_eligible_customers = rnd_state.choice(eligible_customers, size=min(len(eligible_customers), customers_to_remove), replace=False)
        for customer in finally_eligible_customers:
            self.removal_visit_type_update(customer, routes, unassigned)
            routes = [[point for point in route if point[1] is not None] for route in routes]

        return MultiModalState(routes, unassigned)


    def high_cost_removal(self, state, rnd_state):
        destroyer = MultiModalState(state.routes, state.unassigned)
        routes = destroyer.routes
        unassigned = destroyer.unassigned

        repair_instance = Repair() #repair에서 넘어올때 visit_type_update 한번 더 검사
        repair_instance.truck_repair_visit_type_update(routes)
        repair_instance.drone_repair_visit_type_update(routes)
        
        self.unassigned_check(routes, unassigned)
        
        # Get the indices of customers that meet the specified condition
        eligible_customers = [customer for customer in range(1, data["dimension"]) if data["logistic_load"][customer] < data["cargo_limit_drone"] and data["availability_landing_spot"][customer] == 1 and data["customer_drone_preference"][customer] == 1]

        # Calculate objective values for each eligible customer
        objective_values = {}
        for customer in eligible_customers:
            objective_values[customer] = self.calculate_objective(customer, routes, unassigned)

        # Sort eligible customers based on objective values in descending order
        sorted_customers = sorted(eligible_customers, key=lambda customer: objective_values[customer], reverse=True)

        # Choose a subset of customers to remove based on the sorted order
        finally_eligible_customers = sorted_customers[:customers_to_remove]

        # Update routes by removing selected customers
        for customer in finally_eligible_customers:
            self.removal_visit_type_update(customer, routes, unassigned)
            routes = [[point for point in route if point[1] is not None] for route in routes]

        # Return updated MultiModalState
        return MultiModalState(routes, unassigned)
    
    
    def calculate_objective(self, customer, routes, unassigned):
        
        # deep copy를 사용하여 routes 리스트를 복사
        copied_routes = copy.deepcopy(routes)

        self.removal_visit_type_update(customer, copied_routes, unassigned)

        # Update routes by removing customers
        copied_routes = [[point for point in route if point[1] is not None] for route in copied_routes]

        # 복사된 리스트를 사용하여 MultiModalState 객체를 생성
        result_state = MultiModalState(copied_routes,unassigned)

        # 복사된 리스트에 대한 objective 값을 반환
        return result_state.cost_objective()
    
    
    def removal_visit_type_update(self, customer, routes, unassigned):
        
        for route in routes:
            for i in range(0, len(route)-1):
                if route[i][0] == customer:
                    if route[i][1] == IDLE:
                        route[i] = (route[i][0], NULL)
                        if (route[i][0], route[i][1]) not in unassigned:
                            unassigned.append((route[i][0], route[i][1]))

            for i in range(0, len(route)-1):
                if route[i][0] == customer:
                    if route[i][1] == FLY:
                        route[i] = (route[i][0], NULL)
                        if (route[i][0], route[i][1]) not in unassigned:
                            unassigned.append((route[i][0], route[i][1]))
                        
                        j = i + 1
                        while j<= len(route) and (route[j][1] != FLY and route[j][1] != CATCH):
                            if route[j][1] == ONLY_DRONE:
                                route[j] = (route[j][0], NULL)
                                if (route[j][0], route[j][1]) not in unassigned:
                                    unassigned.append((route[j][0], route[j][1]))
                            elif route[j][1] == ONLY_TRUCK and route[j][1] != NULL:
                                route[j] = (route[j][0], IDLE)
                            j += 1
                            
                        if route[j][1] == CATCH:  
                            route[j] = (route[j][0], IDLE)

                        if i >= 2 :
                            k = i - 1
                            while k >= 0 and route[k][1] != FLY:
                                if route[k][1] == ONLY_DRONE:
                                    route[k] = (route[k][0], NULL)
                                    if (route[k][0], route[k][1]) not in unassigned:
                                        unassigned.append((route[k][0], route[k][1]))
                                elif route[k][1] == ONLY_TRUCK:
                                    route[k] = (route[k][0], IDLE)
                                k -= 1
                            
                            if k == 0 and route[k][1] == FLY:
                                route[k] = (route[k][0], IDLE)
                            elif k > 0 and route[k][1] == FLY:
                                if route[k - 1][1] == CATCH:
                                    route[k] = (route[k][0], IDLE)
                                elif route[k - 1][1] == IDLE:
                                    route[k] = (route[k][0], IDLE)
                                elif route[k - 1][1] == ONLY_DRONE:
                                    route[k] = (route[k][0], CATCH)
                                elif route[k - 1][1] == ONLY_TRUCK:
                                    route[k] = (route[k][0], CATCH)


            for i in range(1, len(route)-1):
                if route[i][0] == customer:
                    if route[i][1] == ONLY_DRONE:
                        route[i] = (route[i][0], NULL)
                        if (route[i][0], route[i][1]) not in unassigned:
                            unassigned.append((route[i][0], route[i][1]))
                        
                        if i == 1:
                            if route[i - 1][1] == FLY:
                                route[i - 1] = (route[i - 1][0], IDLE)
                            
                        j = i + 1
                        while j<= len(route) and (route[j][1] != FLY and route[j][1] != CATCH):
                            if route[j][1] == ONLY_DRONE:
                                route[j] = (route[j][0], NULL)
                                if (route[j][0], route[j][1]) not in unassigned:
                                    unassigned.append((route[j][0], route[j][1]))
                            elif route[j][1] == ONLY_TRUCK and route[j][1] != NULL:
                                route[j] = (route[j][0], IDLE)
                            j += 1
                            
                        if route[j][1] == CATCH:  
                            route[j] = (route[j][0], IDLE)
                        
                        
                        if i >= 2 :
                            k = i - 1
                            while k >= 0 and route[k][1] != FLY:
                                if route[k][1] == ONLY_DRONE:
                                    route[k] = (route[k][0], NULL)
                                    if (route[k][0], route[k][1]) not in unassigned:
                                        unassigned.append((route[k][0], route[k][1]))
                                elif route[k][1] == ONLY_TRUCK:
                                    route[k] = (route[k][0], IDLE)
                                k -= 1
                            
                            if k == 0 and route[k][1] == FLY:
                                route[k] = (route[k][0], IDLE)
                            elif k > 0 and route[k][1] == FLY:
                                if route[k - 1][1] == CATCH:
                                    route[k] = (route[k][0], IDLE)
                                elif route[k - 1][1] == IDLE:
                                    route[k] = (route[k][0], IDLE)
                                elif route[k - 1][1] == ONLY_DRONE:
                                    route[k] = (route[k][0], CATCH)
                                elif route[k - 1][1] == ONLY_TRUCK:
                                    route[k] = (route[k][0], CATCH)

        
            for i in range(2, len(route)-1):
                if route[i][0] == customer:
                    if route[i][1] == CATCH:
                        route[i] = (route[i][0], NULL)
                        if (route[i][0], route[i][1]) not in unassigned:
                            unassigned.append((route[i][0], route[i][1]))
                        j = i - 1
                        while j>0 and route[j][1] != FLY:
                            if route[j][1] == ONLY_DRONE:
                                route[j] = (route[j][0], NULL)
                                if (route[j][0], route[j][1]) not in unassigned:
                                    unassigned.append((route[j][0], route[j][1]))
                            elif route[j][1] == ONLY_TRUCK and route[j][1] != NULL:
                                route[j] = (route[j][0], IDLE)
                            j -= 1
                        
                        if j==0:
                            if route[j][1] == FLY:
                                route[j] = (route[j][0], IDLE)
                            
                        elif j>=1:
                            if route[j][1] == FLY:
                                if route[j-1][1] == CATCH and route[j-1][1] != NULL:
                                    route[j] = (route[j][0], IDLE)
                                elif route[j-1][1] == IDLE and route[j-1][1] != NULL:
                                    route[j] = (route[j][0], IDLE)
                                elif route[j-1][1] == ONLY_DRONE and route[j-1][1] != NULL:
                                    route[j] = (route[j][0], CATCH)
                                elif route[j-1][1] == ONLY_TRUCK and route[j-1][1] != NULL:
                                    route[j] = (route[j][0], CATCH)
                        
            for i in range(1,len(route)-1):
                if route[i][0] == customer:
                    if route[i][1] == ONLY_TRUCK:
                        route[i] = (route[i][0], NULL)
                        if (route[i][0], route[i][1]) not in unassigned:
                            unassigned.append((route[i][0], route[i][1]))
                                
        return routes, unassigned
    


    def unassigned_check(self, routes, unassigned):
        
        for node_id in range(1, data["dimension"]):
            is_in_routes = any(node_id == node[0] for route in routes for node in route)
            if not is_in_routes and (node_id, None) not in unassigned:
                unassigned.append((node_id, None))
                
        return unassigned
        