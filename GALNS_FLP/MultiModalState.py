import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List
from itertools import groupby
from sklearn.ensemble import RandomForestRegressor
from RouteGenerator import *
from FileReader import *
import config


vrp_file_path = config.vrp_file_path

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)

IDLE = 0 # 해당 노드에 드론이 트럭에 업힌 상태의 경우
FLY = 1 # 해당 노드에서 트럭이 드론의 임무를 위해 드론을 날려주는 경우
ONLY_DRONE = 2 # 해당 노드에 드론만이 임무를 수행하는 서비스 노드인 경우
CATCH = 3 # 해당 노드에서 트럭이 임무를 마친 드론을 받는 경우
ONLY_TRUCK = 4 # 해당 노드에서 트럭만이 임무를 수행하는 경우 (드론이 업혀있지 않음)
NULL = None # 해당 노드가 전체 ROUTES에 할당이 안되어 있는 경우 (SOLUTION에서 UNVISITED 한 노드)

        
class MultiModalState:
    """
    routes 딕셔너리 집합을 input으로 받아서 copy를 수행한 뒤, 해당 routes 에서의 정보를 추출하는 함수
    output: objective cost value / 특정 customer node를 포함한 route  
    """
    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
        
        unassigned_check=[]
        for node_id in range(1, data["dimension"]):
            is_in_routes = any(node_id == node[0] for route in routes for node in route)
            if not is_in_routes and (node_id, None) not in unassigned_check:
                unassigned_check.append((node_id, None))
        
        self.unassigned = unassigned_check

    def copy(self):
        return MultiModalState(
            copy.deepcopy(self.routes),
            self.unassigned.copy()
        )
        
    def __str__(self):
        return f"Routes: {self.routes}, Unassigned: {self.unassigned}" 
    
    def __iter__(self):
        return iter(self.routes)       
        
    
    def cost_objective(self): 

        total_energy_consumption = self.soc()[2]
        energy_cost = total_energy_consumption * 0.35

        truck_consumption = self.soc()[3]
        drone_consumption = self.soc()[4]
        truck_cost = truck_consumption * 0.35
        drone_cost = drone_consumption * 0.35

        route_count=0
        for route in self.routes:
            route_count += 1
            
        time_of_eTrucks = self.renew_time_arrival()[0]
        carrier_cost = 0
        salary = 15
        for carrier in time_of_eTrucks:
            if carrier:
                carrier_cost += (carrier[0]/60) * salary  # sink에 도착했을 때 carrier의 time arrival
            else:
                print("경고: 빈 리스트가 포함되어 있습니다.")  
        
        carrier_cost += route_count* 30 #고정비용(임금)

        self.carrier_cost = round(carrier_cost, 2)
        self.energy_cost = round(energy_cost, 2)
        self.truck_cost = round(truck_cost, 2)
        self.drone_cost = round(drone_cost, 2)
        self.route_count = route_count

        return self.carrier_cost + self.energy_cost
    
    def get_route_count(self):
        return self.route_count
    
    def get_carrier_cost(self):
        return self.carrier_cost

    def get_energy_cost(self):
        return self.energy_cost
    
    def get_truck_cost(self):
        return self.truck_cost
    
    def get_drone_cost(self):
        return self.drone_cost
    
    
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
    
    def detail_drone_modeling(self,current_logistic_load,edge_time):
        
        drone_consumption = ((((data["mass_d"]+current_logistic_load)*data["speed_d"]*60)/(370*data["lift_to_drag"]*data["power_motor_prop"]))+data["power_elec"])*edge_time
        return drone_consumption
    
    def detail_just_drone_modeling(self,edge_time):
        
        drone_consumption = ((((data["mass_d"])*data["speed_d"]*60)/(370*data["lift_to_drag"]*data["power_motor_prop"]))+data["power_elec"])*edge_time
        return drone_consumption

    def soc(self):
        truck_soc = []
        drone_soc = []
        total_energy_consumption = 0
        total_drone = 0
        total_truck = 0
        for route in self.routes:
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

                    drone_time = (((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]))/data["speed_d"])/60
                    drone_distance = data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]
                        
                    for n, w in current_logistic_loads:
                        if n == route[idx-flee-1][0]:
                            weight = w
                            break

                    drone_energy_consumption_per_meter = self.detail_drone_modeling(weight,drone_time) / (drone_distance*1000)
                    drone_energy_consumption = self.detail_drone_modeling(weight,drone_time)+ drone_energy_consumption_per_meter * 0.96 * data["altitude"]
                    total_energy_consumption += drone_energy_consumption
                    total_energy_consumption += truck_energy_consumption
                    total_drone += drone_energy_consumption
                    total_truck += truck_energy_consumption

                    flee = 0
                    jump = 0
                    truck_kwh -= truck_energy_consumption
                    drone_kwh -= drone_energy_consumption
                    truck_ofvs.append(truck_kwh)
                    drone_ofvs.append(drone_kwh)
                    

                elif route[idx][1] == ONLY_DRONE:
                    drone_time = (((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]))/data["speed_d"])/60
                    drone_distance = data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]
                        
                    for n, w in current_logistic_loads:
                        if n == route[idx-flee-1][0]:
                            weight = w
                            break

                    if fly==1:
        
                        drone_energy_consumption_per_meter = self.detail_drone_modeling(weight,drone_time) / (drone_distance*1000)
                        drone_energy_consumption = drone_energy_consumption_per_meter * 1.18 * data["altitude"] + self.detail_drone_modeling(weight,drone_time)
                        total_energy_consumption += drone_energy_consumption

                        flee = 0
                        drone_kwh -= drone_energy_consumption
                        truck_ofvs.append(truck_kwh)
                        drone_ofvs.append(drone_kwh)
                        jump += 1
                        fly = 0
                        total_drone += drone_energy_consumption
                    
                    
                    elif fly==0:
                        drone_energy_consumption_per_meter = self.detail_drone_modeling(weight,drone_time) / (drone_distance*1000)
                        drone_energy_consumption = self.detail_drone_modeling(weight,drone_time)+drone_energy_consumption_per_meter * 1.18 * data["altitude"]+ drone_energy_consumption_per_meter * 0.96 * data["altitude"]
                        total_energy_consumption += drone_energy_consumption
                        
                        flee = 0
                        drone_kwh -= drone_energy_consumption
                        truck_ofvs.append(truck_kwh)
                        drone_ofvs.append(drone_kwh)
                        jump += 1
                        total_drone += drone_energy_consumption


                elif route[idx][1] == ONLY_TRUCK:
                    truck_energy_consumption = (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) *data["energy_kwh/km_t"]
                    
                    total_energy_consumption += truck_energy_consumption
                    jump = 0
                    truck_kwh -= truck_energy_consumption
                    truck_ofvs.append(truck_kwh)
                    drone_ofvs.append(drone_kwh)
                    flee += 1
                    total_truck += truck_energy_consumption

                elif route[idx][1] in [IDLE, FLY]:
                    
                    if route[idx][1]== FLY:
                        fly=1

                    truck_energy_consumption = (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) * data["energy_kwh/km_t"]
                    
                    total_energy_consumption += truck_energy_consumption
                    jump = 0
                    truck_kwh -= truck_energy_consumption
                    total_truck += truck_energy_consumption
                    
                    truck_ofvs.append(truck_kwh)
                    drone_ofvs.append(drone_kwh)

            soc_t = [(x / data["battery_kwh_t"]) * 100 for x in truck_ofvs]
            soc_d = [(x / data["battery_kwh_d"]) * 100 for x in drone_ofvs]
            truck_soc.append(soc_t)
            drone_soc.append(soc_d)

        return truck_soc, drone_soc, total_energy_consumption, total_truck, total_drone

    def renew_time_arrival(self):
        truck_time_arrivals = []  # 각 route의 트럭 도착 시간을 담은 리스트
        drone_time_arrivals = []  # 각 route의 드론 도착 시간을 담은 리스트

        for route in self.routes:
            truck_time_arrivals.append(self.new_time_arrival_per_route(route)['eTruck'])
            drone_time_arrivals.append(self.new_time_arrival_per_route(route)['eVTOL'])

        return truck_time_arrivals, drone_time_arrivals

    def new_time_arrival_per_route(self, route):
        truck_time = 0
        drone_time = 0
        waiting_time = []
        waiting_time_truck=[]
        truck_time_table = {'source': 0}  # 트럭 도착 시간을 담을 리스트
        drone_time_table = {'source': 0}  # 드론 도착 시간을 담을 리스트
        jump = 0
        flee = 0
        for idx in range(1, len(route)):
            catch = route[idx][1] == CATCH and route[idx][0] != 0
            catch_at_sink = route[idx][1] == CATCH and route[idx][0] == 0
            catch_with_fly = (route[idx][1] == FLY) and (route[idx-1][1] in [ONLY_TRUCK, ONLY_DRONE])
            if catch_at_sink:
                drone_time += ((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]))/ data["speed_d"]
                drone_time += data["takeoff_landing_time"]
                truck_time += (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) / data["speed_t"]
                truck_time_table.update({route[idx][0] : truck_time})
                drone_time_table.update({route[idx][0] : drone_time})
                flee = 0
                jump = 0

            elif catch or catch_with_fly:
                drone_time += ((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]))/ data["speed_d"]
                drone_time += data["takeoff_landing_time"]
                truck_time += (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) / data["speed_t"]
                longer_time = max(drone_time, truck_time)
                if drone_time < truck_time:
                    waiting_time_truck.append(truck_time-drone_time)
                waiting_time.append(abs(drone_time - truck_time))
                truck_time_table.update({route[idx][0] : longer_time})
                drone_time_table.update({route[idx][0] : longer_time})
                drone_time = longer_time
                truck_time = longer_time
                truck_time += data["service_time"]
                flee = 0
                jump = 0

            elif (route[idx][1] in [IDLE, FLY,None]) and (route[idx-1][1] not in [ONLY_TRUCK, ONLY_DRONE]):
                truck_time += (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) / data["speed_t"]
                truck_time_table.update({route[idx][0] : truck_time})
                drone_time = truck_time
                drone_time_table.update({route[idx][0] : drone_time})
                truck_time += data["service_time"]

            elif route[idx][1] == ONLY_DRONE:
                drone_time += ((data["edge_km_d"][route[idx-flee-1][0]][route[idx][0]]*data["curvy_air"]))/ data["speed_d"]
                drone_time += data["takeoff_landing_time"]
                drone_time_table.update({route[idx][0] : drone_time})
                truck_time_table.update({route[idx][0] : None})
                jump += 1
            elif route[idx][1] == ONLY_TRUCK:
                truck_time += (data["edge_km_t"][route[idx-jump-1][0]][route[idx][0]]*data["curvy_road"]) / data["speed_t"]
                truck_time_table.update({route[idx][0] : truck_time})
                truck_time += data["service_time"]
                drone_time_table.update({route[idx][0] : None})
                flee += 1

        return {'eTruck' : truck_time_table, 'eVTOL' : drone_time_table,'Waiting Truck' : waiting_time_truck, 'Over Waiting Time' : any(time > data["max_waiting_time"] for time in waiting_time)}
