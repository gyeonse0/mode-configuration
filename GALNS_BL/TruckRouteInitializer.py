import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List


class TruckRouteInitializer:
    """
    트럭과 드론의 path 분할에 가장 기초가 되는 트럭만의 route를 NN으로 intialize 하는 클래스
    """
    def __init__(self, data):
        self.data = data

    def neighbors_init_truck(self, customer):
        """
        truck의 distance(km) edge data를 기반으로, 해당 customer의 neighbor 노드 탐색
        """
        locations = np.argsort(self.data["edge_km_t"][customer])
        return locations[locations != 0]
    
    def validate_truck_routes(self,truck_routes):
        """
        모든 트럭의 경로가 한번씩의 주행으로 수요를 만족하는지 검증하는 함수/ 만족하면 pass, 만족하지 않으면 error 발생
        """
        for route in truck_routes:
            consecutive_zeros = sum(1 for loc in route if loc == 0)
            if consecutive_zeros > 2:
                raise ValueError("Unable to satisfy demand with the given number of trucks!!")
            
    def nearest_neighbor_init_truck(self):
        """
        트럭의 capacity 조건을 만족하면서, 가까우면서, 방문한적 없는 노드를 truck_init_route에 순차적으로 append하여 
        truck_init_routes 결정 (num_t로 트럭의 fleet 수 고려)-> 이를 통해 딕셔너리 형태로 route를 저장하고, RouteGenerator의 input route로 적용
        """
        truck_init_routes = [[] for _ in range(self.data["num_t"])]
        unvisited = set(range(1, self.data["dimension"]))

        while unvisited:
            for i in range(self.data["num_t"]):
                route = [0] 
                route_demands = 0

                while unvisited:
                    current = route[-1]
                    neighbors = [nb for nb in self.neighbors_init_truck(current) if nb in unvisited]
                    nearest = neighbors[0]

                    if route_demands + self.data["demand"][nearest] > self.data["capacity_t"]:
                        break

                    route.append(nearest)
                    unvisited.remove(nearest)
                    route_demands += self.data["demand"][nearest]

                route.append(0) 
                truck_init_routes[i].extend(route[0:])
        
        self.validate_truck_routes(truck_init_routes)

        return {
            'num_t': len(truck_init_routes),
            'num_d': 0,
            'route': [{'vtype': 'truck', 'vid': f't{i+1}', 'path': path} for i, path in enumerate(truck_init_routes)]
        }