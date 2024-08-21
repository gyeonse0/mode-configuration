import numpy as np
import random

from MultiModalState import *

class RouteInitializer:
    def __init__(self, data):
        self.data = data

    def neighbors_init_truck(self, customer):
        locations = np.argsort(self.data["edge_km_t"][customer])
        return locations[locations != 0]

    def validate_truck_routes(self, truck_routes):
        for route in truck_routes:
            consecutive_zeros = sum(1 for loc in route if loc == 0)
            if consecutive_zeros > 2:
                raise ValueError("Unable to satisfy demand with the given number of trucks!!")

    def nearest_neighbor_init_truck(self):
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

                    if route_demands + self.data["demand"][nearest] > self.data["demand_t"]:
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
            'route': [{'vtype': 'truck', 'vid': f't{i + 1}', 'path': path} for i, path in enumerate(truck_init_routes)]
        }
    
    def init_truck(self):
        # nearest_neighbor_init_truck() 메서드 호출
        truck_init_routes = self.nearest_neighbor_init_truck()
        init_truck=[]
        # 각 경로를 튜플로 변환
        for route in truck_init_routes['route']:
            tuple_route = [(node, 0) for node in route['path']]
            init_truck.append(tuple_route)
        
        return MultiModalState(init_truck)


    def dividing_route(self, route_with_info, route_index):
        truck_route = [value for value in route_with_info if value[1] != 2]
        drone_route = [value for value in route_with_info if value[1] != 4]

        return [
            {'vtype': 'drone', 'vid': 'd' + str(route_index + 1), 'path': drone_route},
            {'vtype': 'truck', 'vid': 't' + str(route_index + 1), 'path': truck_route},
        ]
        
    
    def combine_paths(self, route_data):
        combined_paths = []
        initial_solution = self.nearest_neighbor_init_truck()
        nn_truck_paths = [route['path'] for route in initial_solution['route']]
        
        for i in range(len(route_data)):  # 모든 드론 및 트럭 경로에 대해 반복
            if i % 2 == 0:  # 짝수 인덱스인 경우 드론 경로
                nn_truck_path = nn_truck_paths[i // 2]
                drone_route = route_data[i]['path']
                truck_route = route_data[i + 1]['path']
                
                filled_path = []
                for node, value in drone_route[:-1]:
                    if node not in [point[0] for point in filled_path]:
                        filled_path.append((node, value))
                
                for node, value in truck_route[:-1]:
                    if node not in [point[0] for point in filled_path]:
                        filled_path.append((node, value))
            
                filled_path = sorted(filled_path, key=lambda x: nn_truck_path[:-1].index(x[0]))
                filled_path.append(drone_route[-1])
                combined_paths.append(filled_path)
        
        return combined_paths
