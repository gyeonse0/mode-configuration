import random

def extract_first_elements(path):
    return [x[0] for x in path]

def find_random_sortie(routes):
    return [generate_subroutes(extract_first_elements(route)) for route in routes]

    
max_drone_mission = 4
k = 2
l = 1

def generate_subroutes(each_route):
    SERVICE = 0
    CATCH = 0
    only_drone_index = []
    fly_node_index = []
    catch_node_index = []
    subroutes = []
    depot_end = len(each_route) - 1
    while len(subroutes) < max_drone_mission:
        FLY = random.choice(range(CATCH, len(each_route)))
        SERVICE = FLY + k
        CATCH = SERVICE + l
        if CATCH > depot_end:
            break
        subroute = list(range(FLY, CATCH + 1))
        subroutes.append(subroute)
        fly_node_index.append(FLY)
        only_drone_index.append(SERVICE)
        catch_node_index.append(CATCH)
    visit_type = [0] * len(each_route)
    visit_type = [
        1 if index in fly_node_index else
        2 if index in only_drone_index else
        3 if index in catch_node_index else
        0 for index in range(len(visit_type))
    ]
    for i in [
        i for subroute in subroutes
        for i in subroute[1:-1] if i not in only_drone_index
    ]:
        visit_type[i] = 4
    return list(zip(each_route, visit_type))


def dividing_route(route_with_info, route_index):
    
    truck_route = [value for value in route_with_info if value[1] != 2]
    drone_route = [value for value in route_with_info if value[1] != 4]

    return [
        {'vtype': 'drone', 'vid': str(route_index + 1), 'path': drone_route},
        {'vtype': 'truck', 'vid': str(route_index + 1), 'path': truck_route},
    ]

def apply_dividing_route_to_routes(routes):
    result = []
    for route_index, route_with_info in enumerate(routes):
        result.extend(dividing_route(route_with_info, route_index))
    return result

"""

def make(self):
    #리스트 오브 튜플을 받아 makemake하기
    empty_list = []

    for route_index, route_info in enumerate(self.routes):
        self.depot_end = len(route_info['path']) - 1
        # self.can_fly = len(self.routes['path']) - k - l
        self.SERVICE = 0
        self.CATCH = 0
        self.only_drone_index = []
        self.fly_node_index = []
        self.catch_node_index = []
        self.subroutes = []
        self.generate_subroutes(self.extract_first_elements(route_info['path']))
        
        diclist = {'vtype': 'truck', 'vid': 't'+ str(route_index + 1), 'path': self.route_tuples(route_info['path'])}
        
        empty_list.append(diclist)

    return {
        'num_t' : int(len(empty_list)/2),
        'num_d' : int(len(empty_list)/2),
        'route' : empty_list
    }

def makemake(self):
    #단순히 정보를 가진 튜플만 생성하는 함수
    empty_list = []

    for route_index, route_info in enumerate(self.routes):
        self.depot_end = len(route_info['path']) - 1
        # self.can_fly = len(self.routes['path']) - k - l
        self.SERVICE = 0
        self.CATCH = 0
        self.only_drone_index = []
        self.fly_node_index = []
        self.catch_node_index = []
        self.subroutes = []
        self.generate_subroutes(route_info['path'])
        
        diclist = {'vtype': 'truck', 'vid': 't'+ str(route_index + 1), 'path': self.route_tuples(route_info['path'])}
        
        empty_list.append(diclist)

    return {
        'num_t' : int(len(empty_list)/2),
        'num_d' : int(len(empty_list)/2),
        'route' : empty_list
    }

def makemakemake(self):
    empty_list = []

    for route_index, route_info in enumerate(self.routes):
        self.depot_end = len(route_info['path']) - 1
        # self.can_fly = len(self.routes['path']) - k - l
        self.SERVICE = 0
        self.CATCH = 0
        self.only_drone_index = []
        self.fly_node_index = []
        self.catch_node_index = []
        self.subroutes = []
        self.generate_subroutes(route_info['path'])
        diclist = self.dividing_route(self.route_tuples(route_info['path']), route_index)
        
        empty_list.extend(diclist)

    return {
        'num_t' : int(len(empty_list)/2),
        'num_d' : int(len(empty_list)/2),
        'route' : empty_list
    }

"""