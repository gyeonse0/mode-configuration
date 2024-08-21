import numpy as np
import random
import numpy.random as rnd
import shutil
import os

class FileReader:
    """
    multi_modal_data.vrp, multi_modal_data.sol 파일을 파싱하고 읽어오는 클래스
    TO DO : 새로운 data 입력 및 수정할 때마다 코드 추가 필요!!
    """
    def __init__(self):
        self.data = {
            "name": None,
            "type": None,
            "vehicles": None,
            "dimension": None,
            "num_t": None,
            "num_d": None,
            "maximum_system_duration": None,
            "service_time": None,
            "takeoff_landing_time": None, 
            "max_waiting_time": None,
            "init_soc": None,
            "max_soc": None,
            "min_soc_t": None,
            "min_soc_d": None,
            "demand_t": None,
            "demand_d": None,
            "mass_t":None,
            "mass_d":None,
            "lift_to_drag":None,
            "power_motor_prop":None,
            "power_elec":None,
            "cargo_limit_drone": None,
            "cargo_limit_truck": None,
            "battery_kwh_t": None,
            "battery_kwh_d": None,
            "energy_kwh/km_t": None,
            "energy_kwh/km_d": None,
            "charging_kw_d": None,
            "losistic_kwh/kg_t": None,
            "losistic_kwh/kg_d": None,
            "speed_t": None,
            "speed_d": None,
            "altitude": None,
            "curvy_air": None,
            "curvy_road": None,
            "temperature": None,
            "idle_t": None,
            "node_coord": {},
            "demand": {},
            "logistic_load": {},
            "availability_landing_spot": {},
            "customer_drone_preference": {},
            "price": {},
            "priority_delivery_time": {},
            "depot": None,
            "edge_km_d_type": None,
            "edge_km_t_type": None,
            "edge_km_d_format": None,
            "edge_km_t_format": None,
            "edge_km_d": [],
            "edge_km_t": [],
        }
        self.section = None

    def read_vrp_file(self, file_path):
        """
        multi_modal_data.vrp 파일을 읽어오고, parse_section_data 함수를 이용해서 섹션별로 딕셔너리에 data를 저장해주는 함수
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            keyword = parts[0]
            value = " ".join(parts[1:]).strip()

            if keyword in ["EDGE_KM_D_TYPE:", "EDGE_KM_T_TYPE:"]:
                self.parse_edge_km_type(keyword, value)
                continue
            if self.section and keyword != self.section:
                self.parse_section_data(line)
            if keyword == "EOF":
                break
            elif keyword == "NAME:":
                self.data["name"] = value
            elif keyword == "TYPE:":
                self.data["type"] = value
            elif keyword == "VEHICLES:":
                self.data["vehicles"] = int(value)
            elif keyword == "NUM_T:":
                self.data["num_t"] = int(value)
            elif keyword == "NUM_D:":
                self.data["num_d"] = int(value)
            elif keyword == "DIMENSION:":
                self.data["dimension"] = int(value)
            elif keyword == "MAXIMUM_SYSTEM_DURATION:":
                self.data["maximum_system_duration"] = int(value)
            elif keyword == "SERVICETIME:":
                self.data["service_time"] = int(value)
            elif keyword == "TAKEOFF_LANDING_TIME:":
                self.data["takeoff_landing_time"] = int(value)
            elif keyword == "MAX_WAITING_TIME:":
                self.data["max_waiting_time"] = int(value)
            elif keyword == "ALTITUDE:":
                self.data["altitude"] = int(value)
            elif keyword == "INIT_SOC:":
                self.data["init_soc"] = float(value)
            elif keyword == "MAX_SOC:":
                self.data["max_soc"] = float(value)
            elif keyword == "MIN_SOC_T:":
                self.data["min_soc_t"] = float(value)
            elif keyword == "MIN_SOC_D:":
                self.data["min_soc_d"] = float(value)
            elif keyword == "DEMAND_T:":
                self.data["demand_t"] = float(value)
            elif keyword == "DEMAND_D:":
                self.data["demand_d"] = float(value)
            elif keyword == "MASS_T:":
                self.data["mass_t"] = float(value)
            elif keyword == "MASS_D:":
                self.data["mass_d"] = float(value)  
            elif keyword == "LIFT_TO_DRAG:":
                self.data["lift_to_drag"] = float(value)
            elif keyword == "POWER_MOTOR_PROP:":
                self.data["power_motor_prop"] = float(value)
            elif keyword == "POWER_ELEC:":
                self.data["power_elec"] = float(value)       
            elif keyword == "CARGO_LIMIT_DRONE:":
                self.data["cargo_limit_drone"] = float(value)
            elif keyword == "CARGO_LIMIT_TRUCK:":
                self.data["cargo_limit_truck"] = float(value)
            elif keyword == "BATTERY_KWH_T:":
                self.data["battery_kwh_t"] = float(value)
            elif keyword == "BATTERY_KWH_D:":
                self.data["battery_kwh_d"] = float(value)
            elif keyword == "ENERGY_KWH/KM_T:":
                self.data["energy_kwh/km_t"] = float(value)
            elif keyword == "ENERGY_KWH/KM_D:":
                self.data["energy_kwh/km_d"] = float(value)
            elif keyword == "CHARGING_KW_D:":
                self.data["charging_kw_d"] = float(value)
            elif keyword == "CHARGING_KW_D:":
                self.data["charging_kw_d"] = float(value)
            elif keyword == "LOSISTIC_KWH/KG_T:":
                self.data["losistic_kwh/kg_t"] = float(value)
            elif keyword == "LOSISTIC_KWH/KG_D:":
                self.data["losistic_kwh/kg_d"] = float(value)
            elif keyword == "SPEED_T:":
                self.data["speed_t"] = float(value)
            elif keyword == "SPEED_D:":
                self.data["speed_d"] = float(value)
            elif keyword == "CURVY_AIR:":
                self.data["curvy_air"] = float(value)
            elif keyword == "CURVY_ROAD:":
                self.data["curvy_road"] = float(value)
            elif keyword == "Temperature:":
                self.data["temperature"] = float(value)
            elif keyword == "IDLE_T:":
                self.data["idle_t"] = float(value)
            elif keyword == "NODE_COORD_SECTION":    
                self.section = "NODE_COORD_SECTION"
            elif keyword == "DEMAND_SECTION":
                self.section = "DEMAND_SECTION"
            elif keyword == "LOGISTIC_LOAD_SECTION":
                self.section = "LOGISTIC_LOAD_SECTION"
            elif keyword == "AVAILABILITY_LANDING_SPOT_SECTION":
                self.section = "AVAILABILITY_LANDING_SPOT_SECTION"
            elif keyword == "CUSTOMER_DRONE_PREFERENCE_SECTION":
                self.section = "CUSTOMER_DRONE_PREFERENCE_SECTION"
            elif keyword == "PRICE_SECTION":
                self.section = "PRICE_SECTION"
            elif keyword == "PRIORITY_DELIVERY_TIME_SECTION":
                self.section = "PRIORITY_DELIVERY_TIME_SECTION"
            elif keyword == "DEPOT_SECTION":
                self.section = "DEPOT_SECTION"
            elif keyword == "EDGE_KM_D_FORMAT":
                self.data["edge_km_d_format"] = value
            elif keyword == "EDGE_KM_T_FORMAT":
                self.data["edge_km_t_format"] = value
            elif keyword == "EDGE_KM_D":
                self.section = "EDGE_KM_D"
                self.data["edge_km_d"] = []
            elif keyword == "EDGE_KM_T":
                self.section = "EDGE_KM_T"
                self.data["edge_km_t"] = []

        return self.data

    def parse_section_data(self, line):
        """
        multi_modal_data.vrp 파일의 데이터를 섹션별로 알맞게 파싱한 후 data를 저장해주는 함수
        """
        parts = line.split()
        if not parts or parts[0] == "EOF":
            return
        if self.section == "NODE_COORD_SECTION":
            self.parse_node_coord(parts)
        elif self.section == "DEMAND_SECTION":
            self.parse_demand(parts)
        elif self.section == "LOGISTIC_LOAD_SECTION":
            self.parse_logistic_load(parts)
        elif self.section == "AVAILABILITY_LANDING_SPOT_SECTION":
            self.parse_availability_landing_spot(parts)
        elif self.section == "CUSTOMER_DRONE_PREFERENCE_SECTION":
            self.parse_customer_drone_preference(parts)
        elif self.section == "PRICE_SECTION":
            self.parse_price(parts)
        elif self.section == "PRIORITY_DELIVERY_TIME_SECTION":
            self.parse_priority_delivery_time(parts)
        elif self.section == "DEPOT_SECTION":
            self.parse_depot(parts)
        elif self.section == "EDGE_KM_D":
            self.parse_edge_km_d(parts)
        elif self.section == "EDGE_KM_T":
            self.parse_edge_km_t(parts)

    def parse_node_coord(self, parts):
        try:
            node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            self.data["node_coord"][node_id] = (x, y)
        except (ValueError, IndexError):
            pass
    def parse_demand(self, parts):
        try:
            customer_id, demand = int(parts[0]), int(parts[1])
            self.data["demand"][customer_id] = demand
        except (ValueError, IndexError):
            pass
    def parse_logistic_load(self, parts):
        try:
            customer_id, load = int(parts[0]), int(parts[1])
            self.data["logistic_load"][customer_id] = load
        except (ValueError, IndexError):
            pass
    def parse_availability_landing_spot(self, parts):
        try:
            spot_id, availability = int(parts[0]), int(parts[1])
            self.data["availability_landing_spot"][spot_id] = availability
        except (ValueError, IndexError):
            pass
    def parse_customer_drone_preference(self, parts):
        try:
            customer_id, preference = int(parts[0]), int(parts[1])
            self.data["customer_drone_preference"][customer_id] = preference
        except (ValueError, IndexError):
            pass
    def parse_price(self, parts):
        try:
            customer_id, price = int(parts[0]), int(parts[1])
            self.data["price"][customer_id] = price
        except (ValueError, IndexError):
            pass
    def parse_priority_delivery_time(self, parts):
        try:
            customer_id, time = int(parts[0]), int(parts[1])
            self.data["priority_delivery_time"][customer_id] = time
        except (ValueError, IndexError):
            pass
    def parse_depot(self, parts):
        try:
            self.data["depot"] = int(parts[0])
        except (ValueError, IndexError):
            pass
    def parse_edge_km_d(self, parts):
        try:
            self.data["edge_km_d"].append(list(map(float, parts)))
        except (ValueError, IndexError):
            pass
    def parse_edge_km_t(self, parts):
        try:
            self.data["edge_km_t"].append(list(map(float, parts)))
        except (ValueError, IndexError):
            pass
    def parse_edge_km_type(self, keyword, value):
        if keyword == "EDGE_KM_D_TYPE:":
            self.data["edge_km_d_type"] = value
        elif keyword == "EDGE_KM_T_TYPE:":
            self.data["edge_km_t_type"] = value

    def read_sol_file(self, file_path):
        """
        multi_modal_data.sol 파일 읽어와서 파싱 후 데이터 저장해주는 함수
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        solution = {"routes": [], "cost": [], "vehicle_types": []}
        current_route = None

        for line in lines:
            if line.startswith("Route"):
                if current_route is not None:
                    solution["routes"].append(current_route)
                route_parts = line.split(":")[1].strip().split()
                current_route = list(map(int, route_parts))
            elif line.startswith("cost"):
                solution["cost"] = int(line.split()[1])
            elif line.startswith("Vehicle types"):
                vehicle_types_str = line.split(":")[1].strip()
                vehicle_types = [int(char) for char in vehicle_types_str if char.isdigit()]
                solution["vehicle_types"] = vehicle_types

        if current_route is not None:
            solution["routes"].append(current_route)

        return solution
    
    def randomize_vrp_data(self, data, time=None, max_value=None, key_seed=None, p_first=0.5):
        # 키 선택을 위한 랜덤 시드 설정
        if key_seed is not None:
            np.random.seed(key_seed)
        time_slots = np.array([0, 90])
        
        values = np.array(list(data.values()))
        values_except_depot = values[1:]

        if time is not None:
            p_rest = (1 - p_first) / (len(time_slots) - 1)
            p = [p_first] + [p_rest] * (len(time_slots) - 1)
            
            while True:
                new_values = np.random.choice(time_slots, size=values_except_depot.shape, p=p)
                if np.all(values[node] > time[0][node] for node in data.keys()):
                    break
            values_except_depot[:] = new_values
        elif max_value is not None:
            values_except_depot[:] = np.random.randint(1, max_value, size=values_except_depot.shape)
        else:
            p = 0.8
            values_except_depot[:] = np.random.binomial(n=1, p=p, size=values_except_depot.shape)
        
        updated_data = {key: value for key, value in zip(data.keys(), values)}

        return updated_data

    
    def write_vrp_file(self, output_file_path, data):
        with open(output_file_path, 'w') as file:
            # file.write("NODE_COORD_SECTION\n")
            # for node_id, coord in data['node_coord'].items():
            #     file.write("{} {:.6f} {:.6f}\n".format(node_id, coord[0], coord[1]))

            # file.write("\n")  # 공백 라인 추가

            ## DEMAND_SECTION 추가
            # file.write("DEMAND_SECTION\n")
            # for load in data['demand'].values():
            #     file.write("{}\n".format(load))

            # file.write("\n")  # 공백 라인 추가

            # LOGISTIC_LOAD_SECTION 추가
            file.write("LOGISTIC_LOAD_SECTION\n")
            for load in data['logistic_load'].values():
                file.write("{}\n".format(load))

            file.write("\n")  # 공백 라인 추가

            # AVAILABILITY_LANDING_SPOT_SECTION 추가
            file.write("AVAILABILITY_LANDING_SPOT_SECTION\n")
            for availability in data['availability_landing_spot'].values():
                file.write("{}\n".format(availability))

            file.write("\n")  # 공백 라인 추가

            # CUSTOMER_DRONE_PREFERENCE_SECTION 추가
            file.write("CUSTOMER_DRONE_PREFERENCE_SECTION\n")
            for node_number, pref in data['customer_drone_preference'].items():
                file.write("{}\t{}\n".format(node_number, pref))

            file.write("\n")  # 공백 라인 추가

            # PRICE_SECTION 추가
            file.write("PRICE_SECTION\n")
            for pref in data['price'].values():
                file.write("{}\n".format(pref))

            file.write("\n")  # 공백 라인 추가

            # PRIORITY_DELIVERY_TIME_SECTION 추가
            file.write("PRIORITY_DELIVERY_TIME_SECTION\n")
            for pref in data['priority_delivery_time'].values():
                file.write("{}\n".format(pref))

            file.write("\n")  # 공백 라인 추가

        print("VRP 파일이 작성되었습니다.")

    def update_vrp_file(self, vrp_file_path, updated_data):
        # Target SECTIONS
        target_sections = ["LOGISTIC_LOAD_SECTION", "AVAILABILITY_LANDING_SPOT_SECTION", "CUSTOMER_DRONE_PREFERENCE_SECTION", "PRICE_SECTION", "PRIORITY_DELIVERY_TIME_SECTION"]
        # 원래 파일 읽기
        with open(vrp_file_path, 'r') as f:
            lines = f.readlines()

        # # 수정된 데이터를 읽어옴
        # with open(updated_data_path, "r") as file:
        #     updated_data = file.readlines()

        # updated_data_dict = {}
        # for line in updated_data:
        #     parts = line.strip().split()
        #     if len(parts) >= 2:
        #         node_number = parts[0]
        #         data = "\t".join(parts[1:])
        #         updated_data_dict[node_number] = data

        for i, line in enumerate(lines):
            section = line.strip()
            target_section = section[:-8].lower()
            if section in target_sections:
                # CUSTOMER_DRONE_PREFERENCE_SECTION을 찾으면 그 다음 줄부터 수정된 데이터 적용
                j = i + 1
                while j < len(lines) and lines[j].strip() != "":
                    parts = lines[j].strip().split()
                    if len(parts) >= 1:
                        node_number = int(parts[0])
                        if node_number in updated_data[target_section]:
                            lines[j] = f"{node_number}\t{updated_data[target_section][node_number]}\n"
                    j += 1

        with open(vrp_file_path, 'w') as f:
            f.writelines(lines)

        print("VRP 파일이 작성되었습니다.")

    def create_backup(self, vrp_file_path, backup_file_path):
        # 원본 파일을 백업 파일로 복사
        shutil.copy2(vrp_file_path, backup_file_path)

    def restore_file(self, backup_file_path, original_file_path):
        shutil.copy2(backup_file_path, original_file_path)

    def restore_on_exit(self, backup_file_path, vrp_file_path):
        self.restore_file(backup_file_path, vrp_file_path)
        print("파일이 원래 상태로 복원되었습니다.")