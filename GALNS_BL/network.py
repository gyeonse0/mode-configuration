from haversine import haversine
import numpy as np
import os

# 강남구 위경도 범위
lat_min, lat_max = 37.4643683, 37.530642
lon_min, lon_max = 127.0227814, 127.1114

# 디포 위치 설정 (강남구 중심)
depot_lat, depot_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2

# 노드 수 (디포를 제외한 49개)
num_nodes = 49

# 지니 계수를 사용하여 노드의 위경도를 분산시킴
def distribute_nodes_with_gini(num_nodes, lat_min, lat_max, lon_min, lon_max, gini_coefficient):
    cluster_center = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    cluster_std_lat = (lat_max - lat_min) * (1 - gini_coefficient) / 2
    cluster_std_lon = (lon_max - lon_min) * (1 - gini_coefficient) / 2
    
    latitudes = np.random.normal(cluster_center[0], cluster_std_lat, num_nodes)
    longitudes = np.random.normal(cluster_center[1], cluster_std_lon, num_nodes)
    
    latitudes = np.clip(latitudes, lat_min, lat_max)
    longitudes = np.clip(longitudes, lon_min, lon_max)
    
    return latitudes, longitudes

# 지니 계수 설정 (0에서 1 사이의 값, 예: 0.7)
gini_coefficient = 0.7

# 노드 배치
lat_nodes, lon_nodes = distribute_nodes_with_gini(num_nodes, lat_min, lat_max, lon_min, lon_max, gini_coefficient)

# 디포 위치 포함하여 모든 노드
lat_all = np.append(lat_nodes, depot_lat)
lon_all = np.append(lon_nodes, depot_lon)

# 바탕화면 경로 설정
desktop_path = "C:/Users/User/OneDrive/바탕 화면/coordinates_and_distances.txt"

# 디렉토리가 존재하는지 확인하고, 존재하지 않으면 생성
desktop_dir = os.path.dirname(desktop_path)
if not os.path.exists(desktop_dir):
    os.makedirs(desktop_dir)

# 위경도 데이터 및 거리 행렬 저장
with open(desktop_path, 'w') as file:
    file.write("NODE_COORD_SECTION\n")
    file.write(f"0\t{depot_lat:.8f}\t{depot_lon:.8f}\n")
    for i in range(num_nodes):
        file.write(f"{i+1}\t{lat_nodes[i]:.8f}\t{lon_nodes[i]:.8f}\n")
    
    file.write("\nEDGE_KM_D\n")
    
    # 유클리드 거리 행렬 계산 및 저장
    euclidean_distances_matrix = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            if i != j:
                euclidean_distances_matrix[i, j] = haversine((lat_all[i], lon_all[i]), (lat_all[j], lon_all[j]))
    
    for i in range(50):
        file.write('\t'.join([f"{euclidean_distances_matrix[i, j]:.8f}" for j in range(50)]) + '\n')
    
    file.write("\nEDGE_KM_T\n")
    
    # 맨해튼 거리 행렬 계산 및 저장
    manhattan_distances_matrix = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            if i != j:
                lat1, lon1 = lat_all[i], lon_all[i]
                lat2, lon2 = lat_all[j], lon_all[j]
                horizontal_distance = haversine((lat1, lon1), (lat1, lon2))
                vertical_distance = haversine((lat1, lon2), (lat2, lon2))
                manhattan_distance = horizontal_distance + vertical_distance
                manhattan_distances_matrix[i, j] = manhattan_distance
    
    for i in range(50):
        file.write('\t'.join([f"{manhattan_distances_matrix[i, j]:.8f}" for j in range(50)]) + '\n')
