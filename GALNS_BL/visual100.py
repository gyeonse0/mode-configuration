import os
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster

# 주어진 좌표 데이터
node_coordinates = {
    0: (37.54701783, 127.0665273),
    1: (37.55193887, 127.0693827),
    2: (37.55534026, 127.0712975),
    3: (37.5574868, 127.0725526),
    4: (37.55218712, 127.0686201),
    5: (37.55118021, 127.0677194),
    6: (37.55073854, 127.0687074),
    7: (37.54900258, 127.0677474),
    8: (37.55034437, 127.0670679),
    9: (37.54957429, 127.0664413),
    10: (37.54951566, 127.0665318),
    11: (37.55030467, 127.0671618),
    12: (37.5511351, 127.0678257),
    13: (37.55874099, 127.0756461),
    14: (37.55879373, 127.0749003),
    15: (37.55609891, 127.0824731),
    16: (37.556267, 127.081557),
    17: (37.55449655, 127.0865505),
    18: (37.55120847, 127.0901492),
    19: (37.55242293, 127.0892452),
    20: (37.554926, 127.085229),
    21: (37.54891832, 127.0934986),
    22: (37.54895793, 127.0926145),
    23: (37.545741, 127.102649),
    24: (37.56153286, 127.075203),
    25: (37.56238618, 127.0752287),
    26: (37.55921103, 127.0775414),
    27: (37.55908662, 127.0776319),
    28: (37.56026752, 127.0782794),
    29: (37.56094951, 127.0784952),
    30: (37.56245368, 127.0794725),
    31: (37.56261501, 127.0794161),
    32: (37.5652136, 127.0809935),
    33: (37.56229075, 127.0818721),
    34: (37.55988373, 127.0808553),
    35: (37.55984027, 127.0811552),
    36: (37.56308993, 127.0828472),
    37: (37.56388824, 127.0833601),
    38: (37.56207257, 127.0870619),
    39: (37.55995657, 127.0873831),
    40: (37.55831636, 127.0876755),
    41: (37.56148909, 127.0839868),
    42: (37.56640123, 127.0778997),
    43: (37.56579555, 127.077177),
    44: (37.56549744, 127.0809972),
    45: (37.56634043, 127.081615),
    46: (37.56685307, 127.0817412),
    47: (37.56863463, 127.0828854),
    48: (37.56849153, 127.0826419),
    49: (37.56791599, 127.0860566),
    50: (37.56793345, 127.0855902),
}


# 노드 좌표 추출
x_coords, y_coords = zip(*node_coordinates.values())

# Matplotlib를 사용하여 좌표 시각화
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, color='red', marker='o')

# depot(0번 노드)를 파란색 동그라미로 표시
plt.plot(node_coordinates[0][0], node_coordinates[0][1], 'bo')

# 각 노드에 번호 표시
for node_id, (x, y) in node_coordinates.items():
    plt.text(x, y, str(node_id), fontsize=8, ha='right')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Node Coordinates Visualization')
plt.grid(True)

# Folium을 사용하여 지도 상에 노드 좌표 표시
map_center = [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]
mymap = folium.Map(location=map_center, zoom_start=12,lang='en')

# Depot을 파란색 동그라미로 표시
folium.CircleMarker(location=[node_coordinates[0][0], node_coordinates[0][1]], radius=6, color='red', fill=True).add_to(mymap)

# 나머지 노드들을 빨간색 점으로 표시
for node_id, (lat, lon) in node_coordinates.items():
    if node_id != 0:
        folium.CircleMarker(location=[lat, lon], radius=1, color='blue', fill=True, popup=f'Node {node_id}').add_to(mymap)

# 바탕 화면 경로 설정
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
# Matplotlib와 Folium을 함께 사용하기 위해 HTML 파일로 저장
html_file_path = os.path.join(desktop_path, 'node_coordinates_map.html')
mymap.save(html_file_path)

# 결과 표시
plt.show()