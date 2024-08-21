import pandas as pd
import requests
import json
import folium
from shapely.geometry import Polygon, MultiPolygon, Point
import random

# 서울 구 데이터 가져오기
API_key = '5357464a61716b7236306b6a746b45'
service = 'SearchFAQOfGUListService'
gu_url = f'http://openapi.seoul.go.kr:8088/{API_key}/json/{service}/1/25'
gu_list = requests.get(gu_url).json()
df_gu = pd.DataFrame(gu_list['SearchFAQOfGUListService']['row'])

# GeoJSON 데이터 가져오기
gu_json = []
vwolrd_key = '3C36C01F-FC3D-302C-BEB6-E697864F7DF3'
for gu in df_gu['CD_DESC']:
    url_vworld = f'https://api.vworld.kr/req/data?service=data&version=2.0&request=GetFeature&format=json&errorformat=json&size=10&page=1&data=LT_C_ADSIGG_INFO&attrfilter=sig_kor_nm:like:{gu}&columns=sig_cd,full_nm,sig_kor_nm,sig_eng_nm,ag_geom&geometry=true&attribute=true&key={vwolrd_key}&domain=https://localhost'
    result_dict = requests.get(url_vworld).json()
    gu_json.append(result_dict)

# GeoJSON 데이터 가공
features = []
for gu_data in gu_json:
   gu_name = gu_data['response']['result']['featureCollection']['features'][0]['properties']['sig_kor_nm']
   geometry = gu_data['response']['result']['featureCollection']['features'][0]['geometry']
   feature = {
       "type": "Feature",
       "id": gu_name,  # 구명을 id로 추가
       "geometry": geometry,
       "properties": {
           "name": gu_name
       }
   }
   features.append(feature)

geojson_data = {
   "type": "FeatureCollection",
   "features": features
}

# GeoJSON 파일 경로
file_path = 'C:/Users/User/OneDrive/바탕 화면/seoul_gu_boundaries.geojson'

# GeoJSON 파일로 저장
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(geojson_data, f, ensure_ascii=False, indent=4)

# 랜덤 포인트 생성 함수
def generate_random_points(polygon, num_points):
    points = []
    while len(points) < num_points:
        # 랜덤 위도와 경도 생성
        lon = random.uniform(polygon.bounds[0], polygon.bounds[2])
        lat = random.uniform(polygon.bounds[1], polygon.bounds[3])
        point = Point(lon, lat)
        
        # 폴리곤 내부에 있는지 확인
        if polygon.contains(point):
            points.append((lat, lon))
    return points

# Folium 지도 생성
m = folium.Map(location=[37.5651, 126.98955], zoom_start=11, tiles='cartodb dark_matter')

# GeoJSON 레이어 추가
folium.GeoJson(
    geojson_data,
    style_function=lambda feature: {
        'opacity': 0.7,
        'weight': 1,
        'color': 'white',
        'fillOpacity': 0.2,
        'dashArray': '5, 5',
    }
).add_to(m)

# 각 구의 경계선 폴리곤을 Shapely 폴리곤으로 변환하고 랜덤 포인트 추가
for feature in features:
    gu_name = feature['id']
    coords = feature['geometry']['coordinates']
    
    if feature['geometry']['type'] == 'Polygon':
        # 단일 폴리곤 처리
        polygon = Polygon(coords)
        random_points = generate_random_points(polygon, 1)
        
        for point in random_points:
            folium.CircleMarker(
                location=point,
                radius=2,
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)
    
    elif feature['geometry']['type'] == 'MultiPolygon':
        # 다중 폴리곤 처리
        for coord in coords:
            polygon = Polygon(coord[0])
            random_points = generate_random_points(polygon, 1)
            
            for point in random_points:
                folium.CircleMarker(
                    location=point,
                    radius=1,
                    color='red',
                    fill=True,
                    fill_color='red'
                ).add_to(m)

# 지도 저장
m.save('C:/Users/User/OneDrive/바탕 화면/seoul_gu_boundaries_with_points.html')
