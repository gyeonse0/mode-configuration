import pandas as pd
import requests
import json
import folium

API_key = '5357464a61716b7236306b6a746b45'
service = 'SearchFAQOfGUListService' 
gu_url = f'http://openapi.seoul.go.kr:8088/{API_key}/json/{service}/1/25'
gu_list = requests.get(gu_url).json()
print(gu_list)
df_gu = pd.DataFrame(gu_list['SearchFAQOfGUListService']['row'])
print(df_gu)

gu_json = []
vwolrd_key = '3C36C01F-FC3D-302C-BEB6-E697864F7DF3'
for gu in df_gu['CD_DESC']:
    url_vworld = f'https://api.vworld.kr/req/data?service=data&version=2.0&request=GetFeature&format=json&errorformat=json&size=10&page=1&data=LT_C_ADSIGG_INFO&attrfilter=sig_kor_nm:like:{gu}&columns=sig_cd,full_nm,sig_kor_nm,sig_eng_nm,ag_geom&geometry=true&attribute=true&key={vwolrd_key}&domain=https://localhost'
    result_dict = requests.get(url_vworld).json()
    gu_json.append(result_dict)

features = []
for gu_data in gu_json:  # gu_json 25개 구의 API 응답 데이터 리스트
   gu_name = gu_data['response']['result']['featureCollection']['features'][0]['properties']['sig_kor_nm']
   feature = {
       "type": "Feature",
       "id": gu_name,  # 구명을 id로 추가
       "geometry": gu_data['response']['result']['featureCollection']['features'][0]['geometry'],
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

# GeoJSON 파일을 읽어 JSON 데이터로 로드
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(geojson_data, f, ensure_ascii=False, indent=4)

def style_function(feature):
    return {
        'opacity': 0.7,
        'weight': 1,
        'color': 'white',
        'fillOpacity': 0.2,
        'dashArray': '5, 5',
    }

m = folium.Map(
    location=[37.5651, 126.98955], 
    zoom_start=11,
    tiles='cartodb dark_matter'
    )
    

# GeoJSON 레이어 추가
folium.GeoJson(
    geojson_data,
    style_function=style_function
).add_to(m)
