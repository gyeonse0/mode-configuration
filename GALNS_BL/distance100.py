from haversine import haversine
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

# 주어진 좌표값 (위도, 경도)
coordinates = np.array([
    (	37.51631968	,	127.0423148	)	,
    (	37.489542	,	127.0796221	)	,
    (	37.48496029	,	127.0594065	)	,
    (	37.51718924	,	127.0413014	)	,
    (	37.49816465	,	127.0283079	)	,
    (	37.51119883	,	127.0216579	)	,
    (	37.508862	,	127.063073	)	,
    (	37.53360292	,	127.0283258	)	,
    (	37.48541757	,	127.1045604	)	,
    (	37.52046807	,	127.054044	)	,
    (	37.49400035	,	127.0658114	)	,
    (	37.48473577	,	127.0876176	)	,
    (	37.49044865	,	127.039997	)	,
    (	37.49354047	,	127.0505467	)	,
    (	37.48772527	,	127.0360742	)	,
    (	37.48615258	,	127.056121	)	,
    (	37.49561749	,	127.0577682	)	,
    (	37.48790982	,	127.0981867	)	,
    (	37.50044687	,	127.0605134	)	,
    (	37.50450288	,	127.0489425	)	,
    (	37.48407358	,	127.0844518	)	,
    (	37.47903282	,	127.0657076	)	,
    (	37.52643082	,	127.0284809	)	,
    (	37.49605364	,	127.0788531	)	,
    (	37.47711586	,	127.0518861	)	,
    (	37.48613775	,	127.0424159	)	,
    (	37.49903766	,	127.0392494	)	,
    (	37.51201518	,	127.0575397	)	,
    (	37.50393779	,	127.0603856	)	,
    (	37.52509008	,	127.0513	)	,
    (	37.49160529	,	127.058004	)	,
    (	37.48661527	,	127.0471989	)	,
    (	37.49708719	,	127.0449782	)	,
    (	37.50002424	,	127.0365086	)	,
    (	37.49358351	,	127.0780732	)	,
    (	37.50882291	,	127.0645418	)	,
    (	37.48698084	,	127.106951	)	,
    (	37.49267955	,	127.0306305	)	,
    (	37.49233075	,	127.0745897	)	,
    (	37.47858842	,	127.1114094	)	,
    (	37.48698084	,	127.106951	)	,
    (	37.50528056	,	127.0289194	)	,
    (	37.49147536	,	127.0526126	)	,
    (	37.46682003	,	127.0969337	)	,
    (	37.48005839	,	127.0641545	)	,
    (	37.52269079	,	127.0328729	)	,
    (	37.49828674	,	127.0301038	)	,
    (	37.48811677	,	127.1030688	)	,
    (	37.49012921	,	127.071565	)	,
    (	37.48483867	,	127.0897795	)		
])

# 맨해튼 거리를 저장할 행렬 초기화
haversine_manhattan_distances_matrix = np.zeros((50, 50))

# 좌표 간의 맨해튼 거리 계산 (가로 세로 각각에 대해 하버사인 적용)
for i in range(50):
    for j in range(50):
        if i != j:
            # 좌표 간의 가로, 세로 거리 계산
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            horizontal_distance = haversine((lat1, lon1), (lat1, lon2))
            vertical_distance = haversine((lat1, lon2), (lat2, lon2))
            
            # 가로 세로 거리의 합을 맨해튼 거리로 사용
            haversine_manhattan_distance = horizontal_distance + vertical_distance
            
            # 맨해튼 거리를 행렬에 저장
            haversine_manhattan_distances_matrix[i, j] = haversine_manhattan_distance

# 맨해튼 거리 데이터프레임 생성
df_manhattan = pd.DataFrame(haversine_manhattan_distances_matrix)

# 엑셀 파일로 저장 (바탕 화면 경로)
desktop_path = "C:/Users/User/OneDrive/바탕 화면/"
df_manhattan.to_excel(desktop_path + 'haversine_manhattan_distances.xlsx', index=False)

# 유클리드 거리를 저장할 행렬 초기화
haversine_distances_matrix = np.zeros((50, 50))

# 좌표 간의 유클리드 거리 계산 (haversine 함수 적용)
for i in range(50):
    for j in range(50):
        if i != j:
            # 주석으로 표시된 부분이 행렬의 값입니다.
            haversine_distances_matrix[i, j] = haversine(coordinates[i], coordinates[j])

# 유클리드 거리 데이터프레임 생성
df_euclidean = pd.DataFrame(haversine_distances_matrix)

# 엑셀 파일로 저장 (바탕 화면 경로)
df_euclidean.to_excel(desktop_path + 'haversine_euclidean_distances.xlsx', index=False)
