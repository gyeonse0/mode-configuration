
import os

# 현재 파일 경로, 부모 경로, 조부모 경로 설정
current_file_path = os.path.abspath(__file__)
parent_file_path = os.path.dirname(current_file_path)
grandparent_file_path = os.path.dirname(parent_file_path)

# vrp 파일 경로 설정
vrp_file_path = os.path.join(parent_file_path, 'data', 'multi_modal_data.vrp')

# 두 단계 전 경로의 디렉토리 이름을 얻기
folder_name = os.path.basename(parent_file_path)

# 변수 설정
input_variable = 'speed_t'  # 'servicetime' 또는 'speed_t'로 설정

if input_variable == 'servicetime':
    new_values = [1, 3, 7, 9]  # SERVICETIME
    variable_label = 'servicetime'
    old_variable = 'SERVICETIME: 5'
elif input_variable == 'speed_t':
    new_values = [0.1, 0.5, 0.7, 0.9]  # SPEED_T
    variable_label = 'speed_t'
    old_variable = 'SPEED_T: 0.3'
else:
    raise ValueError("input_variable은 'servicetime' 또는 'speed_t'이어야 합니다.")

# 파일 내용 수정 함수
def modify_vrp_file(file_path, old_variable, new_value):
    with open(file_path, 'r') as file:
        data = file.read()

    data = data.replace(old_variable, f'{old_variable.split(":")[0]}: {new_value}')

    # 수정된 내용 저장
    with open(file_path, 'w') as file:
        file.write(data)

# 파일명 변경 작업 및 파일 내용 수정 작업 수행
for i, new_value in enumerate(new_values):
    old_name_suffix = '' if i == 0 else f' ({i+1})'
    
    old_names = [f'GALNS_BL - 복사본{old_name_suffix}', f'GALNS_FLP - 복사본{old_name_suffix}', f'GALNS_FC - 복사본{old_name_suffix}']
    
    new_names_with_prefix = [f'GALNS_BL_{variable_label}_{new_value}', f'GALNS_FLP_{variable_label}_{new_value}', f'GALNS_FC_{variable_label}_{new_value}']
    
    for old_name, new_name_with_prefix in zip(old_names, new_names_with_prefix):
        old_path = os.path.join(grandparent_file_path, old_name)
        new_path = os.path.join(grandparent_file_path, new_name_with_prefix)
        
        # 폴더가 존재할 때만 이름 변경
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            
            # vrp 파일 수정
            modify_vrp_file(os.path.join(new_path, 'data', 'multi_modal_data.vrp'), old_variable, new_value)