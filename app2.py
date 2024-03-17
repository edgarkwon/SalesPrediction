import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle


# Haversine 공식을 이용하여 거리 계산
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # 지구의 반지름 (km 단위)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

# Streamlit 앱의 제목 설정
st.title('카드 MVP')

# 사용자 입력 받기
location_query = st.text_input('검색할 위치를 입력하세요:', '')

if location_query:
    # Nominatim API를 사용하여 위치 검색
    response = requests.get(f"https://nominatim.openstreetmap.org/search?format=json&q={location_query}")
    locations = response.json()

    # 검색된 위치들의 목록 생성
    location_options = [f"{loc['display_name']} (위도: {loc['lat']}, 경도: {loc['lon']})" for loc in locations]
    
    # Streamlit을 사용하여 위치 목록을 드롭다운으로 표시
    selected_location = st.selectbox('검색 결과에서 위치를 선택하세요:', location_options)

    if selected_location:
        # 사용자가 선택한 위치의 위도와 경도 출력
        selected_index = location_options.index(selected_location)
        lat = float(locations[selected_index]['lat'])
        lon = float(locations[selected_index]['lon'])

        # 'area.csv'에서 가장 가까운 기록 찾기
        df = pd.read_csv('area.csv')
        df['distance'] = df.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
        closest_record = df.loc[df['distance'].idxmin()]

        # 해당 기록의 정보 출력
        st.write("가장 가까운 기록의 정보:")
        st.write(closest_record)
