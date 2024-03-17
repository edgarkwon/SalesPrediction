import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle

# Haversine formula for distance calculation
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

# Streamlit app title
st.title('카드 MVP')

# User input
location_query = st.text_input('검색할 위치를 입력하세요:', '')

if location_query:
    # Using Nominatim API for location search
    response = requests.get(f"https://nominatim.openstreetmap.org/search?format=json&q={location_query}")
    locations = response.json()

    # Creating a list of search results
    location_options = [f"{loc['display_name']} (위도: {loc['lat']}, 경도: {loc['lon']})" for loc in locations]

    # Displaying location options in a dropdown
    selected_location = st.selectbox('검색 결과에서 위치를 선택하세요:', location_options)

    if selected_location:
        # Extracting latitude and longitude of the selected location
        selected_index = location_options.index(selected_location)
        lat = float(locations[selected_index]['lat'])
        lon = float(locations[selected_index]['lon'])

        # Finding the closest record from 'area.csv'
        df = pd.read_csv('area.csv')
        df['distance'] = df.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
        closest_record = df.loc[df['distance'].idxmin()]

        # Displaying information of the closest record
        st.write("가장 가까운 기록의 정보:")
        st.write(closest_record)

        # Load brand, area, and training data
        df_brand = pd.read_csv('brand.csv')
        df_area = pd.read_csv('area.csv')

        # Load the sales model
        with open('sales_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Use 상권_코드 from closest_record to filter area data
        AREA_CODE = closest_record['상권_코드']
        area_data = df_area[df_area['상권_코드'] == AREA_CODE]

        # Prepare data for prediction
        brand_data = df_brand.copy()
        area_data = pd.concat([area_data]*len(brand_data), ignore_index=True)
        data = pd.concat([brand_data.reset_index(drop=True), area_data.reset_index(drop=True)], axis=1)
        data = data.replace({'FALSE': 0, 'TRUE': 1})
        
        # Ensure the data columns match the model's expected input
        df_training = pd.read_csv('final.csv').drop('EST_AMT_TOT', axis=1)
        data = data[df_training.columns.tolist()]

        # Predict sales
        prediction = model.predict(data)
        data['EST_AMT_TOT'] = prediction

        for column in brand_data.columns:
            if column not in data.columns:
                data[column] = brand_data[column]

        # Sort brands by estimated sales amount in descending order
        sorted_brands = data.sort_values(by='EST_AMT_TOT', ascending=False)
        
        # Display sorted brands
        st.write("브랜드 목록 (예상 매출액 순):")
        st.dataframe(sorted_brands[['brand', 'EST_AMT_TOT']])  # Assuming 'brand_name' is a column in your brand data
