from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pandas as pd


class FillLocData:
    def __init__(self, geolocator=Nominatim(user_agent="lazaniaa7@gmail.com")):
        self.geolocator = geolocator

    def get_zipcode(self, lat, lon):
        try:
            location = self.geolocator.reverse((lat, lon), exactly_one=True , timeout=10)
            if location and 'postcode' in location.raw['address']:
                return location.raw['address']['postcode'][:5]
        except GeocoderTimedOut:
            return None
        return None

    def fill_zipcode(self, df):
        for idx, row in df[df['zipcode'].isna()].iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                zipcode = self.get_zipcode(row['latitude'], row['longitude'])
                df.at[idx, 'zipcode'] = zipcode
        return df 
    
    def fill_neighbourhood(self, df):
        for i in df['zipcode'][df['neighbourhood'].isna()]:
            try:
                df.loc[(df['zipcode'] == i) & (df['neighbourhood'].isna()), 'neighbourhood'] = df['neighbourhood'][df['zipcode'] == i].mode()[0]
            except:
             #NNF = NO NEIGHBOURHOOD FOUND
                df.loc[(df['zipcode'] == i) & (df['neighbourhood'].isna()), 'neighbourhood'] = 'NNF'
        return df