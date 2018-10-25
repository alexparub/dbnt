import pandas as pd
import quandl
quandl.ApiConfig.api_key = '2RysfuULya2obtypxpeU'
quandl.ApiConfig.api_version = '2015-04-09'

def get_data(metadata, feature_size = 50):
    DATA = quandl.get(dataset="WIKI/A", start_date="2008-01-01", end_date="2014-01-01").Close
    DATA.name = 'A'

    i = 1
    for code in metadata.code[1:]:
        try: 
            data = quandl.get(dataset="WIKI/" + code, start_date="2008-01-01", end_date="2014-01-01").Close
            data.name = code
        except quandl.NotFoundError :
            pass
        else:
            DATA = pd.concat([DATA, data], axis=1)
            DATA = DATA.dropna(axis=1)
            if i+1 == DATA.shape[1]:
                i += 1
                print(i, code)
                if i == feature_size:
                    break
    return DATA

