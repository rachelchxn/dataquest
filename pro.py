import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def dayOfYear(month, day, year):
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dayOfYear = sum(daysInMonth[:int(month)-1]) + int(day)
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) and dayOfYear > 59:
        dayOfYear -= 1
    return dayOfYear

def preprocessing(df):
    
    df['ArrivalDayOfYear'] = df.apply(lambda row: dayOfYear(row['ArrivalMonth'], row['ArrivalDate'], row['ArrivalYear']), axis=1)
    
    df['TotalNights'] = df['NumWeekendNights'] + df['NumWeekNights']

    df['TotalGuests'] = df['NumAdults'] + df['NumChildren']
    
    df['TotalPrice'] = df['AvgRoomPrice'] * df['TotalNights']

