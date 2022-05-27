from os import walk
from os.path import join
from pandas import read_csv, concat
from numpy import *

#------------------------------------------------------------------------------

#calibrated data directory
dircal = join('..', 'data', 'raw', 'calibrated', 'data_calibrated_env')

#derived data directory
dirder = join('..', 'data', 'raw', 'derived', 'data_derived_env')

#processed data directory
dirout = join('..', 'data', 'pro')

#which fields to select from derived data files
keysder = {
    'ANCILLARY': [
        'SCLK',
        'LTST',
        'SOLAR_LONGITUDE_ANGLE',
        'SOLAR_ZENITHAL_ANGLE',
        'ROVER_POSITION_X',
        'ROVER_POSITION_Y',
        'ROVER_POSITION_Z',
        'ROVER_VELOCITY'
    ],
    'PS': [
        'SCLK',
        'LTST',
        'PRESSURE'
    ],
    'RHS': [
        'SCLK',
        'LTST',
        'LOCAL_RELATIVE_HUMIDITY',
        'VOLUME_MIXING_RATIO'
    ],
    'TIRS': [
        'SCLK',
        'LTST',
        'DOWNWARD_LW_IRRADIANCE',
        'UPWARD_LW_IRRADIANCE'
    ],
    'WS': [
        'SCLK',
        'LTST',
        'HORIZONTAL_WIND_SPEED',
        'VERTICAL_WIND_SPEED',
        'WIND_DIRECTION'
    ]
}

#which fields to select from calibrated data files
keyscal = {
    'TIRS': [
        'SCLK',
        'LTST',
        'AIR_TEMP',
        'GROUND_TEMP'
    ]
}

#------------------------------------------------------------------------------

def lower_cols(df):
    df.columns = [c.lower() for c in df.columns]
    return None

def tofloat32(df):
    for col in df:
        if df[col].dtype is dtype('float64'):
            df[col] = df[col].astype(float32)
    return None

def handlenull(df):
    #this value is specified in readme.txt and does appear in some tables
    df.replace(9999999999, nan, inplace=True)
    return None

def split_datetime(df):
    col = ['sec', 'min', 'hr', 'sol']
    slc = [slice(11,13), slice(8,10), slice(5,7), slice(0,4)]
    for c,s in zip(col, slc):
        df.insert(1, c, [int16(x[s]) for x in df.ltst])
    df.drop('ltst', axis=1, inplace=True)
    return None

def read_tables(datadir, datakeys):
    #empty dictionary of lists to get started
    tables = {k:[] for k in datakeys}
    #scan for csv files
    for root, _, fns in walk(datadir):
        for fn in fns:
            for k in datakeys:
                if (k in fn) and ('.csv' in fn.lower()):
                    df = read_csv(join(root, fn))
                    tables[k].append(df[datakeys[k]])
    #stack all the portions of each table
    for k in tables:
        tables[k] = concat(tables[k], axis=0, ignore_index=True)
    return tables

#------------------------------------------------------------------------------

#read and combine data tables
tables = read_tables(dircal, keyscal)
#make the columns lower case and split the timestamp
for k in tables:
    df = tables[k]
    lower_cols(df)
    split_datetime(df)
    handlenull(df)
    tofloat32(df)