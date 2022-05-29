#%%
from os import walk
from os.path import join
from pandas import read_csv, concat
from numpy import *

#%%----------------------------------------------------------------------------

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

#%%----------------------------------------------------------------------------

def lower_cols(df):
    df.columns = [c.lower() for c in df.columns]
    return(None)

def handle_null(df):
    #this value is specified in readme.txt and does appear in some tables
    df.replace(9999999999, nan, inplace=True)
    return(None)

def split_datetime(df):
    col = ['sec', 'min', 'hr', 'sol']
    slc = [slice(11,13), slice(8,10), slice(5,7), slice(0,4)]
    for c,s in zip(col, slc):
        df.insert(1, c, [int16(x[s]) for x in df.ltst])
    df.drop('ltst', axis=1, inplace=True)
    return None

def process(df):
    lower_cols(df)
    handle_null(df)
    split_datetime(df)
    return(None)

def extract_tables(datadir, datakeys, prefix, dirout):
    #empty dictionary of lists to get started
    tables = {k:[] for k in datakeys}
    #scan for csv files
    for root, _, fns in walk(datadir):
        for fn in fns:
            for k in datakeys:
                if (k in fn) and ('.csv' in fn.lower()):
                    df = read_csv(join(root, fn))
                    tables[k].append(df[datakeys[k]])
    for k in tables:
        #stack all the portions 
        tables[k] = concat(tables[k], axis=0, ignore_index=True)
        #do some initial processing
        process(tables[k])
        #write to file
        p = join(dirout, prefix + '_' + k + '.feather')
        tables[k].to_feather(p)
        print('file written:', p)
    return(None)

#%%----------------------------------------------------------------------------

extract_tables(dircal, keyscal, 'calibrated', dirout)

extract_tables(dirder, keysder, 'derived', dirout)
