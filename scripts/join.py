from os import listdir
from os.path import join
from pandas import read_feather
from numpy import nan

#------------------------------------------------------------------------------

dirpro = join('..', 'data', 'pro')

#------------------------------------------------------------------------------

#read files
dfs = [read_feather(join(dirpro, fn)) for fn in listdir(dirpro)]
#take another pass at nulls and use time stamps for indices
for df in dfs:
    df.replace(999999999.0, nan, inplace=True)
    df.dropna(inplace=True)
    df.drop('sclk', axis=1, inplace=True)
    df.set_index(['sol','hr','min','sec'], inplace=True)
    df.sort_index(inplace=True)
#drop empty frames
dfs = [df for df in dfs if len(df) > 0]
#join everything
df = dfs[0]
for x in dfs[1:]:
    df = df.join(x, how='inner')
    print(df)
df = df.reset_index()
print(df)
df.to_feather(join(dirpro, 'meda.feather'))
