#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Raw data sets

Created on Thu Nov 10 16:28:53 2022
@author: jparedes
"""
import os
import pandas as pd
import numpy as np
import warnings
import pickle
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
#%%
class dataSet_raw_refStation():
    def __init__(self,path,fname,time_agg):
        if fname not in ['PR_Data2018.xlsx','PR_Data2019.xlsx','Joint']:
            raise Exception('Enter a valid reference station file name')
        print(f'Preparing data set: {fname}')
        self.path = path
        self.fname = fname
        self.idx = [0,2,4,3,5,8,9]# idx of sheets used
        self.df = pd.DataFrame()
        self.time_agg = time_agg
        
        if self.fname == 'Joint':
            print(f'{self.fname} data set specified. Concatenating previous data sets')
    
    def load_ds(self,time_period='60min'):
        if self.fname == 'Joint':
            raise Exception('Joint data set is the composition of 2 or more previously loaded data sets.\nFirst load a proper data set')
        for i in self.idx:
            df = pd.read_excel(self.path+'/'+self.fname,sheet_name=i)
            df.rename(columns={df.columns[0]:'date'},inplace=True)
            df.date = pd.to_datetime(df.date)
            if i == 8:# keep only certain entries from Meteo data
                df = df.loc[:,['date','TEMP','HUM','PRES','Vmax']]
            
            #specific adjustments
            if self.fname == 'PR_Data2018.xlsx':
                if i == 2:
                    df.date = df.date.dt.round(freq='5T')
                elif i in [3,4,5]:   
                    df.date.dt.round(freq='10T')
            elif self.fname == 'PR_Data2019.xlsx':
                if i in [2,4]:
                    df.date = df.date.dt.round(freq='5T')
                elif i in [5,6]:
                    df.date = df.date.dt.round(freq='10T')
                
            
            # correct wrong values
            #df.iloc[:,1:] = np.abs(df.iloc[:,1:])# replace with abs val
            #df.iloc[(df.iloc[:,1]<0.0).values,1]=0.0# replace with zero
            #df.iloc[(df.iloc[:,1]<0.0).values,1]=np.nan# replace with NaN
            
      
            df_ = df.groupby(pd.Grouper(key='date',freq=time_period)).mean()
            self.df = pd.concat([self.df,df_],axis=1)
        
        # Dropping missing values
        #self.df.dropna(inplace=True)    
    
    def Merge_datasets(self,*args):
        if self.fname != 'Joint':
            raise Exception(f'Concatenation operation requires different data sets.\nsingle data set loaded {self.fname}.  Must create a Joint data set.')
        if args == ():
            raise Exception('No data sets specified for concatenation')
        if self.df.shape[0] != 0:
            warnings.warn('Data set not empty. New data frames will be concatenated to previous data set')
            
        print('Concatenating data sets')
        for ds in args:
            self.df = pd.concat([self.df,ds],axis=0)
        
        # Set simpler names
        var = ['BC','N','PM1','PM25','PM10','T','RH','P','V','SO2','NO','NO2','O3','CO','NOx']
        self.df.columns = var
        # correct units
        self.df.loc[:,'BC'] = 1e-3*self.df.loc[:,'BC'] #ng to ug 
        self.df.loc[:,'N'] = 1e6*self.df.loc[:,'N'] # n/cm3 to n/m3
        # change index
        #self.df.set_index(self.df.date,drop=True,inplace=True)
        
class dataSet():
    def __init__(self,files_path,device,dataformat,filenames,time_agg):
        device_opt = ['RefSt','VirtualSensor']
        format_opt = ['raw','data_frame']
        if device not in device_opt:
            raise Exception(f'Incorrect deivce. Choose either {device_opt}')
        if dataformat not in format_opt:
            raise Exception(f'Incorrect data source. Choose either {format_opt}')
        
        print('Initializing data set')
        self.device = device
        self.dataformat = dataformat
        self.files_path = files_path
        self.filenames = filenames
        self.time_agg = time_agg
        self.df = pd.DataFrame()
        print(f'Prepared to load {self.device} data set as {self.dataformat} from {self.filenames}')
    
    def load_raw(self):
        print('Loading data set from raw files')
        print(f'Loading data for {self.dataformat}')
        if self.device == 'RefSt':
            path = self.files_path + '/Reference_station'
            fname = self.filenames[0]
            ds18 = dataSet_raw_refStation(path = path, fname = fname,time_agg=self.time_agg)
            ds18.load_ds(time_period=self.time_agg)
            fname = self.filenames[1]
            ds19 = dataSet_raw_refStation(path = path, fname = fname,time_agg=self.time_agg)
            ds19.load_ds(time_period=self.time_agg)
            ds = dataSet_raw_refStation(path=path,fname='Joint',time_agg=self.time_agg)
            ds.Merge_datasets(ds18.df,ds19.df)
            
            
    def load_dataFrame(self,time_freq='60min'):
        if self.device == 'RefSt':
            path = self.files_path + '/Reference_station/'
            print(f'Loading dataFrame from {path}')
            fname = self.filenames[0]
            self.df = pd.read_csv(path+fname,sep=',')
            self.df.set_index(self.df.date,drop=True,inplace=True)
            
    def load_dataSet(self):
        if self.dataformat == 'raw':
            self.load_raw()
            
        elif self.dataformat == 'data_frame':
            self.load_dataFrame()
        

                
        
        



#%% main

def main():
    
    # go to file location
    device = 'RefSt'
    source = 'data_frame'
    ds = dataSet(device=device,source=source) 
    ds.load_dataSet()
    df = ds.df
    scaler = 'Standard'# Standard or MinMax
    ds.Scale(scaler=scaler)
 
    return df,ds
#%%
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    
    df,ds = main()
