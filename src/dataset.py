import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class LootDataset(Dataset):
    '''
    Extreme points for the BM@N:
        x_min, x_max, y_min, y_max = (-73.8, 81.6, -29.4, 16.2)

    Resolution for the BM@N:
        x_res, y_res = (512, 256)

    Usage:
        dataset = LootDataset(root_file,
                              x_max=x_max,
                              x_min=x_min,
                              y_max=y_max,
                              y_min=y_min,
                              x_res=x_res,
                              y_res=y_res)
    '''
    def __init__(self, 
                 file_path,
                 x_max, 
                 x_min,  
                 y_max, 
                 y_min, 
                 x_res,
                 y_res=None,
                 n_stations=None):
        self.__x_max = x_max
        self.__x_min = x_min
        self.__y_max = y_max
        self.__y_min = y_min
        self.__x_res = x_res
        self.__y_res = x_res if y_res is None else y_res
        # read file with dataframe
        self.df = pd.read_csv(file_path, delimiter='\t')
        # some events are missed
        # if not factorized, there will be empty events
        self.df.event = pd.factorize(self.df.event)[0]
        
        if n_stations is None:
            self.n_stations = self.df.station.nunique()
        else:
            self.n_stations = n_stations
            
        # calculate pixel sizes
        self.x_pixel_size = self.get_x_pixel_size()
        self.y_pixel_size = self.get_y_pixel_size()
    
    
    def get_x_pixel_size(self):
        return (self.x_max-self.x_min)/self.x_res
    
    
    def get_y_pixel_size(self):
        return (self.y_max-self.y_min)/self.y_res
      
      
    @property
    def x_max(self):
        return self.__x_max
      
      
    @x_max.setter
    def x_max(self, x_max):
        if hasattr(self, 'x_min') and x_max <= self.x_min:
            raise ValueError("x_max should be greater than x_min!")
            
        self.__x_max = x_max
        self.x_pixel_size = self.get_x_pixel_size()
        
        
    @property
    def x_min(self):
        return self.__x_min
      
      
    @x_min.setter
    def x_min(self, x_min):
        if hasattr(self, 'x_max') and x_min >= self.x_max:
            raise ValueError("x_min should be smaller than x_max!")
            
        self.__x_min = x_min
        self.x_pixel_size = self.get_x_pixel_size()
      
      
    @property
    def y_max(self):
        return self.__y_max
      
      
    @y_max.setter
    def y_max(self, y_max):
        if hasattr(self, 'y_min') and y_max <= self.y_min:
            raise ValueError("y_max should be greater than y_min!")
            
        self.__y_max = y_max
        self.y_pixel_size = self.get_y_pixel_size()
      
      
    @property
    def y_min(self):
        return self.__y_min
      
      
    @y_min.setter
    def y_min(self, y_min):
        if hasattr(self, 'y_max') and y_min >= self.y_max:
            raise ValueError("y_min should be smaller than y_max!")
            
        self.__y_min = y_min
        self.y_pixel_size = self.get_y_pixel_size()
        
        
    @property
    def x_res(self):
        return self.__x_res
      
      
    @x_res.setter
    def x_res(self, x_res):
        if x_res <= 0:
            raise ValueError("Resolution must be greater than zero!")
            
        self.__x_res = x_res
        self.x_pixel_size = self.get_x_pixel_size()
        
        
    @property
    def y_res(self):
        return self.__y_res
      
      
    @y_res.setter
    def y_res(self, y_res):
        if y_res <= 0:
            raise ValueError("Resolution must be greater than zero!")
            
        self.__y_res = y_res
        self.y_pixel_size = self.get_y_pixel_size()
        
        
    def xcoord2pix(self, x):
        return ((x-self.x_min)/self.x_pixel_size).astype(int)
  
  
    def ycoord2pix(self, y):
        return ((y-self.y_min)/self.y_pixel_size).astype(int) 
    
    
    def start_points_xy_shifts(self, event):
        # extract only tracks data
        gp = event[event.track >= 0]\
                .sort_values(by=['z']) \
                .groupby(['event', 'track'])
        # starting point of each track
        start_point_df = gp.first()[['xpix', 'ypix']]
        # join two dataframes
        tracks_df = event.join(
            start_point_df, 
            on=['event', 'track'], 
            rsuffix='_start',
            how='right')
        # calculate shifts with respect of the start point
        tracks_df['xshift'] = tracks_df.xpix - tracks_df.xpix_start
        tracks_df['yshift'] = tracks_df.ypix - tracks_df.ypix_start
        # create the array for shifts
        self.n_shifts = (self.n_stations-1)*2
        # depth dimension
        self.y_ddim = self.n_shifts + 1
        res_arr = np.zeros((self.y_res, self.x_res, self.y_ddim), 
                            dtype=np.int32)
        # fill the array with computed shifts
        # y shifts
        res_arr[tracks_df.ypix_start, 
                tracks_df.xpix_start,
                tracks_df.station*2-1] = tracks_df.yshift
        # x shifts
        res_arr[tracks_df.ypix_start, 
                tracks_df.xpix_start,
                tracks_df.station*2] = tracks_df.xshift
        
        # add start  points' mask to the array
        res_arr[start_point_df.ypix,
                start_point_df.xpix,
                0] = 1
        
        return res_arr
        
        
    def __len__(self):
        return self.df.event.nunique()
    
    
    def event_img(self, idx, with_fakes=True):
        event = self.df[self.df.event==idx]
        # sparse representation
        ypix=self.ycoord2pix(event.y)
        xpix=self.xcoord2pix(event.x)
        # dense representation
        X = np.zeros((self.y_res, self.x_res), dtype=np.int8)
        # true tracks ids start with 0
        # fakes -> 0, true tracks > 0
        increment = 1
        # fake hits label = -1
        if with_fakes:
            # for -1 -> 1 and all tracks' ids > 0
            increment = 2
            
        event.track += increment
        X[ypix, xpix] = event.track
        
        return X
      
      
    def __getitem__(self, idx):
        event = self.df[self.df.event==idx]
        # sparse representation
        event = event.assign(
                    ypix=self.ycoord2pix(event.y),
                    xpix=self.xcoord2pix(event.x)) 
        # labels array
        y = self.start_points_xy_shifts(event)
        # dense representation
        X = np.zeros((self.y_res, self.x_res, self.n_stations), 
                     dtype=np.int8)
        # 1 - because hits and fakes are the same at the input
        X[event.ypix, event.xpix, event.station] = 1
        return X, y