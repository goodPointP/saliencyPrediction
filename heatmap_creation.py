import h5py
import pandas as pd
import numpy as np
import time
import torch
import os


class heatmapper:
    def __init__(self, dataframe, screen_dimensions, parse=True):
        """
        
        need to add: DRAW function
        
        screen_dimensions : tuple or string
        string == experiment name (such as 'Baseline')
        tuple == dimensions (such as (1280, 960))

        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.abstract = [10, 11]
        self.discards = 0
        self.df = dataframe
        self.subjects = dataframe.SUBJECTINDEX.unique()
        self.trials = [dataframe[dataframe.SUBJECTINDEX == sub].trial.iloc[-1] for sub in self.subjects]
        self.target_dim = 224
        
        if type(screen_dimensions) == str:
            meta = pd.read_csv('../../Datasets/nature_dataset/meta.csv', sep='; ', engine='python')
            original_dims = tuple(map(int, meta[meta.columns[-9]][meta.Name == screen_dimensions].str.split('x').iloc[0]))
        else:
            original_dims = screen_dimensions
            
        self.scale = (self.target_dim/original_dims[0], self.target_dim/original_dims[1])
        self.dims = (int(original_dims[0]*self.scale[0]), int(original_dims[1]*self.scale[1]))
        
        if parse == True:
            self.fixations, self.paths = self._parser(self.df, self.subjects, self.trials)
            
        
    
    def _gaussian_(self, x, sx):
        y = x
        sy = sx
        xo = x/2
        yo = y/2
        # matrix of zeros
        M = torch.empty([y,x],dtype=float, device=self.device)
        # gaussian matrix
        for i in range(x):
            for j in range(y):
                M[j,i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )
        return M   

    def _parser(self, df, subjectlist, triallist, debug=False):
        paths = []
        fixinfo = []
        index_errors = []
        for sub_idx, subject in enumerate(subjectlist):
            for trial_idx in range(0, int(triallist[sub_idx])):
                try:
                    df_temp = df.loc[(df.SUBJECTINDEX == subject) & (df.trial == trial_idx+1)]
                    cat, filenr = int(df_temp.category.iloc[0]), int(df_temp.filenumber.iloc[0])
                    _, ext = os.path.splitext(os.listdir('../../Datasets/nature_dataset/{}/'.format(cat))[-1])
                    path = '../../Datasets/nature_dataset/{}/{}{}'.format(cat, filenr, ext)
                    if cat in self.abstract:
                        self.discards += 1
                        continue
                    
                    if os.path.exists(path):
                        paths.append(path)
                        fixinfo.append(torch.Tensor((np.array((df_temp.x.values*self.scale[0], 
                                                               df_temp.y.values*self.scale[1], 
                                                               np.abs(df_temp.start-df_temp.end).values/100)))).to(self.device).T)
                    else:
                        self.discards += 1
                        continue
                except IndexError:
                    self.discards += 1
                    if debug == True:
                        index_errors.append(sub_idx, trial_idx)
                    else:                 
                        pass
        if debug == True:
            return fixinfo, index_errors
        else:
            return fixinfo, paths

    def compute(self, count = None, gwh = 40, stddev = 6):
        if type(count) == int:
            fixationlist = self.fixations[:count]
        
        if type(count) == tuple:
            fixationlist = self.fixations[count[0]:count[1]]
        
        if count == None:
            fixationlist = self.fixations
        
        checkpoints = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
        # heatmaps = torch.empty((len(fixationlist), 1, self.dims[1], self.dims[0]))
        heatmaps = []
        gsdwh = gwh/stddev
        gaus = self._gaussian_(gwh, gsdwh)
        
        strt = int(gwh/2)
        heatmapsize = int(self.dims[1] + 2*strt), int(self.dims[0] + 2*strt)
        for idx, fix in enumerate(fixationlist):
            if idx % 1000 == 0:
                print("computed {} heatmaps".format(idx))
            
            if idx in checkpoints:
                np.savez('heatmaps_up_to:{}.npz'.format(idx), heatmaps=heatmaps)
                print("saving at checkpoint: {}".format(idx))
                heatmaps = []
                
            heatmap = torch.empty(heatmapsize, dtype=float, device=self.device)
            for i in range(0,len(fix)):

                x = int(strt + int(fix[:,0][i]) - int(gwh/2))
                y = int(strt + int(fix[:,1][i]) - int(gwh/2))

                if (not 0 < x < self.dims[0]) or (not 0 < y < self.dims[1]):
                    hadj=[0,gwh];vadj=[0,gwh]
                    if 0 > x:
                        hadj[0] = abs(x)
                        x = 0
                    elif self.dims[0] < x:
                        hadj[1] = gwh - int(x-self.dims[0])
                    if 0 > y:
                        vadj[0] = abs(y)
                        y = 0
                    elif self.dims[1] < y:
                        vadj[1] = gwh - int(y-self.dims[1])
                    try:
                        heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix[:,2][i]
                    except:
                        pass
                else:                
                    heatmap[y:y+gwh,x:x+gwh] += gaus * fix[:,2][i]
                    
            ### ADD 0-255 (or 0-1?) SCALING FOR EACH HEATMAP... or for entire set of heatmaps??
            # heatmaps[idx] = (heatmap.cpu()[np.newaxis, strt:self.dims[1]+strt,strt:self.dims[0]+strt] > .05).astype(int)
            heatmaps.append((heatmap[np.newaxis, strt:self.dims[1]+strt,strt:self.dims[0]+strt].cpu().numpy() > .05).astype(int))
            del heatmap
            torch.cuda.empty_cache()
            
        np.savez('heatmaps_up_to:{}.npz'.format(idx), heatmaps=heatmaps)
        return heatmaps
