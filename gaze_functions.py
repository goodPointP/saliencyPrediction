import numpy as np
import pandas as pd
from matplotlib import pyplot, image
import os

meta = pd.read_csv('../../Datasets/nature_dataset/meta.csv', sep='; ', engine='python')
# def calculate_heatmaps( )

def get_subject_trial(df):
    subjects = list(df.SUBJECTINDEX.unique())
    trials = []
    for subject in subjects:
        trials.append(df[df.SUBJECTINDEX==subject].trial.unique())
    return subjects, trials



    

def image_paths(df, subjects, trials, last_tr=None):
    """
    df : 
        pd.DataFrame for experiment in question.
    subjects : 
        list of subjects
        (n).
    trials : 
        list of trials for each subject to retrieve image paths from
        (n, m).
    last_tr : 
        last trial to include for all subjects
    paths : 
        image paths from each of the subjects included trials

    """
    paths = []
    for sub_idx, subject in enumerate(subjects):
        for trial in trials[sub_idx][:last_tr]:
            series = df[(df.SUBJECTINDEX == subject) & (df.trial == trial)].iloc[0]
            cat, filenr = int(series.category), int(series.filenumber)
            _, ext = os.path.splitext(os.listdir('../../Datasets/nature_dataset/{}/'.format(cat))[-1])
            paths.append('../../Datasets/nature_dataset/{}/{}{}'.format(cat, filenr, ext))
    return paths


def remove_invalid_paths(paths, heatmaps):
    """ 
    input: paths (n, m), heatmaps (n, m)
    if path == invalid removes both path and heatmap 
    
    output: paths (n-i, m) heatmaps (n-i, m))
    """ 
    for idx, path in enumerate(paths):
        if not os.path.exists(path):
            del heatmaps[idx]
            del paths[idx]
    return paths, heatmaps


def compute_heatmap(df, s_index, s_trial, experiment = None, last_tr=None, draw=True): 
    
    """
    df : DataFrame of gaze data
    s_index : subject index in DataFrame
    s_trial : trial index in DataFrame 
    experiment : from which experiment to extract the data e.g. 'Baseline'
    """
    
    # function to prepare data in DF and images in numbered folders for draw_heatmap function
    
    # Dimensions of monitor for experiment in question
    screendims = list(map(int, meta[meta.columns[-9]][meta.Name == experiment].str.split('x').iloc[0]))
    
    # Gaze data for Subject & Trial
    if type(s_index) == int and type(s_trial) == int:
        df = df.loc[(df.SUBJECTINDEX == s_index) & (df.trial == s_trial)]
        fixations = np.array((df.start, df.end, np.abs(df.start-df.end), df.x, df.y)).T
    
    # Retrieving image path
        cat, filenr = int(df.category.iloc[0]), int(df.filenumber.iloc[0])
        _, ext = os.path.splitext(os.listdir('../../Datasets/nature_dataset/{}/'.format(cat))[-1])
        imgpath = '../../Datasets/nature_dataset/{}/{}{}'.format(cat, filenr, ext)
    
    # Drawing heatmap based on draw_heatmap() from Pygazeanalyser
        return draw_heatmap(fixations, screendims, imagefile=imgpath, draw=draw)
    
    else: 
        draw=False
        heatmaps = []
        for s in s_index:
            print(s, last_tr)
            for t in s_trial[int(s)-1][:last_tr]:
                df_t = df.loc[(df.SUBJECTINDEX == s) & (df.trial == t)]
                fixations = np.array((df_t.start, df_t.end, np.abs(df_t.start-df_t.end), df_t.x, df_t.y)).T
                heatmaps.append(draw_heatmap(fixations, screendims, imagefile=None, draw=draw))
        return heatmaps
                

def draw_heatmap(fixations, dispsize, imagefile=None, durationweight=True, alpha=0.5, savefilename=None, draw=True):
    #function from Pygazeanalyser
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.
    
    arguments
    
    fixations        -    a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    dispsize        -    tuple or list indicating the size of the display,
                    e.g. (1024,768)
    
    keyword arguments
    
    imagefile        -    full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    durationweight    -    Boolean indicating whether the fixation duration is
                    to be taken into account as a weight for the heatmap
                    intensity; longer duration = hotter (default = True)
    alpha        -    float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename    -    full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)
    
    returns
    
    fig            -    a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # FIXATIONS
    fix = parse_fixations(fixations)

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh/6
    gaus = gaussian(gwh,gsdwh)
    # matrix of zeroes
    strt = int(gwh/2)
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0,len(fix['dur'])):
        # get x and y coordinates
        #x and y - indexes of heatmap array. must be integers
        x = int(strt + int(fix['x'][i]) - int(gwh/2))
        y = int(strt + int(fix['y'][i]) - int(gwh/2))
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj=[0,gwh];vadj=[0,gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['dur'][i]
            except:
            # fixation was probably outside of display
                pass
        else:                
            # add Gaussian to the current heatmap
            heatmap[y:y+gwh,x:x+gwh] += gaus * fix['dur'][i]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
    
    # IMAGE
    if draw:
        fig, ax = draw_display(dispsize, imagefile=imagefile)
        
        # remove zeros
        lowbound = np.mean(heatmap[heatmap>0])
        heatmap[heatmap<lowbound] = np.NaN
        
        # draw heatmap on top of image
        ax.imshow(heatmap, cmap='jet', alpha=alpha)
        
        # save the figure if a file name was provided
        if savefilename != None:
            fig.savefig(savefilename)
        return fig, heatmap
    
    else:
        return  heatmap

def parse_fixations(fixations):
    #function from Pygazeanalyser
    """Returns all relevant data from a list of fixation ending events
    
    arguments
    
    fixations        -    a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    returns
    
    fix        -    a dict with three keys: 'x', 'y', and 'dur' (each contain
                a numpy array) for the x and y coordinates and duration of
                each fixation
    """
    
    # empty arrays to contain fixation coordinates
    fix = {    'x':np.zeros(len(fixations)),
            'y':np.zeros(len(fixations)),
            'dur':np.zeros(len(fixations))}
    # get all fixation coordinates
    for fixnr in range(len( fixations)):
        stime, etime, dur, ex, ey = fixations[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur
    
    return fix

def draw_display(dispsize, imagefile=None):
    #function from Pygazeanalyser
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it
    
    arguments
    
    dispsize        -    tuple or list indicating the size of the display,
                    e.g. (1024,768)
    
    keyword arguments
    
    imagefile        -    full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    
    returns
    fig, ax        -    matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """
    
    # construct screen (black background)

    data_type = 'float32'
    screen = np.zeros((dispsize[1],dispsize[0],3), dtype=data_type)
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        
        _, ext = os.path.splitext(imagefile)
        ext = ext.lower()
        data_type = 'float32' if ext == '.png' else 'uint8'
        
        # load image
        img = image.imread(imagefile)
        # flip image over the horizontal axis
        # (do not do so on Windows, as the image appears to be loaded with
        # the correct side up there; what's up with that? :/)
        if os.name == 'nt':
            img = np.flipud(img)
        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0]/2 - w/2)
        y = int(dispsize[1]/2 - h/2)
        # draw the image on the screen
        screen[y:y+h,x:x+w,:] += img
        
    # dots per inch
    dpi = 100.0
    
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # plot display
    ax.axis([0,dispsize[0],0,dispsize[1]])
    ax.imshow(screen)#, origin='upper')
    
    return fig, ax


def gaussian(x, sx, y=None, sy=None):
    #function from Pygazeanalyser
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution
    
    arguments
    x        -- width in pixels
    sx        -- width standard deviation
    
    keyword argments
    y        -- height in pixels (default = x)
    sy        -- height standard deviation (default = sx)
    """
    
    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers    
    xo = x/2
    yo = y/2
    # matrix of zeros
    M = np.zeros([y,x],dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j,i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )

    return M


def f_extraction(df, cols):
    """
    
    """

    df_copy = df.copy(deep=True)
    df_copy = df_copy[df_copy.columns.intersection(cols)]
    
    return df_copy


def f_engi(df):
    
    df_copy = df.copy(deep=True)
    df_copy['time'] = np.abs(df_copy.start-df_copy.end)
    
    return df_copy