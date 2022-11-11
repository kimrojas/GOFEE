########################################################################################
#
# This program is designed to make it easy to make proper statistics for
# evolutionary algorithms. It can be used in two differnet ways:
#
# 1: Call the program from the terminal, followed be the files you want statistics for.
#    If you have two data sets you want to compre it could look like this:
#    $ python survival_stats.py event_file1.npy event_file2.npy
#    If you only want statistics for one 
#    The results will be saved in a map called stats/ so you can make all
#    the displayed yourself. Each file have a format of (x,y,y_LCB,y_UCB,censorings),
#    Hazard only holds (x,y)
#    A last input can be given to set labels. it shold be of this from:
#    labels=[label1,label2,label3]
#
# 2: Import the function survival_stats() from this script and give it two lists
#    of inputs. The first should be a list of all the times when an event or censoring 
#    occured. The second second should be a list of binaries indicating if an event 
#    occured at the corresponding time. That is 0 for censorings and 1 for events. 
#    List of list can also be used for input, and will additionally result in log-rank
#    tests being made between the inputs
#
# Event files should be .npy files holding a 2 x n array. eg. [[5,4,10][1,1,0]].
# the first vector holds the times for the EA runs. either the time when the best
# structure was found or when the run ended. The second vector should hold a list
# of zero and ones, where a 1 indicate that the best structure was found, and a 0
# indicate that it was not.
#
########################################################################################
import os
import sys
import numpy as np
import scipy.stats as st
from scipy.special import erfinv
from copy import copy
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *

# Define a class to be used KRR
class Comp(object):
    def get_features(self,x):
        return x
    def get_similarity(self,f1,f2):
        return abs(f1-f2)

# Function for the log-rank test
def logrank(n1,d1,t1,n2,d2,t2):
    """This function returns the p-value for a log-rank test
    
    Inputs:
    n1: Number at risk in population 1 at times indicated by t1
    d1: Number of events in population 1 at times indicated by t1
    t1: times used with the two above inputs
    n2: Number at risk in population 2 at times indicated by t2
    d2: Number of events in population 2 at times indicated by t2
    t2: times used with the two above inputs

    output:
    p-value

    """

    # The first part here is just collecting and ordering the inputs
    # for the calculations
    n1 = copy(n1)
    d1 = copy(d1)
    t1 = copy(t1)
    n2 = copy(n2)
    d2 = copy(d2)
    t2 = copy(t2)
    n = []
    n_1 = []
    n_2 = []
    d = []
    d_1 = []
    d_2 = []
    while t1 or t2:
        if t1 and t2:
            if t1[0] < t2[0]:
                n_1.append(n1.pop(0))
                n_2.append(n_2[-1])
                n.append(n_1[-1]+n_2[-1])
                d_1.append(d1.pop(0))
                d_2.append(0)
                d.append(d_1[-1]+d_2[-1])
                t1.pop(0)
            elif t1[0] > t2[0]:
                n_1.append(n_1[-1])
                n_2.append(n2.pop(0))
                n.append(n_1[-1]+n_2[-1])
                d_1.append(0)
                d_2.append(d2.pop(0))
                d.append(d_1[-1]+d_2[-1])
                t2.pop(0)
            elif t1[0] == t2[0]:
                n_1.append(n1.pop(0))
                n_2.append(n2.pop(0))
                n.append(n_1[-1]+n_2[-1])
                d_1.append(d1.pop(0))
                d_2.append(d2.pop(0))
                d.append(d_1[-1]+d_2[-1])
                t1.pop(0)
                t2.pop(0)
        elif t1:
            n_1.append(n1.pop(0))
            n_2.append(n_2[-1])
            n.append(n_1[-1]+n_2[-1])
            d_1.append(d1.pop(0))
            d_2.append(0)
            d.append(d_1[-1]+d_2[-1])
            t1.pop(0)
        elif t2:
            n_1.append(n_1[-1])
            n_2.append(n2.pop(0))
            n.append(n_1[-1]+n_2[-1])
            d_1.append(0)
            d_2.append(d2.pop(0))
            d.append(d_1[-1]+d_2[-1])
            t2.pop(0)
    # This is where the actual test is performed
    e_1 = []
    v = []
    for i in range(len(n)):
        e1 = n_1[i]*d[i]/float(n[i])
        e_1.append(e1)
        v1 = (d[i] * n_1[i]/float(n[i]) * (1-n_1[i]/float(n[i])) * (n[i]-d[i])) / float(n[i]-1)
        v.append(v1)
    Z = np.sum(np.array(d_1)-np.array(e_1)) / np.sqrt(np.sum(v))
    return st.norm.sf(abs(Z))*2

# This is the real function of interest
def survival_stats(times,events,alpha=0.95,sigma=5000,show_plot=True,legend_outside=False,save=True,get_hazard=True,labels=[], colors=None, linestyles=None, save_dir='stats'):
    """This function calculateds a number of statistics that may beof interest

    inputs:
    times:     Either a single list of times or a list of list of times
    events:    Either a single list of events or a list of list of events.
               0 indicates a censoring and 1 indicates an event.
    alpha:     Is the size of the confidence bound given for the functions.
               Default is 0.95
    sigma:     Is used for the kernel size for the kernel smoothing used to creat
               the hazard curve. Lower the number if the curve seems to flat and
               raise it if the crve is to spikey. Default is 5000
    show_plot: Default is True. Change to False if you don't want to see
               the plots
    save:      Default is True. Change to False if you don't want the statistics saved

    Output: (KM,CDF,NA,Hazard,censoring,logrank_res)
    KM:          Kaplan-Meier. List of tuples containing: time, value of KM , LCB of KM,
                 UCB of KM.
    CDF:         Cumultative distribution function. List of tuples containing: time,
                 value of CDF , LCB of CDF, UCB of CDF.
    NA:          Nelson-Aalen. List of tuples containing: time, value of NA , LCB of NA, UCB of NA.
    Hazard:      List of tuples containing: time, value of Hazard
    censoring:   List of list indicating if censorings occured at the times given by the KM times
    logrank_res: The results of the log-rank tests arranged in a matrix

    All the outer lists are used to seperate multiple inputs.
    """
    # Arrange the input into a standard format
    if hasattr((times[0]),'__len__'):
        n_inputs = len(times)
    else:
        n_inputs = 1
        times = [times]
        events = [events]
    # calculate a z value from the given alpha
    z = np.sqrt(2)*erfinv(2*(alpha+(1-alpha)/2.)-1)

    # Change the input to conviniant format
    time = []
    censoring = []
    n = []
    d = []
    for i in range(n_inputs):
        time.append([0])
        censoring.append([False])
        n.append([len(times[i])])
        d.append([0])
        ds = 0 # dead or censord at this timestep
        sort_index = np.argsort(times[i])
        for j in sort_index:
            if times[i][j] == time[i][-1]:
                ds += 1
                if events[i][j]:
                    d[i][-1] += 1
                else:
                    censoring[i][-1] = True
            else:
                time[i].append(times[i][j])
                n[i].append(n[i][-1]-ds)
                ds = 1
                if events[i][j]:
                    d[i].append(1)
                    censoring[i].append(False)
                else:
                    d[i].append(0)
                    censoring[i].append(True)
        censoring[i][-1] = False
        censoring[i] = np.array(censoring[i])

    # Make Kaplan-Meier 
    KM = []
    for i in range(n_inputs):
        S = [1]
        for j in range(1,len(time[i])):
            S.append(S[-1]*(n[i][j]-d[i][j])/float(n[i][j]))
        KM.append((np.array(time[i]),np.array(S)))

    # Make confidence bounds for Kaplan-Meier
    KM_CB = []
    for i in range(n_inputs):
        S_LCB = [1]
        S_UCB = [1]
        temp = 0
        for j in range(1,len(time[i])):
            if KM[i][1][j] == 1:
                c_L = 1
                c_U = 1
            elif n[i][j] != d[i][j]:
                temp += d[i][j]/float(n[i][j]*(n[i][j]-d[i][j]))
                V = temp/float(np.log(KM[i][1][j])**2)
                c_L = np.log(-np.log(KM[i][1][j])) + z*np.sqrt(V)
                c_U = np.log(-np.log(KM[i][1][j])) - z*np.sqrt(V)
            else:
                V = temp/float(np.log(KM[i][1][j-1])**2)
                c_L = np.log(-np.log(KM[i][1][j-1])) + z*np.sqrt(V)
                c_U = np.log(-np.log(KM[i][1][j-1])) - z*np.sqrt(V)
            S_LCB.append(np.exp(-np.exp(c_L)))
            S_UCB.append(np.exp(-np.exp(c_U)))
        KM_CB.append((np.array(time[i]),np.array(S_LCB),np.array(S_UCB)))

    # Gather all KM stuff
    for i in range(n_inputs):
        KM[i] = (KM[i][0],KM[i][1],KM_CB[i][1],KM_CB[i][2])

    # Make Cumultative distribution function
    CDF = []
    CDF_CB = []
    for i in range(n_inputs):
        CDF.append((KM[i][0],1-KM[i][1]))
        CDF_CB.append((KM_CB[i][0],1-KM_CB[i][1],1-KM_CB[i][2]))

    # Gather all CDF stuff 
    for i in range(n_inputs):
        CDF[i] =(CDF[i][0],CDF[i][1],CDF_CB[i][1],CDF_CB[i][2])

    if show_plot:
        print('')

    if show_plot:# make plots
        labels += range(len(labels),n_inputs)
        f, ax = subplots(1,1, figsize=(7+3*legend_outside,5))
        if colors is None:
            colors = ['b','r','g','y','c','m']
        max_time = 0
        for i in range(n_inputs):
            color_i = colors[i%len(colors)]
            if linestyles is None:
                linestyle_i = '-'
            else:
                linestyle_i = linestyles[i]
            try:
                ax.fill_between(CDF[i][0], CDF[i][2], CDF[i][3], step='post', facecolor=color_i, alpha=0.1)
            except:
                ax.fill_between(CDF[i][0], CDF[i][2], CDF[i][3], facecolor=color_i, alpha=0.1)
            ax.step(CDF[i][0], CDF[i][1],where='post',c=color_i, linestyle=linestyle_i, label=labels[i])
            #ax.plot(CDF[i][0][censoring[i]],CDF[i][1][censoring[i]],marker='+',c='k')
            if CDF[i][0][-1] > max_time:
                max_time = KM[i][0][-1]            
        ax.set_ylabel('Succes Rate')
        ax.set_xlabel('Singlepoint calculations')
        ax.set_xlim([0,max_time])
        ax.set_ylim([0,1])
        ax.set_yticks(np.linspace(0,1,6))
        ax.set_yticklabels(['{} %'.format(int(i)) for i in np.linspace(0,100,6)])
        ax.set_title('CDF')
        if legend_outside:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            tight_layout()
        else:
            ax.legend(loc=2)

        
    if save and show_plot: # Save files
    #if save: # Save files
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(save_dir)):
            os.makedirs(os.path.join(save_dir))
        savefig(save_dir+'/survival_stats.pdf')
        savefig(save_dir+'/survival_stats.png')
        for i in range(n_inputs):
            name = str(i)
            name_CDF = name+'_CDF'
            if os.path.isfile(os.path.join(cwd,save_dir,name_CDF+'.npy')):
                n_name = 1
                while os.path.isfile(os.path.join(cwd,save_dir,name_CDF+'({})'.format(n_name)+'.npy')):
                    n_name += 1
                name_CDF += '({})'.format(n_name)
            name_CDF += '.npy'
            np.save(os.path.join(cwd,save_dir,name_CDF),(CDF[i][0],CDF[i][1],CDF[i][2],CDF[i][3],censoring[i]))
    if show_plot:
        show()

    return CDF

if __name__ == '__main__':
    cwd = os.getcwd()
    # Check how many inputs are given
    try:
        n_inputs = len(sys.argv)-1
    except:
        print('At least one input must be given for this program.')
        raise
    try:
        if 'label' in sys.argv[-1]:
            n_inputs -= 1
            label_str = sys.argv[-1]
            index1 = label_str.find('[')+1
            label_str = label_str[index1:]
            labels = []
            while label_str.find(',') != -1:
                index2 = label_str.find(',')
                label = label_str[:index2]
                labels.append(label)
                label_str = label_str[index2+1:]
            index2 = label_str.find(']')
            label =label_str[:index2]
            labels.append(label)
        else:
            labels = []
    except:
        print('labels should be given as the last input with a format like:\n'\
              +'labels=[label1,label2,label3]')
        raise

    # Prepare the files for input into the function
    times = [None]*n_inputs
    events = [None]*n_inputs
    for i in range(n_inputs):
        times[i], events[i] = np.load(os.path.join(cwd,sys.argv[i+1]))
    # Run function
    survival_stats(times,events,labels=labels)
