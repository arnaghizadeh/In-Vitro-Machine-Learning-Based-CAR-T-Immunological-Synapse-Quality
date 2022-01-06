import numpy as np
import pandas as pd
import glob
from dataset_antigen import Protein
import argparse
import cv2
import postprocessing
import time
from datetime import date
import nms
import math
#import imread_nd2
# import xlwt
# from xlwt import Workbook
import xlsxwriter
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.preprocessing import minmax_scale

def generate(filename):
    xls = pd.ExcelFile(filename)
    totalIntensityPD = pd.read_excel(xls, 'Mean intensity')
    areaPD = pd.read_excel(xls, 'Area')
    #meanIntensityPD = pd.read_excel(xls, 'Mean intensity')


    totalIntensityArr = totalIntensityPD.values
    firstCell = totalIntensityArr[0][0]
    cellNameList = firstCell.split()
    firstUniqueCell = cellNameList[0]
    fActinList = []
    perforinList = []
    tumorAntiList = []
    pZetaList = []
    fActin = []
    perforin = []
    tumorAnti = []
    pZeta = []
    tempList = []



    uniqueCells = set([])
    for i in range(totalIntensityArr.shape[0]):
        tempCell = totalIntensityArr[i, 0].split()[0]
        uniqueCells.add(tempCell)

    uniqueCells = sorted(list(uniqueCells))

    for j in range(len(uniqueCells)):
        fActin = []
        perforin = []
        tumorAnti = []
        pZeta = []
        n = 0
        for i in range(totalIntensityArr.shape[0]):
            if(totalIntensityArr[i, 0].split()[0] == uniqueCells[j] ):
                if(totalIntensityArr[i,0][21:24] == '642'):
                    tumorAnti.append(totalIntensityArr[i, 2])
                if(totalIntensityArr[i,0][21:24] == '405'):
                    fActin.append(totalIntensityArr[i, 2])
                if(totalIntensityArr[i,0][21:24] == '488'):
                    perforin.append(totalIntensityArr[i, 2])
                if(totalIntensityArr[i,0][21:24] == '561'):
                    pZeta.append(totalIntensityArr[i, 2])
        fActinList.append(fActin)
        perforinList.append(perforin)
        tumorAntiList.append(tumorAnti)
        pZetaList.append(pZeta)


    # print(fActinList[0])
    # print(fActinList[1])

    fActinSample = []
    perforinSample = []
    tumorAntiSample = []
    pZetaSample = []

    '''
    for l in range(len(fActinList)):
        fActinSample.append(random.sample(fActinList[l], 100))
        perforinSample.append(random.sample(perforinList[l], 100))
        tumorAntiSample.append(random.sample(tumorAntiList[l], 100))
        pZetaSample.append(random.sample(pZetaList[l], 100))
    fActinList = np.array(fActinSample)
    perforinList = np.array(perforinSample)
    tumorAntiList = np.array(tumorAntiSample)
    pZetaList = np.array(pZetaSample)
    '''
    fActinList = np.array(fActinList)
    perforinList = np.array(perforinList)
    tumorAntiList = np.array(tumorAntiList)
    pZetaList = np.array(pZetaList)

    fActinSorted_0 = (np.sort(fActinList[0]))
    fActinSorted_1 = (np.sort(fActinList[1]))
    perforinSorted_0 = ((np.sort(perforinList[0])))
    perforinSorted_1 = (np.sort(perforinList[1]))
    tumorAntiSorted_0 = (np.sort(tumorAntiList[0]))
    tumorAntiSorted_1 = (np.sort(tumorAntiList[1]))
    pZetaSorted_0 =(np.sort(pZetaList[0]))
    pZetaSorted_1 = (np.sort(pZetaList[1]))

    # print(fActinSorted_0)
    fig,ax = plt.subplots(4,2)
    fig.set_size_inches(20, 12)

    #------------------------------------F-Actin---------------------------------------------------------------------------

    #numBinsFActin = math.floor(np.log(1 + (stats.moment(fActinList[0], moment = 3))/(math.sqrt(6/len(fActinList[0])))))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    _, bins, _ = ax[0,0].hist(fActinList[0], bins=50,  ec='black', alpha=0.5, label = uniqueCells[0], color ='darkred')
    _ = ax[0,0].hist(fActinList[1], bins=bins, alpha=0.5,  ec='black', label = uniqueCells[1], color='royalblue')
    ax[0,0].legend(loc='upper right')
    ax[0,0].set(ylabel='Frequency', title='F-Actin')
    ax[0,0].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    fActinTtest = stats.ttest_ind(fActinList[0], fActinList[1])
    #fActinT = fActinTtest[0]
    fActinPValue = fActinTtest[1]
    #print('fActin: ' + str(fActinPValue))
    fActinStr = 'p-value: ' + str('{:0.5e}'.format(fActinPValue))

    fActinMu_0 = round(np.mean(fActinList[0]), 1)
    fActinMu_1 = round(np.mean(fActinList[1]), 1)
    fActinSEM_0 = round(stats.sem(fActinList[0]), 1)
    fActinSEM_1 = round(stats.sem(fActinList[1]), 1)

    c = None
    if (fActinPValue > 0.05):
        c = u'\u2248'
    elif (fActinMu_0 > fActinMu_1):
        c = '>'
    else:
        c = '<'

    fActinStr = '\n'.join((
        str(uniqueCells[0][:3]) + ': ' + str(fActinMu_0) + ' ' + u"\u00B1" + ' ' + str(fActinSEM_0) + ' (n=' + str(len(fActinList[0])) +')',
        str(uniqueCells[1][:3]) + ': ' + str(fActinMu_1) + ' ' + u"\u00B1" + ' ' + str(fActinSEM_1) + ' (n=' + str(len(fActinList[1])) +')',
        fActinStr,
        '                    ' + str(uniqueCells[0][:4]) + ' ' + c + ' ' + str(uniqueCells[1][:4]) ))


    ax[0,0].text(0.63, 0.55, fActinStr, transform=ax[0,0].transAxes, fontsize=11,
    verticalalignment='top', bbox=props)


    _, bins1, patches1 = ax[0,1].hist(fActinSorted_0, bins = 5000, density=True, cumulative=True, histtype='step' , color ='darkred', label = uniqueCells[0])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    _, _, patches2 = ax[0,1].hist(fActinSorted_1, bins=bins1, density = True, cumulative=True, histtype='step', color ='royalblue', label = uniqueCells[1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])

    # ax[0,1].step(fActinSorted_0, np.arange(fActinSorted_0.size), color ='darkred', label = uniqueCells[0])
    # ax[0,1].step(fActinSorted_1, np.arange(fActinSorted_1.size), color='royalblue', label = uniqueCells[1])
    ax[0,1].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    ax[0,1].legend(loc='lower right')

    #------------------------------------Perofrin---------------------------------------------------------------------------
    #colors = ['darkred', 'royalblue']
    #numBinsPerf = math.floor(np.log(1 + (stats.moment(perforinList[0], moment = 3))/(math.sqrt(6/len(perforinList[0])))))

    _, binsPerforin, _ = ax[1,0].hist(perforinList[0], bins=50,  ec='black', alpha=0.5, label = uniqueCells[0], color ='darkred')
    _ = ax[1,0].hist(perforinList[1], bins=binsPerforin, alpha=0.5, ec='black', label = uniqueCells[1], color='royalblue')
    ax[1,0].legend(loc='upper right')
    ax[1,0].set(ylabel='Frequency', title='Perforin')
    ax[1,0].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    perforinTtest = stats.ttest_ind(perforinList[0], perforinList[1])
    #perforinT = perforinTtest[0]
    perforinPValue = perforinTtest[1]
    #print('perforin: ' + str(perforinPValue))
    perforinStr = 'p-value: ' + str('{:0.5e}'.format(perforinPValue))

    perforinMu_0 = round(np.mean(perforinList[0]), 1)
    perforinMu_1 = round(np.mean(perforinList[1]), 1)
    perforinSEM_0 = round(stats.sem(perforinList[0]), 1)
    perforinSEM_1 = round(stats.sem(perforinList[1]), 1)

    c = None
    if (perforinPValue > 0.05):
        c = u'\u2248'
    elif (perforinMu_0 > perforinMu_1):
        c = '>'
    else:
        c = '<'

    perfStr = '\n'.join((
        str(uniqueCells[0][:3]) + ': ' + str(perforinMu_0) + ' ' + u"\u00B1" + ' ' + str(perforinSEM_0) + ' (n=' + str(len(perforinList[0])) +')',
        str(uniqueCells[1][:3]) + ': ' + str(perforinMu_1) + ' ' + u"\u00B1" + ' ' + str(perforinSEM_1) + ' (n=' + str(len(perforinList[1])) +')',
        perforinStr,
        '                    ' + str(uniqueCells[0][:4]) + ' ' + c + ' ' + str(uniqueCells[1][:4]) ))


    ax[1,0].text(0.65, 0.55, perfStr, transform=ax[1,0].transAxes, fontsize=11,
    verticalalignment='top', bbox=props)

    _, binsPerf, patchesPerf = ax[1,1].hist(perforinSorted_0, bins = 5000, density=True, cumulative=True, histtype='step' , color ='darkred', label = uniqueCells[0])
    patchesPerf[0].set_xy(patchesPerf[0].get_xy()[:-1])
    _, _, patchesPerf2 = ax[1,1].hist(perforinSorted_1, bins=binsPerf, density = True, cumulative=True, histtype='step', color ='royalblue', label = uniqueCells[1])
    patchesPerf2[0].set_xy(patchesPerf2[0].get_xy()[:-1])



    # ax[1,1].step(perforinSorted_0, np.arange(perforinSorted_0.size), color ='darkred', label = uniqueCells[0])
    # ax[1,1].step(perforinSorted_1, np.arange(perforinSorted_1.size), color='royalblue', label = uniqueCells[1])
    ax[1,1].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off\
    ax[1,1].legend(loc='lower right')
    #------------------------------------Tumor Antigen---------------------------------------------------------------------------

    # numBinsAntigen = math.floor(np.log(1 + (stats.moment(tumorAntiList[0], moment = 3))/(math.sqrt(6/len(tumorAntiList[0])))))

    tumorAntiMu_0 = round(np.mean(tumorAntiList[0]), 1)
    tumorAntiMu_1 = round(np.mean(tumorAntiList[1]), 1)
    tumorAntiSEM_0 = round(stats.sem(tumorAntiList[0]), 1)
    tumorAntiSEM_1 = round(stats.sem(tumorAntiList[1]), 1)

    # tumorAntiSD_0 = np.std(tumorAntiList[0])
    # tumorAntiSD_1 = np.std(tumorAntiList[1])
    #
    # minAnti = min(tumorAntiMu_0 - 3*tumorAntiSD_0, tumorAntiMu_1 - 3*tumorAntiSD_1 )
    # maxAnti = max(tumorAntiMu_0 + 3*tumorAntiSD_0, tumorAntiMu_1 + 3*tumorAntiSD_1 )

    _, binsAntigen, _ = ax[2,0].hist(tumorAntiList[0], bins=50, ec='black', alpha=0.5, label = uniqueCells[0], color ='darkred')
    _ = ax[2,0].hist(tumorAntiList[1], bins=binsAntigen, alpha=0.5, ec='black', label = uniqueCells[1], color='royalblue')
    ax[2,0].legend(loc='upper right')
    ax[2,0].set(ylabel='Frequency', title='Tumor Antigen')
    ax[2,0].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    tumorAntiTtest = stats.ttest_ind(tumorAntiList[0], tumorAntiList[1])
    #tumorAntiT = tumorAntiTtest[0]
    tumorAntiPValue = tumorAntiTtest[1]
    #print('tumorAnti: ' + str(tumorAntiPValue))
    tumorAntiStr = 'p-value: ' + str('{:0.5e}'.format(tumorAntiPValue))



    c = None
    if (tumorAntiPValue > 0.05):
        c = u'\u2248'
    elif (tumorAntiMu_0 > tumorAntiMu_1):
        c = '>'
    else:
        c = '<'

    tumorAntiStr = '\n'.join((
        str(uniqueCells[0][:3]) + ': ' + str(tumorAntiMu_0) + ' ' + u"\u00B1" + ' ' + str(tumorAntiSEM_0) + ' (n=' + str(len(tumorAntiList[0])) +')',
        str(uniqueCells[1][:3]) + ': ' + str(tumorAntiMu_1) + ' ' + u"\u00B1" + ' ' + str(tumorAntiSEM_1) + ' (n=' + str(len(tumorAntiList[1])) +')',
        tumorAntiStr,
        '                    ' + str(uniqueCells[0][:4]) + ' ' + c + ' ' + str(uniqueCells[1][:4]) ))


    ax[2,0].text(0.65, 0.55, tumorAntiStr, transform=ax[2,0].transAxes, fontsize=11,
    verticalalignment='top', bbox=props)


    _, binsTumor, patchesTumor = ax[2,1].hist(tumorAntiSorted_0, bins = 5000, density=True, cumulative=True, histtype='step' , color ='darkred', label = uniqueCells[0])
    patchesTumor[0].set_xy(patchesTumor[0].get_xy()[:-1])
    _, _, patchesTumor2 = ax[2,1].hist(tumorAntiSorted_1, bins=binsTumor, density = True, cumulative=True, histtype='step', color ='royalblue', label = uniqueCells[1])
    patchesTumor2[0].set_xy(patchesTumor2[0].get_xy()[:-1])

    # ax[2,1].step(tumorAntiSorted_0, np.arange(tumorAntiSorted_0.size), color ='darkred', label = uniqueCells[0])
    # ax[2,1].step(tumorAntiSorted_1, np.arange(tumorAntiSorted_1.size), color='royalblue', label = uniqueCells[1])
    ax[2,1].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    ax[2,1].legend(loc='lower right')
    #-----------------------------------pZeta---------------------------------------------------------------------------

    pZetaMu_0 = round(np.mean(pZetaList[0]), 1)
    pZetaMu_1 = round(np.mean(pZetaList[1]), 1)
    pZetaSEM_0 = round(stats.sem(pZetaList[0]), 1)
    pZetaSEM_1 = round(stats.sem(pZetaList[1]), 1)

    # pZetaSD_0 = np.std(pZetaList[0])
    # pZetaSD_1 = np.std(pZetaList[1])
    #
    # minZeta = min(pZetaMu_0 - 3*pZetaSD_0,pZetaMu_1 - 3*pZetaSD_1 )
    # maxZeta = max(pZetaMu_0 + 3*pZetaSD_0,pZetaMu_1 + 3*pZetaSD_1 )

    _, binsZeta, _ = ax[3,0].hist(pZetaList[0], bins=50, ec='black', alpha=0.5, label = uniqueCells[0], color ='darkred')
    _ = ax[3,0].hist(pZetaList[1], bins=binsZeta, alpha=0.5, ec='black', label = uniqueCells[1], color='royalblue')
    ax[3,0].legend(loc='upper right')
    ax[3,0].set(ylabel='Frequency', title='pZeta')
    ax[3,0].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    pZetaTtest = stats.ttest_ind(pZetaList[0], pZetaList[1])
    #pZetaT = pZetaTtest[0]
    pZetaPValue = pZetaTtest[1]
    #print('pZeta: ' + str(pZetaPValue))
    pZetaStr = 'p-value: ' + str('{:0.5e}'.format(pZetaPValue))



    c = None
    if (pZetaPValue > 0.05):
        c = u'\u2248'
    elif (pZetaMu_0 > pZetaMu_1):
        c = '>'
    else:
        c = '<'

    pZetaStr = '\n'.join((
        str(uniqueCells[0][:3]) + ': ' + str(pZetaMu_0) + ' ' + u"\u00B1" + ' ' + str(pZetaSEM_0) + ' (n=' + str(len(pZetaList[0])) +')',
        str(uniqueCells[1][:3]) + ': ' + str(pZetaMu_1) + ' ' + u"\u00B1" + ' ' + str(pZetaSEM_1) + ' (n=' + str(len(pZetaList[1])) +')',
        pZetaStr,
        '                    ' + str(uniqueCells[0][:4]) + ' ' + c + ' ' + str(uniqueCells[1][:4]) ))


    ax[3,0].text(0.65, 0.55, pZetaStr, transform=ax[3,0].transAxes, fontsize=11,
    verticalalignment='top', bbox=props)

    _, binsPZeta, patchesPZeta = ax[3,1].hist(pZetaSorted_0, bins = 5000, density=True, cumulative=True, histtype='step' , color ='darkred', label = uniqueCells[0])
    patchesPZeta[0].set_xy(patchesPZeta[0].get_xy()[:-1])
    _, _, patchesPZeta2 = ax[3,1].hist(pZetaSorted_1, bins=binsPZeta, density = True, cumulative=True, histtype='step', color ='royalblue', label = uniqueCells[1])
    patchesPZeta2[0].set_xy(patchesPZeta2[0].get_xy()[:-1])


    # ax[3,1].step(pZetaSorted_0, np.arange(pZetaSorted_0.size), color ='darkred', label = uniqueCells[0])
    # ax[3,1].step(pZetaSorted_1, np.arange(pZetaSorted_1.size), color='royalblue', label = uniqueCells[1])
    ax[3,1].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    ax[3,1].legend(loc='lower right')


    saveImg ='experiments/count_intensities2_Mean_Intensity_Graph.png'
    plt.savefig(saveImg, dpi = 100)
    plt.show()


def main():
    generate('experiments/count_intensities2.xls')

if __name__ == '__main__':
    main()
