# ***************************************************************************
# Copyright 2020, Jianwei Zheng, Chapman University,
# zheng120@mail.chapman.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Jianwei Zheng.
import pandas as pd
import numpy as np
from numpy import mean
import os,glob
import math
import fnmatch
import re
import gc
import shutil
from joblib import Parallel, delayed
import scipy.signal as scipysi
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics import plot_roc_curve
from scipy import interp
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy import stats
import shap
import seaborn as sns
from joblib import dump
from joblib import load
def LocatePeaksForVerification(DataFileName, DataSet, RefPoints, LogFile):
    ####Calculate Peak and Valley information features
    #DataSet = DataList
    #RefPoints = ReferencePoint
    #LogFile=GraphicFolder

    #SR beat
    LeadNames =['aVF', 'aVL', 'aVR', 'I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for iset in range(2):
        plt.ioff()
        plt.axis('off')
        plt.rcParams.update({'font.size': 18})
        for iLead in range(12):
            #SinglePeaksValleysDF = LocatePeaks(DataSet[0].iloc[:,iLead])
            SingleLeadDataSet =-iLead*5+ DataSet[iset].iloc[:,iLead]
            peaksloc, _ = scipysi.find_peaks(SingleLeadDataSet)
            prominences = scipysi.peak_prominences(SingleLeadDataSet, peaksloc)[0]
            contour_heights = SingleLeadDataSet[peaksloc] - prominences
            results_half = scipysi.peak_widths(SingleLeadDataSet, peaksloc, rel_height=0.5)
            results_full = scipysi.peak_widths(SingleLeadDataSet, peaksloc, rel_height=1)
            valleyloc, _ = scipysi.find_peaks(-SingleLeadDataSet)

            plt.plot(SingleLeadDataSet)
            plt.text(0, SingleLeadDataSet[0], LeadNames[iLead], fontsize=12)
            plt.plot(peaksloc, SingleLeadDataSet[peaksloc], 'bo')
            for ipeak in range(len(peaksloc)):
                plt.text(peaksloc[ipeak], SingleLeadDataSet[peaksloc[ipeak]]+1, 'P'+str(ipeak), fontsize=10)

            #plt.vlines(x=peaksloc, ymin=contour_heights, ymax=SingleLeadDataSet[peaksloc])
            plt.vlines(x=RefPoints[iset], ymin=SingleLeadDataSet.min()-0.5, ymax=SingleLeadDataSet.max()+0.5, linestyle='dashed', color='blue')
            #plt.plot(range(len(SingleLeadDataSet)), np.zeros((len(SingleLeadDataSet))))
            # plt.hlines(*peakswidth_half[1:], color="C2")
            # plt.hlines(*peakswidth_full[1:], color="C3")
            # plt.axvline(x=800, C='r', ls='--')
            # plt.axvline(x=400, C='r', ls='--')
            # plt.axvline(x=1200, C='r', ls='--')
            plt.plot(valleyloc, SingleLeadDataSet[valleyloc], 'r+')
            for ivalley in range(len(valleyloc)):
                plt.text(valleyloc[ivalley], SingleLeadDataSet[valleyloc[ivalley]]-1, 'V'+str(ivalley), fontsize=10)
            #plt.hlines(*results_half[1:], color="C2")
            #plt.hlines(*results_full[1:], color="C3")
            #contour_heights = SingleLeadDataSet[peaksloc] - peaksprom[0]
            #plt.vlines(x=peaksloc, ymin=0, ymax=SingleLeadDataSet[peaksloc])
            #plt.vlines(x=valleyloc, ymax=0, ymin=SingleLeadDataSet[valleyloc])
        if iset==0:
            fileName = LogFile + str(DataFileName) + '_Normal_' + '.png'
        else:
            fileName = LogFile + str(DataFileName) + '_PVC_' + '.png'
        plt.savefig(fileName)
        plt.close()

        plt.axis('off')
        SingleLeadDataSet = DataSet[iset].iloc[:, 4]
        peaksloc, other = scipysi.find_peaks(SingleLeadDataSet, width=20)
        prominences = scipysi.peak_prominences(SingleLeadDataSet, peaksloc)[0]
        contour_heights = SingleLeadDataSet[peaksloc] - prominences
        results_half = scipysi.peak_widths(SingleLeadDataSet, peaksloc, rel_height=0.5)
        results_full = scipysi.peak_widths(SingleLeadDataSet, peaksloc, rel_height=1)
        valleyloc, _ = scipysi.find_peaks(-SingleLeadDataSet)

        plt.plot(SingleLeadDataSet)

        plt.plot(peaksloc, SingleLeadDataSet[peaksloc], 'bo')
        plt.text(peaksloc, SingleLeadDataSet[peaksloc]+0.05, 'P2', fontsize=18)

        plt.hlines(*results_half[1:], color="C2")
        plt.hlines(*results_full[1:], color="C3")
        #contour_heights = SingleLeadDataSet[peaksloc] - peaksprom[0]
        plt.vlines(x=peaksloc, ymin=contour_heights, ymax=SingleLeadDataSet[peaksloc])
        plt.plot(results_full[2], SingleLeadDataSet[results_full[2]], 'bo')
        plt.text(results_full[2], SingleLeadDataSet[results_full[2]]+0.05, 'P1', fontsize=18)
        plt.plot(results_full[3], SingleLeadDataSet[int(results_full[3])]-0.01, 'bo')
        plt.text(results_full[3], SingleLeadDataSet[int(results_full[3])]+0.05, 'P4', fontsize=18)
        plt.plot(peaksloc, contour_heights, 'bo')
        plt.text(peaksloc, contour_heights+0.05, 'P3', fontsize=18)

    return True

def LocatePeaks(SingleLeadDataSet):
    ####Calculate Peak information 75 features
    #SingleLeadDataSet = DataSet[0].iloc[:, 11]
    #RefPoint = RefPoints[0]
    peaksloc, _ = scipysi.find_peaks(SingleLeadDataSet)
    peaksprom = scipysi.peak_prominences(SingleLeadDataSet, peaksloc)
    half_peakswidth = scipysi.peak_widths(SingleLeadDataSet, peaksloc, rel_height=0.5)
    full_peakswidth = scipysi.peak_widths(SingleLeadDataSet, peaksloc,  rel_height=1)
    peaksheight = SingleLeadDataSet[peaksloc]

    valleyloc, _ = scipysi.find_peaks(-SingleLeadDataSet)
    valleysprom = scipysi.peak_prominences(-SingleLeadDataSet, valleyloc)
    half_valleyswidth = scipysi.peak_widths(-SingleLeadDataSet, valleyloc, rel_height=0.5)
    full_valleyswidth = scipysi.peak_widths(-SingleLeadDataSet, valleyloc, rel_height=1)
    valleysheight = SingleLeadDataSet[valleyloc]

    PeaksDF = pd.DataFrame(
        {'Location': peaksloc,
         'Prom': peaksprom[0], 'PromLB': peaksprom[1]-peaksloc, 'PromRB': peaksprom[2]-peaksloc,
         'HalfWidth': half_peakswidth[0], 'ContourHeight': peaksprom[0]-peaksheight, 'FullWidth': full_peakswidth[0],
          'Height': peaksheight})
    PeaksDF['IsPeak'] = 1
    ValleysDF = pd.DataFrame(
        {'Location': valleyloc,
         'Prom': valleysprom[0], 'PromLB': valleysprom[1]-valleyloc, 'PromRB': valleysprom[2]-valleyloc,
         'HalfWidth': half_valleyswidth[0], 'ContourHeight': valleysprom[0]-valleysheight, 'FullWidth': full_valleyswidth[0],
         'Height': valleysheight})
    ValleysDF['IsPeak'] = -1

    PeaksValleysDF = pd.concat([PeaksDF, ValleysDF], axis=0, ignore_index=True)
    PeaksValleysDF = PeaksValleysDF.sort_values(by=['Location'], ascending=True)
    PeaksValleysDF = PeaksValleysDF.reset_index(drop=True)
    return PeaksValleysDF

def CalculateSingleFeatures(DataSet, RefPoints, num_LeadGroup, LogFile):
    ####Calculate Peak and Valley information features
    #DataSet = DataList
    #RefPoints = ReferencePoint
    ColNames = ['Location','Prom', 'PromLB', 'PromRB','HalfWidth', 'ContourHeight', 'FullWidth','Height', 'IsPeak']
    PeaksValleysDF = pd.DataFrame(columns=ColNames)
    DiffsPeaksValleyDF = pd.DataFrame(columns=ColNames[0:-1])
    RatioArray_DiffsSinglLead = np.empty((0, len(ColNames)-1), float)
    DiffsArray_RatiosSingleLead = np.empty((0, len(ColNames)-1), float)
    RatioArray_AllLeadsRows = np.empty((0, len(ColNames)-1), float)
    RatioArray_AllLeadsRowsCols = np.empty((0, len(ColNames)-2), float)
    #PeaksMeanVarDF = pd.DataFrame(columns=['Location', 'Prom', 'PromLB', 'PromRB', 'Width', 'WidthHeight', 'WidthLips',
    #     'WidthRips', 'Height'])
    #DiffsPeaksValleyRatioDF = pd.DataFrame(columns=ColNames[0:-1])  # difference of height, proms, width ratio without time difference in sigle lead
    LeadGroup = list(range(0, 12))
    #if num_LeadGroup==1:
    #    LeadGroup = list([3, 4, 6, 7, 8, 9, 10, 11])
    SRLength = [7, 8, 7, 8, 7, 8, 6, 6, 6, 7, 8, 8]
    PVCLength = [4, 5, 5, 7, 4, 4, 6, 4, 5, 5, 4, 4]
    Index =0
    for iSet in range(2):
        for iLead in LeadGroup:
            SinglePeaksValleysDF = LocatePeaks(DataSet[iSet].iloc[:,iLead])
            indexRef = (abs(SinglePeaksValleysDF['Location'] - RefPoints[iSet])).idxmin()
            offset_indexRef =4
            if indexRef <offset_indexRef:
                for i in range(offset_indexRef-indexRef):
                    SinglePeaksValleysDF.loc[-1] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    SinglePeaksValleysDF.index = SinglePeaksValleysDF.index + 1  # shifting index
                    SinglePeaksValleysDF = SinglePeaksValleysDF.sort_index()
            indexRef = (abs(SinglePeaksValleysDF['Location'] - RefPoints[iSet])).idxmin() # refresh index of reference point
            if ((SinglePeaksValleysDF.shape[0] - indexRef) < offset_indexRef):
                for i in range(4 - SinglePeaksValleysDF.shape[0] + indexRef):
                    SinglePeaksValleysDF = SinglePeaksValleysDF.append({'Location':0, 'Prom':0, 'PromLB':0,
                                        'PromRB':0, 'HalfWidth':0, 'ContourHeight':0, 'FullWidth':0,
                                        'Height':0, 'IsPeak':0}, ignore_index=True)
            indexRef = (abs(SinglePeaksValleysDF['Location'] - RefPoints[iSet])).idxmin() # refresh index of reference point
            SinglePeaksValleysDF = SinglePeaksValleysDF.iloc[indexRef-offset_indexRef:indexRef+offset_indexRef,:]
            SinglePeaksValleysDF = SinglePeaksValleysDF.reset_index(drop=True)

            PeaksValleysDF = PeaksValleysDF.append(SinglePeaksValleysDF, sort=False, ignore_index=True)
            #print('Index Range: iDataset+Lead', iSet, ':', iLead)
            StartIndex = Index
            EndIndex = StartIndex +  SinglePeaksValleysDF.size-1
            Index = Index +  SinglePeaksValleysDF.size
            print('Index Range: iDataset+Lead', iSet, ':', iLead, 'Start:End Index', StartIndex,':', EndIndex)

            DiffsSinglePeaksValleyDF = pd.DataFrame(columns=ColNames[0:-1])
            # Difference of location, prom, height, and width have compare ratio
            for iRow in range(0, SinglePeaksValleysDF.shape[0] - 1):
                DiffsSinglePeaksValleyDF = DiffsSinglePeaksValleyDF.append(SinglePeaksValleysDF.iloc[iRow, 0:-1] - SinglePeaksValleysDF.iloc[iRow + 1:, 0:-1], sort=False, ignore_index=True)
            #print(DiffsSinglePeaksValleyDF.shape)
            for iRow in range(0, DiffsSinglePeaksValleyDF.shape[0] - 1):
                RatioArray = np.true_divide(DiffsSinglePeaksValleyDF.iloc[iRow + 1:, 0:].to_numpy(),
                                            DiffsSinglePeaksValleyDF.iloc[iRow, :].to_numpy(), casting='unsafe',
                                            out=np.zeros_like(DiffsSinglePeaksValleyDF.iloc[iRow + 1:, 0:].to_numpy()),
                                            where=DiffsSinglePeaksValleyDF.iloc[iRow, :].to_numpy() != 0)
                RatioArray_DiffsSinglLead = np.append(RatioArray_DiffsSinglLead, RatioArray, axis=0)
                print('RatioArray_DiffsSinglLead:', str(iRow),':', RatioArray.size)

            RatioArray_SingleLead = np.empty((0, 8), float)
            for iRow in range(0, SinglePeaksValleysDF.shape[0] - 1):
                RatioArray = np.true_divide(SinglePeaksValleysDF.iloc[iRow + 1:, 0:-1].to_numpy(),
                                            SinglePeaksValleysDF.iloc[iRow, 0:-1].to_numpy(), casting='unsafe',
                                            out=np.zeros_like(SinglePeaksValleysDF.iloc[iRow + 1:, 0:-1].to_numpy()),
                                            where=SinglePeaksValleysDF.iloc[iRow, 0:-1].to_numpy() != 0)
                RatioArray_SingleLead = np.append(RatioArray_SingleLead, RatioArray, axis=0)

            for iRow in range(0, RatioArray_SingleLead.shape[0] - 1):
                DiffsArray_RatiosSingleLead = np.append(DiffsArray_RatiosSingleLead, RatioArray_SingleLead[iRow,] - RatioArray_SingleLead[iRow + 1:, ], axis=0)
            #print(DiffsPeaksValleyRatioDF.shape)
    PeaksValleysDF = PeaksValleysDF.reset_index(drop=True)


    #PeaksValleyRatioDF = pd.DataFrame(columns=ColNames[0:-1]) #height, proms, width ratio without time difference
    for iRow in range(0, PeaksValleysDF.shape[0]-1):
        DiffsPeaksValleyDF = DiffsPeaksValleyDF.append(PeaksValleysDF.iloc[iRow,0:-1]-PeaksValleysDF.iloc[iRow+1:, 0:-1],sort=False, ignore_index=True)
        RatioArray = np.true_divide(PeaksValleysDF.iloc[iRow + 1:, 0:-1].to_numpy(), PeaksValleysDF.iloc[iRow, 0:-1].to_numpy(), casting='unsafe',
                                    out=np.zeros_like(PeaksValleysDF.iloc[iRow + 1:, 0:-1].to_numpy()),
                                    where=PeaksValleysDF.iloc[iRow, 0:-1].to_numpy() != 0)
        RatioArray_AllLeadsRows = np.append(RatioArray_AllLeadsRows, RatioArray, axis=0)

    # difference of height, proms, width / time difference ratio
    # 440256
    RatioArray_DiffsVSTimeDist = np.true_divide(DiffsPeaksValleyDF.iloc[:, 1:].transpose().to_numpy(), DiffsPeaksValleyDF.iloc[:, 0].transpose().to_numpy(),
                                                    casting='unsafe', out=np.zeros_like(DiffsPeaksValleyDF.iloc[:, 1:].transpose().to_numpy()),
                                                    where=DiffsPeaksValleyDF.iloc[:, 0].transpose().to_numpy() != 0)

    PeaksValleysHorizontalDF = PeaksValleysDF[ColNames[0:-1]]
    RatioArray_AllLeadsCols = np.empty((0, PeaksValleysHorizontalDF.shape[0]), float)
    for iCol in range(0, PeaksValleysHorizontalDF.shape[1] - 1):
        RatioArray = np.true_divide(PeaksValleysHorizontalDF.iloc[:, iCol + 1:].transpose().to_numpy(),
                                    PeaksValleysHorizontalDF.iloc[:, iCol].transpose().to_numpy(), casting='unsafe',
                                    out=np.zeros_like(
                                        PeaksValleysHorizontalDF.iloc[:, iCol + 1:].transpose().to_numpy()),
                                    where=PeaksValleysHorizontalDF.iloc[:, iCol].transpose().to_numpy() != 0)
        RatioArray_AllLeadsCols = np.append(RatioArray_AllLeadsCols, RatioArray, axis=0)

    Index = 573984
    for iRow in range(PeaksValleysDF.shape[0]):
        for iCol in range(PeaksValleysDF.shape[1]-1):
            ColList = list(range(PeaksValleysDF.shape[1]-1))
            ColList.pop(iCol)
            RatioArray = np.true_divide(PeaksValleysDF.iloc[iRow + 1:, ColList].to_numpy(), PeaksValleysDF.iloc[iRow, iCol],
                                    casting='unsafe', out=np.zeros_like(PeaksValleysDF.iloc[iRow + 1:, ColList].to_numpy()),
                                    where=PeaksValleysDF.iloc[iRow, iCol] != 0)
            print('Row:',iRow, 'Col:', iCol)
            StartIndex = Index
            EndIndex = StartIndex + RatioArray.size - 1
            Index = Index + RatioArray.size
            print("Index Range", StartIndex, ':', EndIndex)

            RatioArray_AllLeadsRowsCols = np.append(RatioArray_AllLeadsRowsCols, RatioArray, axis=0)

    ReturnDF = pd.concat([pd.Series(PeaksValleysDF.to_numpy().flatten()), #1,728
                          pd.Series(RatioArray_DiffsSinglLead.flatten()), #72,576         #74304
                          pd.Series(DiffsArray_RatiosSingleLead.flatten()), #72,576       #146880
                          pd.Series(RatioArray_AllLeadsRows.flatten()), #146,688          #293568
                          pd.Series(DiffsPeaksValleyDF.to_numpy().flatten()), #146,688    #440256
                          pd.Series(RatioArray_DiffsVSTimeDist.flatten()), #128,352       #568608
                          pd.Series(RatioArray_AllLeadsCols.flatten()), #5,376            #573984
                          pd.Series(RatioArray_AllLeadsRowsCols.flatten())]) #1,026,816   #1600800

    ReturnDF.replace(to_replace=np.inf, value=0, inplace=True)
    ReturnDF.replace(to_replace=np.nan, value=0, inplace=True)
    return ReturnDF

def GenerateSingleEngFeatureData(DataFileName, num_LeadGroup, peaksInfo, LogFile):
    #for iIndex in range(len(LabelDF)): #Loop for each patient's file
    #DataFileName =654020 DataFiles[0]
    #peaksInfo = PeakInfoDF
    SinglPeakDF = peaksInfo.loc[peaksInfo['HospitalID'] == DataFileName]
    DataDF = pd.read_csv(DataFilePath + str(DataFileName) + '.csv')
    #PVCDF = DataDF.iloc[SinglPeakDF.iloc[0, 4] - 1:SinglPeakDF.iloc[0, 6], :]
    #NormalDF = DataDF.iloc[SinglPeakDF.iloc[0, 1] - 1:SinglPeakDF.iloc[0, 3], :]
    NormalDF = DataDF.iloc[SinglPeakDF.iloc[0, 2] - 215:SinglPeakDF.iloc[0, 2] + 215, :]
    PVCDF = DataDF.iloc[SinglPeakDF.iloc[0, 5] - 335:SinglPeakDF.iloc[0, 5] + 335, :]
    PVCDF = PVCDF.reset_index(drop=True)
    NormalDF = NormalDF.reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    NormalDataDF = pd.DataFrame(scaler.fit_transform(NormalDF))
    PVCDataDF = pd.DataFrame(scaler.fit_transform(PVCDF))
    #ReferencePoint = [SinglPeakDF.iloc[0, 2]-SinglPeakDF.iloc[0, 1],SinglPeakDF.iloc[0, 5]-SinglPeakDF.iloc[0, 4]]
    ReferencePoint = [215, 335]

    DataList = list()
    DataList.append(NormalDataDF)#NormalDF)
    DataList.append(PVCDataDF)#PVCDF)
    FeatureArray = CalculateSingleFeatures(DataList, ReferencePoint, num_LeadGroup, LogFile)

    return FeatureArray

def GenerateEngFeature(DataFileNames, num_LeadGroup, peaksInfo, LogFile):
    print("Start Generating Eng Feature Array:", file=open(LogFile, "a"))
    print("Number of Lead Group:", num_LeadGroup, file=open(LogFile, "a"))
    num_cores = 36
    results = Parallel(n_jobs=num_cores)(delayed(GenerateSingleEngFeatureData)
            (DataFileNames[Item], num_LeadGroup, peaksInfo, LogFile) for Item in range(len(DataFileNames)))
    EngFeatureArray = np.column_stack(results)
    EngFeatureArray = EngFeatureArray.transpose()
    print("Finish Generating Eng Data Feature Array:", file=open(LogFile, "a"))
    print("Eng Data Feature Array Size is:", EngFeatureArray.shape, file=open(LogFile, "a"))
    return EngFeatureArray

DataFilePath = '../CSVECGDenoisingSmoothData_Truncated10S_20190418/' # The data file path storage the ECG data
PeakDataFilePath = '../PeaksInfo20190120_AutoFeatures/' #The data fles path storage the SR Beat and PVC R-wave peak locations
LogFileWithPath = '../20200120JACC_AutoFeatures.csv'
LogFolder = '../LRVOT_AutoFeature_SMOTE03_JACC_0120/'
LabelFile = '../PVCVT_STD_20200315.xlsx' #The diagnosis File
print("Running start:", file=open(LogFileWithPath, "a"))

LabelDF = pd.read_excel(LabelFile, index=False)
LabelDF = LabelDF.loc[(LabelDF['LeftRight']=='Right') | (LabelDF['LeftRight']=='Left')]
LabelDF['LeftRightInOut'] = LabelDF['LeftRight']+ LabelDF['InOut']
#Encode the RVOT as 0 and LVOT as 1
LabelDF['LeftRightInOut'].replace(to_replace='RightOut', value=0, inplace=True)
LabelDF['LeftRightInOut'].replace(to_replace='LeftOut', value=1, inplace=True)
LabelDF['LeftRightInOut'].value_counts()
LabelDF = LabelDF.reset_index(drop=True)
ClassLabelDF = LabelDF.loc[(LabelDF['LeftRightInOut'] == 0) | (LabelDF['LeftRightInOut'] == 1)]

#each peak information file contains HospitalID, SR beat start point, SR beat R-wave peak point, SR beat end point, and similar information for PVCs
PeakInfoDF = pd.DataFrame(columns=['HospitalID', 'Nstart', 'NormalRPeak', 'Nend', 'PVCstart', 'PVCRPeak', 'PVCend'])
for i in range(len(ClassLabelDF['HospitalID'])):
    SinglePeakInfoDF = pd.read_csv(PeakDataFilePath + ClassLabelDF['HospitalID'].iloc[i].astype(str)+'.csv')
    PeakInfoDF = PeakInfoDF.append(SinglePeakInfoDF, ignore_index=True)


ClassLabelDF = ClassLabelDF.reset_index(drop=True)
ClassLabelDF['LeftRightInOut'].value_counts()
print("#2. Compare LVOT VS RVOT", file=open(LogFileWithPath, "a"))
print(ClassLabelDF['LeftRightInOut'].value_counts(), file=open(LogFileWithPath, "a"))
ClassLabelArray = np.array(ClassLabelDF['LeftRightInOut'].to_numpy(), dtype=int)
DataFiles = ClassLabelDF['HospitalID']


FeatureArray = GenerateEngFeature(DataFiles, 0, PeakInfoDF, LogFileWithPath)


np.save(LogFolder+'FeatureArray.npy', FeatureArray)
FeatureArray = np.load(LogFolder+'FeatureArray.npy')
print('ClassLabel size is:', ClassLabelArray.shape, file=open(LogFileWithPath, "a"))
print('FeatureArray size is:', FeatureArray.shape, file=open(LogFileWithPath, "a"))

GSBestClassifier = xgb.XGBClassifier(objective='binary:logistic', tree_method='approx', scale_pos_weight=1,
                               grow_policy='depthwise', n_estimators=4000, learning_rate=0.01, max_depth=50, n_jobs=25)
oversample = SMOTE(sampling_strategy='auto')
Train_X, Train_Y = oversample.fit_resample(FeatureArray, ClassLabelArray)
XGBfit = GSBestClassifier.fit(Train_X, Train_Y)

#After model was trained, save it to joblib data
dump(XGBfit, LogFolder+'pretrainedModle.joblib.dat')

#Load the pretrained model and use it to predict the RVOT and LVOT locations,
loaded_model = load(LogFolder+'pretrainedModle.joblib.dat')
#Test the loaded model
print(metrics.classification_report(ClassLabelArray, loaded_model.predict(FeatureArray)))