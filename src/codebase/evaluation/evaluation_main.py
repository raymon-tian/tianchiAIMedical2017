#coding=utf-8
import os
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogFormatter,StrMethodFormatter,FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from NoduleFinding import NoduleFinding

from tools import csvTools

# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.1 # Mininum value of x-axis of FROC curve
FROC_maxX = 8.5 # Maximum value of x-axis of FROC curve
bLogPlot = True

def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_index_im   = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

    return candidates

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[math.floor(Pz*len(vec))]
        sens_up[i] = vec[math.floor((1.0-Pz)*len(vec))]

    return sens_mean,sens_lb,sens_up

def computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,FROCImList,excludeList,numberOfBootstrapSamples=1000, confidence = 0.95):

    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
    
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)
    
    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:,i:i+1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid],candidate),axis = 1)

    for i in range(numberOfBootstrapSamples):
        print 'computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict,FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
    
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # compute mean and CI
    sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)

    return all_fps, sens_mean, sens_lb, sens_up

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): # Handle border case when there are no false positives and ROC analysis give nan values.
      print "WARNING, this system has no false positives.."
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False,numberOfBootstrapSamples=1000,confidence = 0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: 所有的测试集CT图像名称列表
    @param results_filename: 提交的csv文件，*.csv
    @param outputDir: 存放F-ROC计算结果的文件夹路径
    @param allNodules: 所有的nodule构成的字典，以图像名索引，GT
    @param CADSystemName: 系统名字，用来作为文件的前缀之类
    @param maxNumberOfCADMarks: 一张CT图像最多允许多少条标注
    @param performBootstrapping:
    @param numberOfBootstrapSamples:
    @param confidence:
    '''

    nodOutputfile = open(os.path.join(outputDir,'CADAnalysis.txt'),'w')
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    results = csvTools.readCSV(results_filename)  # 最终的csv文件结果

    allCandsCAD = {}
    
    for seriesuid in seriesUIDs:                  # 对每一个测试图像ID
        
        # collect candidates from result file
        nodules = {}
        header = results[0]                       # csv文件第一行，表头
        
        i = 0
        for result in results[1:]:                # 对于每一个标注
            nodule_seriesuid = result[header.index(seriesuid_label)]  # 该标注的文件名
            
            if seriesuid == nodule_seriesuid:     # 判断该标注的是否是suriesuid
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule  # 同一个ID的所有nodule
                i += 1

        if (maxNumberOfCADMarks > 0):   
            # 如果一张CT图像的标注超过某个值，就按照得分排序，只截取前maxNumberOfCADMarks条记录
            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.iteritems():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True) # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.iteritems():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2
        
        print 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules        # 以图像名称索引nodule字典
    
    # open output files
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')
    
    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0 # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:  # 对于每一张CT图像
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]       # 该图像的预测标注信息
        except KeyError:
            candidates = {}

        totalNumberOfCands += len(candidates.keys())  # 预测标注总个数

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()               # 复制该图像的预测标注信息

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]      # 该图像的GT标注信息
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:              # 对GT标注中的每一条记录
            # increment the number of nodules
            if noduleAnnot.state == "Included":       # 该标注被用来计算结果
                totalNumberOfNodules += 1             # 记录GT标注总数

            x = float(noduleAnnot.coordX)             # GT标注 X坐标
            y = float(noduleAnnot.coordY)             # GT标注 Y坐标
            z = float(noduleAnnot.coordZ)             # GT标注 Z坐标

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm) # GT标注的直径
            if diameter < 0.0:
              diameter = 5
            radiusSquared = pow((diameter / 2.0), 2.0) # GT 半径的平方

            found = False
            noduleMatches = []
            for key, candidate in candidates.iteritems(): # 遍历预测的每一条标注
                x2 = float(candidate.coordX)              # 预测的坐标 X
                y2 = float(candidate.coordY)              # 预测的坐标 Y
                z2 = float(candidate.coordZ)              # 预测的坐标 Z
                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.) # 预测与真实的距离的差
                if dist < radiusSquared:                  # 如果距离小于半径
                    if (noduleAnnot.state == "Included"): # 可被用于测评的标注
                        found = True
                        noduleMatches.append(candidate)   # 将该条预测的标注添加到 noduleMatches
                        if key not in candidates2.keys():
                            print "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), seriesuid, str(noduleAnnot.id))
                        else:
                            del candidates2[key]          # 在candidates2中将相应数据删除
                    elif (noduleAnnot.state == "Excluded"): # an excluded nodule
                        if bOtherNodulesAsIrrelevant: #    delete marks on excluded nodules so they don't count as false positives
                            if key in candidates2.keys():
                                irrelevantCandidates += 1
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
                                del candidates2[key]
            if len(noduleMatches) > 1:    # 如果预测的标注中有至少两个都预测到了GT的某一个nodule
                doubleCandidatesIgnored += (len(noduleMatches) - 1)  # 舍弃多余的标注，记录舍弃的数目
            if noduleAnnot.state == "Included":  # 判断GT中的这条标注可被用来计算F-ROC
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True: # 对该条GT标注，在预测标注中，找到了至少一条符合的
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(noduleMatches)): # 对所有符合条件的预测，寻找最大的概率的一条
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROCGTList.append(1.0)
                    FROCProbList.append(float(maxProb))
                    FPDivisorList.append(seriesuid)
                    excludeList.append(False)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    candTPs += 1
                else:
                    candFNs += 1
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), int(-1), "NA"))
                    nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.iteritems():
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id), float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write("    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
    nodOutputfile.write("    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)
    
    if performBootstrapping:
        fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,seriesUIDs,excludeList,
                                                                  numberOfBootstrapSamples=numberOfBootstrapSamples, confidence = confidence)
        
    # Write FROC curve
    with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
    
    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    
    sens_itp = np.interp(fps_itp, fps, sens)
    
    sum_sensitivity = 0
    for idx in range(len(fps_itp)-1):
        if fps_itp[idx]<0.125 and fps_itp[idx+1]>0.125:
            print("0.125:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
        if fps_itp[idx]<0.25 and fps_itp[idx+1]>0.25:
            print("0.25:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
        if fps_itp[idx]<0.5 and fps_itp[idx+1]>0.5:
            print("0.5:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
        if fps_itp[idx]<1 and fps_itp[idx+1]>1:
            print("1:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
        if fps_itp[idx]<2 and fps_itp[idx+1]>2:
            print("2:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
        if fps_itp[idx]<4 and fps_itp[idx+1]>4:
            print("4:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
        if fps_itp[idx]<8 and fps_itp[idx+1]>8:
            print("8:",sens_itp[idx])
            sum_sensitivity += sens_itp[idx]
    ave_sensitivity = sum_sensitivity/7.0
    print("final score is %d" % ave_sensitivity)
    if performBootstrapping:
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))
        
        if bLogPlot:
            plt.xscale('log', basex=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)
    
def getNodule(annotation, header, state = ""):
    nodule = NoduleFinding()                                # 实例化一个nodule,将各个属性赋予该nodule
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]
    
    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]
    
    if not state == "":
        nodule.state = state

    return nodule
    
def collectNoduleAnnotations(annotations, seriesUIDs):
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:      # 对于测试集里的每一个CT图像
        print 'adding nodule annotations: ' + seriesuid
        
        nodules = []
        numberOfIncludedNodules = 0
        
        # add included findings
        header = annotations[0]       # 读取anno文件的第一行（seriesuid,coordX,coordY,coordZ,probability）
        for annotation in annotations[1:]:  # 遍历anno数据的每一行，寻找所有属于seriesuid的标注
            nodule_seriesuid = annotation[header.index(seriesuid_label)]  # 取出这一行的第一个元素
            
            if seriesuid == nodule_seriesuid:   # 判断该元素与seriesuid是否一致
                nodule = getNodule(annotation, header, state = "Included") # 取出该nodule的各种信息
                nodules.append(nodule)          # 将该nodule作为一个结构体，添加到nodules
                numberOfIncludedNodules += 1    # 属于该suriesuid的nodule 计数加一
            
        allNodules[seriesuid] = nodules  # 将所有属于seriesuid的nodule放在一起，给一个suriesuid索引
        noduleCount      += numberOfIncludedNodules   # 被计数的nodule数据
        noduleCountTotal += len(nodules)              # 所有的nodule总数
    
    print 'Total number of included nodule annotations: ' + str(noduleCount)
    print 'Total number of nodule annotations: ' + str(noduleCountTotal)
    return allNodules
    
    
def collect(annotations_filename,seriesuids_filename):
    annotations = csvTools.readCSV(annotations_filename)   # 读取GT标注文件
    seriesUIDs_csv = csvTools.readCSV(seriesuids_filename) # 读取GT文件名列表
    
    seriesUIDs = []
    for seriesUID in seriesUIDs_csv:         # 将CSV文件内容转化成一个list，每个元素为一个CT图像名
        seriesUIDs.append(seriesUID[0])

    allNodules = collectNoduleAnnotations(annotations,seriesUIDs)  # 返回的是所有的nodule,以CT图像名字索引
    
    return (allNodules, seriesUIDs)
    
    
def noduleCADEvaluation(annotations_filename,seriesuids_filename,results_filename,outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''
    
    print annotations_filename   # 输出ground-Truth 标注文件名
    
    # 输入GT的标注文件和文件名列表
    # 输出所有的nodule,以CT图像名字索引，nodule为结构体类型；所有测试集CT图像的名字列表
    (allNodules, seriesUIDs) = collect(annotations_filename, seriesuids_filename)
    
    evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
                'Tianchi',
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)


if __name__ == '__main__':

    annotations_filename          = './annotations/annotations.csv'   # ground-Truth 标注文件
    seriesuids_filename           = './annotations/seriesuids.csv'    # ground-Truth 文件名列表
    results_filename              = './examplesFiles/submission/final_result.csv'  # 提交的标注文件
    outputDir                     = './examplesFiles/evaluation/'     # F-ROC中间结果
    # execute only if run as a script
    noduleCADEvaluation(annotations_filename,seriesuids_filename,results_filename,outputDir)
    print "Finished!"
