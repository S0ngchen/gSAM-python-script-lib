#!/usr/bin/env python3
# Parameters Must To Be Determined
relativeFileLoc = "../../"              # The Model Out Data files Rekative Directory
identifier = 'DERECHO_3kmDerechoAug10'  # Consist By `Case Name`_`Identifier of Datas`
nBegining = 360                         # Beginning Identifier Index
nEnding = 10800                         # Ending Identifier Index
mode = '2D'                             # Mode Can Be Turned From `'2D'` to `'3D'`

#+------------------------------------------------------------------------------------------------------+#
#|Optional Parameters:                                                                                  |#
#|1. parallelChannelNumber: Number Of Parallel Channels                                                 |#
#|2. SLEEP_SECONDS: Sleep Seconds Between Each Batch                                                    |#
#|3. filesPerRow: How Many Files Shown In One Row For Better Visualized Print                           |#
#+------------------------------------------------------------------------------------------------------+#
parallelChannelNumber = 1               # number of parallel channels
SLEEP_SECONDS = 1
filesPerRow = 2                         # how many files to show on one row, only for butifying visualized procession
OriginalScript = False                  # if use the orignal converting script. False means use rewrited script

#+------------------------------------------------------------------------------------------------------+#
#|                          THE REST OF THIS SCRIPT DOES NOT NEED TO BE EDITED                          |#
#+------------------------------------------------------------------------------------------------------+#
import glob
import subprocess
import time
import os
import re
import sys

if mode == '2D':
    UTILITY = "../../UTIL/2D2nc"
    PYTHON_UTILITY = "./2D2nc_python.py"
    fileLoc = f"{relativeFileLoc}/OUT_2D/"
elif mode == '3D':
    UTILITY = "../../UTIL/3D2nc"
    PYTHON_UTILITY = None
    fileLoc = f"{relativeFileLoc}/OUT_3D/"
else:
    raise ValueError("mode must be '2D' or '3D'")

def FillingZero(inStr):
    # Convert Input Variable To STR
    if isinstance(inStr, str):
        pass
    elif isinstance(inStr, int):
        inStr = str(inStr)
    toFill = 10 - len(inStr) # The number of 0s to be filled 
    if toFill <= 0:
        return inStr
    return '0'*toFill+inStr


def GetConvertList(path):
    # Get All The Files Need To Be Convert
    fileList = glob.glob(f"{path}/{identifier}*.{mode}*")
    # Get The File Identify Number
    fileIndex = []
    if mode == '2D':
        for fullName in fileList:
            baseName = os.path.basename(fullName)
            try:
                idx = int((baseName.split('.')[0]).split('_')[-1])
            except Exception as e:
                # Skip Bad Name
                continue
            fileIndex.append(idx)
        # Convert To DICT For Better Formating
        fileDict = dict(zip(fileIndex, fileList))
        # Sort By Identifier
        fileDict = dict(sorted(fileDict.items()))
        fileList = {value: value for key, value in fileDict.items() if nBegining <= key <= nEnding}
        fileList = list(fileList)
    elif mode == '3D':
        for fullName in fileList:
            baseName = os.path.basename(fullName)
            try:
                idx = int((baseName.split('.')[0]).split('_')[-1])
                apendence = int((baseName.split('_')[-1]).split('.')[0])
            except Exception as e:
                # Skip Bad Name
                continue
            fileIndex.append(f"{str(idx)}_{str(apendence)}")
        # Convert To DICT For Better Formating
        fileDict = dict(zip(fileIndex, fileList))
        # Sort By Identifier 
        # for key, value in fileDict.items():
        fileDict = dict(sorted(fileDict.items(), key=lambda kv: natural_key(kv[0])))

        # TO DO: solve the unmatch problem
        fileList = {value: value for key, value in fileDict.items()\
                     if nBegining <= int(key.split('_')[1]) <= nEnding}
        fileList = list(fileList)
    return fileList

def natural_key(k: str):
    # "360_15" -> (360, 15)
    return tuple(int(part) for part in k.split("_"))




def PrintFormatedList(inList: list):
    for i in range(0, len(inList), filesPerRow):
        print(*inList[i:i+filesPerRow], sep="\t")


def _WriteLogHeader(out_f, logId, taskFile):
    # Write A Clear Separator For Each Task In Same Channel Log
    if OriginalScript:
        cmdStr = f"{UTILITY} {taskFile}"
    else:
        cmdStr = f"{PYTHON_UTILITY} {taskFile}"
    out_f.write("\n")
    out_f.write("=============================================\n")
    out_f.write(f"[CHANNEL {logId}] START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out_f.write(f"[CHANNEL {logId}] FILE : {taskFile}\n")
    out_f.write(f"[CHANNEL {logId}] CMD  : {cmdStr}\n")
    out_f.write("=============================================\n")
    out_f.flush()


def _WriteLogFooter(out_f, logId, taskFile, rc):
    out_f.write("=============================================\n")
    out_f.write(f"[CHANNEL {logId}] END  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out_f.write(f"[CHANNEL {logId}] FILE : {taskFile}\n")
    out_f.write(f"[CHANNEL {logId}] RC   : {rc}\n")
    out_f.write("=============================================\n")
    out_f.write("\n")
    out_f.flush()


def RunParallel(tasks):
    # Run Tasks Parallelly (One Log For One Channel In This Batch)
    procs = []
    for i, f in enumerate(tasks, start=1):
        # Creat A New Log
        out_f = open(f"out{mode}{i}.log", "a")
        # Add Header To Help Debug
        _WriteLogHeader(out_f, i, f)
        # Run Parallelly
        print(f'File {f} Started Converting')
        
        if OriginalScript:
            # Use the initial script to convert NC files
            p = subprocess.Popen([UTILITY, f], stdout=out_f, stderr=subprocess.STDOUT)
        else:
            # Use new script to convert NC files
            if mode != '2D':
                out_f.write("[ERROR] Python converter is only implemented for 2D mode.\n")
                _WriteLogFooter(out_f, i, f, 1)
                out_f.close()
                print(f"[ERROR] Python converter only supports 2D mode: {f}")
                continue
            p = subprocess.Popen([sys.executable, PYTHON_UTILITY, f], stdout=out_f, stderr=subprocess.STDOUT) # type: ignore

        procs.append((p, out_f, f, i))
        print(f'File {f} Finished Converting')

    # Wait All Tasks And Check Return Code
    for p, out_f, f, logId in procs:
        rc = p.wait()
        # Add Footer To Help Debug
        _WriteLogFooter(out_f, logId, f, rc)
        out_f.close()
        if rc != 0:
            print(f"[ERROR] Convert Failed (rc={rc}): {f}")



def InitializeLog(channelNumber):
    for i in range(1, channelNumber+1):
        if os.path.exists(f"out{mode}{i}.log"):
            os.remove(f"out{mode}{i}.log")
    return 

def ParallelConvert(convertList: list, channelNumber=parallelChannelNumber):
    # Determine How Many Batch To Use (Avoid Empty Batch)
    lenth = len(convertList)
    if lenth == 0:
        return

    batchs = int((lenth + channelNumber - 1) / channelNumber)

    # Delate All The Old Task Logs
    try:
        InitializeLog(channelNumber)
    except Exception as e:
        print(f"[WARN] InitializeLog Failed: {e}")

    for index in range(batchs):
        tasks = convertList[index*channelNumber:(index+1)*channelNumber]
        if len(tasks) == 0:
            continue
        RunParallel(tasks)
        time.sleep(SLEEP_SECONDS)

    return


def NCOutCheck():
    netCDFPac = True
    scipyPac = True
    try:
        import netCDF4
    except ImportError:
        netCDFPac = False
    try:
        import scipy
    except ImportError:
        scipyPac = False
    if not (scipyPac and netCDFPac):
        raise ImportError('At Least To Install One Of the Packages: {"scipy", "netCDF4"}')


def New2DConvert():
    pass

def Main():
    NCOutCheck()
    toConvertList = GetConvertList(fileLoc)
    print('Going To Convert The Following Files: ')
    PrintFormatedList(toConvertList)
    ParallelConvert(toConvertList)


if __name__ == "__main__":
    Main()
