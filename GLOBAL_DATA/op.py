#+------------------------------------------------------------------------------------------------------+#
#|WHAT ARE NEEDED:                                                                                      |#
#|0. This File Should Be Put Under `GLOBAL_DATA` Dictionory                                             |#
#|1. The Input Data Should Be Download Seoerately                                                       |#
#|2. A Dictionory For Binary File, e.g. GLOBAL_DATA/BIN_D                                               |#
#|3. A Dictionory For NC File, e.g. GLOBAL_DATA/NC_D                                                    |#
#|4. A Dictionory For NUDGE File, e.g. GLOBAL_DATA/NUDGE                                                |#
#|These Files Are Suggested To Be Put In scracth directory                                              |#
#|NOTE:                                                                                                 |#
#|1. For Unkown Reason, NCAR Changed The ERA5 Directory, Please Check All The ERA5 Related File Paths   |#
#+------------------------------------------------------------------------------------------------------+#

# Parameters Must To Be Determined
relativeLocOfGD = "./"          # the ralative location from this script and `GLOBAL_DATA` directory
lon0 = 260                      # longitude (degrees east) of western boundary. In The Scale Of 360 Degree
lat0 = 40                       # latitude of domain's center
nlon = 768                      # number of grids in x
nlat = 768                      # number of grids in y
dx   = 3000                     # resolution in x
dy   = 3000                     # resolution in y
autoCaculateDomain = True       # auto caculate domain size base on the input lon0, lat0, nlon, nlat and
                                # dx, dy
identifier = '3kmDerechoAug10'    # the case name to be used
startDate = 2020081000          # the simulation starts date, format: YYYYMMDDHH
endDate   = 2020081106          # the simulation ends date, format: YYYYMMDDHH


#+------------------------------------------------------------------------------------------------------+#
#|Optional Parameters:                                                                                  |#
#|If `autoCaculateDomain` Was Set False, Determine The Domain Parameters. These Parameters Should Larger|#
#|Than The Simulation Domain By Around 5 Degrees; If Was Set True, Buffer Can Be Adjusted               |#
#+------------------------------------------------------------------------------------------------------+#
lonMin = 255.
lonMax = 295.
latMin = 25.
latMax = 55.
buffer = 5
makeInvariant = False            # make invariant files
make2DInitial = False            # make 2d initial fields
make3DInitial = False            # make 3d initial fields
makeNudge     = False            # make nudged data
makeOzone     = True            # make Ozone fields
fortranCompiler = "gfortran"    # fortran compiler, gfortran is default one, other compiler can be edited
ncOut = True                    # if output nc file for some data

# This Function Is Not Support Yet
ncar = True                     # If You Using NCAR HPC, Set It True. Else, Set It False
outBinFile = './BIN_D'          # Binary File Output Dictionory, BIN_D Is The Default
outNCFile = './NC_D'            # NC File Output Dictionory, NC_D Is The Default
outNUDGE = './NUDGE'            # NUDGE File Output Dictionory, NC_D Is The Default
region = True                   # If Make Regional Simulation Data, Set It True, Else, Set It False

#+------------------------------------------------------------------------------------------------------+#
#|                          THE REST OF THIS SCRIPT DOES NOT NEED TO BE EDITED                          |#
#+------------------------------------------------------------------------------------------------------+#

import os
import sys
import re
import subprocess
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

if region:
    mode = '_REGION'
else:
    mode = ''

def EditFile(path: str, toEditList: dict, save=True):
    # Test The File Type
    fileType = path.split('.')[-1]
    # fileName = path.replace(f".{fileType}", "")
    if fileType == '.f90' or fileType == '.F90':
        mark = 'Fortran'
    elif fileType == 'ncl':
        mark = 'NCL'
    else:
        mark = None

    # Open And Find The Lines Where The Parameters Are
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    if len(lines) <= 1:
        raise FileNotFoundError(f'The {path} Has Incorrect Lenth')
    
    locations = LocLine(lines, toEditList.keys(), mark)

    # Edit The File
    for key, loc in locations.items():
        toEdit = lines[loc]
        lines[loc] = ReplaceOneLine(toEdit, key, toEditList[key])

    if save:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    else:
        # If Not Save, Enter Debug Mode
        print("".join(lines))

    return f'Sucessful Run file {path}'


def ReplaceOneLine(text: str, var: str, new_value: str) -> str:
    # To Match Different Programing Langurages: Fortran, NCL and Python
    patterns = []
    # Fortran
    fortran_pat = rf"""
(^[ \t]*                                   
 (?:.*?::\s*)?                             
 \b{re.escape(var)}\b\s*=\s*               
)
([^!\n\r,;]+?)                             
(?=[ \t]*(?:!|,|;|$))                      
"""
    # NCL
    ncl_pat = rf"""
(^[ \t]*\b{re.escape(var)}\b\s*(?::=|=)\s*)
([^;\n\r]+?)
(?=([ \t]*(?:;|$)))
"""
    # Python
    py_pat = rf"""
(^[ \t]*\b{re.escape(var)}\b
 (?:\s*:\s*[^=\#\n\r]+)?
 \s*=\s*
)
([^\#\n\r]+?)
(?=([ \t]*(?:\#|$)))
"""
    patterns.append(re.compile(fortran_pat, re.MULTILINE | re.VERBOSE))
    patterns.append(re.compile(ncl_pat, re.MULTILINE | re.VERBOSE))
    patterns.append(re.compile(py_pat, re.MULTILINE | re.VERBOSE))

    def repl(m: re.Match) -> str:
        left = m.group(1)
        old_rhs = m.group(2)

        leading = re.match(r"^\s*", old_rhs).group(0)
        trailing = re.search(r"\s*$", old_rhs).group(0)

        old_core = old_rhs.strip()
        new_core = str(new_value).strip()

        # If The Old Value Is REAL In Fortran, Keep Its Format
        if old_core.endswith(".") and not any(c in new_core for c in ".eEdD"):
            new_core += "."

        # If The Old Value Is String, Keep Its Quotation Mark
        new_core = IfWraped(old_core, new_core)

        returnVar = f"{left}{leading}{new_core}{trailing}"
        return returnVar

    for pat in patterns:
        text, n = pat.subn(repl, text, count=1)
        if n == 1:
            return text

    raise ValueError(f"{var} Not Found")


def IfWraped(old_core: str, new_core: str) -> str:
    if (
        len(old_core) >= 2
        and old_core[0] == old_core[-1]
        and old_core[0] in ("'", '"')
    ):
        quote = old_core[0]
        return f"{quote}{new_core}{quote}"
    return new_core


def LocLine(linedFile: list, prms: list, fileType: str):
    # Use Different Comment Mark for Different Languages 
    if fileType == 'Fortran':
        mark = '!'
    elif fileType == 'NCL':
        mark = ';'
    else:
        mark = '#'
    
    # Locate The Line Number and Content
    reDict = {prm: [] for prm in prms}
    for i in range(len(linedFile)):
        lineContent = linedFile[i]
        if lineContent.lstrip().startswith(mark) or lineContent=='':
            continue
        else:
            pass
        for prm in prms:
            if prm in lineContent:
                reDict[prm].append(i)
                break
    
    # Use The Top One As Defination
    for key, value in reDict.items():
        if len(value) >= 1:
            reDict[key] = value[0]
        else:
            raise ValueError(f'The Parameter {key} Is Missing')
    return reDict


class PrmDict:
    def __init__(self, autoCaculateDomain):
        # The total parameters list
        self.lon0 = self._ToStr(lon0, "lon0")
        self.lat0 = self._ToStr(lat0, "lat0")
        self.nlon = self._ToStr(nlon, "nlon")
        self.nlat = self._ToStr(nlat, "nlat")
        self.dx = self._ToStr(dx, "dx")
        self.dy = self._ToStr(dy, "dy")
        self.startDate = self._ToStr(startDate, "startDate")
        self.endDate = self._ToStr(endDate, "endDate")
        self.ncOut = ncOut

        # Slice Date Before And After Simulation Date For One Month For Ozne Data Appliance
        lastMonth, nextMonth = self.ExpandMonthRange(self.startDate[:6], self.endDate[:6])
        self.year_start = lastMonth[:4]
        self.year_end = nextMonth[:4]
        self.month_start = lastMonth[4:6]
        self.month_end = nextMonth[4:6]

        # The Ozone Data From NCAR Only Supports Till 2022
        if self.year_end >= 2023:
            self.years_add = self.year_end - 2022
        else:
            self.years_add = 0

        if not isinstance(identifier, str):
            raise TypeError("identifier should be str")
        self.identifier = identifier

        # Caculate Domain Base On Lon0 and Lat0
        if autoCaculateDomain:
            self.lonMin, self.lonMax, self.latMin, self.latMax = self.DomainBounds(
                float(lon0), float(lat0), float(nlon), float(nlat), float(dx), float(dy))
        else:
            self.lonMin, self.lonMax, self.latMin, self.latMax = lonMin, lonMax, latMin, latMax
        # Add Buffer and Cut For Better Looking
        self.lonMin, self.lonMax, self.latMin, self.latMax = \
            self.lonMin-buffer, self.lonMax+buffer, self.latMin-buffer, self.latMax+buffer
        self.lonMin, self.lonMax, self.latMin, self.latMax = \
            f"{self.lonMin:.2f}", f"{self.lonMax:.2f}", f"{self.latMin:.2f}", f"{self.latMax:.2f}"

        self.totalDict = {'lon0': self.lon0, 'lat0': self.lat0, 'nlon': self.nlon, 'nlat': self.nlat, 
                            'dx': self.dx, 'dy': self.dy, 'grid': self.identifier, 'date': self.startDate,
                            'date_start': self.startDate, 'date_end': self.endDate, 'lonmin': self.lonMin, 
                            'lonmax': self.lonMax, 'latmin': self.latMin, 'latmax': self.latMax, 
                            'identifier': self.identifier, 'year_add': self.years_add, 'netcdf': self.ncOut, 
                            'year_start': self.year_start, 'year_end': self.year_end, 'netcdf_out': self.ncOut,
                            'month_start': self.month_start, 'month_end': self.month_end}
        
    def _ToStr(self, value, name):
        # Convert Str, Int or Float To Str
        if isinstance(value, (str, int, float)):
            return str(value)
        raise TypeError(f'{name} should be one of [str, int, float], got {type(value).__name__}')

    def DomainBounds(self, lon0: float, lat0: float, nlon: float, nlat: float,
                      dx: float, dy: float, lon_wrap="0_360"):
        """
        Returns (lon_min, lon_max, lat_min, lat_max).
        Assumes a regular lat/lon grid where dy maps to constant dlat,
        and dx maps to dlon computed at the center latitude lat0.
        lon0 is treated as the WESTERN boundary (left edge).
        """

        # degrees per grid step
        R = 6.371e6  # meters, Earth radius (mean)
        dlat = (dy / R) * (180.0 / math.pi)
        dlon = (dx / (R * math.cos(math.radians(lat0)))) * (180.0 / math.pi)

        # latitude bounds from center
        half_span_lat = 0.5 * (nlat - 1) * dlat
        lat_min = lat0 - half_span_lat
        lat_max = lat0 + half_span_lat

        # longitude bounds from western boundary
        lon_min = lon0
        lon_max = lon0 + (nlon - 1) * dlon

        # wrap longitudes if desired
        if lon_wrap == "0_360":
            lon_min = lon_min % 360.0
            lon_max = lon_max % 360.0
        elif lon_wrap == "-180_180":
            lon_min = ((lon_min + 180.0) % 360.0) - 180.0
            lon_max = ((lon_max + 180.0) % 360.0) - 180.0

        return lon_min, lon_max, lat_min, lat_max
    
    def ExpandMonthRange(self, startDate: str, endDate: str):
        """
        startDate, endDate: 'YYYYMM'
        return: (prev_month_of_start, next_month_of_end) as 'YYYYMM'
        """
        start = datetime.strptime(startDate, "%Y%m")
        end = datetime.strptime(endDate, "%Y%m")

        prev_start = start - relativedelta(months=1)
        next_end = end + relativedelta(months=1)

        return prev_start.strftime("%Y%m"), next_end.strftime("%Y%m")

    def GetSubDict(self, toGetList: list):
        reDict = {k: self.totalDict[k] for k in toGetList if k in self.totalDict}
        return reDict


def RunFile(fname, prefix, prms=[], save=True):
    path = f'{relativeLocOfGD}/{fname}{prefix}'

    if len(prms)!=0:
        # If Length Of Parameters List Not Equal To 0, Means Some Parameters Need To be Edited
        
        # Get Parameters Names
        prmDict = PrmDict(autoCaculateDomain)
        prmList = prmDict.GetSubDict(prms)
        print(EditFile(path, prmList, save=save))
    else:
        save = False
        print(f'File {fname} Does Not Need To Be Edited')

    # Run Files Depends On Its Type
    if prefix == '.f90' or prefix == '.F90':
        print(f'Star Compiling {fname}{prefix}')
        subprocess.run([fortranCompiler, path, "-o", fname], check=True)
        print(f'Star Excecuting {fname}{prefix}')
        subprocess.run([f"./{fname}"], check=True)
    elif prefix == '.ncl':
        print(f'Star Excecuting {fname}{prefix}')
        subprocess.run([f"module load ncl && ncl ./{fname}{prefix}"],
                        check=True, shell=True, executable="/bin/bash")
    elif prefix == '.py':
        print(f'Star Excecuting {fname}{prefix}')
        subprocess.run([f"python ./{fname}{prefix}"], check=True)
    else:
        raise ValueError(f'Invalid File Type {prefix}')
    return f'{fname} Was Run Sucessfully'


def IfExists(fileList: list):
    # Test If All The Files To Generate Exist
    for f in fileList:
        if not os.path.exists(f):
            raise FileNotFoundError(f'Failed To Generate File {f}')
    return 0

def Main():
    workFlow = '''
    Work Flow: Invariant Data:
                make_2D_terr_REGION.f90 -> make_2D_landtype_REGION.f90 -> bin2nc_terr.ncl && bin2nc_landtype.ncl
                -> rename: out_terr.nc && out_landtype.nc && out_latlon.txt -> make_2D_invar.ncl
               Initial Fields:
                make_2D_init_soil_era5_ncar.ncl -> make_2D_init_snow_era5_ncar.ncl 
                -> make_2D_evol_sst_ci_era5_ncar.ncl -> make_3D_init_REGION_ncar.ncl
               Nudge Data && Ozone Data:
                make_3D_nudge_REGION_ncar.ncl -> make_3D_ozone_REGION_ncar.ncl
    '''
    print(workFlow)
    prms = PrmDict(autoCaculateDomain)
    if makeInvariant:
        status = 0          # To Summarize If All The File Generated Sucessfully
        print('Start Making Invariant Data')
        # Edit And Run The `make_2D_terr_REGION` File
        print(RunFile(f'make_2D_terr{mode}', '.f90', prms=['lat0', 'nlon', 'nlat', 'dx', 'dy'], save=True))
        
        # Run The `make_2D_landtype_REGION.f90` File
        print(RunFile(f'make_2D_landtype{mode}', '.f90', save=False))
        fList = [f'{relativeLocOfGD}/{outBinFile}/out_terr.bin',
                 f'{relativeLocOfGD}/{outBinFile}/out_latlon.bin',
                 f'{relativeLocOfGD}/{outBinFile}/out_landtype.bin']
        IfExists(fList)
        
        # Run The `bin2nc_terr.ncl` File and `bin2nc_landtype.ncl` File
        print(RunFile('bin2nc_terr', '.ncl', save=False))
        print(RunFile('bin2nc_landtype', '.ncl', save=False))
        fList = [f'{relativeLocOfGD}/{outNCFile}/out_terr.bin',
            f'{relativeLocOfGD}/{outNCFile}/out_latlon.bin']
        IfExists(fList)
        
        # Rename File `out_terr.nc`, `out_landtype.nc` and `out_latlon.txt`
        os.rename(f'{relativeLocOfGD}/{outNCFile}/out_landtype.nc', 
                  f'{relativeLocOfGD}/{outNCFile}/landtype_{identifier}.nc')
        os.rename(f'{relativeLocOfGD}/{outNCFile}/out_terr.nc',
                  f'{relativeLocOfGD}/{outNCFile}/terrain_{identifier}.nc')
        os.rename(f'{relativeLocOfGD}/{outBinFile}/out_latlon.txt',
                  f'{relativeLocOfGD}/{outBinFile}/latlon_{identifier}.txt')
        
        # Edit And Run The `make_2D_invar.ncl` File
        print(RunFile('make_2D_invar', '.ncl', prms=['grid', 'netcdf_out'], save=True))
        fList = [f'{relativeLocOfGD}/{outBinFile}/lai_{identifier}.bin',
                 f'{relativeLocOfGD}/{outBinFile}/landtype_{identifier}.bin',
                 f'{relativeLocOfGD}/{outBinFile}/soil_{identifier}.bin',
                 f'{relativeLocOfGD}/{outBinFile}/landmask_{identifier}.bin']
        IfExists(fList)
        print('Successfully Making Invariant Data')
    else:
        print('Skip Making Invariant Data')
    
    if make2DInitial:
        print('Start Making Initial Field Data')
        # Edit And Run `make_2D_init_soil_era5_ncar.ncl` File
        print(RunFile('make_2D_init_soil_era5_ncar', '.ncl',
                       prms=['grid', 'date', 'netcdf_out'], save=True))
        fList = [f'{relativeLocOfGD}/{outBinFile}/soil_init_{startDate}_{identifier}.bin']
        IfExists(fList)
        
        # Edit And Run `make_2D_init_snow_era5_ncar.ncl` File
        print(RunFile('make_2D_init_snow_era5_ncar', '.ncl',
                       prms=['grid', 'date', 'netcdf_out'], save=True))
        fList = [f'{relativeLocOfGD}/{outBinFile}/snow_{startDate}_{identifier}.bin',
                 f'{relativeLocOfGD}/{outBinFile}/snowt_{startDate}_{identifier}.bin']
        IfExists(fList)

        # Edit And Run `make_2D_evol_sst_ci_era5_ncar.ncl` File
        print(RunFile('make_2D_evol_sst_ci_era5_ncar', '.ncl',
                       prms=['grid', 'date_start', 'date_end', 'netcdf_out'], save=True))
        fList = [f'{relativeLocOfGD}/{outBinFile}/icemask_{startDate}-{endDate}_{identifier}.bin',
                 f'{relativeLocOfGD}/{outBinFile}/sst_{startDate}-{endDate}_{identifier}.bin']
        IfExists(fList)
        print('Sucessfully Making 2D Initial Field Data')
    else:
        print('Skip Making 2D Initial Field Data')

    if make3DInitial:
        print('Start Making 3D Initial Field Data')
        # Edit And Run `make_3D_init_REGION_ncar.ncl` File
        print(RunFile(f'make_3D_init{mode}_ncar', '.ncl', 
                    prms=['date', 'lonmin', 'lonmax', 'latmin', 'latmax',
                           'identifier', 'netcdf_out'], save=True))
        fList = [f'{relativeLocOfGD}/{outBinFile}/init_era5_{startDate}_{identifier}.bin']
        IfExists(fList)
        print('Sucessfully Making 3D Initial Field Data')
    else:
        print('Skip Making 3D Initial Field Data')
        
    if makeNudge:
        print('Start Making Nudge Data')
        # Edit And Run `make_3D_nudge_REGION_ncar.ncl` File
        print(RunFile(f'make_3D_nudge{mode}_ncar', '.ncl', 
                    prms=['date_start', 'date_end', 'lonmin', 'lonmax', 'latmin', 'latmax', 
                          'identifier', 'netcdf'], save=True))
        fList = [f'{relativeLocOfGD}/{outNUDGE}/nudge_era5_{startDate}-{endDate}_{identifier}_FILELIST.ascii']
        IfExists(fList)
        print('Sucessfully Making Nudge Data')
    else:
        print('Skip Making Nudge Data')

    if makeOzone:
        print('Start Making Ozone Data')
        # Edit And Run `make_3D_ozone_REGION_ncar.ncl` File
        print(RunFile(f'make_3D_ozone{mode}_ncar', '.ncl', 'year_add',
                prms=['year_start', 'month_start', 'year_end', 'month_end', 'netcdf_out',
                    'lonmin', 'lonmax', 'latmin', 'latmax', 'identifier'], save=True))
        # Caculate Ozone File Start And End Date
        addYear = prms.years_add
        startMonth = str(prms.year_start+addYear)+str(prms.month_start)
        endMonth = str(prms.year_end+addYear)+str(prms.month_end)
        fList = [f'{relativeLocOfGD}/{outBinFile}/ozone_era5_monthly_{startMonth}-{endMonth}_{identifier}.bin']
        IfExists(fList)
        print('Sucessfully Making Ozone Data')
    else:
        print('Skip Making Ozone Data')
    print('Simulation Data Are All Set!')


Main()