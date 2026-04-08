#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D2nc_python.py
================

把 Fortran 程序 2D2nc.f 改写成 Python 版本。

设计目标
--------
1. 尽量保持与原 Fortran 程序一致的输入、输出和处理逻辑。
2. 代码尽量写得直白，便于单步调试。
3. 不使用装饰器、dataclass、property 这类会增加“跳转层”的语法。
4. numpy 是可选依赖；没有 numpy 时，尽量退回纯 Python 路径。

支持的命令行
------------
    python 2D2nc_python.py input.2D
    python 2D2nc_python.py input.2D latlon

行为说明
--------
- 读取 Fortran unformatted sequential 的 .2D 文件。
- 输出文件名为: <input_stem>_<comp>.nc
- 支持两种字段编码：
    * isbin = True  -> 直接存储的 REAL*4
    * isbin = False -> INT*2 打包数据 + (fmax, fmin) 反量化
- 按 nsubsx / nsubsy 把子域拼接回全局二维网格。
- 仅当 time 严格递增时才写入 NetCDF，这与原程序一致。

依赖说明
--------
- numpy 为可选依赖。
- NetCDF 写出后端需要 netCDF4 或 scipy。
- 如果系统没有任何 NetCDF 后端，脚本会报错退出。

可选环境变量
------------
TWO_D2NC_DISABLE_NUMPY=1
    即使系统里安装了 numpy，也强制使用纯 Python 路径。
"""

import argparse
import math
import os
import struct


# -----------------------------------------------------------------------------
# 可选依赖：numpy
# -----------------------------------------------------------------------------
USE_NUMPY = True
# _disable_numpy = os.environ.get("TWO_D2NC_DISABLE_NUMPY", "").strip().lower()
# if _disable_numpy in ("1", "true", "yes", "y"):
#     USE_NUMPY = False

np = None
if USE_NUMPY:
    try:
        import numpy as _numpy
        np = _numpy
    except Exception:
        np = None
        USE_NUMPY = False


# -----------------------------------------------------------------------------
# 可选依赖：NetCDF 后端
# -----------------------------------------------------------------------------
NetCDF4Dataset = None
try:
    from netCDF4 import Dataset as _NetCDF4Dataset
    NetCDF4Dataset = _NetCDF4Dataset
except Exception:
    NetCDF4Dataset = None

ScipyNetCDFFile = None
try:
    from scipy.io import netcdf_file as _ScipyNetCDFFile
    ScipyNetCDFFile = _ScipyNetCDFFile
except Exception:
    ScipyNetCDFFile = None


class FortranRecordError(RuntimeError):
    pass


class Field(object):
    """
    保存单个物理变量的信息。

    name       变量名，来自 Fortran 中的 NAME
    long_name  变量长描述
    units      单位
    isbin      是否为直接 REAL*4 存储
    data       变量二维场，形状为 [ny_gl][nx_gl] 或 numpy 2D array
    """

    def __init__(self, name, long_name, units, isbin, data):
        self.name = name
        self.long_name = long_name
        self.units = units
        self.isbin = isbin
        self.data = data


class Timestep(object):
    """
    保存单个时间步全部信息。
    """

    def __init__(self):
        self.nstep = 0
        self.dolatlon = False
        self.timesec = 0.0
        self.datechar = ""
        self.time_utsec = 0.0
        self.comp = ""

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.nsubs = 0
        self.nsubsx = 0
        self.nsubsy = 0
        self.nfields = 0

        self.dx = 0.0
        self.dy = 0.0

        self.lat = None
        self.lon = None
        self.latv = None
        self.lonu = None
        self.wgt = None
        self.y = None
        self.yv = None

        self.time = 0.0
        self.fields = []

        self.nx_gl = 0
        self.ny_gl = 0


class FortranSequentialReader(object):
    """
    读取 Fortran unformatted sequential 文件。

    Fortran 这类文件的每条记录通常长这样：
        [记录长度marker][记录数据][记录长度marker]

    这里自动探测：
    - marker 是 4 字节还是 8 字节
    - little endian 还是 big endian
    """

    def __init__(self, path):
        self.path = str(path)
        self.fp = open(self.path, "rb")
        self.filesize = os.path.getsize(self.path)
        detected = self.detect_format()
        self.marker_size = detected[0]
        self.endian = detected[1]

    def close(self):
        self.fp.close()

    def rewind(self):
        self.fp.seek(0)

    def tell(self):
        return self.fp.tell()

    def seek(self, pos):
        self.fp.seek(pos)

    def unpack_marker(self, raw, endian, marker_size):
        if marker_size == 4:
            return struct.unpack(endian + "i", raw)[0]
        if marker_size == 8:
            return struct.unpack(endian + "q", raw)[0]
        raise ValueError("unsupported marker size: %r" % (marker_size,))

    def detect_format(self):
        pos0 = self.fp.tell()
        self.fp.seek(0)
        first8 = self.fp.read(8)
        self.fp.seek(0)

        if len(first8) < 4:
            raise FortranRecordError("file too small to contain a Fortran record")

        candidates = []

        for marker_size in (4, 8):
            if len(first8) < marker_size:
                continue

            raw = first8[:marker_size]
            for endian in ("<", ">"): 
                try:
                    n = self.unpack_marker(raw, endian, marker_size)
                except Exception:
                    continue

                if n <= 0:
                    continue
                if n > self.filesize:
                    continue

                trailer_pos = marker_size + n
                if trailer_pos + marker_size > self.filesize:
                    continue

                self.fp.seek(trailer_pos)
                trailer = self.fp.read(marker_size)
                self.fp.seek(0)
                if len(trailer) != marker_size:
                    continue

                try:
                    n2 = self.unpack_marker(trailer, endian, marker_size)
                except Exception:
                    continue

                if n == n2:
                    candidates.append((marker_size, endian))

        self.fp.seek(pos0)

        if not candidates:
            raise FortranRecordError("could not detect Fortran record format")

        # 优先使用最常见的 Linux 小端 4-byte marker。
        if (4, "<") in candidates:
            return (4, "<")
        return candidates[0]

    def read_record(self):
        """
        读一条完整的 Fortran 记录。

        返回：
            bytes 数据体
        若到文件尾且没有更多记录：
            返回 None
        """
        header = self.fp.read(self.marker_size)
        if not header:
            return None

        if len(header) != self.marker_size:
            raise FortranRecordError("truncated record marker at EOF")

        n = self.unpack_marker(header, self.endian, self.marker_size)
        if n < 0:
            raise FortranRecordError("negative record length: %r" % (n,))

        data = self.fp.read(n)
        if len(data) != n:
            raise FortranRecordError(
                "truncated record body: expected %d bytes, got %d" % (n, len(data))
            )

        trailer = self.fp.read(self.marker_size)
        if len(trailer) != self.marker_size:
            raise FortranRecordError("missing trailing record marker")

        n2 = self.unpack_marker(trailer, self.endian, self.marker_size)
        if n != n2:
            raise FortranRecordError(
                "record marker mismatch: start=%d end=%d" % (n, n2)
            )

        return data


# -----------------------------------------------------------------------------
# 一些尽量直白的辅助函数
# -----------------------------------------------------------------------------
def decode_char(raw):
    return raw.decode("latin1", errors="replace").rstrip(" \x00")


def expect_len(raw, expected, what):
    if len(raw) != expected:
        raise FortranRecordError(
            "unexpected record length for %s: got %d, expected %d"
            % (what, len(raw), expected)
        )


def unpack_values(endian, fmt, raw):
    return struct.unpack(endian + fmt, raw)


def unpack_numeric_array(raw, endian, kind, count):
    """
    从 bytes 中解析一段定长数值数组。

    kind:
        f4 -> float32
        f8 -> float64
        i2 -> int16
    """
    if USE_NUMPY:
        if kind == "f4":
            dtype = np.dtype(endian + "f4")
        elif kind == "f8":
            dtype = np.dtype(endian + "f8")
        elif kind == "i2":
            dtype = np.dtype(endian + "i2")
        else:
            raise ValueError("unsupported array kind: %r" % (kind,))
        return np.frombuffer(raw, dtype=dtype, count=count).copy()

    fmt_map = {
        "f4": "f",
        "f8": "d",
        "i2": "h",
    }
    if kind not in fmt_map:
        raise ValueError("unsupported array kind: %r" % (kind,))
    fmt = endian + str(count) + fmt_map[kind]
    return list(struct.unpack(fmt, raw))


def make_zeros_1d(length, value):
    if USE_NUMPY:
        return np.full(length, value, dtype=np.float64)
    out = []
    i = 0
    while i < length:
        out.append(float(value))
        i += 1
    return out


def make_x_coords(nx_gl, dx):
    if USE_NUMPY:
        return (np.arange(nx_gl, dtype=np.float32) * np.float32(dx)).astype(np.float32)
    out = []
    i = 0
    while i < nx_gl:
        out.append(float(i) * float(dx))
        i += 1
    return out


def to_float32_array(values):
    if USE_NUMPY:
        return np.asarray(values, dtype=np.float32)
    out = []
    for v in values:
        out.append(float(v))
    return out


def to_float64_array(values):
    if USE_NUMPY:
        return np.asarray(values, dtype=np.float64)
    out = []
    for v in values:
        out.append(float(v))
    return out


def values_min(values):
    if USE_NUMPY:
        return float(np.min(values))
    return float(min(values))


def values_max(values):
    if USE_NUMPY:
        return float(np.max(values))
    return float(max(values))


def values_sum(values):
    if USE_NUMPY:
        return float(np.sum(values))
    total = 0.0
    for v in values:
        total += float(v)
    return total


def unpack_packed_field(packed, fmin, fmax):
    """
    反量化公式，直接照搬原 Fortran：

        fld = fmin + (byte + 32000) * (fmax - fmin) / 64000.
    """
    if USE_NUMPY:
        out = (
            np.float32(fmin)
            + (np.asarray(packed, dtype=np.float32) + np.float32(32000.0))
            * (np.float32(fmax) - np.float32(fmin))
            / np.float32(64000.0)
        )
        return out.astype(np.float32, copy=False)

    out = []
    scale = float(fmax - fmin) / 64000.0
    for v in packed:
        out.append(float(fmin + (float(v) + 32000.0) * scale))
    return out


def reshape_block(raw, ny, nx):
    """
    把一维块重排成二维块，形状为 [ny][nx]。
    """
    if USE_NUMPY:
        return np.asarray(raw, dtype=np.float32).reshape((ny, nx), order="C")

    out = []
    m = 0
    j = 0
    while j < ny:
        row = []
        i = 0
        while i < nx:
            row.append(float(raw[m]))
            m += 1
            i += 1
        out.append(row)
        j += 1
    return out


def unpack_subdomains(raw, nx, ny, nsubsx, nsubsy):
    """
    把 Fortran 中按子域顺序存放的一维数据恢复成整张二维场。

    原 Fortran 逻辑等价于：
        先按子域块遍历
        每个子域块内部再按 j,i 顺序写入 fld(it+i, jt+j)

    Python 这里返回 [ny_gl][nx_gl]，更适合 NetCDF 写出。
    """
    nx_gl = nx * nsubsx
    ny_gl = ny * nsubsy

    if USE_NUMPY:
        out = np.empty((ny_gl, nx_gl), dtype=np.float32)
    else:
        out = []
        j = 0
        while j < ny_gl:
            row = []
            i = 0
            while i < nx_gl:
                row.append(0.0)
                i += 1
            out.append(row)
            j += 1

    m = 0
    jt = 0
    while jt < nsubsy * ny:
        it = 0
        while it < nsubsx * nx:
            block = reshape_block(raw[m:m + nx * ny], ny, nx)
            j = 0
            while j < ny:
                i = 0
                while i < nx:
                    out[jt + j][it + i] = block[j][i]
                    i += 1
                j += 1
            m += nx * ny
            it += nx
        jt += ny

    total_points = nx_gl * ny_gl
    if m != total_points:
        raise ValueError(
            "layout reconstruction consumed %d points, expected %d" % (m, total_points)
        )

    return out


def compute_weights(ts):
    """
    计算纬向权重。

    若 dolatlon=True：
        wgt(j) = (latv(j+1)-latv(j)) * cos(lat(j)*pi/180)
        然后按原程序方式归一化，使平均权重大约为 1

    若 dolatlon=False：
        权重全为 1
    """
    if ts.dolatlon:
        if ts.wgt is not None:
            if USE_NUMPY:
                x = np.zeros(ts.nx_gl, dtype=np.float32)
                return x, np.asarray(ts.wgt, dtype=np.float64)
            x = make_zeros_1d(ts.nx_gl, 0.0)
            return x, ts.wgt

        if USE_NUMPY:
            pi = np.arccos(-1.0)
            wgt = (ts.latv[1:] - ts.latv[:-1]) * np.cos(pi / 180.0 * ts.lat)
            mean = float(np.sum(wgt))
            if mean != 0.0:
                wgt = wgt / mean * ts.ny_gl
            else:
                wgt = np.ones(ts.ny_gl, dtype=np.float64)
            x = np.zeros(ts.nx_gl, dtype=np.float32)
            return x, wgt

        wgt = []
        j = 0
        while j < ts.ny_gl:
            v = (float(ts.latv[j + 1]) - float(ts.latv[j]))
            v = v * math.cos(math.pi / 180.0 * float(ts.lat[j]))
            wgt.append(v)
            j += 1
        s = values_sum(wgt)
        if s != 0.0:
            j = 0
            while j < len(wgt):
                wgt[j] = wgt[j] / s * ts.ny_gl
                j += 1
        else:
            wgt = make_zeros_1d(ts.ny_gl, 1.0)
        x = make_zeros_1d(ts.nx_gl, 0.0)
        return x, wgt

    x = make_x_coords(ts.nx_gl, ts.dx)
    wgt = make_zeros_1d(ts.ny_gl, 1.0)
    return x, wgt


def field_min(field2d):
    if USE_NUMPY:
        return float(np.min(field2d))
    vmin = None
    for row in field2d:
        for v in row:
            if vmin is None or float(v) < vmin:
                vmin = float(v)
    return float(vmin)


def field_max(field2d):
    if USE_NUMPY:
        return float(np.max(field2d))
    vmax = None
    for row in field2d:
        for v in row:
            if vmax is None or float(v) > vmax:
                vmax = float(v)
    return float(vmax)


def weighted_field_mean(field2d, wgt, nx_gl, ny_gl):
    if USE_NUMPY:
        return float(np.sum(field2d * np.asarray(wgt)[:, None]) / (nx_gl * ny_gl))

    total = 0.0
    j = 0
    while j < ny_gl:
        row_sum = 0.0
        i = 0
        while i < nx_gl:
            row_sum += float(field2d[j][i])
            i += 1
        total += row_sum * float(wgt[j])
        j += 1
    return total / float(nx_gl * ny_gl)


class Parser(object):
    """
    从 FortranSequentialReader 中逐时间步解析 .2D 文件。
    """

    def __init__(self, reader):
        self.reader = reader
        self.endian = reader.endian

    def parse_y_record(self, rec, ts):
        old_len = 4 + 8 * ts.ny_gl + 8 * (ts.ny_gl + 1) + 4 * ts.ny_gl + 4 * (ts.ny_gl + 1)
        new_len = old_len + 8 * ts.ny_gl

        if len(rec) == old_len:
            has_wgt = False
        elif len(rec) == new_len:
            has_wgt = True
        else:
            raise FortranRecordError(
                "unexpected record length for dx/lat/latv/y/yv[/wgt]: got %d, expected %d or %d"
                % (len(rec), old_len, new_len)
            )

        offset = 0
        ts.dx = struct.unpack_from(self.endian + "f", rec, offset)[0]
        offset += 4
        ts.lat = unpack_numeric_array(rec[offset:offset + 8 * ts.ny_gl], self.endian, "f8", ts.ny_gl)
        offset += 8 * ts.ny_gl
        ts.latv = unpack_numeric_array(rec[offset:offset + 8 * (ts.ny_gl + 1)], self.endian, "f8", ts.ny_gl + 1)
        offset += 8 * (ts.ny_gl + 1)
        ts.y = unpack_numeric_array(rec[offset:offset + 4 * ts.ny_gl], self.endian, "f4", ts.ny_gl)
        offset += 4 * ts.ny_gl
        ts.yv = unpack_numeric_array(rec[offset:offset + 4 * (ts.ny_gl + 1)], self.endian, "f4", ts.ny_gl + 1)
        offset += 4 * (ts.ny_gl + 1)

        if has_wgt:
            ts.wgt = unpack_numeric_array(rec[offset:offset + 8 * ts.ny_gl], self.endian, "f8", ts.ny_gl)
        else:
            ts.wgt = None

    def parse_x_record(self, rec, ts):
        old_len = 4 + 8 * ts.nx_gl + 8 * ts.nx_gl
        new_len = 4 + 8 * ts.nx_gl + 8 * (ts.nx_gl + 1)

        if len(rec) == old_len:
            lonu_len = ts.nx_gl
        elif len(rec) == new_len:
            lonu_len = ts.nx_gl + 1
        else:
            raise FortranRecordError(
                "unexpected record length for dy/lon/lonu: got %d, expected %d or %d"
                % (len(rec), old_len, new_len)
            )

        offset = 0
        ts.dy = struct.unpack_from(self.endian + "f", rec, offset)[0]
        offset += 4
        ts.lon = unpack_numeric_array(rec[offset:offset + 8 * ts.nx_gl], self.endian, "f8", ts.nx_gl)
        offset += 8 * ts.nx_gl
        ts.lonu = unpack_numeric_array(rec[offset:offset + 8 * lonu_len], self.endian, "f8", lonu_len)

    def read_timestep(self, force_latlon=False):
        rec = self.reader.read_record()
        if rec is None:
            raise EOFError

        if len(rec) < 38:
            raise FortranRecordError(
                "header record too short: got %d bytes, expected at least 38" % len(rec)
            )

        ts = Timestep()

        offset = 0
        ts.nstep = struct.unpack_from(self.endian + "i", rec, offset)[0]
        offset += 4
        dolatlon_i = struct.unpack_from(self.endian + "i", rec, offset)[0]
        offset += 4
        ts.timesec = struct.unpack_from(self.endian + "d", rec, offset)[0]
        offset += 8
        ts.datechar = decode_char(rec[offset:offset + 14])
        offset += 14
        ts.time_utsec = struct.unpack_from(self.endian + "d", rec, offset)[0]
        offset += 8
        ts.dolatlon = (dolatlon_i != 0)
        if force_latlon:
            ts.dolatlon = True

        rec = self.reader.read_record()
        if rec is None:
            raise FortranRecordError("unexpected EOF while reading comp")
        ts.comp = decode_char(rec)

        rec = self.reader.read_record()
        if rec is None:
            raise FortranRecordError("unexpected EOF while reading grid integers")
        expect_len(rec, 7 * 4, "nx..nfields")
        vals = unpack_values(self.endian, "7i", rec)
        ts.nx = vals[0]
        ts.ny = vals[1]
        ts.nz = vals[2]
        ts.nsubs = vals[3]
        ts.nsubsx = vals[4]
        ts.nsubsy = vals[5]
        ts.nfields = vals[6]
        ts.nx_gl = ts.nx * ts.nsubsx
        ts.ny_gl = ts.ny * ts.nsubsy

        rec = self.reader.read_record()
        if rec is None:
            raise FortranRecordError("unexpected EOF while reading dx/lat/latv/y/yv")
        self.parse_y_record(rec, ts)

        rec = self.reader.read_record()
        if rec is None:
            raise FortranRecordError("unexpected EOF while reading dy/lon/lonu")
        self.parse_x_record(rec, ts)

        rec = self.reader.read_record()
        if rec is None:
            raise FortranRecordError("unexpected EOF while reading time")
        expect_len(rec, 8, "time")
        ts.time = unpack_values(self.endian, "d", rec)[0]

        ts.fields = []
        field_index = 0
        while field_index < ts.nfields:
            rec = self.reader.read_record()
            if rec is None:
                raise FortranRecordError("unexpected EOF while reading field metadata")
            if len(rec) < 100:
                raise FortranRecordError(
                    "field metadata record too short: got %d bytes, expected at least 100"
                    % len(rec)
                )

            name = decode_char(rec[0:8])
            long_name = decode_char(rec[9:89])
            units = decode_char(rec[90:100])

            rec = self.reader.read_record()
            if rec is None:
                raise FortranRecordError("unexpected EOF while reading isbin")
            expect_len(rec, 4, "isbin")
            isbin_i = unpack_values(self.endian, "i", rec)[0]
            isbin = (isbin_i != 0)

            total_points = ts.nx_gl * ts.ny_gl

            if isbin:
                rec = self.reader.read_record()
                if rec is None:
                    raise FortranRecordError("unexpected EOF while reading REAL*4 field")
                expect_len(rec, total_points * 4, "REAL*4 field")
                raw = unpack_numeric_array(rec, self.endian, "f4", total_points)
                raw = to_float32_array(raw)
            else:
                rec = self.reader.read_record()
                if rec is None:
                    raise FortranRecordError("unexpected EOF while reading fmax/fmin")
                expect_len(rec, 8, "fmax/fmin")
                fmax, fmin = unpack_values(self.endian, "2f", rec)

                rec = self.reader.read_record()
                if rec is None:
                    raise FortranRecordError("unexpected EOF while reading INT*2 packed field")
                expect_len(rec, total_points * 2, "INT*2 packed field")
                packed = unpack_numeric_array(rec, self.endian, "i2", total_points)
                raw = unpack_packed_field(packed, fmin, fmax)

            data = unpack_subdomains(raw, ts.nx, ts.ny, ts.nsubsx, ts.nsubsy)
            ts.fields.append(Field(name, long_name, units, isbin, data))
            field_index += 1

        return ts


class NCWriter(object):
    """
    封装两个 NetCDF 后端：
    - netCDF4
    - scipy.io.netcdf_file

    这里刻意只保留很薄的一层，方便调试时直接看到底层对象 ds.variables。
    """

    def __init__(self, path):
        self.path = str(path)
        self.backend = None
        self.ds = None

        if NetCDF4Dataset is not None:
            self.backend = "netCDF4"
            self.ds = NetCDF4Dataset(self.path, "w", format="NETCDF3_64BIT_DATA")
        elif ScipyNetCDFFile is not None:
            self.backend = "scipy"
            self.ds = ScipyNetCDFFile(self.path, mode="w", version=2)
        else:
            raise RuntimeError("No NetCDF writer available. Install netCDF4 or scipy.")

    def close(self):
        if self.ds is not None:
            self.ds.close()

    def create_dimension(self, name, size):
        self.ds.createDimension(name, size)

    def create_variable(self, name, dtype, dims):
        return self.ds.createVariable(name, dtype, dims)

    def set_attr(self, var, name, value):
        if self.backend == "netCDF4":
            var.setncattr(name, value)
        else:
            setattr(var, name, value)


class Converter(object):
    def __init__(self, input_path, force_latlon=False):
        self.input_path = input_path
        self.force_latlon = force_latlon

    def output_path_for(self, comp):
        base = os.path.basename(self.input_path)
        root, ext = os.path.splitext(base)
        if ext != ".2D":
            raise ValueError("wrong file name extension!")

        comp_clean = comp.strip()
        if comp_clean == "":
            comp_clean = "UNK"

        out_name = root + "_" + comp_clean + ".nc"
        out_path = os.path.join(os.path.dirname(self.input_path), out_name)

        if os.path.abspath(out_path) == os.path.abspath(self.input_path):
            raise ValueError("attempt to overwrite binary file!")

        return out_path

    def get_dims(self, ts):
        if ts.dolatlon:
            xdim = "lon"
            ydim = "lat"
        else:
            xdim = "x"
            ydim = "y"

        if ts.ny_gl != 1:
            return ("time", ydim, xdim), xdim, ydim
        return ("time", xdim), xdim, ydim

    def define_file(self, writer, ts, x, wgt):
        field_dims, xdim, ydim = self.get_dims(ts)

        # scipy 后端对 unlimited 维度顺序更敏感，因此 time 放最前面。
        if writer.backend == "scipy":
            writer.create_dimension("time", None)
            writer.create_dimension(xdim, ts.nx_gl)
            if ts.ny_gl != 1:
                writer.create_dimension(ydim, ts.ny_gl)
        else:
            writer.create_dimension(xdim, ts.nx_gl)
            if ts.ny_gl != 1:
                writer.create_dimension(ydim, ts.ny_gl)
            writer.create_dimension("time", None)

        if ts.dolatlon:
            vlon = writer.create_variable("lon", "f8", ("lon",))
            writer.set_attr(vlon, "units", "degrees_east")
            writer.set_attr(vlon, "long_name", "longitude")

            if ts.ny_gl != 1:
                vlat = writer.create_variable("lat", "f8", ("lat",))
                writer.set_attr(vlat, "units", "degrees_north")
                writer.set_attr(vlat, "long_name", "latitude")

                vwgt = writer.create_variable("wgt", "f8", ("lat",))
                writer.set_attr(vwgt, "long_name", "averaging weights")
        else:
            vx = writer.create_variable("x", "f4", ("x",))
            writer.set_attr(vx, "units", "m")
            writer.set_attr(vx, "long_name", "x")

            if ts.ny_gl != 1:
                vy = writer.create_variable("y", "f4", ("y",))
                writer.set_attr(vy, "units", "m")
                writer.set_attr(vy, "long_name", "y")

        vtime = writer.create_variable("time", "f8", ("time",))
        if ts.time_utsec > 0.0:
            writer.set_attr(vtime, "units", "seconds since 1900-01-01 00:00:00.0")
            writer.set_attr(vtime, "long_name", "time")
            writer.set_attr(vtime, "calendar", "gregorian")

            vday = writer.create_variable("day", "f8", ("time",))
            writer.set_attr(vday, "units", "day")
            writer.set_attr(vday, "long_name", "day")
        else:
            writer.set_attr(vtime, "units", "day")
            writer.set_attr(vtime, "long_name", "time")

        vtimesec = writer.create_variable("timesec", "f8", ("time",))
        writer.set_attr(vtimesec, "units", "s")
        writer.set_attr(vtimesec, "long_name", "run time in sec")

        for field in ts.fields:
            v = writer.create_variable(field.name, "f4", field_dims)
            writer.set_attr(v, "long_name", field.long_name)
            writer.set_attr(v, "units", field.units)

        ds = writer.ds
        if ts.dolatlon:
            if USE_NUMPY:
                ds.variables["lon"][:] = np.asarray(ts.lon, dtype=np.float64)
                if ts.ny_gl != 1:
                    ds.variables["lat"][:] = np.asarray(ts.lat, dtype=np.float64)
                    ds.variables["wgt"][:] = np.asarray(wgt, dtype=np.float64)
            else:
                ds.variables["lon"][:] = ts.lon
                if ts.ny_gl != 1:
                    ds.variables["lat"][:] = ts.lat
                    ds.variables["wgt"][:] = wgt
        else:
            if USE_NUMPY:
                ds.variables["x"][:] = np.asarray(x, dtype=np.float32)
                if ts.ny_gl != 1:
                    ds.variables["y"][:] = np.asarray(ts.y, dtype=np.float32)
            else:
                ds.variables["x"][:] = x
                if ts.ny_gl != 1:
                    ds.variables["y"][:] = ts.y

    def write_timestep(self, writer, ts, time_index):
        ds = writer.ds

        for field in ts.fields:
            if ts.ny_gl != 1:
                if USE_NUMPY:
                    ds.variables[field.name][time_index, :, :] = np.asarray(field.data, dtype=np.float32)
                else:
                    ds.variables[field.name][time_index, :, :] = field.data
            else:
                if USE_NUMPY:
                    ds.variables[field.name][time_index, :] = np.asarray(field.data, dtype=np.float32).reshape(-1)
                else:
                    ds.variables[field.name][time_index, :] = field.data[0]

        if ts.time_utsec > 0.0:
            ds.variables["time"][time_index] = float(ts.time_utsec)
            ds.variables["day"][time_index] = float(ts.time)
        else:
            ds.variables["time"][time_index] = float(ts.time)

        ds.variables["timesec"][time_index] = float(ts.timesec)

    def print_brief_step(self, ts):
        print("NSTEP=%s" % ts.nstep)
        print("DATE=%s" % ts.datechar)
        print("timeUTsec=%s" % ts.time_utsec)
        print("timesec=%s" % ts.timesec)

    def print_header(self, ts, reader):
        print("Detected Fortran record markers: %s-byte, endian=%s" % (reader.marker_size, reader.endian))
        self.print_brief_step(ts)
        print("nx,ny,nz,nsubs,nsubsx,nsubsy,nfields:")
        print(ts.nx, ts.ny, ts.nz, ts.nsubs, ts.nsubsx, ts.nsubsy, ts.nfields)
        print("lat:  %.6g %.6g" % (values_min(ts.lat), values_max(ts.lat)))
        print("lon:  %.6g %.6g" % (values_min(ts.lon), values_max(ts.lon)))
        print("latv: %.6g %.6g" % (values_min(ts.latv), values_max(ts.latv)))
        print("lonu: %.6g %.6g" % (values_min(ts.lonu), values_max(ts.lonu)))
        print("y:    %.6g %.6g" % (values_min(ts.y), values_max(ts.y)))
        print("yv:   %.6g %.6g" % (values_min(ts.yv), values_max(ts.yv)))
        print("time=%s dx=%s dy=%s" % (ts.time, ts.dx, ts.dy))
        print("nx_gl=%s dx=%s" % (ts.nx_gl, ts.dx))
        print("ny_gl=%s dy=%s" % (ts.ny_gl, ts.dy))

    def print_field_stats(self, ts, wgt):
        print("                Max             Min           Mean")
        for field in ts.fields:
            vmax = field_max(field.data)
            vmin = field_min(field.data)
            mean = weighted_field_mean(field.data, wgt, ts.nx_gl, ts.ny_gl)
            print("%-8s %14.6g %14.6g %14.6g" % (field.name, vmax, vmin, mean))

    def validate_compatibility(self, first, ts, var_names):
        fields_now = []
        for f in ts.fields:
            fields_now.append(f.name)

        if ts.nx_gl != first.nx_gl:
            raise RuntimeError("incompatible nx_gl in later timestep")
        if ts.ny_gl != first.ny_gl:
            raise RuntimeError("incompatible ny_gl in later timestep")
        if ts.dolatlon != first.dolatlon:
            raise RuntimeError("incompatible dolatlon in later timestep")
        if fields_now != var_names:
            raise RuntimeError("incompatible field list in later timestep")

    def convert(self):
        reader = FortranSequentialReader(self.input_path)
        parser = Parser(reader)
        writer = None

        try:
            first = parser.read_timestep(force_latlon=self.force_latlon)
            out_path = self.output_path_for(first.comp)
            writer = NCWriter(out_path)

            self.print_header(first, reader)
            x, wgt = compute_weights(first)

            var_names = []
            for f in first.fields:
                var_names.append(f.name)

            self.define_file(writer, first, x, wgt)

            time_index = 0
            time_old = -1.0

            if first.time > time_old:
                self.write_timestep(writer, first, time_index)
                self.print_field_stats(first, wgt)
                time_old = first.time
                time_index += 1
            else:
                self.print_field_stats(first, wgt)

            while True:
                try:
                    ts = parser.read_timestep(force_latlon=self.force_latlon)
                except EOFError:
                    break

                self.validate_compatibility(first, ts, var_names)
                self.print_brief_step(ts)
                x_now, wgt_now = compute_weights(ts)
                # x_now 这里不用于写出，因为坐标在第一帧就已经定义。
                _ = x_now
                self.print_field_stats(ts, wgt_now)

                if ts.time > time_old:
                    self.write_timestep(writer, ts, time_index)
                    time_old = ts.time
                    time_index += 1

            return out_path
        finally:
            if writer is not None:
                writer.close()
            reader.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Fortran sequential .2D files to NetCDF"
    )
    parser.add_argument("input", help="input .2D file")
    parser.add_argument(
        "gridtype",
        nargs="?",
        default=None,
        help="optional override: latlon",
    )
    args = parser.parse_args()

    if args.gridtype is not None and args.gridtype != "latlon":
        raise SystemExit("Optional argument not latlon.")

    return args


def main():
    args = parse_args()

    if not str(args.input).endswith(".2D"):
        raise SystemExit("wrong file name extension!")

    force_latlon = (args.gridtype == "latlon")
    converter = Converter(args.input, force_latlon=force_latlon)
    out_path = converter.convert()
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
