"""图片隐私检查 —— 解析 PNG / JPEG 内嵌 EXIF 的拍摄时间、相机信息与 GPS。

结果只在本次调用内组装返回：不写磁盘、不建任务、不触碰水印任务 / 攻击记录 /
报告归档。损坏或无法解码的 EXIF 只产生 warning，绝不影响图片本身的打开与检查。
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime

from PIL import ExifTags, Image, ImageOps

# 仅允许 PNG / JPEG（隐私检查入口明确限定这两种格式）
SUPPORTED_FORMATS = {"JPEG", "PNG"}

# 各字段的 EXIF tag 编号
_T_MAKE = 271
_T_MODEL = 272
_T_DATETIME = 306
_T_DATETIME_ORIGINAL = 36867
_T_DATETIME_DIGITIZED = 36868
_T_OFFSET_TIME_ORIGINAL = 36880
_T_OFFSET_TIME = 34858
_T_GPS_LAT_REF = 1
_T_GPS_LAT = 2
_T_GPS_LON_REF = 3
_T_GPS_LON = 4
_T_GPS_ALT = 6

_EXIF_DT_RE = re.compile(r"^(\d{4})[:./-](\d{2})[:./-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


class PrivacyImageError(ValueError):
    """图片本身无法解码（非图片 / 不支持的格式 / 尺寸越界）。"""


@dataclass
class PrivacyResult:
    filename: str
    format: str
    width: int
    height: int
    has_exif: bool
    taken_at: str | None = None        # 拍摄时间（EXIF 原文字符串）
    taken_at_iso: str | None = None    # 尽力归一化后的 ISO-8601
    camera_make: str | None = None
    camera_model: str | None = None
    gps: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "has_exif": self.has_exif,
            "taken_at": self.taken_at,
            "taken_at_iso": self.taken_at_iso,
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "gps": self.gps,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------- 基础工具

def _clean_text(v) -> str | None:
    """EXIF 字符串字段：去掉尾部 NUL 与空白；非字符串 / 空值返回 None。"""
    if v is None:
        return None
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", "replace")
        except Exception:
            return None
    s = str(v).strip().strip("\x00").strip()
    return s or None


def _dms_to_decimal(dms, ref: str | None) -> tuple[float | None, str | None]:
    """(度, 分, 秒) + 方向字母 -> 带符号十进制度。

    返回 (decimal, warning)。分量缺失或非法时 decimal 为 None。
    方向只接受 N/S（纬度）/ E/W（经度），大小写不敏感；无方向时按正值还原。
    """
    if not isinstance(dms, (tuple, list)) or len(dms) < 3:
        return None, "GPS 坐标分量不完整（需要度/分/秒三项），已忽略该坐标。"
    vals: list[float] = []
    for x in dms[:3]:
        try:
            vals.append(abs(float(x)))
        except (TypeError, ValueError, OverflowError):
            return None, "GPS 度分秒含无法解析的数值，已忽略该坐标。"
    d, m, s = vals
    if m >= 60 or s >= 60:
        return None, f"GPS 分/秒越界（{m:g}′ {s:g}″），已忽略该坐标。"
    dec = d + m / 60.0 + s / 3600.0
    r = (ref or "").strip().upper()
    if r in ("S", "W"):
        dec = -dec
    elif r not in ("N", "E", ""):
        return None, f"GPS 方向标记非法（{ref!r}），已忽略该坐标。"
    return round(dec, 8), None


def _parse_exif_datetime(raw: str | None, offset: str | None) -> str | None:
    """把 EXIF 的 ``YYYY:MM:DD HH:MM:SS`` 尽量归一化为 ISO-8601。"""
    if not raw:
        return None
    m = _EXIF_DT_RE.match(raw.strip())
    if not m:
        return None
    y, mo, d, h, mi, se = (int(g) for g in m.groups())
    try:
        dt = datetime(y, mo, d, h, mi, se)
    except ValueError:
        return None
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if offset:
        off = offset.strip()
        # OffsetTime 形如 +08:00 / -05:00 / Z
        if off.upper() == "Z":
            return iso + "Z"
        if re.match(r"^[+-]\d{2}:\d{2}$", off):
            return iso + off
    return iso


# ---------------------------------------------------------------- GPS

def _parse_gps(gps_ifd: dict, warnings_out: list[str]) -> dict | None:
    lat_raw = gps_ifd.get(_T_GPS_LAT)
    lon_raw = gps_ifd.get(_T_GPS_LON)
    lat_ref = _clean_text(gps_ifd.get(_T_GPS_LAT_REF))
    lon_ref = _clean_text(gps_ifd.get(_T_GPS_LON_REF))

    if lat_raw is None and lon_raw is None:
        return None

    lat_dec, lat_warn = _dms_to_decimal(lat_raw, lat_ref)
    lon_dec, lon_warn = _dms_to_decimal(lon_raw, lon_ref)
    for w in (lat_warn, lon_warn):
        if w:
            warnings_out.append(w)

    # 范围校验
    if lat_dec is not None and not -90 <= lat_dec <= 90:
        warnings_out.append(f"纬度 {lat_dec} 超出 ±90°，已忽略。")
        lat_dec = None
    if lon_dec is not None and not -180 <= lon_dec <= 180:
        warnings_out.append(f"经度 {lon_dec} 超出 ±180°，已忽略。")
        lon_dec = None

    gps = {
        "latitude": lat_dec,
        "longitude": lon_dec,
        "latitude_ref": lat_ref,       # N / S；缺失为 None
        "longitude_ref": lon_ref,      # E / W；缺失为 None
        "latitude_dms": _format_dms(lat_raw),
        "longitude_dms": _format_dms(lon_raw),
    }

    # 海拔（可选补充字段，缺失即不出现在结果里也不影响主流程）
    alt = gps_ifd.get(_T_GPS_ALT)
    if alt is not None:
        try:
            gps["altitude_m"] = round(float(alt), 2)
        except (TypeError, ValueError, OverflowError):
            pass

    if lat_dec is None and lon_dec is None:
        # GPS IFD 存在但坐标都用不了：标记存在但坐标缺失
        gps["present_but_unusable"] = True
    return gps


def _format_dms(dms) -> str | None:
    """把度分秒元组渲染成展示字符串，如 ``31°14′12″``。"""
    if not isinstance(dms, (tuple, list)) or len(dms) < 3:
        return None
    parts: list[str] = []
    for x in dms[:3]:
        try:
            parts.append(f"{float(x):g}")
        except (TypeError, ValueError, OverflowError):
            return None
    return f"{parts[0]}°{parts[1]}′{parts[2]}″"


# ---------------------------------------------------------------- 主入口

def check_privacy(data: bytes, filename: str = "") -> PrivacyResult:
    """解析上传图片字节并提取隐私 EXIF 字段。

    - 图片打不开 / 非 PNG·JPEG：抛 :class:`PrivacyImageError`（接口层转 400）。
    - 图片可打开但 EXIF 损坏 / 缺失：正常返回，``has_exif`` 与 ``warnings``
      如实反映情况，绝不抛异常。
    """
    result = PrivacyResult(
        filename=filename or "",
        format="",
        width=0,
        height=0,
        has_exif=False,
    )
    # JPEG 的 APP1 在 Image.open() 阶段即被解析，损坏 EXIF 会在此刻发警告，
    # 因此整个打开 + 读取过程都包在 catch_warnings 内。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            im = Image.open(io.BytesIO(data))
            # load() 触发完整解码，确保截断 / 伪装文件在此处暴露
            im.load()
        except Exception as e:
            raise PrivacyImageError(
                "无法解析上传的图片，文件可能已损坏。") from e

        result.format = (im.format or "").upper()
        if result.format not in SUPPORTED_FORMATS:
            raise PrivacyImageError(
                f"仅支持 PNG / JPEG 图片，收到的是 {im.format or '未知'} 格式。")
        result.width, result.height = im.size

        # EXIF 读取整体兜底：损坏段只记录 warning，绝不抛出影响检查。
        try:
            top = im.getexif()
            exif_ifd = top.get_ifd(ExifTags.IFD.Exif)
            gps_ifd = top.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception as e:  # 极端损坏的 TIFF 结构
            result.warnings.append(
                f"EXIF 数据已损坏，无法解析（{type(e).__name__}）。")
            top = None
            exif_ifd = {}
            gps_ifd = {}

    for w in caught:
        msg = str(w.message)
        if issubclass(w.category, UserWarning) and "EXIF" in msg.upper():
            line = f"EXIF 数据已损坏：{msg}"
            if line not in result.warnings:
                result.warnings.append(line)

    if top is not None:
        try:
            has_any = len(top) > 0 or len(exif_ifd) > 0 or len(gps_ifd) > 0
        except Exception:
            has_any = False
        result.has_exif = bool(has_any)

        result.camera_make = _clean_text(top.get(_T_MAKE))
        result.camera_model = _clean_text(top.get(_T_MODEL))

        raw_dt = (
            _clean_text(exif_ifd.get(_T_DATETIME_ORIGINAL))
            or _clean_text(exif_ifd.get(_T_DATETIME_DIGITIZED))
            or _clean_text(top.get(_T_DATETIME))
        )
        offset = (
            _clean_text(exif_ifd.get(_T_OFFSET_TIME_ORIGINAL))
            or _clean_text(top.get(_T_OFFSET_TIME))
        )
        result.taken_at = raw_dt
        result.taken_at_iso = _parse_exif_datetime(raw_dt, offset)

        try:
            result.gps = _parse_gps(dict(gps_ifd), result.warnings)
        except Exception as e:  # 任何意外的 GPS 结构都不能阻断检查
            result.gps = {"latitude": None, "longitude": None,
                          "latitude_ref": None, "longitude_ref": None,
                          "latitude_dms": None, "longitude_dms": None,
                          "present_but_unusable": True}
            result.warnings.append(f"GPS 信息结构异常，已忽略（{type(e).__name__}）。")

    return result


def open_rgb_safely(data: bytes):
    """供水印任务使用的“带 EXIF 容错”读图。

    先用普通方式打开并做方向矫正；若 EXIF 损坏导致矫正失败，则退回
    未矫正的 RGB 图，保证损坏 EXIF 不会阻断嵌入 / 检测任务创建。
    """
    im = Image.open(io.BytesIO(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            im = Image.open(io.BytesIO(data))
    return im.convert("RGB")
