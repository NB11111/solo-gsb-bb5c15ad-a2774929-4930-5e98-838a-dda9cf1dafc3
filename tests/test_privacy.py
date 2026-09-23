"""图片隐私检查（EXIF）：单元解析 + 独立 API 端到端测试。

覆盖：
- PNG / JPEG 的拍摄时间、相机品牌/型号解析；
- GPS 度分秒 + N/S、E/W 还原为带符号十进制度；
- 字段缺失明确返回 None（前端显示“无”）；
- 损坏 / 无法解码的 EXIF 不阻断检查（200 + warnings），
  也不阻断含水印任务的创建（/api/embed）；
- 隐私检查不落盘：不创建任何任务目录、不触碰归档。

运行（需项目依赖）：
    python -m pytest tests/test_privacy.py -q
或直接：
    python tests/test_privacy.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ExifTags

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.privacy import (  # noqa: E402
    PrivacyImageError,
    _dms_to_decimal,
    check_privacy,
)


# ---------------------------------------------------------------- 夹具

def _plain_image(fmt="JPEG", size=(256, 256), color=(40, 90, 140)) -> bytes:
    im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def _image_with_exif(fmt="JPEG", make="TestCam", model="X100",
                     dto="2026:09:23 12:34:56", offset=None,
                     lat=None, lat_ref="N", lng=None, lng_ref="E",
                     altitude=None) -> bytes:
    im = Image.new("RGB", (256, 192), (40, 90, 140))
    exif = Image.Exif()
    if make is not None:
        exif[ExifTags.Base.Make] = make
    if model is not None:
        exif[ExifTags.Base.Model] = model
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    if dto is not None:
        sub[ExifTags.Base.DateTimeOriginal] = dto
    if offset is not None:
        sub[36880] = offset  # OffsetTimeOriginal
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if lat is not None:
        gps[ExifTags.GPS.GPSLatitudeRef] = lat_ref
        gps[ExifTags.GPS.GPSLatitude] = lat
    if lng is not None:
        gps[ExifTags.GPS.GPSLongitudeRef] = lng_ref
        gps[ExifTags.GPS.GPSLongitude] = lng
    if altitude is not None:
        gps[ExifTags.GPS.GPSAltitude] = altitude
    buf = io.BytesIO()
    im.save(buf, format=fmt, exif=exif)
    return buf.getvalue()


def _corrupt_exif_jpeg() -> bytes:
    """合法 JPEG 图片 + 结构损坏的 APP1/EXIF 段（IFD 条目数越界）。"""
    raw = _plain_image("JPEG")
    payload = b"II*\x00\x08\x00\x00\x00\xff\xff" + b"\x00" * 40
    seg = b"Exif\x00\x00" + payload
    app1 = b"\xff\xe1" + (len(seg) + 2).to_bytes(2, "big") + seg
    return raw[:2] + app1 + raw[2:]


def _garbage_tiff_exif_jpeg() -> bytes:
    """合法 JPEG + APP1 声明完好但 TIFF 内容是随机字节。"""
    raw = _plain_image("JPEG")
    payload = b"GARBAGE-TIFF" + b"\x7f" * 60
    seg = b"Exif\x00\x00" + payload
    app1 = b"\xff\xe1" + (len(seg) + 2).to_bytes(2, "big") + seg
    return raw[:2] + app1 + raw[2:]


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-privacy-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    for mod in ("app.main", "app.archive"):
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR  # noqa: E402

    return TestClient(app), Path(DATA_DIR)


# ---------------------------------------------------------------- 单元：DMS

def test_dms_decimal_signs_and_bounds():
    v, w = _dms_to_decimal((31.0, 14.0, 12.0), "N")
    assert w is None and abs(v - (31 + 14 / 60 + 12 / 3600)) < 1e-7

    v, w = _dms_to_decimal((31.0, 14.0, 12.0), "S")
    assert w is None and v < 0 and abs(v + (31 + 14 / 60 + 12 / 3600)) < 1e-7

    v, w = _dms_to_decimal((121.0, 28.0, 30.0), "E")
    assert w is None and v > 0
    v, w = _dms_to_decimal((121.0, 28.0, 30.0), "W")
    assert w is None and v < 0

    # 无方向：按正值
    v, w = _dms_to_decimal((10.0, 0.0, 0.0), None)
    assert w is None and v == 10.0

    # 非法方向 / 越界分秒 / 分量缺失
    assert _dms_to_decimal((10.0, 0.0, 0.0), "X") == (None, None) or \
           _dms_to_decimal((10.0, 0.0, 0.0), "X")[0] is None
    v, w = _dms_to_decimal((10.0, 70.0, 0.0), "N")
    assert v is None and w
    v, w = _dms_to_decimal((10.0, 30.0), "N")
    assert v is None and w
    v, w = _dms_to_decimal(("x", 0, 0), "N")
    assert v is None and w


# ---------------------------------------------------------------- 单元：解析

def test_jpeg_full_exif():
    data = _image_with_exif(
        "JPEG", lat=(31.0, 14.0, 12.0), lng=(121.0, 28.0, 30.0),
        altitude=12.5)
    r = check_privacy(data, "shot.jpg")
    assert r.format == "JPEG"
    assert r.has_exif is True
    assert r.camera_make == "TestCam"
    assert r.camera_model == "X100"
    assert r.taken_at == "2026:09:23 12:34:56"
    assert r.taken_at_iso == "2026-09-23T12:34:56"
    assert r.warnings == []
    gps = r.gps
    assert gps["latitude_ref"] == "N" and gps["longitude_ref"] == "E"
    assert gps["latitude_dms"] == "31°14′12″"
    assert gps["longitude_dms"] == "121°28′30″"
    assert abs(gps["latitude"] - 31.23666667) < 1e-6
    assert abs(gps["longitude"] - 121.475) < 1e-6
    assert gps["altitude_m"] == 12.5


def test_png_full_exif_negative_hemispheres():
    # 悉尼：南纬 / 西经 -> 均为负数
    data = _image_with_exif(
        "PNG", make="Acme", model="S2",
        lat=(33.0, 51.0, 54.5), lat_ref="S",
        lng=(151.0, 12.0, 33.2), lng_ref="W")
    r = check_privacy(data, "sydney.png")
    assert r.format == "PNG"
    assert r.camera_make == "Acme" and r.camera_model == "S2"
    assert r.gps["latitude"] < 0 and r.gps["longitude"] < 0
    assert r.gps["latitude_ref"] == "S" and r.gps["longitude_ref"] == "W"
    expected_lat = -(33 + 51 / 60 + 54.5 / 3600)
    expected_lng = -(151 + 12 / 60 + 33.2 / 3600)
    assert abs(r.gps["latitude"] - expected_lat) < 1e-6
    assert abs(r.gps["longitude"] - expected_lng) < 1e-6


def test_no_exif_fields_are_none():
    r = check_privacy(_plain_image("JPEG"), "clean.jpg")
    assert r.has_exif is False
    assert r.taken_at is None and r.taken_at_iso is None
    assert r.camera_make is None and r.camera_model is None
    assert r.gps is None
    assert r.warnings == []

    # PNG 无 EXIF 同理
    r2 = check_privacy(_plain_image("PNG"), "clean.png")
    assert r2.has_exif is False and r2.gps is None


def test_partial_fields():
    # 只有品牌与纬度，没有型号 / 经度 / 时间
    data = _image_with_exif(model=None, dto=None,
                            lat=(10.0, 0.0, 0.0), lat_ref="N",
                            lng=None, lng_ref=None)
    r = check_privacy(data, "half.jpg")
    assert r.camera_make == "TestCam"
    assert r.camera_model is None
    assert r.taken_at is None
    gps = r.gps
    assert gps["latitude"] == 10.0
    assert gps["longitude"] is None and gps["longitude_ref"] is None
    assert gps["latitude_dms"] == "10°0′0″"
    assert gps["longitude_dms"] is None


def test_datetime_iso_normalization():
    data = _image_with_exif(dto="2026:09:23 12:34:56", offset="+08:00")
    r = check_privacy(data, "tz.jpg")
    assert r.taken_at_iso == "2026-09-23T12:34:56+08:00"

    data2 = _image_with_exif(dto="not-a-date", offset=None)
    r2 = check_privacy(data2, "bad-dt.jpg")
    assert r2.taken_at == "not-a-date"
    assert r2.taken_at_iso is None  # 归一化失败但原文保留


def test_corrupt_exif_does_not_block_check():
    for blob in (_corrupt_exif_jpeg(), _garbage_tiff_exif_jpeg()):
        r = check_privacy(blob, "corrupt.jpg")
        assert r.format == "JPEG"
        assert r.width == 256 and r.height == 256
        # 字段缺失显式为 None
        assert r.taken_at is None and r.camera_make is None
        assert r.camera_model is None and r.gps is None
        # 损坏至少在 has_exif=False 或 warnings 非空中体现
        assert r.has_exif is False or r.warnings


def test_unreadable_and_wrong_format():
    with pytest_raises(PrivacyImageError):
        check_privacy(b"not an image" * 50, "x.jpg")
    with pytest_raises(PrivacyImageError):
        check_privacy(b"\x89PNG\r\n\x1a\n" + b"broken" * 40, "x.png")
    with pytest_raises(PrivacyImageError):
        check_privacy(_plain_image("BMP"), "x.bmp")
    with pytest_raises(PrivacyImageError):
        # GIF 也不允许
        check_privacy(_plain_image("GIF"), "x.gif")


# 轻量 pytest.raises 替身，便于免 pytest 直接运行
class pytest_raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), \
            f"expected {self.exc.__name__}"
        return True


# ---------------------------------------------------------------- API

def test_api_full_and_missing_fields():
    client, data_dir = _fresh_client()
    data = _image_with_exif(
        "JPEG", lat=(31.0, 14.0, 12.0), lng=(121.0, 28.0, 30.0))
    r = client.post(
        "/api/privacy-check",
        files={"image": ("shot.jpg", data, "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["camera_make"] == "TestCam"
    assert j["taken_at"] == "2026:09:23 12:34:56"
    assert j["gps"]["latitude"] > 0 and j["gps"]["longitude"] > 0

    # 无 EXIF：字段为 null
    r2 = client.post(
        "/api/privacy-check",
        files={"image": ("clean.png", _plain_image("PNG"), "image/png")})
    assert r2.status_code == 200, r2.text
    j2 = r2.json()
    assert j2["format"] == "PNG"
    assert j2["has_exif"] is False
    assert j2["taken_at"] is None and j2["camera_make"] is None
    assert j2["camera_model"] is None and j2["gps"] is None

    # 隐私检查不产生任何任务目录 / 归档文件
    entries = sorted(p.name for p in data_dir.iterdir())
    assert entries == [], f"隐私检查不得落盘，发现：{entries}"


def test_api_corrupt_exif_returns_200():
    client, _ = _fresh_client()
    r = client.post(
        "/api/privacy-check",
        files={"image": ("corrupt.jpg", _corrupt_exif_jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["format"] == "JPEG" and j["width"] == 256
    assert j["taken_at"] is None and j["gps"] is None


def test_api_rejects_bad_and_unsupported():
    client, _ = _fresh_client()
    r = client.post(
        "/api/privacy-check",
        files={"image": ("x.jpg", b"garbage" * 30, "image/jpeg")})
    assert r.status_code == 400

    r = client.post(
        "/api/privacy-check",
        files={"image": ("x.bmp", _plain_image("BMP"), "image/bmp")})
    assert r.status_code == 400
    assert "PNG / JPEG" in r.json()["detail"]

    r = client.post(
        "/api/privacy-check",
        files={"image": ("x.gif", _plain_image("GIF"), "image/gif")})
    assert r.status_code == 400


def test_corrupt_exif_does_not_block_embed_job():
    """损坏 EXIF 的图片仍可创建水印任务（与隐私检查互不干扰）。"""
    client, data_dir = _fresh_client()
    r = client.post(
        "/api/embed",
        files={"image": ("corrupt.jpg", _corrupt_exif_jpeg(), "image/jpeg")},
        data={"text": "EXIF-损坏-不阻断", "strength": "medium"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert (data_dir / job_id / "meta.json").exists()
    assert (data_dir / job_id / "watermarked.png").exists()
    # 归档日志不存在
    assert not (data_dir / "archive.log.jsonl").exists()


def test_privacy_check_does_not_touch_archive():
    client, data_dir = _fresh_client()
    # 先制造一个真实水印任务
    r = client.post(
        "/api/embed",
        files={"image": ("a.png", _plain_image("PNG", size=(384, 384)),
                         "image/png")},
        data={"text": "已有任务", "strength": "medium"})
    assert r.status_code == 200
    before = sorted(p.name for p in data_dir.iterdir())

    # 多次隐私检查：任务目录集合不变、无归档产生
    for _ in range(3):
        rr = client.post(
            "/api/privacy-check",
            files={"image": ("p.jpg",
                             _image_with_exif(lat=(1.0, 2.0, 3.0),
                                              lng=(4.0, 5.0, 6.0)),
                             "image/jpeg")})
        assert rr.status_code == 200
    after = sorted(p.name for p in data_dir.iterdir())
    assert before == after
    assert not (data_dir / "archive.log.jsonl").exists()
    assert not (data_dir / "reports").exists()


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个图片隐私检查测试全部通过 ✔")
