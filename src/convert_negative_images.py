from pathlib import Path
import shutil

from PIL import Image

image_dir = Path(
    "/home/dylan/Documents/YOLO/2026vollayball/images/train"
)
backup_dir = Path(
    "/home/dylan/Documents/YOLO/volleyball-training/"
    "backups/negative_jpeg_before_conversion"
)

files = sorted(image_dir.glob("volleyball_negative_20260914_*.jpg"))
assert len(files) == 57, f"预期57张JPEG，实际{len(files)}张，请先核对"

# 先检查目标路径，避免覆盖已有文件。
for source in files:
    assert not source.with_suffix(".png").exists(), f"PNG已存在：{source}"
    assert not (backup_dir / source.name).exists(), f"备份已存在：{source}"

backup_dir.mkdir(parents=True, exist_ok=True)

# 先备份全部原图。
for source in files:
    shutil.copy2(source, backup_dir / source.name)

# 转成PNG，并核对解码后的尺寸和像素。
for source in files:
    target = source.with_suffix(".png")

    with Image.open(source) as image:
        original = image.convert("RGB")
        original.save(target, format="PNG")

    with Image.open(target) as image:
        converted = image.convert("RGB")
        assert converted.size == original.size, f"尺寸变化：{target}"
        assert converted.tobytes() == original.tobytes(), f"像素变化：{target}"

# 所有转换都通过后，移除训练目录中的旧JPEG。
# 原JPEG仍保存在上面的backup_dir中。
for source in files:
    assert source.read_bytes() == (backup_dir / source.name).read_bytes()
    source.unlink()

print(f"完成：{len(files)}张 JPEG 转成 PNG，像素检查通过。")
print(f"旧JPEG备份：{backup_dir}")
