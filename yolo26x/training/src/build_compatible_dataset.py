"""Create a pixel-checked YOLO-compatible derivative; keep the source snapshot intact."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

from PIL import Image
from ultralytics.data.utils import verify_image_label

SOURCE = Path('/datasets/volleyball')
TARGET = Path('/workspace/data/volleyball_yolo_v1_20260917_compatible')
STAGING = TARGET.with_name(TARGET.name + '.building')
AUDIT = Path('/workspace/diagnostics/dataset_loader_audit.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    if TARGET.exists() or STAGING.exists():
        raise FileExistsError('Refusing to overwrite an existing derivative or staging directory')
    rows = json.loads((SOURCE / 'manifest.json').read_text())
    audit = json.loads(AUDIT.read_text())
    rejected = {(split, item['image']) for split, entry in audit.items() for item in entry['rejected']}
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (STAGING / kind / split).mkdir(parents=True, exist_ok=True)

    def copy_row(row):
        row = dict(row)
        split, old_name = row['split'], row['image']
        source_image = SOURCE / 'images' / split / old_name
        source_label = SOURCE / 'labels' / split / (Path(old_name).stem + '.txt')
        assert sha(source_image) == row['image_sha256'], old_name
        assert sha(source_label) == row['label_sha256'], old_name
        converted = (split, old_name) in rejected
        new_name = Path(old_name).with_suffix('.png').name if converted else old_name
        target_image = STAGING / 'images' / split / new_name
        target_label = STAGING / 'labels' / split / (Path(new_name).stem + '.txt')
        if target_image.exists() or target_label.exists():
            raise FileExistsError(new_name)
        if converted:
            with Image.open(source_image) as original:
                assert original.getexif().get(274, 1) == 1, old_name
                pixels = original.convert('RGB')
                assert pixels.size == (row['width'], row['height']), old_name
                original_pixels = hashlib.sha256(pixels.tobytes()).hexdigest()
                pixels.save(target_image, 'PNG', compress_level=1)
            with Image.open(target_image) as saved:
                assert saved.size == (row['width'], row['height']), new_name
                assert hashlib.sha256(saved.convert('RGB').tobytes()).hexdigest() == original_pixels, new_name
            row['compatibility_conversion'] = {
                'original_image': old_name, 'original_image_sha256': row['image_sha256'],
                'method': 'JPEG decoded to RGB, saved as lossless PNG',
                'decoded_rgb_sha256': original_pixels, 'pixels_and_dimensions_verified': True,
            }
        else:
            shutil.copy2(source_image, target_image)
            assert sha(target_image) == row['image_sha256'], new_name
        shutil.copy2(source_label, target_label)
        assert sha(target_label) == row['label_sha256'], new_name
        row['image'] = new_name
        row['image_sha256'] = sha(target_image) if converted else row['image_sha256']
        return row

    print(f'Building derivative: {len(rows)} images; {len(rejected)} JPEG-to-PNG conversions', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        output_rows = []
        for i, row in enumerate(pool.map(copy_row, rows), 1):
            output_rows.append(row)
            if i % 500 == 0:
                print(f'Prepared {i}/{len(rows)} images', flush=True)

    for name in ['classes.txt', 'split_report.json', 'split_groups.json', 'online_manifest.json',
                 'online_图片来源与署名.md', 'configure_dataset.py', 'verify_dataset.py']:
        shutil.copy2(SOURCE / name, STAGING / name)
    shutil.copy2(SOURCE / 'README.md', STAGING / 'README-source.md')
    (STAGING / 'manifest.json').write_text(json.dumps(output_rows, ensure_ascii=False, indent=2))
    (STAGING / 'data.yaml').write_text(
        'path: ' + json.dumps(str(TARGET)) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: volleyball\n'
    )
    counts = {}
    for split in ('train', 'val', 'test'):
        images = sorted((STAGING / 'images' / split).iterdir())
        args = [(str(p), str(STAGING / 'labels' / split / (p.stem + '.txt')), '', False, 1, 0, 0, False) for p in images]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(verify_image_label, args))
        errors = [r[-1] for r in results if r[0] is None]
        assert not errors, errors[:5]
        counts[split] = {'images': len(results), 'negative': sum(len(r[1]) == 0 for r in results),
                         'boxes': sum(len(r[1]) for r in results), 'rejected': 0}
        print(split, counts[split], flush=True)
    (STAGING / 'compatibility_report.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'converted_images': len(rejected), 'stats': counts, 'source_snapshot_modified': False,
        'labels_and_splits_unchanged': True, 'converted_pixels_and_dimensions_verified': True,
        'purpose': 'Avoid JPEG trailing-data repair and sample exclusion in the YOLO loader.'
    }, ensure_ascii=False, indent=2))
    STAGING.rename(TARGET)
    print('COMPLETE:', TARGET, flush=True)


if __name__ == '__main__':
    main()
