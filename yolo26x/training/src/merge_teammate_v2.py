"""Build and independently verify a new dataset release; preserve both inputs."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil

from PIL import Image
from ultralytics.data.utils import verify_image_label

BASE = Path('/workspace/data/volleyball_yolo_v1_20260917_compatible')
EXTRA = Path('/workspace/imports/volleyball_teammate_reviewed_20260918')
TARGET = Path('/workspace/data/volleyball_yolo_v2_20260918')
STAGING = TARGET.with_name(TARGET.name + '.building')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(path):
    with Image.open(path) as im:
        im.load()
        rgb = im.convert('RGB')
        h = hashlib.sha256(str(rgb.size).encode())
        h.update(rgb.tobytes())
        return h.hexdigest(), rgb.size


def main():
    assert not TARGET.exists() and not STAGING.exists(), 'Refusing to overwrite a release/staging directory'
    base_rows = json.loads((BASE / 'manifest.json').read_text())
    supplement = json.loads((EXTRA / 'manifest.json').read_text())
    new_rows = supplement['records']
    old_hashes = {r['image_sha256'] for r in base_rows}
    assert not old_hashes.intersection(r['image_sha256'] for r in new_rows), 'Duplicate bytes require review'
    # Hash decoded images with the same dimensions as the supplement to catch re-encoded duplicates.
    sizes = {(r['width'], r['height']) for r in new_rows}
    old_pixels = {}
    for r in base_rows:
        if (r['width'], r['height']) in sizes:
            h, _ = pixel_hash(BASE / 'images' / r['split'] / r['image'])
            old_pixels[h] = r
    new_pixels = set()
    for r in new_rows:
        h, size = pixel_hash(EXTRA / 'images/train' / r['image'])
        assert size == (r['width'], r['height'])
        assert h not in old_pixels, f'Decoded-pixel duplicate: {r["image"]}'
        assert h not in new_pixels, f'Duplicate within supplement: {r["image"]}'
        new_pixels.add(h)
    print('Decoded-pixel duplicate check passed.', flush=True)
    near = []
    for r in new_rows:
        for old in base_rows:
            if old['split'] != 'train':
                distance = (int(r['dhash'], 16) ^ int(old['dhash'], 16)).bit_count()
                if distance <= 5:
                    near.append([r['image'], old['image'], old['split'], distance])
    assert not near, 'Near-duplicate evaluation images need review before merging'
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (STAGING / kind / split).mkdir(parents=True, exist_ok=True)

    def copy_base(row):
        for kind, filename, expected in [('images', row['image'], row['image_sha256']),
                                         ('labels', Path(row['image']).stem + '.txt', row['label_sha256'])]:
            source = BASE / kind / row['split'] / filename
            target = STAGING / kind / row['split'] / filename
            shutil.copy2(source, target)
            assert sha(target) == expected, filename
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(copy_base, base_rows), 1):
            if i % 1000 == 0:
                print(f'Copied and hashed {i}/{len(base_rows)} original samples', flush=True)

    rows = list(base_rows)
    stems = {Path(r['image']).stem for r in rows if r['split'] == 'train'}
    annotation_dir = STAGING / 'annotations/train'
    annotation_dir.mkdir(parents=True)
    for source_row in new_rows:
        row = dict(source_row)
        name, stem = row['image'], Path(row['image']).stem
        assert stem not in stems, name
        stems.add(stem)
        for source, target, expected in [
            (EXTRA / 'images/train' / name, STAGING / 'images/train' / name, row['image_sha256']),
            (EXTRA / 'labels/train' / (stem + '.txt'), STAGING / 'labels/train' / (stem + '.txt'), row['label_sha256']),
            (EXTRA / 'images/train' / (stem + '.json'), annotation_dir / (stem + '.json'), row['annotation_sha256']),
        ]:
            shutil.copy2(source, target)
            assert sha(target) == expected, name
        row.update(split='train', source_id='teammate_20260918', group_id=row['original_group_id'],
                   source_image=str(EXTRA / 'images/train' / name),
                   source_label=str(EXTRA / 'images/train' / (stem + '.json')),
                   review_label_sha256=row['annotation_sha256'],
                   assignment_reason='Added reviewed teammate capture batch to train; existing val/test unchanged.')
        rows.append(row)

    counts = defaultdict(Counter)
    groups = defaultdict(list)
    for r in rows:
        split = r['split']
        counts[split]['images'] += 1
        counts[split]['positive' if r['boxes'] else 'negative'] += 1
        counts[split]['boxes'] += r['boxes']
        groups[r['capture_group_id']].append(r)
    assert all(len({r['split'] for r in group}) == 1 for group in groups.values())
    for split in ('train', 'val', 'test'):
        images = sorted((STAGING / 'images' / split).iterdir())
        args = [(str(p), str(STAGING / 'labels' / split / (p.stem + '.txt')), '', False, 1, 0, 0, False) for p in images]
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(verify_image_label, args))
        errors = [r[-1] for r in checked if r[0] is None]
        assert not errors, errors[:5]
        assert len(checked) == counts[split]['images']
        assert sum(len(r[1]) for r in checked) == counts[split]['boxes']
        print(split, dict(counts[split]), 'YOLO accepted:', len(checked), flush=True)

    for name in ('classes.txt', 'online_manifest.json', 'online_图片来源与署名.md', 'verify_dataset.py', 'configure_dataset.py'):
        shutil.copy2(BASE / name, STAGING / name)
    provenance = STAGING / 'provenance'
    provenance.mkdir()
    shutil.copy2(BASE / 'compatibility_report.json', provenance / 'v1_compatibility_report.json')
    shutil.copy2(EXTRA / 'manifest.json', provenance / 'teammate_manifest.json')
    shutil.copy2(EXTRA / 'source_annotations_original.zip', provenance / 'teammate_original_annotations.zip')
    report = {'version': TARGET.name, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'total_images': len(rows), 'total_boxes': sum(r['boxes'] for r in rows),
              'stats': counts, 'base_dataset': str(BASE), 'supplement': str(EXTRA),
              'source_zip_sha256': '3021dc5dacfdc35d39ae76551df21ab0fc768118651a4c30ed412b97e20a2578',
              'added_images': len(new_rows), 'validation_and_test_unchanged': True,
              'new_vs_existing_byte_duplicates': 0, 'new_vs_existing_same_size_pixel_duplicates': 0,
              'new_vs_val_test_dhash_candidates_at_distance_5': near,
              'limitations': ['Similarity screening does not prove independent capture or unseen locations.',
                             'Supplement review status comes from the supplied package; automatic verification checks format and consistency.']}
    (STAGING / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (STAGING / 'split_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    group_report = {k: {'split': group[0]['split'], 'images': [r['image'] for r in group]} for k, group in groups.items()}
    (STAGING / 'split_groups.json').write_text(json.dumps(group_report, ensure_ascii=False, indent=2))
    (STAGING / 'data.yaml').write_text('path: ' + json.dumps(str(STAGING)) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: volleyball\n')
    # Reuse the reviewed standard-library validator to verify every copied hash and label.
    spec = importlib.util.spec_from_file_location('release_verify', str(STAGING / 'verify_dataset.py'))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify(STAGING)
    (STAGING / 'data.yaml').write_text('path: ' + json.dumps(str(TARGET)) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: volleyball\n')
    (STAGING / 'README.md').write_text('''# 排球数据 v2（2026-09-18）

在 v1 兼容副本上新增队友已复核的 234 张素材，全部加入训练集。
训练：3432 张（2947 张正样本、485 张负样本），2960 个框。
验证：626 张（539 张正样本、87 张负样本），540 个框。
测试：209 张（75 张正样本、134 张负样本），86 个框。
合计：4267 张，3586 个框。唯一类别：0 = volleyball。

原有图片、标签及验证/测试划分保持不变，旧数据版本未修改。
新增素材的 JSON 保存在 annotations/train，原始标注备份及来源在 provenance。
图片字节校验、标签坐标、YOLO 实际读取、分组及补充素材与旧数据的重复筛查均通过。
相似度筛查不能证明拍摄场景完全独立；人工复核状态来自补充包的记录。
data.yaml 使用容器路径；迁移后可运行 configure_dataset.py 更新，再运行 verify_dataset.py。
''', encoding='utf-8')
    STAGING.rename(TARGET)
    print('COMPLETE:', TARGET, flush=True)


if __name__ == '__main__':
    main()
