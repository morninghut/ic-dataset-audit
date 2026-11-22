"""Cache deterministic frozen encoder outputs; keep token moments for exact normalization."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from revision.common import CACHE, DATA, ROOT, write_json, read_json, sha256, setup

def catalog(cache):
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'catalog.json').exists():
        return read_json(cache / 'catalog.json')
    from evaluate import load_records
    from polos.evaluation_data import pascal_trials
    texts, images, text_ids, image_ids, datasets = [], [], {}, {}, {}
    def intern(value, values, lookup):
        if value not in lookup:
            lookup[value] = len(values)
            values.append(value)
        return lookup[value]
    for name in ['polaris_train', 'polaris_val', 'polaris_test', 'flickr8k_expert',
                 'flickr8k_cf', 'composite', 'pascal50s', 'foil_1ref', 'foil_4ref']:
        records, source, skipped = load_records(DATA, name, seed=42)
        original_count = len(records)
        pairwise = name == 'pascal50s' or name.startswith('foil_')
        if pairwise:
            if name == 'pascal50s':
                records = pascal_trials(records, 'paper', 42)
            write_json(cache / f'{name}_pairs.json',
                       [{k: row[k] for k in ['id', 'category', 'preferred', 'trial'] if k in row}
                        for row in records])
            records = [dict(id=f"{row.get('trial', 0)}:{row['id']}:{side}",
                            caption=row[f'caption_{side}'], references=row['references'], image=row['image'])
                       for row in records for side in ['a', 'b']]
        n, max_refs = len(records), max(len(r['references']) for r in records)
        cand = np.empty(n, dtype=np.int32)
        img = np.empty(n, dtype=np.int32)
        refs = np.full((n, max_refs), -1, dtype=np.int32)
        counts = np.empty(n, dtype=np.int16)
        targets = np.full(n, np.nan, dtype=np.float32)
        ids = []
        for i, row in enumerate(records):
            cand[i] = intern(row['caption'], texts, text_ids)
            img[i] = intern(str(Path(row['image']).resolve()), images, image_ids)
            counts[i] = len(row['references'])
            refs[i, :counts[i]] = [intern(s, texts, text_ids) for s in row['references']]
            targets[i] = row.get('human_score', np.nan)
            ids.append(row['id'])
        np.savez(cache / f'{name}.npz', candidate=cand, image=img, references=refs,
                 counts=counts, target=targets, ids=np.array(ids))
        datasets[name] = dict(count=n, original_count=original_count, source=str(source),
                              source_sha256=sha256(source), skipped=skipped, pairwise=pairwise,
                              text_prefix_count=len(texts), image_prefix_count=len(images))
        print(name, n, len(texts), len(images), flush=True)
        del records
        gc.collect()
    result = dict(version=1, texts=texts, images=images, datasets=datasets,
                  text_prefix='A photo depicts ', dtype='float32', encoder_mode='eval',
                  simcse='princeton-nlp/sup-simcse-roberta-base',
                  normalization_cache='raw CLS + per-layer per-sentence all-valid-token mean, variance, count',
                  test_labels_used_for_training=False, built_at=time.time())
    write_json(cache / 'catalog.json', result)
    return result

def mmap(cache, key, shape):
    path = cache / f'{key}.npy'
    if path.exists():
        out = np.load(path, mmap_mode='r+')
        if out.shape != shape or out.dtype != np.float32:
            raise ValueError(f'Cache shape or dtype mismatch: {key}')
        return out
    return np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)

def extract(args):
    setup()
    cache = args.cache
    meta = catalog(cache)
    part = args.part
    state_path = cache / f'{part}_status.json'
    lock = (cache / f'.{part}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read_json(state_path) if state_path.exists() else {'completed_texts': 0, 'completed_images': 0}
    if state.get('status') == 'complete':
        return
    state.update(status='running', pid=os.getpid(), started=time.time(), catalog_sha256=sha256(cache / 'catalog.json'))
    write_json(state_path, state)
    texts, images = meta['texts'], meta['images']
    required = meta['datasets'].get('polaris_test', dict(text_prefix_count=len(texts), image_prefix_count=len(images)))
    text_limit = min(len(texts), required['text_prefix_count']) if args.scope == 'polaris' else len(texts)
    image_limit = min(len(images), required['image_prefix_count']) if args.scope == 'polaris' else len(images)
    state.update(scope=args.scope, required_texts=text_limit, required_images=image_limit)
    device = args.device
    try:
        if part == 'simcse':
            from argparse import Namespace
            from polos.models.encoders.bert import BERTEncoder
            model = BERTEncoder.from_pretrained(Namespace(pretrained_model=meta['simcse'])).to(device).eval()
            model.requires_grad_(False)
            arrays = {key: mmap(cache, 'simcse_' + key, shape) for key, shape in
                      dict(cls=(len(texts), 13, 768), mean=(len(texts), 13),
                           var=(len(texts), 13), count=(len(texts),)).items()}
            batch_size = args.batch_size or 64
            with torch.no_grad():
                for start in range(state['completed_texts'], text_limit, batch_size):
                    end = min(start + batch_size, text_limit)
                    inputs = {k: v.to(device) for k, v in model.prepare_sample(texts[start:end]).items()}
                    enc = model(**inputs)
                    mask = enc['mask'].float().unsqueeze(-1)
                    count = mask.sum((1, 2)) * 768
                    arrays['count'][start:end] = count.cpu().numpy()
                    for layer, hidden in enumerate(enc['all_layers']):
                        mean = (hidden * mask).sum((1, 2)) / count
                        var = (((hidden - mean[:, None, None]) ** 2) * mask).sum((1, 2)) / count
                        arrays['cls'][start:end, layer] = hidden[:, 0].float().cpu().numpy()
                        arrays['mean'][start:end, layer] = mean.cpu().numpy()
                        arrays['var'][start:end, layer] = var.cpu().numpy()
                    for a in arrays.values():
                        a.flush()
                    state.update(completed_texts=end, total_texts=len(texts), updated=time.time(),
                                 seconds_per_text=(time.time() - state['started']) / max(1, end))
                    write_json(state_path, state)
                    if start % (batch_size * 50) == 0:
                        print(part, end, len(texts), flush=True)
        else:
            if part == 'clip':
                import polos.clip as clip
                # Polos builds CLIP on CPU (float32) before Lightning moves it to CUDA.
                # Loading directly on CUDA would silently use the OpenAI fp16 path.
                model, preprocess = clip.load('ViT-B/32', device='cpu')
                model.to(device)
                tokenize = clip.tokenize
                dimension, image_batch = 512, 64
            else:
                from polos.siglip_adapter import SiglipAdapter
                model = SiglipAdapter(str(ROOT / 'models/siglip-so400m-patch14-384'), 4).to(device)
                preprocess, tokenize = model.preprocess, model.tokenize
                dimension, image_batch = 1152, 4
            model.eval().requires_grad_(False)
            txt = mmap(cache, part + '_text', (len(texts), dimension))
            img = mmap(cache, part + '_image', (len(images), dimension))
            batch_size = args.batch_size or 64
            with torch.no_grad():
                for start in range(state['completed_texts'], text_limit, batch_size):
                    end = min(start + batch_size, text_limit)
                    tokens = tokenize(['A photo depicts ' + s for s in texts[start:end]], truncate=True).to(device)
                    txt[start:end] = model.encode_text(tokens).float().cpu().numpy()
                    txt.flush()
                    state.update(completed_texts=end, total_texts=len(texts), updated=time.time())
                    write_json(state_path, state)
                    if start % (batch_size * 100) == 0:
                        print(part, 'texts', end, len(texts), flush=True)
                for start in range(state['completed_images'], image_limit, image_batch):
                    end = min(start + image_batch, image_limit)
                    tensors = []
                    for path in images[start:end]:
                        with Image.open(path) as source:
                            tensors.append(preprocess(source.convert('RGB')))
                    img[start:end] = model.encode_image(torch.stack(tensors).to(device)).float().cpu().numpy()
                    img.flush()
                    state.update(completed_images=end, total_images=len(images), updated=time.time())
                    write_json(state_path, state)
                    if start % (image_batch * 100) == 0:
                        print(part, 'images', end, len(images), flush=True)
        state.update(status='complete' if args.scope=='all' else 'polaris_complete', finished=time.time())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(state_path, state)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['catalog', 'simcse', 'clip', 'siglip'], required=True)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--scope', choices=['all','polaris'], default='all')
    args = p.parse_args()
    if args.part == 'catalog':
        catalog(args.cache)
    else:
        extract(args)
