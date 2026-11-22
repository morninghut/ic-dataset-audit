"""Pearl's four frozen encoders, run once over the distinct pictures and texts.

The released pipelines in save_features/ walk one CSV at a time and re-encode a
text or a picture once per row, which would cost several times more here because
the same pictures and captions occur in many splits.  This script keeps the
authors' model loading, preprocessing and output format and only changes the
iteration: every distinct image file of pearl_data/images and every distinct
text of the Pearl CSVs is encoded once, and the result is written to
features/<model>/all/clip_{img,mt,ref}_features_{1..5}.npz, the layout
load_clip_features expects.

The mt and ref dictionaries hold the same keys (Pearl looks a reference up in
`ref_features` and a candidate in `mt_features`, and both occur in both roles).
For CLIP++ the dictionaries additionally hold the "A photo depicts " prefixed
keys, because the released evaluation code looks texts up with that prefix while
training looks them up raw.

Usage: extract_features.py <clippp|blip2|beit3|stella> [batch size]
"""
import os as _os
_ROOT = _os.environ.get('CAPAUDIT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_POLOS_HOME = _os.environ.get('CAPAUDIT_POLOS_HOME') or _os.path.join(_ROOT, 'Polos-revised')
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_ROOT)
DATA = ROOT / 'pearl_data'
PEARL = ROOT / 'external/Pearl'
CHECKPOINTS = ROOT / 'external/weights'
CHUNKS = 5
CLIP_PREFIX = 'A photo depicts '

csv.field_size_limit(10 ** 9)
sys.path.insert(0, str(PEARL / 'save_features'))


def items():
    """Distinct image file names and texts over every Pearl CSV."""
    images, texts = set(), set()
    for path in sorted((DATA / 'csv').glob('*.csv')):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                images.add(row['imgid'])
                texts.add(row['mt'])
                texts.update(ast.literal_eval(row['refs']))
    return sorted(images), sorted(texts)


def save(model_name, images, texts, image_features, text_features, prefixed=None, seconds=None):
    """Write the five-chunk dictionaries the Pearl loader merges."""
    out = ROOT / 'features' / model_name / 'all'
    out.mkdir(parents=True, exist_ok=True)
    image_parts = np.array_split(np.arange(len(images)), CHUNKS)
    text_parts = np.array_split(np.arange(len(texts)), CHUNKS)
    for chunk in range(CHUNKS):
        image_chunk = {images[i]: image_features[i] for i in image_parts[chunk]} if image_features is not None else {}
        text_chunk = {texts[i]: text_features[i] for i in text_parts[chunk]}
        if prefixed is not None:
            text_chunk.update({CLIP_PREFIX + texts[i]: prefixed[i] for i in text_parts[chunk]})
        np.savez(out / f'clip_img_features_{chunk + 1}.npz', features=image_chunk)
        np.savez(out / f'clip_mt_features_{chunk + 1}.npz', features=text_chunk)
        np.savez(out / f'clip_ref_features_{chunk + 1}.npz', features=text_chunk)
    with open(out / 'extraction.json', 'w') as f:
        json.dump(dict(model=model_name, images=len(images) if image_features is not None else 0,
                       texts=len(texts), prefixed=prefixed is not None, seconds=seconds), f)


def batched(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def progress(tag, done, total, started):
    elapsed = time.time() - started
    rate = done / max(elapsed, 1e-9)
    print(f'[{tag}] {done}/{total} {rate:.0f}/s eta {(total - done) / max(rate, 1e-9) / 60:.1f} min', flush=True)


def encode_images(tag, images, batch_size, preprocess, forward, workers=12):
    """Preprocess image files in worker processes, run the encoder on the main process.

    REUSE_IMAGE_FEATURES=<dir> takes the image features from an earlier run's
    npz files instead, for a re-run that only changes the text side.
    """
    reuse = os.environ.get('REUSE_IMAGE_FEATURES')
    if reuse:
        stored = {}
        for chunk in range(CHUNKS):
            with np.load(Path(reuse) / f'clip_img_features_{chunk + 1}.npz', allow_pickle=True) as f:
                stored.update(f['features'].item())
        print(f'[{tag}] reusing {len(stored)} image features from {reuse}', flush=True)
        return np.stack([stored[name] for name in images])

    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class Images(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            with Image.open(DATA / 'images' / images[index]) as image:
                return preprocess(image.convert('RGB'))

    loader = DataLoader(Images(), batch_size=batch_size, num_workers=workers, shuffle=False)
    out, started = [], time.time()
    for index, batch in enumerate(loader):
        with torch.no_grad():
            out.append(forward(batch.cuda()).cpu().numpy())
        if index % 50 == 0:
            progress(tag, index * batch_size, len(images), started)
    return np.concatenate(out)


def encode_texts(tag, texts, batch_size, forward):
    out, started = [], time.time()
    for index, batch in enumerate(batched(texts, batch_size)):
        with torch.no_grad():
            out.append(forward(batch))
        if index % 200 == 0:
            progress(tag, index * batch_size, len(texts), started)
    return np.concatenate(out)


def run_clippp(images, texts, batch_size=128):
    from pipelines.save_clippp_spica_features import load_clippp_model
    sys.path.append(str(PEARL / 'pacscore'))
    from models import clip  # noqa: F401  (clip.tokenize comes back from the loader)
    model, preprocess, tokenizer = load_clippp_model(
        'ViT-L/14', 4, CHECKPOINTS / 'PAC++_clip_ViT-L-14.pth', torch.device('cuda'))

    image_features = encode_images('clippp img', images, batch_size, preprocess,
                                   lambda x: model.encode_image(x.float()))

    def text_forward(batch):
        return model.encode_text(tokenizer(batch, truncate=True).cuda()).cpu().numpy()

    text_features = encode_texts('clippp txt', texts, batch_size, text_forward)
    prefixed = encode_texts('clippp pfx', [CLIP_PREFIX + t for t in texts], batch_size, text_forward)
    return image_features, text_features, prefixed


def run_blip2(images, texts, batch_size=32):
    from pipelines.save_blip2_spica_features import load_blip2_model
    model, vis_processors, text_processors = load_blip2_model(
        'blip2_image_text_matching', 'pretrain', torch.device('cuda'))

    def image_forward(batch):
        return model.extract_features({'image': batch}, mode='image').image_embeds_proj

    image_features = encode_images('blip2 img', images, batch_size, vis_processors['eval'], image_forward)

    def text_forward(batch):
        processed = [text_processors['eval'](t) for t in batch]
        out = model.extract_features({'text_input': processed}, mode='text')
        return out.text_embeds_proj[:, 0, :].detach().cpu().numpy()

    text_features = encode_texts('blip2 txt', texts, batch_size, text_forward)
    return image_features, text_features, None


def run_beit3(images, texts, batch_size=128):
    from transformers import XLMRobertaTokenizer
    from pipelines.save_beit3_spica_features import build_model, prepare_preprocess
    model = build_model(torch.device('cuda'), CHECKPOINTS / 'beit3_base_itc_patch16_224.pth')
    tokenizer = XLMRobertaTokenizer(str(CHECKPOINTS / 'beit3.spm'))
    preprocess = prepare_preprocess()

    image_features = encode_images('beit3 img', images, batch_size, preprocess,
                                   lambda x: model(image=x)[0])

    def text_forward(batch):
        tokens = tokenizer([t or '[EMPTY]' for t in batch], return_tensors='pt',
                           padding=True, truncation=True, max_length=77)
        return model(text_description=tokens['input_ids'].cuda())[1].cpu().numpy()

    # The released pipeline passes no padding mask to BEiT-3's text tower, so a
    # text's feature depends on the longest text in its batch.  Batching texts of
    # equal token length removes the padding altogether, which makes the features
    # reproducible and independent of the CSV order.  BEIT3_BATCH_PADDING=1
    # reproduces the released behaviour instead, for the comparison in the paper.
    if os.environ.get('BEIT3_BATCH_PADDING') == '1':
        text_features = encode_texts('beit3 txt', texts, batch_size, text_forward)
        return image_features, text_features, None

    by_length = {}
    for index, text in enumerate(texts):
        by_length.setdefault(len(tokenizer.tokenize(text or '[EMPTY]')[:75]), []).append(index)
    text_features, started, done = None, time.time(), 0
    for length in sorted(by_length):
        for batch in batched(by_length[length], batch_size):
            with torch.no_grad():
                values = text_forward([texts[i] for i in batch])
            if text_features is None:
                text_features = np.zeros((len(texts), values.shape[-1]), dtype=values.dtype)
            text_features[batch] = values
            done += len(batch)
        progress('beit3 txt', done, len(texts), started)
    return image_features, text_features, None


def run_stella(images, texts, batch_size=128):
    # The released loader needs xformers, which its own environment does not
    # install and whose current builds for torch 2.2 no longer match the model's
    # remote code.  The model card's documented alternative is used instead; it
    # masks padding in ordinary attention and gives the same vectors.
    from transformers import AutoModel, AutoTokenizer
    from pipelines.save_stella_spica_features import encode_texts as stella_encode
    path = CHECKPOINTS / 'stella_en_400M_v5'
    model = AutoModel.from_pretrained(path, trust_remote_code=True, use_memory_efficient_attention=False,
                                      unpad_inputs=False).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    projection = torch.nn.Linear(model.config.hidden_size, 768)
    state = torch.load(path / '2_Dense_768' / 'pytorch_model.bin', map_location='cpu')
    projection.load_state_dict({k.replace('linear.', ''): v for k, v in state.items()})
    projection = projection.cuda().eval()
    text_features = encode_texts(
        'stella txt', texts, batch_size,
        lambda batch: stella_encode(batch, tokenizer, model, projection, torch.device('cuda')))
    return None, text_features, None


def main(model_name, batch_size=None):
    images, texts = items()
    print(model_name, 'images', len(images), 'texts', len(texts), flush=True)
    run = dict(clippp=run_clippp, blip2=run_blip2, beit3=run_beit3, stella=run_stella)[model_name]
    started = time.time()
    arguments = (images, texts) if batch_size is None else (images, texts, int(batch_size))
    image_features, text_features, prefixed = run(*arguments)
    save(model_name, images, texts, image_features, text_features, prefixed, time.time() - started)
    print(model_name, 'done in', round((time.time() - started) / 60, 1), 'min', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
