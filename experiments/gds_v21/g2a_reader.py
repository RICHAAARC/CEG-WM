"""Bounded clean RGB reader pilot on saved G1 only. No generation or attacks."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from geometry import groups, addresses

ARMS = ('N', 'B0', 'B1', 'B2', 'P', 'P-E')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_split(source_ids, config):
    ids = sorted(set(source_ids), key=lambda s: (
        hashlib.sha256((config['split_domain'] + s).encode()).digest(), s))
    if len(ids) != 8:
        raise ValueError('Expected exactly eight G1 sources')
    return {s: ('train' if i < 4 else 'dev' if i < 6 else 'heldout')
            for i, s in enumerate(ids)}


def load_saved_g1(source, config):
    """Private labels are supervision/scoring only; network receives RGB alone."""
    source = Path(source).resolve()
    hashes = {}

    def checked(relative):
        p = (source / relative).resolve()
        if source not in p.parents or not p.is_file():
            raise ValueError(f'Missing or unsafe input: {relative}')
        hashes[relative] = digest(p)
        return p

    report = json.loads(checked('report.json').read_text())
    rows = report.get('attempts', [])
    if (report.get('stage') != 'G1' or report.get('status') != 'complete'
            or len(rows) != 48 or report.get('generation_attempts_started') != 48
            or any(r.get('status') != 'complete' or r.get('generator_forwards') != 100 for r in rows)):
        raise ValueError('Completed 48-unit G1 required; no automatic recovery')
    expected = {(f'g1-{i:03}', a) for i in range(8) for a in ARMS}
    if {(r['source_id'], r['arm']) for r in rows} != expected:
        raise ValueError('Unexpected source/arm roster')
    if report['files']['geometry.py'] != digest(Path(__file__).with_name('geometry.py')):
        raise ValueError('Public group mapping source differs from original G1')
    split = source_split([r['source_id'] for r in rows], config)
    data = []
    for r in sorted(rows, key=lambda x: (x['source_id'], ARMS.index(x['arm']))):
        unit = r['source_id'] + '_' + r['arm']
        if r['unit'] != unit:
            raise ValueError('Unit identity mismatch')
        p = checked(unit + '/image.png')
        with Image.open(p) as im:
            if im.size != (256, 256) or im.mode != 'RGB':
                raise ValueError(f'{unit}: RGB 256x256 required, no silent resize')
            rgb = np.array(im, dtype=np.uint8)
        with np.load(checked(unit + '/private_tokens.npz'), allow_pickle=False) as z:
            times, tokens = z['timesteps'], z['tokens']
        if (not np.array_equal(times, np.arange(99, -1, -1))
                or tokens.shape != (100, 1024) or tokens.dtype.kind not in 'iu'
                or np.any((tokens < 0) | (tokens > 4096))
                or np.any(tokens[-1] == 4096)):
            raise ValueError(f'{unit}: invalid saved Token trajectory')
        data.append(dict(unit=unit, source_id=r['source_id'], arm=r['arm'],
                         split=split[r['source_id']], rgb=rgb,
                         target=groups()[tokens[-1]].reshape(32, 32)))
    # Same image in different sources would defeat source isolation.
    image_hashes = [hashes[d['unit'] + '/image.png'] for d in data]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError('Duplicate RGB files: stop and inspect source independence')
    return data, hashes, split


class Block(nn.Sequential):
    def __init__(self, ci, co):
        super().__init__(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.ReLU())


class Reader(nn.Module):
    """The frozen paper architecture; 256 RGB -> 64x64x16 group logits."""
    def __init__(self):
        super().__init__()
        self.e1, self.e2 = Block(3, 32), Block(32, 64)
        self.e3, self.e4 = Block(64, 128), Block(128, 256)
        self.d3, self.d2, self.d1 = Block(384, 128), Block(192, 64), Block(96, 32)
        self.head = nn.Conv2d(32, 16, 1)

    def forward(self, rgb):
        x1 = self.e1(rgb)
        x2 = self.e2(F.avg_pool2d(x1, 2))
        x3 = self.e3(F.avg_pool2d(x2, 2))
        x4 = self.e4(F.avg_pool2d(x3, 2))
        def up(x, skip, block):
            return block(torch.cat([F.interpolate(x, size=skip.shape[-2:],
                       mode='bilinear', align_corners=False), skip], dim=1))
        x = up(up(up(x4, x3, self.d3), x2, self.d2), x1, self.d1)
        return F.avg_pool2d(self.head(x), 4)


def rgb_tensor(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1.


def target64(target):
    return np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)


def token_probs(prob64):
    p = np.asarray(prob64, dtype=np.float64)
    if p.shape != (64, 64, 16):
        raise ValueError('Expected 64x64x16 probabilities')
    return p.reshape(32, 2, 32, 2, 16).mean(axis=(1, 3))


def metrics(prob, truth):
    p = np.asarray(prob, dtype=np.float64).reshape(-1, 16)
    y = np.asarray(truth, dtype=np.int64).reshape(-1)
    if (len(p) != len(y) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=2e-6)
            or np.any((y < 0) | (y >= 16))):
        raise ValueError('Invalid probability/label input')
    pred = p.argmax(1)  # lowest class index breaks exact ties, also for baselines
    confusion = np.bincount(16*y + pred, minlength=256).reshape(16, 16)
    support = confusion.sum(1)
    present = support > 0
    result = dict(accuracy=float(np.mean(pred == y)),
        balanced_accuracy=float(np.mean(confusion.diagonal()[present] / support[present])),
        ce=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
        present_classes=int(present.sum()), tokens=int(len(y)))
    return result, confusion


def make_priors(data):
    selected = [d for d in data if d['split'] == 'train' and d['arm'] in ('N', 'B0')]
    if len(selected) != 8:
        raise ValueError('Exactly eight training images required')
    labels = np.stack([d['target'] for d in selected])
    position = np.ones((32, 32, 16), dtype=np.float64)
    for label in labels:
        position += np.eye(16)[label]
    position /= position.sum(-1, keepdims=True)
    freq = np.bincount(labels.ravel(), minlength=16).astype(float) + 1.
    freq /= freq.sum()
    return dict(uniform=np.full((32, 32, 16), 1/16),
                frequency=np.broadcast_to(freq, (32, 32, 16)).copy(),
                position=position,
                address=np.eye(16)[addresses().reshape(32, 32)])


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    outputs = {}
    for d in data:
        x = rgb_tensor(d['rgb']).unsqueeze(0).to(device)
        p = model(x).softmax(1)[0].permute(1, 2, 0).cpu().numpy()
        outputs[d['unit']] = p
    return outputs


def evaluate(data, outputs, priors, seed):
    rows, matrices = [], {}
    for d in data:
        peer_ids = sorted(x['source_id'] for x in data
                          if x['split'] == d['split'] and x['arm'] == d['arm'])
        peer = peer_ids[(peer_ids.index(d['source_id']) + 1) % len(peer_ids)] + '_' + d['arm']
        if peer == d['unit']:
            raise ValueError('Cross-source shuffle needs at least two sources per split')
        p = token_probs(outputs[d['unit']])
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
            f'{seed}:{d["unit"]}:spatial-control'.encode()).digest()[:8], 'big'))
        controls = dict(priors, reader=p, shuffled_rgb=token_probs(outputs[peer]),
                        shuffled_space=p.reshape(-1, 16)[rng.permutation(1024)].reshape(32, 32, 16))
        for name, q in controls.items():
            m, confusion = metrics(q, d['target'])
            rows.append(dict(unit=d['unit'], source_id=d['source_id'], arm=d['arm'],
                split=d['split'], method=name, control_peer=peer if name == 'shuffled_rgb' else '', **m))
            matrices[d['unit'] + '__' + name] = confusion
    return rows, matrices


def signal_flags(rows):
    units = sorted({r['unit'] for r in rows if r['split'] == 'heldout' and r['arm'] in ('N', 'B0')})
    flags = []
    for unit in units:
        table = {r['method']: r for r in rows if r['unit'] == unit}
        r = table['reader']
        ce_control = min(table[x]['ce'] for x in
                         ('uniform', 'frequency', 'position', 'shuffled_rgb', 'shuffled_space'))
        ba_control = max(table[x]['balanced_accuracy'] for x in
                         ('uniform', 'frequency', 'position', 'address'))
        flags.append(dict(unit=unit, ce_advantage=ce_control-r['ce'],
                          balanced_accuracy_advantage=r['balanced_accuracy']-ba_control,
                          signal=bool(r['ce'] < ce_control-1e-6 and r['balanced_accuracy'] > ba_control+1e-6)))
    if len(flags) != 4:
        raise ValueError('Expected exactly four heldout N/B0 units')
    return flags


def save_checkpoint(path, model, optimizer, step, config, input_hashes):
    tmp = Path(str(path) + '.tmp')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), step=step, config=config,
        input_hashes=input_hashes, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all()), tmp)
    os.replace(tmp, path)


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    table = pd.read_csv(out/'metrics.csv')
    train = pd.read_csv(out/'training.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train.step, train.loss, label='Training batch CE')
    dev = train.dropna(subset=['dev_ce'])
    ax.plot(dev.step, dev.dev_ce, 'o-', label='Dev N/B0 CE (no checkpoint selection)')
    ax.axhline(np.log(16), color='grey', linestyle='--', label='Uniform CE')
    ax.set(xlabel='Optimizer step', ylabel='Cross entropy', title='Clean reader: fixed training budget')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'learning_curve.png', dpi=140); plt.close(fig)
    subset = table[(table.split == 'heldout') & table.arm.isin(['N','B0'])]
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    for a, key in zip(ax, ['balanced_accuracy','ce']):
        # Address is a hard guess; its clipped CE is not comparable to soft priors.
        names = ['reader','frequency','position'] + (['address'] if key=='balanced_accuracy' else []) + ['shuffled_rgb','shuffled_space']
        for unit, frame in subset.groupby('unit'):
            a.plot(range(len(names)), frame.set_index('method').loc[names,key], 'o-', label=unit)
        a.set_xticks(range(len(names)), names, rotation=25, ha='right')
        a.set_ylabel(key); a.set_title('Heldout: two sources, four N/B0 images')
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'heldout_controls.png',dpi=140); plt.close(fig)


def run(source, out):
    root = Path(__file__).resolve().parent
    config = json.loads((root/'g2a_config.json').read_text())
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Output must be separate from G1')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(stage=config['stage'], status='running', config=config, step=0,
        training_attempts=0, new_generations=0, source=str(source), expected_units=48,
        quality_status='record only; not evaluated as passed',
        scope='Existing G1 clean small-data readability; no new confirmation, geometry, JPEG or FPR',
        automatic_expansion=False, heldout_evaluation='not_run')
    hashes = {}; data = []; model = optimizer = None; old_handler = None
    write_json(out/'report.json', status)
    try:
        # Fail-fast independently of the notebook UI and preserve original input hashes.
        def deadline(signum, frame):
            raise TimeoutError('90-minute whole-job budget exhausted; no extension')
        old_handler = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(config['max_run_seconds'])
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU runtime. No slow CPU research-training fallback.')
        torch.set_num_threads(2)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        status.update(gpu=torch.cuda.get_device_name(0),
                      versions={p:importlib.metadata.version(p) for p in ('torch','numpy','Pillow','pandas','matplotlib')},
                      code_hashes={p.name:digest(p) for p in root.iterdir() if p.suffix in ('.py','.json','.txt')})
        data, hashes, split = load_saved_g1(source, config)
        write_json(out/'input_hashes.json', hashes)
        write_json(out/'source_split.json', split)
        write_json(out/'config.json', config)
        (out/'protocol.txt').write_text((root/'G2A_README.txt').read_text())
        ledger = [{k:d[k] for k in ('unit','source_id','arm','split')} | {'evaluation':'not_run'} for d in data]
        write_json(out/'unit_ledger.json', ledger)
        status.update(source_split=split, input_files=len(hashes), expected_units=48,
                      train_images=8, dev_images=4, heldout_images=12)
        priors = make_priors(data)
        np.savez_compressed(out/'training_only_priors.npz', **priors)
        fit = [d for d in data if d['split']=='train' and d['arm'] in config['train_arms']]
        dev = [d for d in data if d['split']=='dev' and d['arm'] in config['train_arms']]
        x = torch.stack([rgb_tensor(d['rgb']) for d in fit])
        y = torch.from_numpy(np.stack([target64(d['target']) for d in fit])).long()
        device = torch.device('cuda')
        model = Reader().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        status.update(parameters=sum(p.numel() for p in model.parameters()), training_attempts=1)
        write_json(out/'report.json',status)
        torch.cuda.reset_peak_memory_stats()
        sampler = torch.Generator().manual_seed(config['seed'] + 1)
        with (out/'training.csv').open('w', newline='') as log:
            writer = csv.DictWriter(log, fieldnames=['step','loss','dev_ce','wall_seconds'])
            writer.writeheader(); log.flush()
            for step in range(1, config['max_steps']+1):
                model.train()
                ids = torch.randperm(len(fit), generator=sampler)[:config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[ids].to(device))
                loss = F.cross_entropy(logits.float(), y[ids].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                status['step'] = step
                if step == 1 or step % 20 == 0:
                    dev_ce = ''
                    if step % config['dev_every'] == 0:
                        pred = predict(model, dev, device)
                        dev_ce = float(np.mean([metrics(token_probs(pred[d['unit']]),d['target'])[0]['ce'] for d in dev]))
                    record = dict(step=step,loss=float(loss.detach().cpu()),dev_ce=dev_ce,
                                  wall_seconds=time.monotonic()-started)
                    writer.writerow(record); log.flush()
                    print(json.dumps(record), flush=True)
                if step % config['checkpoint_every'] == 0:
                    save_checkpoint(out/'latest.pt',model,optimizer,step,config,hashes)
                    status.update(wall_seconds=time.monotonic()-started,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    write_json(out/'report.json',status)
        # Final-step checkpoint only; no best-dev or heldout-selected checkpoint.
        outputs = predict(model, data, device)
        np.savez_compressed(out/'group_probabilities_64.npz', **outputs)
        rows, matrices = evaluate(data, outputs, priors, config['seed'])
        write_csv(out/'metrics.csv', rows)
        np.savez_compressed(out/'confusions.npz', **matrices)
        flags = signal_flags(rows)
        status.update(status='complete', heldout_evaluation='complete', signal_flags=flags,
            decision=('clean_signal_on_this_small_pilot_review_before_geometry' if all(f['signal'] for f in flags)
                      else 'clean_signal_not_established_under_this_small_data_budget_stop_and_review'))
        for row in ledger: row['evaluation'] = 'complete'
        write_json(out/'unit_ledger.json', ledger)
        plots(out)
    except BaseException as error:
        status.update(status='stopped', error=f'{type(error).__name__}: {error}',
                      decision='incomplete_stop_no_retry_no_expansion')
        (out/'failure.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        signal.alarm(0)
        if old_handler is not None: signal.signal(signal.SIGALRM, old_handler)
        if hashes:
            changed = [name for name, h in hashes.items() if not (source/name).is_file() or digest(source/name)!=h]
            status['original_files_unchanged'] = not changed
            status['changed_inputs'] = changed
            if changed:
                status.update(status='stopped',decision='input_integrity_failure_stop')
        status['wall_seconds'] = time.monotonic()-started
        write_json(out/'report.json', status)
        # Large recoverable checkpoint remains on Drive; small review archive is separate.
        with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.iterdir()):
                if path.is_file() and path.suffix != '.pt' and not path.name.endswith('.tmp'):
                    z.write(path, path.name)
    print(json.dumps(status,ensure_ascii=False,indent=2), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = run(args.source,args.output)
    if result['status'] != 'complete':
        raise SystemExit(2)
