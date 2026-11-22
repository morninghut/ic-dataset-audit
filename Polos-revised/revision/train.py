"""Train the scoring model on cached encoders, saving true initialization and optimizer state."""
import argparse
import fcntl
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from revision.common import CACHE, RUNS, REPO, FeatureStore, batches, load_rows, setup, metrics, write_json, read_json, sha256
from revision.model import make_model

def config_for(backbone, variant, seed):
    config = dict(backbone=backbone, variant=variant, seed=seed, normalization='sample',
                  standardize=True, mix_lr=.003, head_lr=.00003, head='mlp',
                  betas=[.9, .98], batch_size=64, accumulate=4, max_epochs=100,
                  patience=5, gradient_clip=1., dropout=.1, loss='sum_squared_error',
                  validation_batch_size=32, selection='kendall_tau_c', encoder_mode='eval')
    if variant == 'legacy_cached':
        config.update(normalization='batch', standardize=False, mix_lr=.00003)
    elif variant == 'sample_raw':
        config.update(standardize=False, mix_lr=.00003)
    elif variant == 'standard_low':
        config.update(mix_lr=.00003)
    elif variant == 'linear':
        config.update(head='linear', mix_lr=0., head_lr=.001)
    elif variant != 'full':
        raise ValueError(variant)
    return config

def score(model, store, rows, device='cuda', batch_size=32):
    result = np.empty(len(rows['candidate']), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for indices in batches(rows, batch_size):
            result[indices] = model(store.batch(rows, indices, device)).cpu().numpy()
    return result

def diagnostics(model, store, rows, device, initial=None):
    result = dict(layer_weights=model.mix.logits.softmax(0).detach().cpu().tolist(),
                  layer_logits=model.mix.logits.detach().cpu().tolist(), gamma=float(model.mix.gamma.detach()))
    if model.kind != 'mlp':
        return result
    selected = np.random.default_rng(20261104).choice(len(rows['candidate']), min(512, len(rows['candidate'])), replace=False)
    subset = {k: v[selected] for k, v in rows.items()}
    zs, activations = [], []
    model.eval()
    with torch.no_grad():
        for indices in batches(subset, 64):
            x = model.features(store.batch(subset, indices, device))
            z = model.head[0](x).flatten(0, 1).cpu()
            zs.append(z)
            activations.append(z.tanh())
    z, a = torch.cat(zs), torch.cat(activations)
    saturated = (z.abs() > 3).all(0)
    result.update(sampled_train_rows=len(selected), first_saturated_fraction=float((z.abs() > 3).float().mean()),
                  first_always_saturated_units=int(saturated.sum()),
                  first_always_saturated_indices=torch.where(saturated)[0].tolist(),
                  first_almost_constant_units=int((a.std(0, unbiased=False) < .001).sum()),
                  first_preactivation_rms=float(z.square().mean().sqrt()))
    if 0 < int(saturated.sum()) < saturated.numel():
        from sklearn.metrics import roc_auc_score
        current = model.head[0].weight[:, 2892].detach().abs().cpu().numpy()
        result['current_column_2892_saturation_auc'] = float(roc_auc_score(saturated.numpy(), current))
        if initial is not None:
            original = initial['head.0.weight'][:, 2892].abs().numpy()
            result['initial_column_2892_saturation_auc'] = float(roc_auc_score(saturated.numpy(), original))
    if initial is not None:
        before = initial['head.0.weight']
        after = model.head[0].weight.detach().cpu()
        result.update(first_weight_actual_delta_rms=float((after - before).square().mean().sqrt()),
                      first_weight_initial_std=float(before.std()), first_weight_final_std=float(after.std()),
                      first_weight_delta_initial_covariance=float(((after-before) * (before-before.mean())).mean()))
    return result

def save_torch(value, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)

def run(args):
    setup(args.seed)
    config = config_for(args.backbone, args.variant, args.seed)
    name = f'{args.variant}_{args.backbone}_s{args.seed}'
    output = args.output or RUNS / name
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'training_summary.json').exists():
        if read_json(output / 'training_summary.json')['status'] == 'complete':
            return
    if args.smoke:
        config.update(max_epochs=args.smoke_epochs, smoke=True)
    if args.epoch_limit is not None:
        config['max_epochs'] = args.epoch_limit
    store = FeatureStore(args.backbone, args.cache)
    stats = torch.load(args.stats or args.cache / f'{args.backbone}_stats.pt', map_location='cpu', weights_only=True)
    model = make_model(config, stats).to(args.device)
    train, val = load_rows(args.train_name, args.cache), load_rows(args.val_name, args.cache)
    if args.smoke:
        train = {k: v[:768] for k, v in train.items()}
        val = {k: v[:128] for k, v in val.items()}
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    groups = [{'params': model.head.parameters(), 'lr': config['head_lr']}]
    if config['mix_lr']:
        groups.append({'params': model.mix.parameters(), 'lr': config['mix_lr']})
    optimizer = torch.optim.Adam(groups, betas=tuple(config['betas']))
    start_epoch, best, stale, global_step = 0, -float('inf'), 0, 0
    if (output / 'last.pt').exists():
        last = torch.load(output / 'last.pt', map_location=args.device, weights_only=False)
        if {k:v for k,v in last['config'].items() if k != 'max_epochs'} != {k:v for k,v in config.items() if k != 'max_epochs'}:
            raise ValueError('Resume config does not match saved checkpoint')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        start_epoch, best, stale, global_step = last['epoch'] + 1, last['best'], last['stale'], last['global_step']
        torch.set_rng_state(last['torch_rng'].cpu())
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in last['cuda_rng']])
        np.random.set_state(last['numpy_rng'])
        random.setstate(last['python_rng'])
        initial = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True)
    else:
        save_torch(initial, output / 'initial.pt')
        write_json(output / 'initial_diagnostics.json', diagnostics(model, store, train, args.device))
    write_json(output / 'config.json', config)
    write_json(output / 'provenance.json', dict(catalog_sha256=sha256(args.cache / 'catalog.json'),
               train_name=args.train_name, validation_name=args.val_name,
               train_npz_sha256=sha256(args.cache / f'{args.train_name}.npz'),
               stats_sha256=sha256(args.stats or args.cache / f'{args.backbone}_stats.pt'),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               encoder_parameters_in_optimizer=0, all_references_retained=True,
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               source_files_sha256={p.name:sha256(p) for p in sorted((REPO/'revision').glob('*.py'))},
               torch_version=torch.__version__, numpy_version=np.__version__, cuda_version=torch.version.cuda,
               resumed_from=str(output / 'last.pt') if start_epoch else None))
    began = time.time()
    try:
        for epoch in range(start_epoch, config['max_epochs']):
            tick = time.time()
            model.train()
            groups = batches(train, config['batch_size'], seed=args.seed * 1000 + epoch)
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for i, indices in enumerate(groups):
                batch = store.batch(train, indices, args.device)
                pred = model(batch)
                loss = (pred - batch['target']).square().sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss / config['accumulate']).backward()
                total_loss += float(loss.detach())
                if (i + 1) % config['accumulate'] == 0 or i + 1 == len(groups):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                if i % 100 == 0 or i + 1 == len(groups):
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    write_json(output / 'status.json', dict(phase='training', pid=os.getpid(), epoch=epoch,
                               batch=i + 1, total_batches=len(groups), global_step=global_step,
                               seconds_per_batch=(time.time()-tick)/(i+1), updated=time.time()))
            training_seconds = time.time() - tick
            write_json(output / 'status.json', dict(phase='validation', pid=os.getpid(), epoch=epoch,
                       global_step=global_step, updated=time.time()))
            pred = score(model, store, val, args.device, config['validation_batch_size'])
            result = metrics(pred, val['target'])
            improved = result['kendall_tau_c'] > best
            if improved:
                best, stale = result['kendall_tau_c'], 0
            else:
                stale += 1
            epoch_result = dict(epoch=epoch, global_step=global_step, train_mse=total_loss/len(train['target']),
                                validation=result, training_seconds=training_seconds,
                                epoch_seconds=time.time()-tick, improved=improved, stale=stale,
                                layer_weights=model.mix.logits.softmax(0).detach().cpu().tolist(),
                                gamma=float(model.mix.gamma.detach()))
            snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                            epoch=epoch, best=best, stale=stale, global_step=global_step,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [],
                            numpy_rng=np.random.get_state(), python_rng=random.getstate(), validation=result)
            save_torch(snapshot, output / 'last.pt')
            if improved:
                save_torch(dict(model=model.state_dict(), config=config, epoch=epoch, validation=result), output / 'best.pt')
                np.savez(output / 'validation_predictions.npz', prediction=pred, target=val['target'], ids=val['ids'], image=val['image'])
            with (output / 'epochs.jsonl').open('a') as f:
                import json
                f.write(json.dumps(epoch_result) + '\n')
            print(epoch_result, flush=True)
            if stale >= config['patience']:
                break
        best_state = torch.load(output / 'best.pt', map_location=args.device, weights_only=True)
        model.load_state_dict(best_state['model'])
        diag = diagnostics(model, store, train, args.device, initial)
        if model.kind == 'mlp':
            before = set(read_json(output/'initial_diagnostics.json')['first_always_saturated_indices'])
            after = set(diag['first_always_saturated_indices'])
            diag['saturation_transition'] = dict(initial=len(before), final=len(after),
                    retained=len(before & after), no_longer_saturated=len(before-after), newly_saturated=len(after-before))
        write_json(output / 'best_diagnostics.json', diag)
        summary = dict(status='complete', best_epoch=best_state['epoch'], best_validation=best_state['validation'],
                       completed_epochs=epoch+1, global_step=global_step, seconds=time.time()-began,
                       checkpoint=str(output / 'best.pt'), smoke=args.smoke)
        write_json(output / 'training_summary.json', summary)
        write_json(output / 'status.json', dict(phase='complete', pid=os.getpid(), updated=time.time(), **summary))
    except BaseException as exc:
        write_json(output / 'status.json', dict(phase='failed', pid=os.getpid(), error=repr(exc), updated=time.time()))
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', choices=['clip', 'siglip'], required=True)
    p.add_argument('--variant', choices=['full', 'legacy_cached', 'sample_raw', 'standard_low', 'linear'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-epochs', type=int, default=2)
    p.add_argument('--epoch-limit', type=int)
    # Defaults reproduce the released Polaris runs; the names select another training pool.
    p.add_argument('--train-name', default='polaris_train')
    p.add_argument('--val-name', default='polaris_val')
    p.add_argument('--stats', type=Path)
    run(p.parse_args())
