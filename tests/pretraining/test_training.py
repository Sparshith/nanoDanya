import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from model_loading import load_model
from training.pretrain import train


@pytest.mark.parametrize('mode,model_type', [
    ('coverage', 'gpt'), ('mixture', 'gpt'), ('hybrid', 'gpt'), ('coverage', 'student')])
def test_real_training_resumes_exactly_and_exports_usable_checkpoints(recipe, tmp_path, mode, model_type):
    recipe['model_type'] = model_type
    if model_type == 'student':
        recipe['model_config']['n_embd'] = 64
    if mode != 'coverage':
        recipe['sampling'] = dict(mode='mixture', windows_per_source={'archive': 2, 'engine': 2})
        if mode == 'hybrid':
            recipe['sampling']['coverage_source'] = 'archive'
        else:
            recipe['sampling']['steps'] = 6
        recipe['loss_weights'] = {'archive': .3, 'engine': .7}
    output = tmp_path / 'resumed'
    train(recipe, output, stop_after_step=0)
    initial = torch.load(output / 'latest.pt', weights_only=False)

    def interrupt():
        if torch.load(output / 'latest.pt', weights_only=False)['step'] == 2:
            raise RuntimeError('preemption')

    if mode == 'mixture':
        with pytest.raises(RuntimeError, match='preemption'):
            train(recipe, output, commit=interrupt)
    else:
        paused = train(recipe, output, stop_after_step=2)
        assert paused['status'] == 'stopped'
    assert not (output / 'final.pt').exists()
    resumed = train(recipe, output)
    clean = train(recipe, tmp_path / 'clean')
    assert resumed == clean
    assert resumed['status'] == 'complete'
    a = torch.load(output / 'latest.pt', weights_only=False)
    b = torch.load(tmp_path / 'clean/latest.pt', weights_only=False)
    assert a['step'] == a['total_steps'] > 2
    assert a['sampler'] == b['sampler']
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    assert a['history'] == b['history']
    assert any(not torch.equal(a['model'][key], initial['model'][key]) for key in a['model'])
    assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])
    assert a['optimizer']['param_groups'] == b['optimizer']['param_groups']
    assert a['optimizer']['param_groups'][0]['lr'] == pytest.approx(recipe['min_learning_rate'])
    assert a['optimizer']['state']
    for key, state in a['optimizer']['state'].items():
        assert state['step'] == a['step']
        assert all(torch.equal(value, b['optimizer']['state'][key][name]) for name, value in state.items())
    for name in recipe['datasets']:
        if mode == 'coverage' or (mode == 'hybrid' and name == 'archive'):
            tokens = np.fromfile(Path(recipe['datasets'][name]) / 'train.bin', dtype='<u2')
            assert resumed['trained_tokens'][name] == np.count_nonzero(tokens != 0)
    best = torch.load(output / 'best.pt', weights_only=False)
    selected = min(a['history'], key=lambda entry: entry['selection_ce'])
    assert best['step'] == selected['step']
    assert best['best_metric'] == selected['selection_ce']
    model, _, _, _ = load_model(str(output / 'final.pt'), 'cpu')
    assert all(torch.equal(model.state_dict()[key], a['model'][key]) for key in a['model'])
    logits = model(torch.tensor([[0, 2, 3]]))
    assert logits.shape == (1, 3, recipe['model_config']['vocab_size'])
    assert torch.isfinite(logits).all()
    assert train(recipe, output) == resumed


@pytest.mark.parametrize('changed', ['recipe', 'data', 'tokenizer'])
def test_resume_rejects_incompatible_inputs(recipe, tmp_path, changed):
    output = tmp_path / 'run'
    train(recipe, output, stop_after_step=1)
    saved = (output / 'latest.pt').read_bytes()
    folder = Path(recipe['datasets']['engine'])
    if changed == 'recipe':
        recipe['seed'] += 1
    elif changed == 'data':
        tokens = np.fromfile(folder / 'train.bin', dtype='<u2')
        tokens[1] = 120
        tokens.tofile(folder / 'train.bin')
    else:
        meta = pickle.loads((folder / 'meta.pkl').read_bytes())
        meta['stoi']['2'], meta['stoi']['3'] = meta['stoi']['3'], meta['stoi']['2']
        (folder / 'meta.pkl').write_bytes(pickle.dumps(meta))
    with pytest.raises(ValueError, match='Recipe differs|dataset contents differ|tokenizer does not match'):
        train(recipe, output)
    assert (output / 'latest.pt').read_bytes() == saved


@pytest.mark.parametrize('preset', ['l16-all-once', 'l16-mixed-windows'])
def test_recipe_cli_trains_and_resumes_with_shipped_settings(recipe, tmp_path, preset):
    settings = json.loads((Path(__file__).resolve().parents[2] / 'training/recipes' / (preset + '.json')).read_text())
    settings.update({key: recipe[key] for key in ['model_config', 'batch_size', 'microbatch_size',
        'device', 'compile', 'val_batches', 'eval_interval', 'checkpoint_interval', 'warmup_steps']})
    settings['datasets'] = dict(zip(settings['datasets'], recipe['datasets'].values()))
    mixture = settings['sampling']['mode'] == 'mixture'
    if mixture:
        settings['sampling'].update(windows_per_source={'human': 2, 'leela': 2}, steps=6)
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(settings))
    output = tmp_path / 'run'
    command = [sys.executable, '-m', 'training.train', '--recipe', str(path), '--output', str(output)]
    if not mixture:
        command += ['--sources', 'human']
    env = {key: value for key, value in os.environ.items()
           if key not in ('TRAIN_RECIPE', 'TRAIN_SOURCES', 'MODAL_TIMEOUT', 'MODAL_CHECKPOINT_COMMIT_FILE')}
    subprocess.run(command + ['--stop-after-step', '1'], env=env, check=True, capture_output=True, text=True)
    paused = torch.load(output / 'latest.pt', weights_only=False)
    assert paused['optimizer']['param_groups'][0]['lr'] == settings['learning_rate']
    assert not (output / 'final.pt').exists()
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'complete'
    assert set(report['trained_tokens']) == ({'human', 'leela'} if mixture else {'human'})
    if not mixture:
        tokens = np.fromfile(Path(settings['datasets']['human']) / 'train.bin', dtype='<u2')
        assert report['trained_tokens']['human'] == np.count_nonzero(tokens != 0)
    final = torch.load(output / 'latest.pt', weights_only=False)
    assert final['optimizer']['param_groups'][0]['lr'] == pytest.approx(settings['min_learning_rate'])


def test_weighted_training_rejects_a_source_without_supervised_targets(recipe, tmp_path):
    np.array([0, *([2] * 64), 1], dtype='<u2').tofile(Path(recipe['datasets']['archive']) / 'train.bin')
    recipe['sampling'] = dict(mode='mixture', windows_per_source={'archive': 2, 'engine': 2}, steps=6)
    recipe['loss_weights'] = {'archive': .5, 'engine': .5}
    with pytest.raises(ValueError, match='supervised targets from every source'):
        train(recipe, tmp_path / 'run')
    assert not (tmp_path / 'run/final.pt').exists()
