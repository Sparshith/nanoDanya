import json
import os
import subprocess
import signal
import sys

import torch
import pytest

from training.pretrain import train


def launch(recipe_path=None, output=None, stop=None):
    command = [sys.executable, '-m', 'torch.distributed.run', '--rdzv-backend=c10d',
               '--rdzv-endpoint=127.0.0.1:0', '--nproc_per_node=2']
    command += (['-m', 'training.train', '--recipe', str(recipe_path), '--output', str(output)]
                if recipe_path else ['tests/pretraining/distributed_worker.py'])
    if stop is not None:
        command += ['--stop-after-step', str(stop)]
    with subprocess.Popen(command, env=dict(os.environ, OMP_NUM_THREADS='1'),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          start_new_session=True) as process:
        try:
            output_text, _ = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output_text, _ = process.communicate()
            raise AssertionError(output_text)
        assert process.returncode == 0, output_text


@pytest.mark.skipif(sys.platform != 'linux', reason='torchrun integration is verified in the Linux training image')
@pytest.mark.parametrize('sampling', ['coverage', 'mixture'])
def test_distributed_training_matches_single_and_resumes(recipe, tmp_path, sampling):
    recipe['datasets'] = {'archive': recipe['datasets']['archive']}
    recipe['microbatch_size'] = 1
    if sampling == 'mixture':
        recipe['sampling'] = dict(mode='mixture', windows_per_source={'archive': 4}, steps=5)
        recipe['loss_weights'] = {'archive': 1.0}
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    train(recipe, tmp_path / 'single')
    launch(path, tmp_path / 'distributed')
    launch(path, tmp_path / 'resumed', stop=2)
    launch(path, tmp_path / 'resumed')
    single, parallel, resumed = [torch.load(tmp_path / name / 'latest.pt', weights_only=False)
                                 for name in ['single', 'distributed', 'resumed']]
    assert single['trained_tokens'] == parallel['trained_tokens'] == resumed['trained_tokens']
    assert single['step'] == parallel['step'] == resumed['step']
    for name in single['model']:
        torch.testing.assert_close(single['model'][name], parallel['model'][name], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(parallel['model'][name], resumed['model'][name], atol=0, rtol=0)
    assert parallel['history'] == resumed['history']


@pytest.mark.skipif(sys.platform != 'linux', reason='torchrun integration is verified in the Linux training image')
def test_distributed_gradients_match_with_unequal_counts_and_empty_rank():
    launch()
