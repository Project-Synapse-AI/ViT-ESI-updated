"""Audit A5 real-data smokes, including normal Linux CUDA test exit and timing."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from a4_compare import METRICS, PROFILE, artifact, digest, require

SOURCE = "c5706f169bcda3d0693a17ce0d549bbd86660f8e"
PARAMS = {"1DCNN":3564876, "LSTM":3557700}


def audit(root, model, *, full=False):
    root=root.resolve()
    require(model in PARAMS, 'Unsupported matched model')
    kind, total, heldout_count, epochs_count = ('full',8000,2000,100) if full else ('smoke',100,25,2)
    j=json.loads((root/'job_manifest.json').read_text())
    require(j['status']=='passed' and j['args']['job']==kind and j['args']['model']==model,
            f'Not the expected passed {kind}')
    require(j['args']['a5_checks'] is True, 'Missing mandatory Linux CUDA check gate')
    env=j['environment']
    require(env['git_commit']==SOURCE and not env['git_dirty'], 'Wrong/dirty source')
    require(env['gpu_used']==['Tesla T4'] and env['cuda_visible_devices']=='0', 'Wrong GPU')
    require(j['training_profile_sha256']==PROFILE and not j['requirements_problems'], 'Profile/dependency mismatch')
    expected_stages={'install','a5_model_checks','generate_train8k_s0','generate_test2k_s1',
                     'fingerprint_train8k_s0','fingerprint_test2k_s1','build_model_folder',f'train_{model}',
                     'eval_train8k_s0_heldout','eval_test2k_s1'}
    require({s['stage'] for s in j['stages']}==expected_stages and all(s['returncode']==0 for s in j['stages']),
            'Failed/missing stages (including Linux CUDA tests)')
    paths=list(root.glob('training/**/run_manifest.json'));require(len(paths)==1,'Expected one training manifest')
    t=json.loads(paths[0].read_text());a=t['args']
    require(t['status']=='completed' and t['completed_epochs']==epochs_count,'Incomplete training')
    require(t['git_commit']==SOURCE and not t['git_dirty'],'Wrong training source')
    require(t['seed']==0 and not t['early_stopping_enabled'] and t['optimizer']=='Adam' and t['learning_rate']==.001,
            'Wrong seed/stopping/optimizer')
    fixed=dict(model=model,simu_name='train8k_s0',to_load=total,per_valid=.25,ep=epochs_count,bs=8,loss='cosine',
               scaler='linear',eeg_snr=5,n_times=500,checkpoint_monitor='validation_loss',require_cuda=True,
               num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=1,devices=1,precision='32-true',
               inter_layer=2604 if model=='1DCNN' else 4096,kernel_size=5,
               lstm_hidden_size=295 if model=='LSTM' else 85)
    for key,value in fixed.items():require(a[key]==value,f'Wrong training {key}')
    require(t['parameter_count']==PARAMS[model],'Wrong matched parameter count')
    require(len(t['train_indices'])==len(set(t['train_indices']))==total-heldout_count and
            len(t['heldout_indices'])==len(set(t['heldout_indices']))==heldout_count,'Wrong split sizes')
    require(set(t['train_indices']).isdisjoint(t['heldout_indices']) and
            set(t['train_indices']+t['heldout_indices'])==set(range(total)),'Bad split')
    chosen=t['checkpoints']['validation_loss']
    require(chosen['monitor']=='validation_loss' and 0<=chosen['epoch']<epochs_count,'Wrong selected checkpoint')
    require(digest(artifact(root,chosen['weights']))==chosen['weights_sha256'],'Selected weight hash mismatch')
    if full:
        require(set(t['checkpoints'])=={'train_loss','validation_loss'},'Missing checkpoint exports')
        for ckpt in t['checkpoints'].values():
            for key in ('weights','checkpoint'):
                require(digest(artifact(root,ckpt[key]))==ckpt[key+'_sha256'],f'Bad {key} hash')
        last=list(root.glob('training/**/last.ckpt'))
        require(len(last)==1 and last[0].stat().st_size>0,'Missing last checkpoint')
    for dataset,n in [('train8k_s0_heldout',heldout_count),('test2k_s1',2000 if full else 100)]:
        p=root/'eval'/dataset;s=json.loads((p/'summary.json').read_text());ea=s['args'];heldout=dataset=='train8k_s0_heldout'
        require(s['git_commit']==SOURCE and not s['git_dirty'] and s['ckpt_sha256']==chosen['weights_sha256'],
                'Eval source/weights mismatch')
        require(s['device']=='cuda' and s['n_evaluated']==n,'Wrong eval device/count')
        expected=dict(model=model,inter_layer=fixed['inter_layer'],kernel_size=5,lstm_hidden_size=fixed['lstm_hidden_size'],
                      seed=0,eeg_snr=5,legacy_order=False,heldout=heldout,to_load=total if heldout else n,per_valid=.25 if heldout else 1.,
                      simu_name='train8k_s0' if heldout else 'test2k_s1')
        for key,value in expected.items():require(ea[key]==value,f'Wrong evaluation {key}')
        with (p/'per_sample.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        ids=[int(r['index']) for r in rows]
        require(len(ids)==n and set(ids)==(set(t['heldout_indices']) if heldout else set(range(n))), 'Bad sample IDs')
        require(all(int(r['id'])==int(r['index'])+1 for r in rows),'Sample ID/index mismatch')
        for metric in METRICS:
            values=np.array([float(r[metric]) for r in rows])
            require(np.isfinite(values).all() and np.isclose(values.mean(),s['metrics'][metric]['mean'],rtol=1e-10,atol=1e-12),
                    f'Bad finite metric/mean {metric}')
    events=[json.loads(line) for line in (paths[0].parent/'progress.jsonl').read_text().splitlines()]
    epochs=[e for e in events if e['event']=='validation_end']
    require(len(epochs)==epochs_count and all(e['epoch_s']>0 for e in epochs),'Missing epoch timing')
    # 100 smoke examples versus8000 full examples:80x examples/epoch,100epochs.
    # This is deliberately a rough scheduling estimate, not a speed claim.
    estimate=None if full else epochs[-1]['epoch_s']*80*100
    return dict(status='passed',scope='full single-seed run' if full else 'smoke only; not a full scientific result',source_commit=SOURCE,
                model=model,parameters=PARAMS[model],epochs=epochs_count,linux_cuda_suite_exit=0,
                selected_weights_sha256=chosen['weights_sha256'],selected_weight_binary_verified=True,
                exported_checkpoint_hashes_verified=full,
                resumable_checkpoint_hashes_verified=False,epoch_timings=epochs,
                rough_full_training_seconds=estimate,
                scheduling_caution='Small smoke timing includes fixed overhead; confirm against live full epochs. '
                                   'Budget with margin and setup/evaluation allowance; no100epoch budget reduction.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=list(PARAMS),required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();result=audit(args.out,args.model);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
