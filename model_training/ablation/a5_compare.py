"""Audit parameter-matched baseline outputs and compare all five paired metrics."""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path

import numpy as np

from a2_compare import RemoveA2Options, git
from a4_compare import METRICS, SETS, SOURCE as CONTROL_SOURCE, compatible, digest, interval, load_run, require
from a5_smoke_audit import SOURCE, PARAMS, audit

NOTICE = ("Preliminary training-seed0 results. 95% descriptive paired sample-bootstrap intervals "
          "(5000 draws, RNGseed0) condition on these checkpoints, not training-seed variation, "
          "and are not multiplicity-adjusted. Held-out data selected weights; test2k_s1 is independent. "
          "Parameter matching does not match compute or establish pure architecture causation. "
          "Further training-set/seed and real-data verification is required.")


class RemoveA5Options(RemoveA2Options):
    def visit_Expr(self, node):
        v=node.value
        if (isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr=='add_argument'
                and v.args and isinstance(v.args[0],ast.Constant) and v.args[0].value=='-lstm_hidden_size'):
            return None
        return super().visit_Expr(node)

    def visit_Dict(self,node):
        for i,key in enumerate(node.keys):
            if isinstance(key,ast.Constant) and key.value=='hidden_size':node.values[i]=ast.Constant(value=85)
        return super().visit_Dict(node)

    def visit_Call(self,node):
        for kw in node.keywords:
            if kw.arg=='hidden_size':kw.value=ast.Constant(value=85)
        return super().visit_Call(node)


def source_audit():
    unchanged=['data_generation','anatomy','model_training/loaders.py','model_training/utils',
               'model_training/ablation/requirements-lock.txt','kaggle/training_profile.json',
               'model_training/models/cnn_1d.py','model_training/models/lstm.py']
    require(not git('diff','--name-only',CONTROL_SOURCE,SOURCE,'--',*unchanged).strip(),
            'Data/metrics/profile/requirements/baseline architecture source changed')
    for path in ('model_training/main_train.py','model_training/ablation/evaluate.py'):
        old=ast.parse(git('show',f'{CONTROL_SOURCE}:{path}').decode())
        new=RemoveA5Options().visit(ast.parse(git('show',f'{SOURCE}:{path}').decode()))
        require(ast.dump(old)==ast.dump(new),f'Non-option training/evaluation change: {path}')
    # Reuse the independently CUDA-tested A2 attention-default compatibility chain.
    require(git('show',f'{SOURCE}:model_training/models/vit.py')==
            git('show','a999ec5:model_training/models/vit.py'),'Unreviewed attention model change')
    return dict(control_source=CONTROL_SOURCE,variant_source=SOURCE,
                data_metrics_anatomy_baseline_architectures_profile_requirements_unchanged=True,
                train_eval_ast_equal_after_removing_reviewed_option_plumbing=True,
                attention_source_identical_to_default_parity_tested_A2=True,
                runner_difference='Mandatory matched-model CUDA-check subprocess before scientific stages; exit0 required')


def load_variant(root,model):
    report=audit(root,model,full=True);root=root.resolve()
    train=json.loads(next(root.glob('training/**/run_manifest.json')).read_text())
    evaluations={}
    for dataset in SETS:
        p=root/'eval'/dataset
        with (p/'per_sample.csv').open(newline='') as f:rows=sorted(csv.DictReader(f),key=lambda r:int(r['index']))
        evaluations[dataset]=dict(rows=rows,values=np.array([[float(r[m]) for m in METRICS] for r in rows]),
                                  summary=json.loads((p/'summary.json').read_text()),csv_sha256=digest(p/'per_sample.csv'))
    return dict(root=root,model=model,job=json.loads((root/'job_manifest.json').read_text()),
                train=train,eval=evaluations,audit=report)


def paired_compatible(control,variant):
    require(control['depth']==6 and control['train']['git_commit']==CONTROL_SOURCE,'Wrong original6-layer control')
    require(variant['train']['git_commit']==SOURCE,'Wrong variant source')
    model=variant['model'];require(model in PARAMS,'Unexpected model')
    require(variant['train']['parameter_count']==PARAMS[model] and
            abs(PARAMS[model]/control['train']['parameter_count']-1)<=.05,'Not parameter matched')
    normalized=copy.deepcopy(variant)
    argslist=[normalized['train']['args'],*[normalized['eval'][d]['summary']['args'] for d in SETS]]
    for args in argslist:
        require(args['model']==model,'Model mismatch')
        require(args['inter_layer']==(2604 if model=='1DCNN' else 4096) and args['kernel_size']==5,'Unplanned CNN width/kernel')
        require(args.pop('lstm_hidden_size')==(295 if model=='LSTM' else 85),'Unplanned LSTM width')
        require(args.pop('vit_mixer')=='attention' and args.pop('vit_window_radius')==5,'Unplanned ViT options')
        require(args['vit_depth']==6,'Unplanned reference options')
        args['model']='VIT';args['inter_layer']=4096
    compatible(control,normalized)  # validates all remaining settings/runtime/packages and paired sample metadata


def compare(control,variants):
    require(len({v['model'] for v in variants})==len(variants),'Duplicate architecture')
    records=[]
    for run in [control,*variants]:
        if run is not control:paired_compatible(control,run)
        for dataset in SETS:
            values=run['eval'][dataset]['values'];delta=values-control['eval'][dataset]['values']
            ci,dci=interval(values,5000,0),interval(delta,5000,0)
            for j,metric in enumerate(METRICS):
                records.append(dict(model=run.get('model','VIT'),parameters=run['train']['parameter_count'],
                                    dataset=dataset,n=len(values),metric=metric,mean=float(values[:,j].mean()),
                                    ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),delta=float(delta[:,j].mean()),
                                    delta_ci_low=float(dci[0,j]),delta_ci_high=float(dci[1,j])))
    return records


def write_outputs(runs,records,provenance,out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report=dict(notice=NOTICE,scope='CNN complete; LSTM full run pending quota' if len(runs)==2 else 'Initial matched comparison',
                analysis_commit=git('rev-parse','HEAD').decode().strip(),source_audit=provenance,
                script_sha256=digest(Path(__file__)),auditor_sha256=digest(Path(__file__).with_name('a5_smoke_audit.py')),
                runs=[dict(model=r.get('model','VIT'),source=r['train']['git_commit'],environment=r['train']['environment'],
                           profile_sha256=r['job']['training_profile_sha256'],
                           package_inventory_sha256=digest(r['root']/'pip_freeze.txt'),
                           checkpoint=r['train']['checkpoints']['validation_loss'],
                           csv_sha256={d:r['eval'][d]['csv_sha256'] for d in SETS}) for r in runs],records=records)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Parameter-matched baseline comparison','',report['scope'],'',NOTICE,'',
           '| Model | Dataset | Metric | Mean [95% interval] | Delta vs ViTESI [95% interval] |',
           '|---|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['metric']} | {r['mean']:.6g} "
                     f"[{r['ci_low']:.6g}, {r['ci_high']:.6g}] | {r['delta']:+.6g} "
                     f"[{r['delta_ci_low']:+.6g}, {r['delta_ci_high']:+.6g}] |")
    lines+=['','## Provenance','']
    for r in report['runs']:
        lines.append(f"- {r['model']}: source `{r['source']}`, selected weights `{r['checkpoint']['weights_sha256']}`, "
                     f"host `{r['environment']['platform']}`, packages `{r['package_inventory_sha256']}`.")
    lines+=['','Host kernel releases may differ; architecture/glibc,all other recorded runtime fields and packages match. '
            'Host environments are not identical; timing comparisons are limited. Compute estimates use the documented '
            'dense-matrix/convolution convention, not measured full hardware FLOPs/inference speed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,label,name in [('LE_mm','Localization error (mm)','localization'),('time_err_ms','Temporal error (ms)','temporal_error')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,dataset in zip(axes,SETS):
            rs=[r for r in records if r['dataset']==dataset and r['metric']==metric];y=np.array([r['mean'] for r in rs])
            ax.errorbar(np.arange(len(rs)),y,yerr=[y-[r['ci_low'] for r in rs],[r['ci_high'] for r in rs]-y],fmt='o',capsize=4)
            ax.set_xticks(np.arange(len(rs)),[r['model'] for r in rs]);ax.set(title=dataset,ylabel=label);ax.grid(alpha=.2)
        fig.suptitle('Parameter-matched baselines: seed0, evaluation-sample 95% intervals');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--cnn',type=Path,required=True);p.add_argument('--lstm',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();provenance=source_audit();control=load_run(args.control)
    variants=[load_variant(args.cnn,'1DCNN')]
    if args.lstm:variants.append(load_variant(args.lstm,'LSTM'))
    records=compare(control,variants);write_outputs([control,*variants],records,provenance,args.out)
    for run in variants:(run['root'].parents[1]/'audit.json').write_text(json.dumps(run['audit'],indent=2)+'\n')
    print('Audited',len(variants),'matched baselines; wrote',args.out)
