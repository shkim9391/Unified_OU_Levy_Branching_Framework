from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize, linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, adjusted_rand_score

@dataclass(frozen=True)
class Config:
    seed:int=20260805; n_lineages:int=180; n_timepoints:int=34; t_end:float=12.0
    jump_time:float=4.1; branch_time:float=6.0; holdout_points:int=4
    ancestral_mu:float=0.1; ancestral_theta:float=0.85; ancestral_sigma:float=0.28; jump_size:float=1.2
    branch_mus:tuple=(-0.85,0.45,1.65); branch_thetas:tuple=(0.70,0.95,1.15); branch_sigmas:tuple=(0.34,0.28,0.31)
    branch_probs:tuple=(0.30,0.40,0.30); replicates:int=120

def ou_step(x,dt,mu,theta,sigma,rng):
    a=np.exp(-theta*dt); m=mu+(x-mu)*a; v=sigma**2*(1-np.exp(-2*theta*dt))/(2*theta)
    return float(rng.normal(m,np.sqrt(max(v,1e-12))))

def simulate(cfg,rng,jump_time=None):
    t=np.linspace(0,cfg.t_end,cfg.n_timepoints); jt=cfg.jump_time if jump_time is None else jump_time
    ji=int(np.argmin(abs(t-jt))); bi=int(np.argmin(abs(t-cfg.branch_time)))
    z=rng.choice(3,cfg.n_lineages,p=cfg.branch_probs); X=np.empty((cfg.n_lineages,cfg.n_timepoints))
    for n,b in enumerate(z):
        X[n,0]=rng.normal(cfg.ancestral_mu,cfg.ancestral_sigma/np.sqrt(2*cfg.ancestral_theta))
        for i in range(1,cfg.n_timepoints):
            if i<bi: mu,th,sg=cfg.ancestral_mu,cfg.ancestral_theta,cfg.ancestral_sigma
            else: mu,th,sg=cfg.branch_mus[b],cfg.branch_thetas[b],cfg.branch_sigmas[b]
            X[n,i]=ou_step(X[n,i-1],t[i]-t[i-1],mu,th,sg,rng)
            if i==ji: X[n,i]+=cfg.jump_size
    return t,X,z,ji,bi

def detect_jump(t,X,bi,cfg):
    s=np.full(len(t)-1,np.nan)
    for i in range(1,bi):
        dt=t[i]-t[i-1]; a=np.exp(-cfg.ancestral_theta*dt)
        m=cfg.ancestral_mu+(X[:,i-1]-cfg.ancestral_mu)*a
        v=cfg.ancestral_sigma**2*(1-np.exp(-2*cfg.ancestral_theta*dt))/(2*cfg.ancestral_theta)
        s[i-1]=np.median(abs(X[:,i]-m))/np.sqrt(v)
    return int(np.nanargmax(s))+1

def features(X,bi,end):
    P=X[:,bi:end]; q=np.arange(P.shape[1]); slopes=np.array([np.polyfit(q,r,1)[0] for r in P])
    return np.c_[P.mean(1),P[:,-1],slopes,P[:,-1]-X[:,bi]]

def infer(X,true,bi,end,cfg):
    gm=GaussianMixture(3,n_init=6,random_state=cfg.seed,reg_covar=1e-6).fit(features(X,bi,end))
    raw=gm.predict(features(X,bi,end)); probs=gm.predict_proba(features(X,bi,end))
    cm=confusion_matrix(true,raw,labels=np.arange(3)); r,c=linear_sum_assignment(-cm); mp={int(cc):int(rr) for rr,cc in zip(r,c)}
    lab=np.array([mp[int(v)] for v in raw]); ap=np.zeros_like(probs)
    for comp,br in mp.items(): ap[:,br]=probs[:,comp]
    return lab,ap

def nll(p,t,V):
    mu=p[0]; th=np.exp(p[1]); sg=np.exp(p[2]); dt=np.tile(np.diff(t),V.shape[0]); a=np.exp(-th*dt)
    xp=V[:,:-1].ravel(); xn=V[:,1:].ravel(); m=mu+(xp-mu)*a; v=sg**2*(1-np.exp(-2*th*dt))/(2*th)
    return .5*np.sum(np.log(2*np.pi*v)+(xn-m)**2/v)

def fit_params(t,X,lab,bi,end,cfg):
    rows=[]; tt=t[bi:end]
    for b in range(3):
        V=X[lab==b,bi:end]; p0=[V.mean(),np.log(.8),np.log(max(np.std(np.diff(V,axis=1))/np.sqrt(np.mean(np.diff(tt))),.1))]
        res=minimize(lambda p:nll(p,tt,V),p0,method='L-BFGS-B',bounds=[(-4,4),(np.log(.05),np.log(4)),(np.log(.03),np.log(2))])
        est=[res.x[0],np.exp(res.x[1]),np.exp(res.x[2])]; tru=[cfg.branch_mus[b],cfg.branch_thetas[b],cfg.branch_sigmas[b]]
        for name,a,z in zip(('mu','theta','sigma'),tru,est): rows.append({'branch':b+1,'parameter':name,'true':a,'estimated':z})
    return pd.DataFrame(rows)

def predict(t,X,lab,pt,end):
    pars={(b-1):{r.parameter:r.estimated for _,r in g.iterrows()} for b,g in pt.groupby('branch')}; rows=[]
    for n in range(len(X)):
        p=pars[int(lab[n])]; m=X[n,end-1]; v=0.0
        for i in range(end,len(t)):
            dt=t[i]-t[i-1]; a=np.exp(-p['theta']*dt); m=p['mu']+(m-p['mu'])*a
            v=v*a*a+p['sigma']**2*(1-np.exp(-2*p['theta']*dt))/(2*p['theta']); sd=np.sqrt(v)
            rows.append({'lineage':n+1,'time':t[i],'observed':X[n,i],'predicted':m,'lower':m-1.96*sd,'upper':m+1.96*sd})
    return pd.DataFrame(rows)

def jump_benchmark(cfg):
    ss=np.random.SeedSequence(cfg.seed+500).spawn(cfg.replicates); rows=[]
    small=Config(seed=cfg.seed,n_lineages=80,replicates=cfg.replicates)
    for k,s in enumerate(ss,1):
        rng=np.random.default_rng(s); target=rng.uniform(3,5); t,X,_,ji,bi=simulate(small,rng,target); dj=detect_jump(t,X,bi,small)
        rows.append({'replicate':k,'true_jump_time':t[ji],'detected_jump_time':t[dj],'time_error':t[dj]-t[ji]})
    return pd.DataFrame(rows)

def style(ax,l,title):
    ax.text(-.13,1.08,l,transform=ax.transAxes,fontsize=14,fontweight='bold',va='top'); ax.set_title(title,fontsize=11.5,fontweight='bold',pad=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=8.5)

def build(cfg,t,X,true,lab,ji,dj,bi,pt,jr,pred):
    fig,ax=plt.subplots(2,3,figsize=(14.2,8.8)); fig.subplots_adjust(left=.07,right=.985,bottom=.1,top=.88,wspace=.29,hspace=.38)
    end=cfg.n_timepoints-cfg.holdout_points; names=['Branch 1','Branch 2','Branch 3']
    for b in range(3):
        S=X[true==b]
        for r in S[:15]: ax[0,0].plot(t,r,color=f'C{b}',alpha=.15,lw=.8)
        ax[0,0].plot(t,S.mean(0),color=f'C{b}',lw=2.2,label=names[b])
    ax[0,0].axvline(t[ji],ls='--',lw=1.3); ax[0,0].axvline(t[bi],ls=':',lw=1.4); ax[0,0].axvspan(t[end],t[-1],alpha=.08); ax[0,0].legend(frameon=False,fontsize=8); ax[0,0].set(xlabel='Time',ylabel='Latent state'); style(ax[0,0],'A','Simulated full OULB trajectories')
    for b in range(3):
        S=X[lab==b]
        for r in S[:15]: ax[0,1].plot(t,r,color=f'C{b}',alpha=.15,lw=.8)
        ax[0,1].plot(t,S.mean(0),color=f'C{b}',lw=2.2)
    ax[0,1].axvline(t[dj],ls='--',lw=1.3); ax[0,1].axvline(t[bi],ls=':',lw=1.4); ax[0,1].axvspan(t[end],t[-1],alpha=.08); ax[0,1].set(xlabel='Time',ylabel='Latent state'); style(ax[0,1],'B','Inferred full OULB trajectories')
    a=ax[0,2]; a.scatter(jr.true_jump_time,jr.detected_jump_time,s=22,alpha=.55); lo=min(jr.true_jump_time.min(),jr.detected_jump_time.min())-.2; hi=max(jr.true_jump_time.max(),jr.detected_jump_time.max())+.2; a.plot([lo,hi],[lo,hi],ls='--',lw=1.6); med=np.median(abs(jr.time_error)); ex=np.mean(jr.time_error==0); a.text(.05,.95,f'Median |error| = {med:.2f}\nExact recovery = {ex:.2f}',transform=a.transAxes,va='top',bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor='.75')); a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='True jump time',ylabel='Detected jump time'); style(a,'C','Jump-time recovery')
    a=ax[1,0]; cm=confusion_matrix(true,lab,labels=np.arange(3)); cn=cm/np.maximum(cm.sum(1,keepdims=True),1); im=a.imshow(cn,vmin=0,vmax=1)
    for i in range(3):
        for j in range(3): a.text(j,i,f'{cm[i,j]}\n({cn[i,j]:.2f})',ha='center',va='center',fontsize=8.8)
    a.set_xticks(range(3),names,rotation=25,ha='right'); a.set_yticks(range(3),names); a.set(xlabel='Inferred branch',ylabel='True branch'); style(a,'D','Branch reconstruction'); fig.colorbar(im,ax=a,fraction=.046,pad=.04,label='Row-normalized proportion')
    a=ax[1,1]; mk={'mu':'o','theta':'s','sigma':'^'}
    for p in ('mu','theta','sigma'):
        q=pt[pt.parameter==p]; a.scatter(q.true,q.estimated,s=65,marker=mk[p],label={'mu':'$\\mu$','theta':'$\\theta$','sigma':'$\\sigma$'}[p])
        for _,r in q.iterrows(): a.annotate(f'B{int(r.branch)}',(r.true,r.estimated),xytext=(4,4),textcoords='offset points',fontsize=7.5)
    vals=np.r_[pt.true,pt.estimated]; lo,hi=vals.min()-.15,vals.max()+.15; a.plot([lo,hi],[lo,hi],ls='--',lw=1.6); rm=np.sqrt(np.mean((pt.estimated-pt.true)**2)); rr=np.corrcoef(pt.true,pt.estimated)[0,1]; a.text(.05,.95,f'$r$ = {rr:.2f}\nRMSE = {rm:.2f}',transform=a.transAxes,va='top',bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor='.75')); a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='True parameter value',ylabel='Estimated parameter value'); a.legend(frameon=False,fontsize=8); style(a,'E','Branch-specific parameter recovery')
    a=ax[1,2]; a.scatter(pred.observed,pred.predicted,s=18,alpha=.42); vals=np.r_[pred.observed,pred.predicted]; lo,hi=np.percentile(vals,[1,99]); pad=.08*(hi-lo); lo-=pad; hi+=pad; a.plot([lo,hi],[lo,hi],ls='--',lw=1.6); prm=np.sqrt(np.mean((pred.predicted-pred.observed)**2)); pr=np.corrcoef(pred.observed,pred.predicted)[0,1]; cov=np.mean((pred.observed>=pred.lower)&(pred.observed<=pred.upper)); a.text(.05,.95,f'$r$ = {pr:.2f}\nRMSE = {prm:.2f}\n95% coverage = {cov:.2f}',transform=a.transAxes,va='top',bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor='.75')); a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed held-out state',ylabel='Predicted held-out state'); style(a,'F','Held-out prediction')
    acc=np.mean(true==lab); bal=balanced_accuracy_score(true,lab); mf=f1_score(true,lab,average='macro'); ari=adjusted_rand_score(true,lab)
    fig.suptitle('Supplementary Figure S5. Joint recovery of the complete OULB model',fontsize=15,fontweight='bold',y=.985)
    summary=pd.DataFrame({'metric':['median_absolute_jump_time_error','exact_jump_time_recovery','branch_accuracy','balanced_accuracy','macro_f1','ari','parameter_correlation','parameter_rmse','prediction_correlation','prediction_rmse','prediction_coverage_95'],'value':[med,ex,acc,bal,mf,ari,rr,rm,pr,prm,cov]})
    return fig,summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--outdir',type=Path,default=Path('.')); p.add_argument('--stem',default='supplementary_figure_S5_joint_OULB_recovery'); p.add_argument('--replicates',type=int,default=120); p.add_argument('--seed',type=int,default=20260805); p.add_argument('--dpi',type=int,default=600); a=p.parse_args()
    cfg=Config(seed=a.seed,replicates=a.replicates); rng=np.random.default_rng(cfg.seed); t,X,true,ji,bi=simulate(cfg,rng); end=cfg.n_timepoints-cfg.holdout_points; dj=detect_jump(t,X,bi,cfg); lab,probs=infer(X,true,bi,end,cfg); pt=fit_params(t,X,lab,bi,end,cfg); pred=predict(t,X,lab,pt,end); jr=jump_benchmark(cfg); fig,summary=build(cfg,t,X,true,lab,ji,dj,bi,pt,jr,pred)
    a.outdir.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'): fig.savefig(a.outdir/f'{a.stem}.{ext}',dpi=a.dpi if ext=='png' else None,bbox_inches='tight')
    assign=pd.DataFrame({'lineage_id':np.arange(1,cfg.n_lineages+1),'true_branch':true+1,'inferred_branch':lab+1,'correct':true==lab}); [assign.__setitem__(f'prob_branch_{b+1}',probs[:,b]) for b in range(3)]
    assign.to_csv(a.outdir/'supplementary_figure_S5_lineage_assignments.csv',index=False); pt.to_csv(a.outdir/'supplementary_figure_S5_parameter_recovery.csv',index=False); jr.to_csv(a.outdir/'supplementary_figure_S5_jump_recovery.csv',index=False); pred.to_csv(a.outdir/'supplementary_figure_S5_prediction_results.csv',index=False); summary.to_csv(a.outdir/'supplementary_figure_S5_summary.csv',index=False); plt.close(fig); print('Saved Figure S5 outputs to',a.outdir.resolve())
if __name__=='__main__': main()
