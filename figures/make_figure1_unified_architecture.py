from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import numpy as np

DEFAULT_OUTDIR = Path("/Unified_OU_Levy_Branching_Framework/figures")
FIGSIZE = (18, 12)
DPI = 600

NAVY = "#173B6C"
BLUE = "#4C78A8"
LIGHT_BLUE = "#DCEAF7"
PURPLE = "#7A5195"
LIGHT_PURPLE = "#EEE5F4"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#C44E52"
LIGHT_RED = "#F6E1E2"
GRAY = "#6B7280"
LIGHT_GRAY = "#F4F5F7"
DARK = "#202124"
WHITE = "#FFFFFF"


def add_panel_box(ax, title: str, label: str, edge=BLUE, face=WHITE) -> None:
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.4,
        edgecolor=edge, facecolor=face, transform=ax.transAxes, clip_on=False))
    ax.text(0.025, 0.96, label, transform=ax.transAxes, fontsize=17,
            fontweight="bold", color=NAVY, va="top")
    ax.text(0.085, 0.96, title, transform=ax.transAxes, fontsize=14.5,
            fontweight="bold", color=NAVY, va="top")


def arrow(ax, xy1, xy2, color=GRAY, lw=1.8, mutation_scale=14, zorder=3):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>",
        mutation_scale=mutation_scale, linewidth=lw, color=color,
        transform=ax.transAxes, zorder=zorder))


def draw_gene_program(ax, cx, cy):
    n, cell = 5, 0.022
    vals = np.array([[.2,.4,.7,.9,.5],[.8,.5,.3,.7,.9],[.1,.6,.8,.4,.3],
                     [.9,.7,.2,.5,.8],[.4,.3,.9,.6,.2]])
    cmap = plt.get_cmap("RdYlGn_r")
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((cx+j*cell, cy+(n-1-i)*cell), cell, cell,
                transform=ax.transAxes, facecolor=cmap(vals[i,j]),
                edgecolor=WHITE, linewidth=.3))


def draw_cell_cluster(ax, cx, cy):
    offsets=[(0,0),(.04,.02),(-.04,.02),(.02,-.04),(-.03,-.04),(0,.06)]
    colors=[PURPLE,"#9C6DB0","#B58BC6",BLUE,"#6C8CC6","#C9A6D8"]
    for (dx,dy),color in zip(offsets,colors):
        ax.add_patch(Circle((cx+dx,cy+dy),.035,transform=ax.transAxes,
            facecolor=color,edgecolor=NAVY,linewidth=.7))


def draw_branch_tree(ax, cx, cy):
    nodes={"r":(cx,cy+.07),"a":(cx-.05,cy),"b":(cx+.05,cy),
           "c":(cx-.075,cy-.075),"d":(cx-.015,cy-.075),"e":(cx+.075,cy-.075)}
    edges=[("r","a"),("r","b"),("a","c"),("a","d"),("b","e")]
    for u,v in edges:
        x1,y1=nodes[u]; x2,y2=nodes[v]
        ax.plot([x1,x2],[y1,y2],transform=ax.transAxes,color=GRAY,lw=1.4,zorder=1)
    for (_, (x,y)),c in zip(nodes.items(),[BLUE,PURPLE,ORANGE,BLUE,PURPLE,GREEN]):
        ax.add_patch(Circle((x,y),.018,transform=ax.transAxes,
            facecolor=c,edgecolor=NAVY,linewidth=.6,zorder=2))


def draw_observations(ax, cx, cy):
    rng=np.random.default_rng(3)
    for color,xoff in [(BLUE,-.04),(PURPLE,0),(ORANGE,.04)]:
        pts=rng.normal(size=(24,2))*np.array([.014,.035])
        pts[:,0]+=cx+xoff; pts[:,1]+=cy
        ax.scatter(pts[:,0],pts[:,1],s=7,color=color,transform=ax.transAxes,
                   alpha=.85,edgecolors="none")
    ax.plot([cx-.09,cx+.09],[cy-.085,cy-.085],color=DARK,lw=.9,transform=ax.transAxes)
    for x,lab in zip([cx-.07,cx,cx+.07],[r"$t_1$",r"$t_2$",r"$t_n$"]):
        ax.plot([x,x],[cy-.09,cy-.08],color=DARK,lw=.8,transform=ax.transAxes)
        ax.text(x,cy-.115,lab,transform=ax.transAxes,fontsize=8,ha="center")


def panel_a(ax):
    add_panel_box(ax,"Multiscale state representation","A",edge=BLUE)
    xs=[.12,.37,.62,.86]; y=.54
    draw_gene_program(ax,xs[0]-.055,y-.05); draw_cell_cluster(ax,xs[1],y)
    draw_branch_tree(ax,xs[2],y+.02); draw_observations(ax,xs[3],y)
    for i in range(3): arrow(ax,(xs[i]+.11,y),(xs[i+1]-.11,y))
    titles=["Molecular\nprograms","Cellular\nstates","Ecological contexts\nand lineages","Longitudinal\nobservations"]
    subtitles=["pathway activity\nand signatures","state compositions\nand cell programs","branch-specific\nmultiscale structure","irregular, noisy,\npossibly multimodal"]
    dims=[r"$p$",r"$q$",r"$K$",r"$N$ observations"]
    for x,t,s,d in zip(xs,titles,subtitles,dims):
        ax.text(x,.79,t,transform=ax.transAxes,ha="center",va="center",fontsize=11.5,fontweight="bold")
        ax.text(x,.26,s,transform=ax.transAxes,ha="center",va="center",fontsize=9.5,color=GRAY,linespacing=1.25)
        ax.text(x,.12,d,transform=ax.transAxes,ha="center",fontsize=10,color=NAVY)
    ax.add_patch(FancyBboxPatch((.035,.035),.93,.08,boxstyle="round,pad=.01",
        transform=ax.transAxes,facecolor=LIGHT_BLUE,edgecolor=BLUE,linewidth=.8))
    ax.text(.5,.075,r"Latent multiscale state $X_t\in\mathbb{R}^{q}$; observed data are noisy projections.",
            transform=ax.transAxes,ha="center",va="center",fontsize=10.2,color=NAVY)


def panel_b(ax):
    add_panel_box(
        ax,
        "Unified latent process",
        "B",
        edge=PURPLE,
    )

    ax.text(
        .5,
        .79,
        (
            r"$\mathrm{d}X_t="
            r"\Theta_{B_t}\{\mu_{B_t}(t)-X_t\}\,\mathrm{d}t"
            r"+\Sigma_{B_t}\,\mathrm{d}W_t"
            r"+\mathrm{d}J_t$"
        ),
        transform=ax.transAxes,
        fontsize=22,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=.35",
            facecolor=WHITE,
            edgecolor=PURPLE,
            linewidth=1.4,
        ),
    )

    items = [
        (
            r"$\Theta_{B_t}$",
            "branch-specific restoring matrix",
            BLUE,
        ),
        (
            r"$\mu_{B_t}(t)$",
            "possibly time-varying attractor",
            PURPLE,
        ),
        (
            r"$\Sigma_{B_t}$",
            "branch-specific diffusion scale",
            ORANGE,
        ),
        (
            r"$W_t$",
            "Wiener process",
            GREEN,
        ),
        (
            r"$J_t$",
            "compound-Poisson / Lévy-like jump process",
            RED,
        ),
    ]

    for i, (sym, desc, color) in enumerate(items):
        y = .58 - i * .085

        ax.text(
            .09,
            y,
            sym,
            transform=ax.transAxes,
            fontsize=13,
            color=color,
            fontweight="bold",
        )

        ax.text(
            .28,
            y,
            desc,
            transform=ax.transAxes,
            fontsize=10.5,
            color=DARK,
        )

    ax.add_patch(
        FancyBboxPatch(
            (.04, .04),
            .92,
            .13,
            boxstyle="round,pad=.012",
            transform=ax.transAxes,
            facecolor=LIGHT_PURPLE,
            edgecolor=PURPLE,
            linewidth=.8,
        )
    )

    ax.text(
        .5,
        .105,
        (
            "Deterministic reversion + continuous diffusion + "
            "discontinuous jumps,\n"
            "all conditional on the current branch state."
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=PURPLE,
    )


def panel_c(ax):
    add_panel_box(ax,"Therapy-dependent attractor shift","C",edge=ORANGE)
    ax.text(.5,.80,r"$\mu_{B_t}(t)=\mu_{0,B_t}+\Delta_{B_t}u(t)$",transform=ax.transAxes,
        fontsize=21,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=.35",facecolor=WHITE,edgecolor=ORANGE,linewidth=1.4))
    for y,sym,desc,color in [(.65,r"$\mu_{0,B_t}$","pre-therapy branch attractor",PURPLE),
                             (.55,r"$\Delta_{B_t}$","therapy-induced displacement",ORANGE),
                             (.45,r"$u(t)$","treatment exposure or dose",NAVY)]:
        ax.text(.07,y,sym,transform=ax.transAxes,fontsize=12.5,color=color)
        ax.text(.28,y,desc,transform=ax.transAxes,fontsize=10.5)
    ax.plot([.13,.43],[.23,.23],color=PURPLE,lw=2.2,transform=ax.transAxes)
    ax.plot([.43,.43],[.23,.33],color=PURPLE,lw=2.2,transform=ax.transAxes)
    ax.plot([.43,.87],[.33,.33],color=PURPLE,lw=2.2,transform=ax.transAxes)
    ax.text(.41,.16,r"$t_{\mathrm{treat}}$",transform=ax.transAxes,fontsize=9)
    ax.text(.09,.22,"0",transform=ax.transAxes,fontsize=9); ax.text(.88,.32,"1",transform=ax.transAxes,fontsize=9)
    ax.text(.89,.19,r"$t$",transform=ax.transAxes,fontsize=10); ax.text(.07,.30,r"$u(t)$",transform=ax.transAxes,fontsize=11,color=PURPLE)
    ax.text(.5,.07,"Step functions, continuous dose functions, and branch-specific responses are all allowed.",
            transform=ax.transAxes,ha="center",fontsize=9.6,color=GRAY)


def panel_d(ax):
    add_panel_box(ax,"Branch-state (lineage) process","D",edge=GREEN)
    ax.text(.06,.80,r"$B_t\in\{1,\ldots,K\},\qquad B_t\sim\mathrm{CTMC}(Q)$",transform=ax.transAxes,fontsize=17)
    
    ax.text(.55,.65,r"$q_{k\ell}\geq0\;(k\neq\ell)$",transform=ax.transAxes,fontsize=12,color=GREEN)
    ax.text(.55,.55,r"$q_{kk}=-\sum_{\ell\neq k}q_{k\ell}$",transform=ax.transAxes,fontsize=12)
    centers=[(.62,.27),(.78,.34),(.78,.16),(.92,.25)]
    for i,((x,y),c) in enumerate(zip(centers,[BLUE,PURPLE,ORANGE,GREEN])):
        ax.add_patch(Circle((x,y),.035,transform=ax.transAxes,facecolor=c,edgecolor=NAVY,linewidth=.8))
        ax.text(x,y,str(i+1),transform=ax.transAxes,ha="center",va="center",fontsize=9,color=WHITE,fontweight="bold")
    for i in range(len(centers)-1): arrow(ax,centers[i],centers[i+1],color=GRAY,lw=1.2,mutation_scale=10)
    ax.text(.06,.09,"Branch transitions represent lineage changes, niche shifts,\nstate conversion, or ecological reorganization.",
            transform=ax.transAxes,fontsize=10.2,color=GRAY)


def panel_e(ax):
    add_panel_box(ax,"Observation model","E",edge=BLUE)
    ax.text(.5,.76,r"$Y_i=H_iX_{t_i}+\varepsilon_i,\qquad \varepsilon_i\sim\mathcal{N}(0,R_i)$",
        transform=ax.transAxes,fontsize=20,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=.35",facecolor=WHITE,edgecolor=BLUE,linewidth=1.4))
    for i,(sym,desc) in enumerate([(r"$Y_i$","observed data at time, site, or platform"),
                                   (r"$H_i$","projection / loading / cross-scale mapping"),
                                   (r"$R_i$","measurement-error covariance")]):
        y=.55-i*.105; ax.text(.11,y,sym,transform=ax.transAxes,fontsize=13,color=NAVY)
        ax.text(.27,y,desc,transform=ax.transAxes,fontsize=10.5)
    ax.add_patch(FancyBboxPatch((.05,.07),.90,.15,boxstyle="round,pad=.012",
        transform=ax.transAxes,facecolor=LIGHT_BLUE,edgecolor=BLUE,linewidth=.8))
    ax.text(.5,.145,"Supports irregular sampling, heterogeneous modalities,\nmissing observations, and known measurement variance.",
            transform=ax.transAxes,ha="center",va="center",fontsize=10.2,color=NAVY)


def panel_f(ax):
    add_panel_box(ax,"Nested special cases","F",edge=RED)
    rows=[("Brownian motion",r"$\Theta=0,\ J_t=0$","diffusion only"),
          ("Standard OU",r"$\Theta>0,\ J_t=0$","mean reversion"),
          ("Treatment-shifted OU",r"$\mu(t)=\mu_0+\Delta u(t)$","therapy response"),
          ("OU with jumps",r"$J_t\neq0$","rare discontinuities"),
          ("OU with branching",r"$B_t\ \mathrm{varies}$","state switching"),
          ("Full OULB",r"$\Theta_{B_t},\mu_{B_t}(t),\Sigma_{B_t},J_t$","all mechanisms")]
    ypos=np.linspace(.78,.19,len(rows))
    for i,((name,cond,meaning),y) in enumerate(zip(rows,ypos)):
        face=LIGHT_RED if i==len(rows)-1 else LIGHT_GRAY; edge=RED if i==len(rows)-1 else "#D0D3D8"
        ax.add_patch(FancyBboxPatch((.08,y-.045),.84,.075,boxstyle="round,pad=.01",
            transform=ax.transAxes,facecolor=face,edgecolor=edge,linewidth=1.0))
        ax.text(.11,y,str(i),transform=ax.transAxes,fontsize=10,color=WHITE,ha="center",va="center",
                bbox=dict(boxstyle="circle,pad=.22",facecolor=RED if i==len(rows)-1 else GRAY,edgecolor="none"))
        ax.text(.18,y,name,transform=ax.transAxes,fontsize=10.5,fontweight="bold" if i==len(rows)-1 else "normal",
                color=RED if i==len(rows)-1 else DARK,va="center")
        ax.text(.50,y,cond,transform=ax.transAxes,fontsize=10,va="center")
        ax.text(.75,y,meaning,transform=ax.transAxes,fontsize=9.5,color=RED if i==len(rows)-1 else GRAY,va="center")
    ax.text(.03,.50,"Simpler",transform=ax.transAxes,fontsize=9,color=GRAY,rotation=90,va="center")
    arrow(ax,(.045,.75),(.045,.17),color=PURPLE,lw=2.2,mutation_scale=18)
    ax.text(.015,.13,"More general",transform=ax.transAxes,fontsize=9,color=PURPLE,rotation=90,va="center")
    ax.text(.5,.065,"The full model is a coherent superset of all simpler cases.",
            transform=ax.transAxes,ha="center",fontsize=10.2,color=RED,fontweight="bold")


def add_takeaway(fig):
    ax=fig.add_axes([.03,.025,.94,.105]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.012,rounding_size=.02",
        transform=ax.transAxes,facecolor="#FFF8E8",edgecolor="#D9A441",linewidth=1.1))
    ax.text(.02,.70,"Key takeaway",transform=ax.transAxes,fontsize=14,fontweight="bold",color="#8A4B08")
    ax.text(.02,.34,"OULB unifies constrained stochastic evolution, therapy-driven attractor shifts, rare jumps, lineage branching, cross-scale mapping, and noisy observation within one interpretable generative framework.",
            transform=ax.transAxes,fontsize=11.2,color=DARK)


def build_figure(outdir: Path, dpi: int=DPI) -> None:
    outdir.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=FIGSIZE,constrained_layout=False)
    fig.text(.5,.965,"Figure 1. Unified Ornstein–Uhlenbeck–Lévy–Branching architecture",
             ha="center",va="top",fontsize=22,fontweight="bold",color=DARK)
    fig.text(.5,.93,"A general stochastic framework for multiscale cancer evolution under therapy, jumps, branching, and observation error",
             ha="center",va="top",fontsize=13.5,color=GRAY,style="italic")
    gs=fig.add_gridspec(2,3,left=.03,right=.97,top=.89,bottom=.17,
                        width_ratios=[1.05,1.05,1.20],height_ratios=[1.08,1.00],wspace=.025,hspace=.04)
    panel_a(fig.add_subplot(gs[0,0])); panel_b(fig.add_subplot(gs[0,1])); panel_c(fig.add_subplot(gs[0,2]))
    panel_d(fig.add_subplot(gs[1,0])); panel_e(fig.add_subplot(gs[1,1])); panel_f(fig.add_subplot(gs[1,2]))
    add_takeaway(fig)
    stem=outdir/"Figure1_unified_OULB_architecture"
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"),dpi=dpi,bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {stem.with_suffix('.svg')}")
    print(f"[SAVED] {stem.with_suffix('.pdf')}")
    print(f"[SAVED] {stem.with_suffix('.png')}")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Generate Figure 1: unified OULB architecture and nested special cases.")
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTDIR)
    p.add_argument("--dpi",type=int,default=DPI)
    return p.parse_args()


def main() -> None:
    args=parse_args(); build_figure(args.output_dir.expanduser(),dpi=args.dpi)


if __name__=="__main__":
    main()
