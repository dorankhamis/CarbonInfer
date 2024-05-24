import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from pathlib import Path
from scipy.signal import (correlate, correlation_lags, find_peaks,
                          peak_prominences, peak_widths)
from sklearn.linear_model import HuberRegressor


from utils import score

fpath = "./output/default_params/"
run_meta = pd.read_csv(fpath + "/run_metadata.csv")

plot_outdir = './plots/default_params/'
Path(plot_outdir).mkdir(exist_ok=True, parents=True)

def plot_scatter(xvar, yvar, xerrvar=None, yerrvar=None, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(xvar, yvar, 'o', markersize=2.5)
    ax.errorbar(xvar, yvar, yerr=yerrvar, xerr=xerrvar,
             fmt='none', ecolor=None, elinewidth=None, capsize=None, barsabove=False,
             lolims=False, uplims=False, xlolims=False, xuplims=False,
             errorevery=1, capthick=None, alpha=0.4)
    xx = np.mean(ax.get_xlim())
    ax.axline((xx,xx), slope=1, linestyle='--', color='k')
    
    thisscore = score(yvar, xvar)
    newlab = f'R^2 = {np.around(thisscore, 3)}'
    xx = np.quantile(ax.get_xlim(), 0.05)
    yy = np.quantile(ax.get_ylim(), 0.925)
    ax.text(xx, yy, newlab, fontsize = 11)
    return ax

def efficiencies(y_pred, y_true):
    alpha = np.std(y_pred) / np.std(y_true)    
    beta_nse = (np.mean(y_pred) - np.mean(y_true)) / np.std(y_true)
    beta_kge = np.mean(y_pred) / np.mean(y_true)
    rho = np.corrcoef(y_pred, y_true)[1,0]
    NSE = -beta_nse*beta_nse - alpha*alpha + 2*alpha*rho # Nash-Sutcliffe
    KGE = 1 - np.sqrt((beta_kge-1)**2 + (alpha-1)**2 + (rho-1)**2) # Kling-Gupta
    return {'NSE':NSE, 'KGE':KGE}

def calc_metrics(df, var1, var2): # ytrue, ypred
    effs = efficiencies(df[[var1, var2]].dropna()[var2],
                        df[[var1, var2]].dropna()[var1]) # ypred, ytrue
    effs['mae'] = sk.metrics.mean_absolute_error(df[[var1, var2]].dropna()[var1],
                                              df[[var1, var2]].dropna()[var2])
    effs['medae'] = sk.metrics.median_absolute_error(df[[var1, var2]].dropna()[var1],
                                                  df[[var1, var2]].dropna()[var2])
    return effs

PLOT = False
alldat = pd.DataFrame()
for site_id in run_meta.site_id:
    try:
        dat = pd.read_parquet(fpath + f'/{site_id}/output_iter_0.parquet')
        alldat = pd.concat([alldat, dat], axis=0)
        if PLOT:
            subdat = (dat.resample('7D').sum()
                .reset_index()[['DATE_TIME', 'NEE', 'nee_gb']]
            )
            
            savename = plot_outdir + f'/timeseries_dayweek_{site_id}.png'
            fig, ax = plt.subplots(2, 1, figsize=(11,8), sharex=True)    
            sns.lineplot(x='DATE_TIME', y='value', hue='variable', ax=ax[0],
                         data=dat.reset_index()[['DATE_TIME', 'NEE', 'nee_gb']].melt(id_vars='DATE_TIME'))
            ax[0].fill_between(
                x=dat.index,
                y1=dat['NEE'] - dat['NEE_sd'],
                y2=dat['NEE'] + dat['NEE_sd'],
                alpha=0.4
            )
            sns.lineplot(x='DATE_TIME', y='value', hue='variable',
                         data=subdat.melt(id_vars='DATE_TIME'), ax=ax[1])    
            ax[0].set_ylabel("NEE, micromoles of CO2")
            ax[1].set_ylabel("NEE, micromoles of CO2")
            ax[0].set_title("daily")
            ax[1].set_title("weekly")
            fig.suptitle(site_id)
            #plt.show()
            plt.savefig(savename, bbox_inches='tight')
            plt.close()
            
            
            savename = plot_outdir + f'/scatter_dayweek_{site_id}.png'
            fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)
            ax[0] = plot_scatter(dat['NEE'], dat['nee_gb'], xerrvar=dat['NEE_sd'],
                                 fig=fig, ax=ax[0])
            ax[1] = plot_scatter(subdat['NEE'], subdat['nee_gb'],
                                 fig=fig, ax=ax[1])
            ax[0].set_title("daily")
            ax[1].set_title("weekly")
            ax[0].set_ylabel("JULES NEE")
            ax[0].set_xlabel("Observed NEE")
            ax[1].set_xlabel("Observed NEE")
            fig.suptitle(site_id)
            #plt.show()
            plt.savefig(savename, bbox_inches='tight')
            plt.close()
    except:
        continue

weekly = (alldat.drop('NEE_sd', axis=1)
    .groupby('site_id')
    .resample('7D').sum()
    .reset_index()
)
PLOT = False
all_site_metrics = pd.DataFrame()
day_metrics = pd.DataFrame()
huber = HuberRegressor(alpha=0.0, epsilon=1)
for site_id in run_meta.site_id:
    try:
        thisdat = alldat[alldat['site_id']==site_id]
        daymetrics = calc_metrics(thisdat, 'NEE', 'nee_gb')
        daymetrics['residual_bias'] = (thisdat['NEE'] - thisdat['nee_gb']).mean()
        day_metrics = pd.concat([
            day_metrics,
            pd.DataFrame(daymetrics, index=[0]).assign(site_id = site_id)],
            axis=0)
        
        # aggregate to weekly to find robust lag and residual bias
        thisweeklydat = weekly[weekly['site_id']==site_id]

        if PLOT:
            sns.lineplot(x='DATE_TIME', y='value', hue='variable',
                        data=thisweeklydat[['DATE_TIME','NEE', 'nee_gb']].melt(id_vars='DATE_TIME'))
            plt.title(site_id)
            plt.ylabel("NEE, micromoles of CO2")
            plt.show()

        thisweeklydat = thisweeklydat[['nee_gb', 'NEE']].dropna()
        corr_w = correlate(thisweeklydat['nee_gb']/ abs(thisweeklydat['nee_gb']).quantile(0.95),
                           thisweeklydat['NEE'] / abs(thisweeklydat['NEE']).quantile(0.95),                       
                           method="fft")
        lags_w = correlation_lags(len(thisweeklydat['nee_gb']), len(thisweeklydat['NEE']))
        lags_w = lags_w[(len(lags_w)//2 - 25):(len(lags_w)//2 + 25)]
        corr_w = corr_w[(len(corr_w)//2 - 25):(len(corr_w)//2 + 25)]
        if PLOT:
            plt.plot(lags_w, corr_w)
            plt.show()
        
        # model lag behind observation. negative means model is ahead of obs
        dynamic_lag = lags_w[np.argmax(corr_w)]
        
        # model metrics
        metrics_week = calc_metrics(thisweeklydat, 'NEE', 'nee_gb')
        metrics_day = calc_metrics(thisdat, 'NEE', 'nee_gb')
        metrics = {f'{k}_week' : metrics_week[k] for k in metrics_week}
        metrics = {**metrics, **metrics_day}
        metrics['lag'] = dynamic_lag  
                
        huber.fit(thisweeklydat['nee_gb'].values[..., None],
                  (thisweeklydat['nee_gb'] - thisweeklydat['NEE']).values)
        x = np.linspace(thisweeklydat['nee_gb'].min()-1, thisweeklydat['nee_gb'].max()+1, 7)
        coef_ = huber.coef_ * x + huber.intercept_
        metrics['residual_heteroscedasticity_week'] = huber.coef_[0]
        if PLOT:
            fig, ax = plt.subplots(figsize=(4,4))
            plt.hlines(0, x[0]-2, x[-1]+2, colors='k', linestyles='--', alpha=0.5)
            plt.scatter(thisweeklydat['nee_gb'], thisweeklydat['nee_gb'] - thisweeklydat['NEE'], s=3.2)        
            plt.plot(x, coef_)
            plt.xlim(thisweeklydat['nee_gb'].min()-0.25, thisweeklydat['nee_gb'].max()+0.25)
            plt.xlabel("JULES NEE")
            plt.ylabel("Residual (JULES - Observed)")
            savename = plot_outdir + f'/value_vs_residual_week_{site_id}.png'
            plt.savefig(savename, bbox_inches='tight')
            plt.close()
            #plt.show()
        
        huber.fit(thisdat['nee_gb'].values[..., None],
                  (thisdat['nee_gb'] - thisdat['NEE']).values)
        x = np.linspace(thisdat['nee_gb'].min()-1, thisdat['nee_gb'].max()+1, 7)
        coef_ = huber.coef_ * x + huber.intercept_
        metrics['residual_heteroscedasticity'] = huber.coef_[0]
        if PLOT:
            fig, ax = plt.subplots(figsize=(4,4))
            plt.hlines(0, x[0]-2, x[-1]+2, colors='k', linestyles='--', alpha=0.5)
            plt.scatter(thisdat['nee_gb'], thisdat['nee_gb'] - thisdat['NEE'], s=3.2)
            plt.plot(x, coef_, c='orange')
            plt.xlim(thisdat['nee_gb'].min()-0.25, thisdat['nee_gb'].max()+0.25)
            plt.xlabel("JULES NEE")
            plt.ylabel("Residual (JULES - Observed)")
            savename = plot_outdir + f'/value_vs_residual_day_{site_id}.png'
            plt.savefig(savename, bbox_inches='tight')
            plt.close()
            #plt.show()
                
        metrics['residual_bias'] = (thisdat['nee_gb'] - thisdat['NEE']).mean()
        metrics['residual_bias_week'] = (thisweeklydat['nee_gb'] - thisweeklydat['NEE']).mean()
        print(metrics)
        
        all_site_metrics = pd.concat([
            all_site_metrics,
            pd.DataFrame(metrics, index=[0]).assign(site_id = site_id)],
            axis=0)
    except:
        continue

site_ids = ['BD-OG',
 'MoorHouse',
 'Redmere1',
 'Conwy',
 'BakersFen',
 'Cairngorms',
 'AuchencorthMoss',
 'Anglesey1',
 'LincsMiscanthus',
 'EF-SA',
 'Rosedene',
 'Redmere2',
 'TadhamMoor',
 'Crosslochs',
 'Anglesey2',
 'EF-LN'
]

site_codes = ['BD-OG',
 'MRHS',
 'RDMR1',
 'CNWY',
 'BAKFN',
 'CAIGM',
 'AUCMS',
 'ANGSY1',
 'LNCMSC',
 'EF-SA',
 'ROSDN',
 'RDMR2',
 'TADMR',
 'CRSLCH',
 'ANGSY2',
 'EF-LN'
]
site_shorts = pd.DataFrame({'site_id':site_ids, 'site_code':site_codes})
all_site_metrics = all_site_metrics.merge(site_shorts, on='site_id')

fig, ax = plt.subplots(figsize=(6,5))
sns.barplot(x='mae', y='site_code', hue='mae',
            data=all_site_metrics, dodge=False, ax=ax,
            palette=sns.light_palette("seagreen", n_colors=24, as_cmap=False))
plt.legend([],[], frameon=False)
ax.set_xlabel("Mean absolute error")
savename = plot_outdir + f'/mae.png'
plt.savefig(savename, bbox_inches='tight')
plt.close()
plt.show()


fig, ax = plt.subplots(figsize=(6,5))
sns.barplot(x='lag', y='site_code', hue='lag',
            data=all_site_metrics, dodge=False, ax=ax,
            palette='vlag')
plt.legend([],[], frameon=False)
ax.set_xlabel("Lag, weeks")
savename = plot_outdir + f'/lag_weeks.png'
plt.savefig(savename, bbox_inches='tight')
plt.close()
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
sns.barplot(x='residual_heteroscedasticity', y='site_code',
            hue='residual_heteroscedasticity',
            data=all_site_metrics, dodge=False, ax=ax,
            palette='vlag')
plt.legend([],[], frameon=False)
ax.set_xlabel("Residual heteroscedasticity")
savename = plot_outdir + f'/residualheter_day.png'
plt.savefig(savename, bbox_inches='tight')
plt.close()
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
sns.barplot(x='residual_bias', y='site_code', hue='residual_bias',
            data=all_site_metrics, dodge=False, ax=ax,
            palette='vlag')
ax.set_xlabel("Residual bias, (JULES - Observation)")
plt.legend([],[], frameon=False)
savename = plot_outdir + f'/residualbias_day.png'
plt.savefig(savename, bbox_inches='tight')
plt.close()
plt.show()
