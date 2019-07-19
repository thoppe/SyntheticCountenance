import glob, os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pylab as plt
import seaborn as sns
from scipy.stats import linregress

save_dest = 'averages'
save_dest_plots = 'averages/plots'

os.system(f'mkdir -p {save_dest}')
os.system(f'mkdir -p {save_dest_plots}')

cutoff = 10**4

featurelist = {
    
    #'data/AgeGender_score/' : ['age', 'female_score'],
    #'data/MEMNET_score/' : ["memnet_score"],
    #'data/NIMA_score/' : [ "NIMA_mean", "NIMA_std",],
    
    'data/FER_score/' :
    ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],

    'data/Aesthetics_score/' :
    ["Aesthetic", "BalancingElement", "ColorHarmony", "Content", "DoF",
     "Light", "MotionBlur", "Object", "Repetition", "RuleOfThrids",
     "Symmetry", "VividColor",],
}


for load_dest in featurelist:

    F_JSON = glob.glob(os.path.join(load_dest, "*.json"))

    data = []
    for f in tqdm(F_JSON):
        with open(f) as FIN:
            js = json.load(FIN)
            data.append(js)

        if len(data)>cutoff:
            break

    df = pd.DataFrame(data)
    df = df.dropna()

    feature = 'fear'
    k = 5
    bins = np.linspace(df[feature].min()*0.9, df[feature].max(), k)
    df['bin'] = pd.cut(df[feature],bins,labels=False)
    min_bin_count = df.groupby('bin').size().min()
    df['is_in_sample'] = False

    for i in range(k-1):
        idx = df[df['bin']==i].sample(n=min_bin_count).index
        df.loc[idx, 'is_in_sample'] = True

    df = df[df.is_in_sample]
    print(f"{feature} len df {len(df)}")    

    Z = []
    for f_img in tqdm(df.f_img):
        f_latent = f_img.replace('data/images', 'data/latent_vectors')
        f_latent = f_latent.replace('.jpg', '.npy')
        if not os.path.exists(f_latent):
            print(f_img, f_latent)
            Z.append(np.zeros(shape=(512,)))
            continue

            
        Z.append(np.load(f_latent))

    Z = np.array(Z)

    zero_idx = Z.sum(axis=1)==0
    if zero_idx.sum():
        Z = Z[~zero_idx]
        df = df[~zero_idx]
        print("Found some zero vectors wtf", zero_idx.sum())
                     

    for key in featurelist[load_dest]:
        
        f_save = os.path.join(save_dest, f'{key}.json')
        f_image = os.path.join(save_dest_plots, f'{key}.png')

        y = df[key].values

        '''
        # Do some filtering here to help fits?
        if load_dest == 'data/FER_score/':
            idx = (y>0.01) & (y<0.99)

            if idx.sum() < 1000:
                print(f"Not enough data for {key}")
                continue

        y = y[idx]
        ZF = Z[idx]
        '''
        ZF = Z

        # Best linear fit
        x, residuals, rank, s = np.linalg.lstsq(ZF, y, rcond=None)

        # Check out fits, see how well they regress back on
        Z_norm = (ZF.T / np.linalg.norm(ZF, axis=1)).T
        x_norm = x / np.linalg.norm(x)

        q = Z_norm.dot(x_norm)
        res = linregress(q, y)
        item = {
            'feature' : key,
            'dataset' : load_dest,
            'slope' : res.slope,
            'intercept' : res.intercept,
            'rvalue' : res.rvalue,
            'pvalue' : res.pvalue,
            'stderr' : res.stderr,
            'latent' : x.tolist(),
        }

        js = json.dumps(item, indent=2)

        xfit = np.linspace(q.min(), q.max(), 100)
        yfit = xfit*res.slope + res.intercept

        plt.figure()
        plt.scatter(q, y, s=1, alpha=0.5)
        plt.plot(xfit, yfit, 'r--', alpha=0.5,
                 label=f'$r^2=${res.rvalue**2:0.2f}')
        sns.despine()
        plt.title(f"{load_dest.split('_')[1]} {key}")
        plt.xlabel('latent projection')
        plt.ylabel(key)
        plt.ylim(y.min(), y.max())
        plt.legend()

        with open(f_save, 'w') as FOUT:
            FOUT.write(js)

        plt.savefig(f_image)
        plt.show()
        exit()
plt.show()

