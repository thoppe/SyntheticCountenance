import pandas as pd
import pylab as plt
import seaborn as sns

df = pd.read_csv("latent_gender_and_emotion_training.csv")

sns.distplot(df.woman,norm_hist=False,bins=20,kde=False)
sns.distplot(df.man,norm_hist=False,bins=20,kde=False)

plt.show()
