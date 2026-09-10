import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# sample data

data = {
    'customer':['rehman','faizan','asad','hassan','hussain'],
    'age':[25,50,45,35,27],
    'spending':[100,200,300,400,500]
}

df  = pd.DataFrame(data)

x = df[['age','spending']]

model = KMeans(n_clusters=2,random_state=42,n_init=10)
df['group'] = model.fit_predict(x)


plt.figure(figsize=(6,5))

for group in df['group'].unique():
    group_data = df[df['group']==group]
    plt.scatter(group_data['age'],group_data['spending'],label = f'group{group}')
plt.xlabel('age')
plt.ylabel('spending')
plt.title("customer segments (K_Means)")
plt.legend()
plt.grid(True)   
plt.show() 
    
print(df)