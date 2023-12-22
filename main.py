import numpy.random
import scipy.io as scio
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

filepath='D:\大学\科研资料\脑智明珠\课程\脑智信息数据分析与模拟\期末小组汇报\实验课2\实验课2\DataProcessing'
file=os.listdir(filepath)

for i in range(len(file)):  # 数据合并
    path=os.path.join(filepath,file[i])
    Predata=scio.loadmat(path)
    if i==0:
        label=Predata['datalabel']
        data=Predata['pre']
    else:
        label=np.concatenate((label,Predata['datalabel']))
        data=np.concatenate((data,Predata['pre']),axis=2)

data=data.transpose(2,0,1)

# 计算MRS
data_std=np.zeros((len(label),12))
for i in range(len(label)):
    data_temp=data[i,:,:]**2
    data_std[i,:]=np.sqrt(np.mean(data_temp,axis=1))

# 打乱数据集
idx=np.arange(len(label))
idx_shuffle=np.arange(len(label))
numpy.random.shuffle(idx_shuffle)
data_shuffle=data_std[idx_shuffle,:]
label_shuffle=label[idx_shuffle]

train_data,test_data,train_label,test_label=train_test_split(data_shuffle,label_shuffle,test_size=0.2)

# 数据标准化
stdScaler=StandardScaler().fit(train_data)
train_data_std=stdScaler.transform(train_data)
test_data_std=stdScaler.transform(test_data)

RF=RandomForestClassifier(n_estimators=140,random_state=42) #n_estimators=150
knn=KNN(n_neighbors=6)
mlp=MLPClassifier(hidden_layer_sizes=(100,),max_iter=500,random_state=42)
gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)


RF.fit(train_data_std,train_label)
knn.fit(train_data_std,train_label)
mlp.fit(train_data_std,train_label)
gbc.fit(train_data_std,train_label)


RF_pred_label=RF.predict(test_data_std)
KNN_pred_label=knn.predict(test_data_std)
MLP_pred_label=mlp.predict(test_data_std)
GBC_pred_label=gbc.predict(test_data_std)


RF_acc=accuracy_score(test_label,RF_pred_label)
RF_cm=confusion_matrix(test_label,RF_pred_label)

KNN_acc=accuracy_score(test_label,KNN_pred_label)
KNN_cm=confusion_matrix(test_label,KNN_pred_label)

MLP_acc=accuracy_score(test_label,MLP_pred_label)
MLP_cm=confusion_matrix(test_label,MLP_pred_label)

GBC_acc=accuracy_score(test_label,GBC_pred_label)
GBC_cm=confusion_matrix(test_label,GBC_pred_label)

print(f'Random Forest accuracy:{RF_acc*100:.2f}%')
print(f'KNN accuracy:{KNN_acc*100:.2f}%')
print(f'MLP accuracy:{MLP_acc*100:.2f}%')
print(f'GBC accuracy:{GBC_acc*100:.2f}%')

RF_precision = precision_score(test_label, RF_pred_label, average='weighted')
RF_recall = recall_score(test_label, RF_pred_label, average='weighted')
RF_f1 = f1_score(test_label, RF_pred_label, average='weighted')
print(f'Random Forest Precision: {RF_precision:.2f}')
print(f'Random Forest Recall: {RF_recall:.2f}')
print(f'Random Forest F1 Score: {RF_f1:.2f}')

KNN_precision = precision_score(test_label, KNN_pred_label, average='weighted')
KNN_recall = recall_score(test_label, KNN_pred_label, average='weighted')
KNN_f1 = f1_score(test_label, KNN_pred_label, average='weighted')
print(f'KNN Precision: {KNN_precision:.2f}')
print(f'KNN Recall: {KNN_recall:.2f}')
print(f'KNN F1 Score: {KNN_f1:.2f}')

MLP_precision = precision_score(test_label, MLP_pred_label, average='weighted')
MLP_recall = recall_score(test_label, MLP_pred_label, average='weighted')
MLP_f1 = f1_score(test_label, MLP_pred_label, average='weighted')
print(f'MLP Precision: {MLP_precision:.2f}')
print(f'MLP Recall: {MLP_recall:.2f}')
print(f'MLP F1 Score: {MLP_f1:.2f}')

GBC_precision = precision_score(test_label, GBC_pred_label, average='weighted')
GBC_recall = recall_score(test_label, GBC_pred_label, average='weighted')
GBC_f1 = f1_score(test_label, GBC_pred_label, average='weighted')
print(f'梯度提升 Precision: {GBC_precision:.2f}')
print(f'梯度提升 Recall: {GBC_recall:.2f}')
print(f'R梯度提升 F1 Score: {GBC_f1:.2f}')

plt.figure(figsize=(6,6))
plt.subplot(221)
sns.heatmap(RF_cm,annot=True,cmap='Blues',xticklabels=[2,4,6,8,10,12],yticklabels=[2,4,6,8,10,12])
plt.xlabel('Predict label')
plt.ylabel('True label')
plt.title('Random Forest')


plt.subplot(222)
sns.heatmap(KNN_cm,annot=True,cmap='Blues',xticklabels=[2,4,6,8,10,12],yticklabels=[2,4,6,8,10,12])
plt.xlabel('Predict label')
plt.ylabel('True label')
plt.title('KNN')

plt.subplot(223)
sns.heatmap(MLP_cm,annot=True,cmap='Blues',xticklabels=[2,4,6,8,10,12],yticklabels=[2,4,6,8,10,12])
plt.xlabel('Predict label')
plt.ylabel('True label')
plt.title('MLP')

plt.subplot(224)
sns.heatmap(GBC_cm,annot=True,cmap='Blues',xticklabels=[2,4,6,8,10,12],yticklabels=[2,4,6,8,10,12])
plt.xlabel('Predict label')
plt.ylabel('True label')
plt.title('GBC')

plt.figure()
acc_all=np.array([RF_acc,KNN_acc,MLP_acc,GBC_acc])
algorithm=['Random Forest','KNN','MLP','GradientBoosting']
plt.bar(x=algorithm,
        height=acc_all,
        width=0.6,
        align='center',
        color='blue')
plt.show()

