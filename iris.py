
#author: xiao_yu

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#load data
#path: the path of iris data in your laptop
iris = pd.read_csv('D:/baidu/IrisCode/Iris Code -- Machine Learning 3/Iris.csv')

#iris = pd.read_csv('D:/baidu/IrisCode/Iris Code -- Machine Learning 3/Iris_test.csv')

iris.drop('Id',axis=1,inplace=True)

# 显示数据前五行
print("数据集前五行:")
print(iris.head())

# 显示数据的基本信息，检查缺失值
print("\n数据集信息:")
print(iris.info())

# 显示描述性统计
print("\n描述性统计:")
print(iris.describe())

###
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')#刺芒野古草
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)#杂色
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

###
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

###
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

###
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)

###
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

###
#plt.figure(figsize=(7,4))
#sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
#plt.show()
#绘制配对图
# 绘制配对图，按物种着色
sns.pairplot(iris, hue='Species', palette='viridis')
#plt.suptitle('Iris Dataset Pair Plot by Species', y=1.02)
plt.show()


''' 
train, test = train_test_split(iris, test_size = 0.3)
print(train.shape)
print(test.shape)
'''


#k-means
# 提取用于聚类的特征
features =['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = iris[features]
#T = test[features]


# 改为 MinMaxScaler 提升聚类稳定性
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 稳定版轮廓系数测试
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K = {k:2d}, Silhouette Score = {score:.15f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='-', color='purple')
plt.title('Silhouette Coefficient for Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()

wcss=[]
# 使用手肘法寻找最佳K值
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 绘制手肘图
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# 使用最佳K值进行K-Means聚类
best_k = 3
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
final_clusters = kmeans_final.fit_predict(X_scaled)

# 将聚类结果添加回原始DataFrame
iris['Cluster'] = final_clusters
# 获取质心并逆变换回原始尺度
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# 创建质心DataFrame
centroid_df = pd.DataFrame(centroids_original, columns=features)
print("K-Means Cluster Centroids (Original Scale):")
print(centroid_df)

# 计算真实物种的特征均值
true_means_df = iris.groupby('Species')[features].mean()
print("\nTrue Species Feature Means:")
print(true_means_df)


# 应用PCA将数据降至2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
#T_pca = pca.fit_transform(T_scaled)

# 创建包含PCA结果和标签的DataFrame
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Cluster'] = final_clusters

# 绘制图
fig, axes = plt.subplots(1, 1, figsize=(7, 7))

# 图1: K-Means聚类结果
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Cluster', data=pca_df, palette='viridis', ax=axes, s=70)
axes.set_title('K-Means Clustering Results (PCA-reduced)')

# 绘制质心
centroids_pca = pca.transform(centroids_scaled)
axes.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='red', marker='X', label='Centroids')
axes.legend()

plt.show()


#评估最终模型的轮廓系数
final_silhouette_score = silhouette_score(X_scaled, final_clusters)
print(f"Final Silhouette Score for K=3: {final_silhouette_score:.4f}")


#其他数据对比
# 映射聚类标签到真实标签
# 创建交叉表
#ct = pd.crosstab(iris, iris['Cluster'])
##print("Contingency Table (Crosstab):")
#print(ct)

# 手动或自动确定映射关系
# 从交叉表可以看出: Cluster 1 -> setosa, Cluster 0 -> versicolor, Cluster 2 -> virginica
mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
mapped_clusters = iris['Cluster'].map(mapping)

# 计算映射后的准确率
accuracy = accuracy_score(iris['Species'], mapped_clusters)
print(f"\nMapped Accuracy: {accuracy:.4f}")

# 计算其他外部验证指标
ari = adjusted_rand_score(iris['Species'], iris['Cluster'])
homogeneity = homogeneity_score(iris['Species'], iris['Cluster'])
completeness = completeness_score(iris['Species'], iris['Cluster'])

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Homogeneity Score: {homogeneity:.4f}")
print(f"Completeness Score: {completeness:.4f}")






