#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib as plt
import seaborn as sns
import sklearn


# In[2]:


import sys
print("Versión de Python: {}".format(sys.version))
print("Versión de pandas: {}".format(pd.__version__))
print("Versión de matplotlib: {}".format(plt.__version__))
print("Versión de NumPy: {}".format(np.__version__))
print("Versión de seaborn: {}".format(sns.__version__))
print("Versión de scikit-learn: {}".format(sklearn.__version__))


# In[3]:


data = pd.read_csv("C:/Users/Angela/OneDrive - Universidad de Almeria/TFG/datos/gbsg.csv")


# In[4]:


data.head(5)


# In[5]:


data.info()


# In[6]:


data.drop(columns=['Unnamed: 0','pid'],inplace=True);


# In[7]:


data.meno.value_counts()


# In[8]:


data.hormon.value_counts()


# In[9]:


data.grade.value_counts()


# # Análisis descriptivo

# In[10]:


data.describe()


# # Histogramas

# In[11]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout
data.hist(bins=50,figsize=(11,7))
tight_layout()


# In[12]:


atributos=['age', 'meno','size', 'grade', 'nodes', 'pgr', 'er','hormon', 'rfstime']
plt.figure(figsize=(12,8))
for i, atributo in enumerate(atributos):
    plt.subplot(3,3,i+1)
    sns.histplot(data[data['status'] == 0][atributo], color='blue', label='Vivo sin recurrencia', kde=True)
    sns.histplot(data[data['status'] == 1][atributo], color='red', label='Recurrencia o muerto por cancer', kde=True)
    plt.title(f'Histograma de {atributo}')
    plt.xlabel(atributo)
    plt.ylabel('Frecuencia')
    plt.legend()
plt.tight_layout()
plt.show()


# # Correlaciones

# In[13]:


matrizcorrelacion=data.corr()
matrizcorrelacion


# In[14]:


atributosnume=['age','size', 'nodes', 'pgr', 'er', 'rfstime']
sns.pairplot(data, hue='status', vars=atributosnume, palette={0: 'blue', 1: 'red'})
plt.suptitle('Pair Plot of Numerical Features based on Status')
plt.tight_layout()
plt.show()


# In[15]:


sns.heatmap(matrizcorrelacion,annot=True,cmap='coolwarm',fmt=".3f")
plt.title('Mapa de calor de la matriz de correlacion')


# # Tratamiento variables categóricas 

# In[16]:


datos = data.drop(columns=['status'])
varobjetivo = data['status']


# In[17]:


print("Cantidad por clase:\n{}".format(
 {n: v for n, v in zip(data.status, np.bincount(data.status))}))


# In[18]:


datos=pd.get_dummies(datos,columns=['grade'],dtype=int)


# In[19]:


datos.head(3)


# In[20]:


datos[['grade_1','grade_2','grade_3']].describe()


# In[21]:


datos[['grade_1','grade_2','grade_3']].hist(bins=50,figsize=(11,7))
plt.tight_layout()


# In[22]:


datos.drop(columns='grade_1',inplace=True)


# In[23]:


datos.head(3)


# # Dividir datos entrenamiento y prueba

# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datos, varobjetivo,
 random_state=1)


# In[25]:


print(X_train.shape)
print(X_test.shape)


# # Escalado de los datos

# In[26]:


from sklearn.preprocessing import StandardScaler
scalerStandard=StandardScaler()
scalerStandard.fit(X_train)
X_train_scaled=scalerStandard.transform(X_train)
X_test_scaled=scalerStandard.transform(X_test)


# # Regresión logística

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


modlogreg=LogisticRegression().fit(X_train,y_train)
print(modlogreg.score(X_train,y_train))
print(modlogreg.score(X_test,y_test))


# In[29]:


modlogreg=LogisticRegression().fit(X_train_scaled,y_train)
print(modlogreg.score(X_train_scaled,y_train))
print(modlogreg.score(X_test_scaled,y_test))


# In[30]:


modlogreg.coef_.T


# In[31]:


modlogreg.intercept_


# In[32]:


X_train


# In[33]:


modlogreg.predict(np.array([34,0, 24, 6, 400, 56, 1, 3456, 1 ,0]).reshape(1,-1))


# # Árbol de decisión

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier(random_state=0)
modeloarbol10=tree1.fit(X_train, y_train)
print("Exactitud en los datos de entrenamiento: {:.3f}".format(modeloarbol10.score(X_train, y_train)))
print("Exactitud en los datos de prueba: {:.3f}".format(modeloarbol10.score(X_test, y_test)))


# In[36]:


tree = DecisionTreeClassifier(max_depth=5, random_state=0)
modarbolred=tree.fit(X_train, y_train)
print("Exactitud en los datos de entrenamiento: {:.3f}".format(modarbolred.score(X_train, y_train)))
print("Exactitud en los datos de prueba: {:.3f}".format(modarbolred.score(X_test, y_test)))


# In[37]:


from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["0", "1"],
 feature_names=datos.columns.tolist(), impurity=False, filled=True)


# In[38]:


import graphviz
with open("tree.dot") as f:
     dot_graph = f.read()
graphviz.Source(dot_graph)


# In[39]:


print("Importancia variables:\n{}".format(modarbolred.feature_importances_))


# In[40]:


def plot_feature_importances(model):
    n_features = datos.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),datos.columns.tolist())
    plt.xlabel("Importancia de las variables")
    plt.ylabel("Variables")

plot_feature_importances(modarbolred)


# In[41]:


modarbolred.predict(np.array([34,0, 24, 6, 400, 56, 1, 3456, 1, 0]).reshape(1,-1))


# # Random forest

# In[42]:


from sklearn.ensemble import RandomForestClassifier


# In[43]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
modeloarbolran10=forest.fit(X_train, y_train)


# In[44]:


print("Exactitud en los datos de entrenamiento: {:.3f}".format(modeloarbolran10.score(X_train,y_train)))
print("Exactitud en los datos de prueba: {:.3f}".format(modeloarbolran10.score(X_test,y_test)))


# In[45]:


plot_feature_importances(modeloarbolran10)


# In[46]:


modeloarbolran10.predict(np.array([34,0, 24, 6, 400, 56, 1, 3456, 1, 0]).reshape(1,-1))


# # Cross-validation, matrices de confusión y métricas

# In[47]:


import mglearn
mglearn.__version__


# In[48]:


mglearn.plots.plot_cross_validation()


# In[49]:


from sklearn.model_selection import cross_val_score
scalerStandard.fit(datos)
datos_scaled=scalerStandard.transform(datos)
scores=cross_val_score(modlogreg,datos_scaled,varobjetivo,cv=5)
print(scores)
print(scores.mean())


# In[50]:


mglearn.plots.plot_binary_confusion_matrix()


# In[51]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[52]:


y_pred = modlogreg.predict(X_test_scaled)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred))
disp.plot()


# In[53]:


from sklearn.metrics import classification_report


# In[54]:


print(classification_report(y_test,y_pred))


# # Selección de variables y validaciones de los modelos reducidos

# In[55]:


from sklearn.feature_selection import SelectPercentile
rng= np.random.RandomState(42)
select= SelectPercentile(percentile=50) 
select.fit(X_train_scaled,y_train)
X_train_selected= select.transform(X_train_scaled)


# In[56]:


mask=select.get_support()
print(mask)
plt.matshow(mask.reshape(1,-1),cmap='gray_r')


# In[57]:


X_test_selected= select.transform(X_test_scaled)

print(LogisticRegression().fit(X_train_selected, y_train).score(X_train_selected,y_train))
print(LogisticRegression().fit(X_train_selected, y_train).score(X_test_selected,y_test))


# In[58]:


rng= np.random.RandomState(42)
select= SelectPercentile(percentile=60) 
select.fit(X_train_scaled,y_train)
X_train_selected60= select.transform(X_train_scaled)

X_test_selected60= select.transform(X_test_scaled)

scoretrainUS=LogisticRegression().fit(X_train_selected60, y_train).score(X_train_selected60,y_train)
scoretestUS=LogisticRegression().fit(X_train_selected60, y_train).score(X_test_selected60,y_test)
print(scoretrainUS)
print(scoretestUS)


# In[59]:


mask=select.get_support()
print(mask)
plt.matshow(mask.reshape(1,-1),cmap='gray_r')


# In[60]:


rng= np.random.RandomState(42)
select= SelectPercentile(percentile=70)
select.fit(X_train_scaled,y_train)
X_train_selected= select.transform(X_train_scaled)

X_test_selected= select.transform(X_test_scaled)


print(LogisticRegression().fit(X_train_selected, y_train).score(X_train_selected,y_train))
print(LogisticRegression().fit(X_train_selected, y_train).score(X_test_selected,y_test))


# In[61]:


rng= np.random.RandomState(42)
select= SelectPercentile(percentile=80) 
select.fit(X_train_scaled,y_train)
X_train_selected= select.transform(X_train_scaled)

X_test_selected= select.transform(X_test_scaled)


print(LogisticRegression().fit(X_train_selected, y_train).score(X_train_selected,y_train))
print(LogisticRegression().fit(X_train_selected, y_train).score(X_test_selected,y_test))


# In[62]:


modreducido=LogisticRegression().fit(X_train_selected60, y_train)


# In[63]:


modreducido.coef_.T


# In[64]:


modreducido.intercept_


# In[65]:


treered60 = DecisionTreeClassifier(max_depth=5, random_state=0)
modarbolred60=treered60.fit(X_train_selected60, y_train)
print("Exactitud en los datos de entrenamiento: {:.3f}".format(modarbolred60.score(X_train_selected60, y_train)))
print("Exactitud en los datos de prueba: {:.3f}".format(modarbolred60.score(X_test_selected60, y_test)))


# In[66]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
modeloarbolran10red=forest.fit(X_train_selected60, y_train)
print("Exactitud en los datos de entrenamiento: {:.3f}".format(modeloarbolran10red.score(X_train_selected60,y_train)))
print("Exactitud en los datos de prueba: {:.3f}".format(modeloarbolran10red.score(X_test_selected60,y_test)))


# In[67]:


from sklearn.model_selection import cross_val_score
scoresred=cross_val_score(modreducido,datos_scaled,varobjetivo,cv=5)
print(scoresred)
print(scoresred.mean())


# In[68]:


y_pred1 = modreducido.predict(X_test_selected60)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred1))
disp.plot()


# In[69]:


print(classification_report(y_test,y_pred1))


# In[70]:


scores2=cross_val_score(modarbolred60,datos,varobjetivo)
print(scores2)
print(scores2.mean())


# In[71]:


y_predarbolred60 = modarbolred60.predict(X_test_selected60)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_predarbolred60))
disp.plot()


# In[72]:


print(classification_report(y_test, y_predarbolred60))


# In[73]:


scores1=cross_val_score(modeloarbolran10,datos,varobjetivo)
print(scores1)
print(scores1.mean())


# In[74]:


y_predarbolran10 = modeloarbolran10.predict(X_test)
disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_predarbolran10))
disp.plot()


# In[75]:


print(classification_report(y_test, y_predarbolran10))


# # Estudio del umbral de decision

# In[76]:


modreducido.predict_proba(X_test_selected60)


# In[77]:


y_pred_lower_threshold = modreducido.predict_proba(X_test_selected60)[:, 1]  >0.4


# In[78]:


disp=ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred_lower_threshold))
disp.plot()


# In[79]:


print(classification_report(y_test, y_pred_lower_threshold))


# In[80]:


y_pred_lower_threshold = modreducido.predict_proba(np.array([30,3, 201, 85,0, 2456]).reshape(1, -1) )[:, 1]  >0.4
y_pred_lower_threshold


# In[81]:


modreducido.predict(np.array([512,418,1, 0,0, 123]).reshape(1, -1))

