import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

dataset = pd.read_csv(r'C:\Users\user\Desktop\Machine Learning\Iris\Iris.csv')
df = pd.DataFrame(dataset)

categories = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

for i in range(df.shape[0]):
    df.iloc[i, 4] = categories[df.iloc[i, 4]]

df['class'] = df['class'].astype('int32')
df = df.sample(frac=1, random_state=24)

X = df.drop(columns='class')
Y = df['class']

tree = DecisionTreeClassifier(random_state=24)
tree.fit(X, Y)

pickle.dump(tree, open('Iris.pkl', 'wb'))

model = pickle.load(open('Iris.pkl', 'rb'))
model.predict([[4.9, 3.7, 1.9]])
