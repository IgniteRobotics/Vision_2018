import pickle

f = open('training.pkl', 'rb')
x3 = pickle.load(f)
f = open('training_y.pkl', 'rb')
y3 = pickle.load(f)

f = open('training_2.pkl', 'rb')
x2 = pickle.load(f)
f = open('training_y_2.pkl', 'rb')
y2 = pickle.load(f)

f = open('training_1.pkl', 'rb')
x1 = pickle.load(f)
f = open('training_y_1.pkl', 'rb')
y1 = pickle.load(f)

x = x1 + x2 + x3
y = y1 + y2 + y3

print(x)
print(y)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

from sklearn import svm
clf = svm.SVC(gamma=0.001)

clf.fit(x, y)

f = open('clf.pkl', 'wb')
pickle.dump(clf, f)

print()
j = 0
for i in x:
    a = clf.predict([i])
    print(i, a, y[j]) 
    j += 1
