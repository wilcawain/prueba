from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos a comparar
models = {
    'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

results = {}
best_model = None
best_score = 0

# Entrenar y evaluar cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'model': model
    }
    
    if results[name]['accuracy'] > best_score:
        best_score = results[name]['accuracy']
        best_model = model

print("Resultados de los modelos:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Exactitud: {metrics['accuracy']:.3f}")
    print(f"  Precisión: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-score:  {metrics['f1']:.3f}")

# Matriz de confusión del mejor modelo
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Mejor Modelo')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.show()
