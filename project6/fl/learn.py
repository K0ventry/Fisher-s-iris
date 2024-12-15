import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib  # Импортируем библиотеку для сохранения модели

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)

# Сохранение модели в файл
joblib.dump(model, 'logistic_regression_model.joblib')

# Прогнозирование и оценка модели
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Загрузка модели из файла (пример, как её использовать позже)
loaded_model = joblib.load('logistic_regression_model.joblib')