from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Загрузка модели
model = joblib.load('random_forest_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение данных из формы
        pclass = int(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        sex = int(request.form['sex'])  # 0 - женский, 1 - мужской
        embarked = int(request.form['embarked'])  # 0 - C, 1 - Q, 2 - S

        # Создание DataFrame для предсказания
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Sex_male': [sex],  # 0 - женский, 1 - мужской
            'Embarked_Q': [1 if embarked == 1 else 0],  # 1 - Q, иначе 0
            'Embarked_S': [1 if embarked == 2 else 0]   # 1 - S, иначе 0
        })

        # Убедимся, что порядок столбцов совпадает с порядком при обучении
        input_data = input_data.reindex(columns=[
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'
        ], fill_value=0)

        # Предсказание
        prediction = model.predict(input_data)
        result = 'Выжил' if prediction[0] == 1 else 'Не выжил'

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
