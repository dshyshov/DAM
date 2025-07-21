import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, request
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Прогноз цін РДН</title>
</head>
<body>
    <h1>Прогноз цін на РДН (Linear Regression)</h1>
    <form method="get">
        <label for="start">Початкова дата:</label>
        <input type="date" id="start" name="start" required value="{{ start }}">
        <label for="end">Кінцева дата:</label>
        <input type="date" id="end" name="end" required value="{{ end }}">
        <input type="submit" value="Оновити">
    </form>
    {% if error %}
        <p style="color: red;">{{ error }}</p>
    {% endif %}
    {% if plot_url %}
        <p>Цей графік показує прогнозовані та фактичні ціни на ринку на добу наперед.</p>
        <img src="data:image/png;base64,{{ plot_url }}">
    {% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    try:
        # Витягуємо параметри дати з GET-запиту
        start = request.args.get("start")
        end = request.args.get("end")

        # 1. Завантаження даних з API
        uuid = "5a616fba-fbc9-4073-9532-9161592faca8"
        url = f"https://map.ua-energy.org/api/v1/datasets/{uuid}/download/?format=csv"
        dl = requests.get(url)
        dl.raise_for_status()
        df = pd.read_csv(io.StringIO(dl.text), parse_dates=['timestamp'])

        # 2. Підготовка даних
        df = df.sort_values('timestamp').set_index('timestamp')
        df = df[['price']].dropna()

        # Фільтрація за датами, якщо вони передані
        if start and end:
            df = df.loc[start:end]

        if df.empty:
            return render_template_string(HTML_TEMPLATE, plot_url=None, start=start, end=end, error="Немає даних у вибраному діапазоні.")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        def make_sequences(data, window=24):
            X, y = [], []
            for i in range(len(data) - window):
                X.append(data[i:i+window].flatten())
                y.append(data[i+window])
            return np.array(X), np.array(y)

        window = 24
        X, y = make_sequences(scaled, window)
        if len(X) == 0:
            return render_template_string(HTML_TEMPLATE, plot_url=None, start=start, end=end, error="Недостатньо даних для побудови прогнозу.")

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_test).reshape(-1, 1)
        pred_inv = scaler.inverse_transform(pred)
        y_test_inv = scaler.inverse_transform(y_test)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test_inv, label='Фактична')
        ax.plot(pred_inv, label='Прогноз')
        ax.set_title('Прогноз цін на РДН (Linear Regression)')
        ax.set_xlabel('Часовий індекс')
        ax.set_ylabel('Ціна, грн/МВт·год')
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

        return render_template_string(HTML_TEMPLATE, plot_url=plot_url, start=start or '', end=end or '', error=None)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, plot_url=None, start=start or '', end=end or '', error=str(e))

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
        app.run(host="127.0.0.1", port=port, debug=False)
    except Exception as e:
        print(f"Не вдалося запустити сервер: {e}")
