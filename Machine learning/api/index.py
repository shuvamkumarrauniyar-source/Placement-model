"""Vercel serverless function for the placement-prediction website."""

from html import escape
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs
import pickle

import numpy as np


# The model is stored at the project root, one directory above api/.
BASE_DIR = Path(__file__).resolve().parent.parent

with (BASE_DIR / "placement_model.pkl").open("rb") as model_file:
    saved_data = pickle.load(model_file)

model = saved_data["model"]
scaler = saved_data["scaler"]


def page(result="", cgpa="", iq="", error=""):
    message = (
        f'<p class="result">{escape(result)}</p>' if result else ""
    ) + (f'<p class="error">{escape(error)}</p>' if error else "")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Placement Predictor</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
      font-family: Arial, sans-serif; color: #172033;
      background: linear-gradient(135deg, #e8f0ff, #f7f4ff); }}
    main {{ width: min(92%, 460px); padding: 36px; border-radius: 20px;
      background: white; box-shadow: 0 15px 45px #26345a2b; }}
    h1 {{ margin: 0 0 10px; color: #253b88; }}
    p {{ line-height: 1.5; color: #5f6675; }}
    label {{ display: block; margin-top: 20px; font-weight: bold; }}
    input {{ width: 100%; margin-top: 8px; padding: 12px; border: 1px solid #ccd2e0;
      border-radius: 8px; font-size: 16px; }}
    button {{ width: 100%; margin-top: 26px; padding: 13px; border: 0; border-radius: 8px;
      background: #314fb5; color: white; font-size: 16px; font-weight: bold; cursor: pointer; }}
    button:hover {{ background: #253d91; }}
    .result, .error {{ padding: 13px; border-radius: 8px; font-weight: bold; }}
    .result {{ background: #e8f8ee; color: #18723c; }}
    .error {{ background: #fff0f0; color: #ad2424; }}
  </style>
</head>
<body>
  <main>
    <h1>Placement Predictor</h1>
    <p>Enter a student's CGPA and IQ to predict placement.</p>
    <form method="post">
      <label for="cgpa">CGPA</label>
      <input id="cgpa" name="cgpa" type="number" min="0" max="10" step="0.01" value="{escape(cgpa)}" required>
      <label for="iq">IQ</label>
      <input id="iq" name="iq" type="number" min="0" step="0.01" value="{escape(iq)}" required>
      <button type="submit">Predict placement</button>
    </form>
    {message}
  </main>
</body>
</html>"""


class handler(BaseHTTPRequestHandler):
    def send_page(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def do_GET(self):
        self.send_page(page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = parse_qs(self.rfile.read(length).decode("utf-8"))
        cgpa = data.get("cgpa", [""])[0]
        iq = data.get("iq", [""])[0]

        try:
            cgpa_value, iq_value = float(cgpa), float(iq)
            if not 0 <= cgpa_value <= 10 or iq_value < 0:
                raise ValueError
            features = scaler.transform(np.array([[cgpa_value, iq_value]]))
            prediction = int(model.predict(features)[0])
            result = "Prediction: Placed" if prediction == 1 else "Prediction: Not placed"
            self.send_page(page(result=result, cgpa=cgpa, iq=iq))
        except ValueError:
            self.send_page(page(cgpa=cgpa, iq=iq, error="Enter a CGPA from 0 to 10 and a non-negative IQ."))

    def log_message(self, format, *args):
        return
