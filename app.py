# ==========================================
# 1. IMPORT LIBRARIES & INITIALIZE
# ==========================================
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your machine learning assets (Make sure these are in your VS Code workspace folder)
model = joblib.load('best_air_quality_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# ==========================================
# 2. ROUTE HANDLERS
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = [
            float(request.form['pt08_s1']),
            float(request.form['nmhc']),
            float(request.form['c6h6']),
            float(request.form['pt08_s2']),
            float(request.form['nox']),
            float(request.form['pt08_s3']),
            float(request.form['no2']),
            float(request.form['pt08_s4']),
            float(request.form['pt08_s5']),
            float(request.form['t']),
            float(request.form['rh']),
            float(request.form['ah']),
            float(request.form['hour']),
            float(request.form['month']),
            float(request.form['dayofweek'])
        ]

        input_df = pd.DataFrame([form_data], columns=feature_names)
        predicted_co = model.predict(input_df)[0]

        if predicted_co <= 2.0:
            status = "🟢 GOOD (Clean Ambient Air)"
            bg_class = "text-bg-success"
        elif predicted_co <= 4.0:
            status = "🟡 MODERATE (Acceptable Air Quality)"
            bg_class = "text-bg-warning" 
        else:
            status = "🔴 DANGEROUS / CRITICAL POLLUTION LEVEL"
            bg_class = "text-bg-danger"

        result_text = f"Predicted CO Concentration: {predicted_co:.4f} mg/m³"

        return render_template('index.html',
                               prediction_text=result_text,
                               air_status=status,
                               status_class=bg_class)

# ==========================================
# 3. RUN LOCAL SERVER ENGINE
# ==========================================
if __name__ == '__main__':
    print(f"\n✨ Dark Theme Web App Booting Locally!")
    print(f"🔗 Open your browser and go to: http://127.0.0.1:5000\n")
    
    # In VS Code, debug=True is great because it auto-restarts when you save changes
    app.run(port=5000, debug=True)