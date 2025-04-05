from flask import Flask, render_template, request, redirect, session
import os, yaml, json, subprocess
from werkzeug.utils import secure_filename
from predict import predict

app = Flask(__name__)
app.secret_key = 'your-secret-key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

users = {
    'admin': {'password': 'adminpass', 'role': 'admin'},
    'user': {'password': 'userpass', 'role': 'user'}
}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname]['password'] == pwd:
            session['username'] = uname
            session['role'] = users[uname]['role']
            return redirect('/dashboard')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if session['role'] == 'admin':
        metrics = {}
        if request.method == 'POST':
            n_estimators = int(request.form['n_estimators'])
            with open('../dvclive/params.yaml', 'w') as f:
                yaml.dump({'n_estimators': n_estimators}, f)
            subprocess.run(['dvc', 'repro'], cwd='../')
            with open('../dvclive/metrics.json', 'r') as m:
                metrics = json.load(m)
        return render_template('dashboard_admin.html', metrics=metrics)

    elif session['role'] == 'user':
        if request.method == 'POST':
            file = request.files['datafile']
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filepath)
                preds = predict(filepath)
                return render_template('prediction_result.html', predictions=preds)
        return render_template('dashboard_user.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
