import os
from flask import Flask, render_template, request, send_from_directory
import preprocessing

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_erd():
    if request.method == 'POST':
        text = request.form['text2erd'] if 'text2erd' in request.form else None 
        if text is None:
            return render_template('index.html', final_relation="None")
        final_relation = preprocessing.process_relation(text)
        preprocessing.generate_erd(final_relation)
        return render_template('index.html', final_relation=final_relation)
    return

@app.route('/show_erd')
def show_erd():
    return render_template('show_erd.html')

if __name__ == '__main__':
    app.run(debug=True)