from flask import Flask, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    # Load your dataset
    with open('file_name.pkl', 'rb') as f:
        data_set = pickle.load(f)
    
    # Pass the dataset to your HTML template
    return render_template('index.html', data_set=data_set)

if __name__ == '__main__':
    app.run(debug=True)
