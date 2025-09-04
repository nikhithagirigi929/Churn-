import os
import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

from flask import Flask, request, render_template
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'json', 'txt'}

# Globals for model and data
model = None
X_full = None
ids_full = None
y_full = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    elif ext == '.txt':
        try:
            return pd.read_csv(file_path)
        except:
            return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f'Unsupported file format: {ext}')

def detect_and_drop_id_columns(df):
    possible_id_cols = ['CustomerID', 'customer_id', 'ID', 'id', 'client_id', 'ClientID', 'cust_id']
    cols_to_drop = []
    for col in possible_id_cols:
        if col in df.columns:
            cols_to_drop.append(col)
    for col in df.columns:
        if df[col].nunique() == len(df) and col not in cols_to_drop:
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def detect_target_column(df, user_input=None):
    if user_input and user_input in df.columns:
        return user_input
    possible_targets = ['Churn', 'Exited', 'Target', 'Label', 'Outcome', 'Response', 'IsChurned']
    for col in possible_targets:
        if col in df.columns:
            return col
    raise ValueError("No known target column found in the dataset. Please specify the target column name.")

def universal_preprocess(df, target_col='Churn', id_col_list=None):
    if id_col_list is None:
        id_col_list = ['CustomerID', 'customer_id', 'ID', 'id', 'client_id', 'ClientID', 'cust_id']
    ids = None
    for col in id_col_list:
        if col in df.columns:
            ids = df[col]
            break
    if ids is None:
        ids = df.index

    if not isinstance(ids, pd.Series):
        ids = pd.Series(ids)

    mask = df[target_col].notna()
    df_proc = df.loc[mask].copy()
    ids = ids.loc[mask]

    df_proc = detect_and_drop_id_columns(df_proc)

    for col in df_proc.select_dtypes(include=['object']).columns:
        df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    return X, y, ids

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global model, X_full, ids_full, y_full
    if request.method == 'POST':
        target_col_input = request.form.get('target_col', '').strip()
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                df = load_data(file_path)
                target_col = detect_target_column(df, user_input=target_col_input)
                X, y, ids = universal_preprocess(df, target_col=target_col)
                columns_used = list(X.columns)

                ids = ids.astype(str).str.strip()
                sample_ids = ids.head(10).tolist()
                print("Sample Customer IDs:", sample_ids)

                model_new = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
                model_new.fit(X, y)

                preds = model_new.predict(X)
                probs = model_new.predict_proba(X)[:, 1]
                report = classification_report(y, preds, output_dict=True)
                roc_auc = roc_auc_score(y, probs)
                accuracy = accuracy_score(y, preds)

                model = model_new
                X_full = X
                ids_full = ids
                y_full = y

                return render_template(
                    'upload.html',
                    message=f'File uploaded and processed successfully! Target column: {target_col}',
                    report=report,
                    roc_auc=roc_auc,
                    accuracy=accuracy,
                    columns_used=columns_used,
                    sample_ids=sample_ids
                )
            except Exception as e:
                return render_template('upload.html', message=f'Error processing file: {str(e)}')
        else:
            return render_template('upload.html', message='Unsupported file type')
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict_customer():
    global model, ids_full, X_full
    if model is None:
        return render_template('upload.html', message='Please upload and process data file first.')

    cust_id = request.form.get('cust_id', '').strip()
    if not cust_id:
        return render_template('upload.html', message='Please enter a Customer ID.')

    ids_norm = ids_full

    if cust_id not in ids_norm.values:
        return render_template('upload.html', message=f'Customer ID {cust_id} not found.')

    idx = ids_norm[ids_norm == cust_id].index[0]
    customer_features = X_full.loc[[idx]]
    prediction = model.predict(customer_features)[0]
    prediction_prob = model.predict_proba(customer_features)[0][1]

    # Estimate expected months until churn with a heuristic
    max_months = 12  # can be adjusted to your business context
    expected_months = round(max_months * (1 - prediction_prob))
    if expected_months == 0:
        expected_months = 1

    customer_data = customer_features.iloc[0].to_dict()

    message = (f'Prediction for Customer ID {cust_id}: '
               f'{"Churn" if prediction == 1 else "Not Churn"} '
               f'(Probability: {prediction_prob:.2%}). '
               f'Estimated time to churn: With In {expected_months} months.')

    return render_template(
        'upload.html',
        message=message,
        columns_used=list(X_full.columns),
        report=None,
        roc_auc=None,
        accuracy=None,
        customer_data=customer_data
    )

if __name__ == '__main__':
    app.run(debug=True)
