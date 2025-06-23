from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import os
from CAT_egine import *

app = Flask(__name__)
app.secret_key = 'replace_with_a_secure_key'


app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.abspath('./.flask_session/')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

Session(app)

@app.route('/result', methods=['GET', 'POST'])
def result():
    if 'user' in session:
        user = session['user']
        items = session['items']

        if request.method == "POST":
            if 'allfalse' in request.form:
                # print("all false")
                response = [0] * len(items)
                user.administer_items(items= items, responses=response)
                return redirect(url_for('dashboard'))


            if 'alltrue' in request.form:
                response = [1] * len(items)
                # print("all true")
                user.administer_items(items= items, responses=response)
                return redirect(url_for('dashboard'))

            result_str = request.form.get('results')
            result_list = json.loads(result_str)

            

            user.administer_items(items= items, responses=result_list)

            data = user.get_test_info()
            session['user_data'] = data

            return redirect(url_for('dashboard'))
    

    return redirect(url_for('home'))
   
    

@app.route('/test', methods=['GET', 'POST'])
def test():
    if 'user' in session:
        user = session['user']

        if request.method == "POST":
            num_questions = int(request.form.get('num_questions', 0))
            ques_index = user.select_next_item(num_questions)
            session['items'] = ques_index
            questions = index_to_question(ques_index)
            # return f"{questions}"
            # print(type(questions))
            # # print(questions[0].answer_option_list)
            # print(questions[0])
            # print(questions[0]['answer_option_list'])
            # print(type(questions[0]['answer_option_list'][0][0]))
            print("test page")
            print(len(questions))
            return render_template('test.html', questions = questions)

    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        user_name = request.form.get('user-name', '').strip() or 'Alice'
        session['user_name'] = user_name
        return redirect(url_for('dashboard'))

    user_name = session.get('user_name', 'Alice')

    if 'user' in session:
        user = session['user']
        data = user.get_test_info()
        # return f"Loaded from session: {data}"
        return render_template('dashboard.html', user=data)

    user = ALManualIRT(user_name, item_bank_file="./frontend/item_bank.csv")
    session['user'] = user  # ✅ storing full object
    data = user.get_test_info()
    print("created...................................")
    # return f"Created user: {data}"
    return render_template('dashboard.html', user= data)

@app.route('/logout', methods= ['POST', 'GET'])
def logout():
    session.clear()
    return redirect(url_for('home'))


if __name__ == '__main__':
    os.makedirs('./.flask_session/', exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
