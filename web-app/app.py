from flask import Flask
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
from preprocessing import preprocess_data 

app = Flask(__name__)

# Flask-WTF requires an encryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

# Flask-Bootstrap requires this line
Bootstrap(app)

class NameForm(FlaskForm):
    sdesc = StringField('Short Description', validators=[DataRequired()])
    desc = StringField('Description', validators=[DataRequired()])
    caller = StringField('Caller', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = NameForm()
    message = ""
    new_data=pd.DataFrame()
    sdesc = ""
    desc = ""
    caller = ""
    pro_data=pd.DataFrame()
    pred=""
    
    if form.validate_on_submit():
        sdesc = form.sdesc.data
        desc = form.desc.data
        caller = form.caller.data
        if sdesc.lower() !="" and desc.lower() !="" and caller.lower() !="":
            message = "Success!"
            form_data=[[sdesc,desc,caller]]
            columns=['Short description','Description','Caller']
            new_data= pd.DataFrame(form_data, columns = columns)
            pro_data=preprocess_data(new_data)
            pred=pro_data[0]
            
        else:
            message = "That actor is not in our database."
        
    return render_template('index.html', form=form, pred=pred,message=message)   

    if __name__ == '__main__':
        app.run() 

    # if __name__ == '__main__':
    #     app.run(host='0.0.0.0', port=8000)