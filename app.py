from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import uuid

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def hello_world():
   request_str = request.method
   if request_str == 'GET':
       return render_template('index.html', href = 'static/base_pic.svg')
   else:
       text = request.form['floatstr']
       randstr = uuid.uuid4().hex
       path='static/'+randstr+'.svg'
       np_arr = string_toarr(text)
       model = load('model.joblib')
       make_picture('AgesAndHeights.pkl',model,np_arr, path )

       return render_template('index.html', href = path)


def make_picture(training_data_filename,model,new_inp_np_arr,output_file):
  data = pd.read_pickle(training_data_filename)
  data = data[data['Age']>0]
  ages = data['Age']
  heights = data['Height']
  x_new = np.array(list(range(19))).reshape(-1,1)
  preds = model.predict(x_new)
  fig = px.scatter(x=ages,y=heights, title = 'Height vs Age', labels = {'x':'Age (years)','y':'Height (inches)'})
  fig.add_trace(go.Scatter(x=x_new.reshape(19),y=preds, mode = 'lines',name = 'model'))
  new_preds = model.predict(new_inp_np_arr)
  fig.add_trace(go.Scatter(x = new_inp_np_arr.reshape(len(new_inp_np_arr)), y = new_preds,name= 'new outputs',mode = 'markers', marker = dict(color = 'pink',size = 20,line = dict(color = 'pink',width = 2))))
  fig.write_image(output_file,width = 800, engine = 'kaleido')
  fig.show()

def string_toarr(string):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False  
  floats = np.array([float(x) for x in string.split(',') if is_float(x)])
  return floats.reshape(-1,1)