from flask import Flask
from flask import Flask, request, render_template
import Clustering
import QueryExecutor


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"


@app.route('/')  
def input_form(): 
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():

    clusters = request.form['clusters']
    comments = request.form['comments']
    topwords = request.form['topwords']

    #print(type(clusters))
    #print(clusters)
    #print(comments)
    #print(topwords)

    clusterObj = Clustering.Clustering(int(clusters), int(comments), int(topwords))
    clusterObj.main_func()

    return render_template('output.html')


@app.route("/queryexec", methods=['GET', 'POST'])
def queryexec():
    if request.method == 'POST':
        
        query = request.form["username"]

        obj = QueryExecutor.QueryExecutor(query)
        obj.save_to_html()

        return render_template('sqloutput.html')

        
    
    return render_template('queryexec.html')


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':   
    app.run() 
