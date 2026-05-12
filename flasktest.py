from flask import Flask, request, jsonify


app = Flask(__name__) 


@app.route("/") 
def mainRoute():  
    username = request.args.get("username")  
    return f"{username}" 


app.run(debug=True)

