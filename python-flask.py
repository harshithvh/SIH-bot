from flask import Flask
import googlescraper

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Welcome to EduChamp Api"

@app.route('/about')
def index():
    return "This is an api build for EduChamp Chatbot.To have seamless interaction"

@app.route('/<query>')
def response(query):
    result = googlescraper.search(query)
    return result

if __name__=='__main__':
    app.run(debug=True)

# heroku config:set WEB_CONCURRENCY=1
