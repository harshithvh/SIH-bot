from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import googlescraper

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Welcome to EduChamp Api"

@app.route('/about')
def index():
    return "This is an api build for EduChamp Chatbot.To have seamless interaction"

#@app.route('/<query>')
#def response(query):
#    result = googlescraper.search(query)
#    return result

@app.route('/query', methods=['POST'])
def response():
    msg = request.form.get('Body')
    resp = MessagingResponse()
    resp.message(googlescraper.search(msg))
    return str(resp)
    

if __name__=='__main__':
    app.run(debug=True)

# heroku config:set WEB_CONCURRENCY=1
'''
Step1: Sign in to the twilio account
Step2: Under My First Twilio account > messaging > Try it out > send a whatsapp message > follow the 5 steps to activate
Step3: Under Twilio Sandbox for WhatsApp > add the URL given by ngrok (Ex: https://fb8f8f8f.ngrok.io/query)
Step4: Run python-flask.py
Step5: Go to downloads > ngrok > run the ngrok.exe file
Step6: In the ngrok.exe enter 'ngrok http 5000' to generate the URL
'''

