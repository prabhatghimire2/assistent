import pyttsx3
import os
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import smtplib
import time

class Jarvis:
    """docstring for Jarvis"""
    def __init__(self):
        self.active = True
        self.contact_list = {'prabhat': 'prabhat.ghimire2@gmail.com', 'bikrant': 'vikrantabasyal@gmail.com'}
        self.app_list = {'sublime' : 'subl', 'blender': 'blender', 'gimp':'gimp', 'firefox':'firefox','':''}

        # sapi5, nsss, espeak
        self.engine = pyttsx3.init('espeak')
        self.voices = self.engine.getProperty('voices')
        # print(voices)
        # for voice of girl voices[1]
        self.engine.setProperty('voice', self.voices[0])

    def sendEmail(self, to, content):
        '''send email'''
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.ehlo()
        server.starttls()
        server.login('prabhu10612@gmail.com','Gulmi!512')
        server.sendmail('prabhu10612@gmail.com', to, content)
        server.close()

    def speak(self, audio):
        '''it speak the given sentence'''
        if self.active :
            self.engine.say(audio)
            self.engine.runAndWait()

    def wishme(self):
        ''' it take time and wish according to time'''
        hour = int(datetime.datetime.now().hour)
        if hour>=0 and hour<12 :
            print('good morning!')
            self.speak('good morning!')
        elif hour>=12 and hour<18 :
            print('good aftrnoon!')
            self.speak('good aftrnoon!')
        else :
            print('good evening!')
            self.speak('good evening!')
        print('I am jarvis, how can i help you')
        self.speak('I am jarvis, how can i help you')

    def takeCommand(self):
        '''it takes microphone input from the user and return the string'''
        r  = sr.Recognizer()
        with sr.Microphone() as source:
            print('Listening...')
            r.pause_threshold = 1
            r.adjust_for_ambient_noise(source)
            audio = r.record(source, duration=4)

        try:
            print('Recognizing...')
            query = r.recognize_google(audio)
            print(f' You said : {query}\n')
        except Exception as e:
            print(e)
            print('say that again please')
            return 'None'
        # else:
        #     pass
        # finally:
        #     pass
        return query


