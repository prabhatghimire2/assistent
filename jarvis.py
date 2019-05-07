#!/usr/bin/env python3
# Requires PyAudio and PySpeech.

import pyttsx3
import os
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import smtplib
import time

contact_list = {'prabhat': 'prabhu10612@gmail.com'}
app_list = {'sublime' : 'subl', 'blender': 'blender', 'gimp':'gimp', 'firefox':'firefox','':''}

# sapi5, nsss, espeak
engine = pyttsx3.init('espeak')
voices = engine.getProperty('voices')
# print(voices)
# for voice of girl voices[1]
engine.setProperty('voice', voices[0])

def sendEmail(to, content):
    '''send email'''
    seerver = smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('youremail@gmail.com','password')
    server.sendmail('yourgmail@gmail.com', to, content)
    server.close()
def speak(audio):
    '''it speak the given sentence'''
    engine.say(audio)
    engine.runAndWait()

def wishme():
    ''' it take time and wish according to time'''
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12 :
        print('good morning!')
        speak('good morning!')
    elif hour>=12 and hour<18 :
        print('good aftrnoon!')
        speak('good aftrnoon!')
    else :
        print('good evening!')
        speak('good evening!')
    print('I am jarvis, how can i help you')
    speak('I am jarvis, how can i help you')

def takeCommand():
    '''it takes microphone input from the user and return the string'''
    r  = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 1
        audio = r.record(source, duration=4)

    try:
        print('Recognizing...')
        query = r.recognize_google(audio)
        print(f'User said {query}\n')
    except Exception as e:
        # print(e)
        print('say that again please')
        return 'None'
    # else:
    #     pass
    # finally:
    #     pass
    return query

if __name__ == '__main__':
    # main()
    # initialization
    wishme()
    # time.sleep(2)
    # while True:
    if True :
        query = takeCommand().lower()
        # query = 'what time is it'
        # logic for executing tasks based on query
        if 'wikipedia' in query :
            speak('searching wikipedia...')
            query = query.replace('wikipedia','')
            result = wikipedia.summary(query, sentences = 2)
            print(f'according to wiki {result}')
            speak(f'according to wiki {result}')

        elif 'youtube' in query :
            speak('opening youtube')
            webbrowser.open('youtube.com')

        elif 'google' in query :
            speak('opening google')
            webbrowser.open('google.com')

        elif 'stackoverflow' in query :
            speak('opening stackoverflow')
            webbrowser.open('stackoverflow.com')

        elif 'music' in query :
            speak('searching for songs')
            music_dir = '/root/Documents/BackUp/mobile/Music'
            songs = os.listdir(music_dir)
            os.system(os.path.join(music_dir, songs[3]))

        elif 'what time is it' in query :
            speak('checking time')
            time_str = datetime.datetime.now().strftime('%H:%S:%M')
            print(f'now the time is { time_str }')
            speak(f'now the time is { time_str }')

        elif 'open' in query and 'app' in query :
            query = query.replace('open ','')
            query = query.replace(' app','')
            speak(f'opening {app_list[query]}')
            os.system(app_list[query])

        elif 'email' in query :
            try :
                speak('what should i say???')
                content = takeCommand()
                sendEmail(to, content)
                speak('email send')
            except Exception as e :
                print(e)
                speak('sorry email send error')

        elif "where is" in query:
            data = data.split(" ")
            location = data[2]
            speak("Hold on Frank, I will show you where " + location + " is.")
            os.system("chromium-browser https://www.google.nl/maps/place/" + location + "/&amp;")

        else :
            print('what can i do for you???')
            speak('what can i do for you???')



