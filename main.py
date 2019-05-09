from jarvis import *


if __name__ == '__main__':
    # main()
    # initialization
    jarvis = Jarvis()
    jarvis.wishme()
    # time.sleep(2)
    while True :
    # if True :
        query = jarvis.takeCommand().lower()
        # query = 'what time is it'
        # logic for executing tasks based on query
        if 'jarvis' in query :
            jarvis.active = True
            jarvis.speak('how can i help you')

        elif 'wikipedia' in query :
            jarvis.speak('searching wikipedia...')
            query = query.replace('wikipedia','')
            result = wikipedia.summary(query, sentences = 2)
            print(f'according to wiki {result}')
            jarvis.speak(f'according to wiki {result}')

        elif 'youtube' in query :
            query = query.replace('search ','')
            query = query.replace(' in ','')
            query = query.replace(' youtube','')
            jarvis.speak('opening youtube')
            if 'open' in query:
                webbrowser.open('youtube.com')
            else:
                webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ','+')}")

        elif 'duckduckgo' in query :
            query = query.replace('search ','')
            query = query.replace(' in ','')
            query = query.replace(' youtube','')
            jarvis.speak('opening duckduckgo')
            if 'open' in query:
                webbrowser.open('duckduckgo.com')
            else:
                webbrowser.open(f"https://duckduckgo.com/?q={query.replace(' ','+')}&t=h_&ia=web")


        elif 'stackoverflow' in query :
            jarvis.speak('opening stackoverflow')
            webbrowser.open('stackoverflow.com')

        elif 'music' in query :
            jarvis.speak('searching for songs')
            music_dir = '/root/Documents/BackUp/mobile/Music'
            songs = os.listdir(music_dir)
            os.system(os.path.join(music_dir, songs[3]))

        elif 'what time is it' in query :
            jarvis.speak('checking time')
            time_str = datetime.datetime.now().strftime('%H:%S:%M')
            print(f'now the time is { time_str }')
            jarvis.speak(f'now the time is { time_str }')

        elif 'open' in query or 'app' in query :
            query = query.replace('open ','')
            query = query.replace(' app','')
            jarvis.speak(f'opening {jarvis.app_list[query]}')
            os.system(jarvis.app_list[query])

        elif 'mail' in query :
            to = query.replace('send mail to ','')
            try :
                jarvis.speak('what should i say???')
                content = jarvis.takeCommand()
                jarvis.sendEmail(jarvis.contact_list[to], content)
                jarvis.speak('email send')
            except Exception as e :
                print(e)
                jarvis.speak('sorry email send error')

        elif "where is" in query:
            query = query.split(" ")
            location = query[2]
            jarvis.speak("Hold on Prabhat, I will show you where " + location + " is.")
            # os.system()
            webbrowser.open("https://www.google.nl/maps/place/" + location + "/&amp;")

        elif 'nothing' in query or 'go to sleep' or 'good  job' in query:
            jarvis.speak('sleeping...')
            jarvis.active = False

        else :
            print('what can i do for you???')
            jarvis.speak('what can i do for you???')