import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import nltk
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import warnings
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def speek(counter):
        
    engine = pyttsx3.init()
    engine.say('good')
    if counter==5:
        engine.say("you are doing grate its already 5")
    engine.say(counter)
    # for i in range(0,10):
    #     a=i
    #     s=str(a)
    #     engine.say(s)


    # engine.say('Hello mister samaditya Jatar we are from your college SVVV and, we wanted to inform you, that you are selected for internship at TCS ')

    newVoiceRate = 145
    engine.setProperty('rate',newVoiceRate)

    # run and wait method, it processes the voice commands. 
    engine.runAndWait()
    return





def camm():
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    # speek(counter)
                    counter +=1
                    print(counter)
                        
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    return counter



# def camm():
#     cap = cv2.VideoCapture(0)
#     counter = 0 
#     stage = None
#     ## Setup mediapipe instance
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image= cv2.flip(image,1)
#             image.flags.writeable = False
        
#             # Make detection
#             results = pose.process(image)
        
#             # Recolor back to BGR
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # Extract landmarks
#             try:
#                 landmarks = results.pose_landmarks.landmark
                
#                 # Get coordinates
#                 # shh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                 # ell = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#                 # weed = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
               
#                 face = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                
#                 # Calculate angle
#                 angle = 67
#                 # acc=56777
#                 # Visualize angle
#                 cv2.putText(image, str("helll"), 
#                             tuple(np.multiply(face, [640, 480]).astype(int)), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                     )

#                 if angle < 100:
#                     stage = "down"
#                 if angle < 30 and stage =='down':
#                     stage="up"
#                     counter +=1
#                     print(counter)        
#             except:
#                 pass
            
            
#             # Render detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                     )               
            
#             cv2.imshow('Mediapipe Feed', image)

#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     return counter
def chat(a):
    user_input = a
    warnings.filterwarnings("ignore")

    nltk.download("punkt", quiet=True)

    #article = Article("https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521")
    article = Article("https://www.everydayhealth.com/fitness/best-home-exercises-stronger-back/")
    article.download()
    article.parse()
    article.nlp()

    corpus = article.text
    text = corpus
    sentence_list = nltk.sent_tokenize(text)


    def greeting_response(text):
        text = text.lower()

        bot_greetings = ["hello", "hi", "hey", "howdy", "hola", "greetings", "welcome", "good day"]
        user_greetings = ["hello", "hi", "hey", "howdy", "hola"]

        for word in text.split():
            if word in user_greetings:
                return random.choice(bot_greetings)


    def index_sort(list_var):
        length = len(list_var)
        list_index = list(range(0, length))
        x = list_var

        for i in range(length):
            for j in range(length):
                if x[list_index[i]] > x[list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp

        return list_index


    def bot_response(user_input):
        user_input = user_input.lower()
        sentence_list.append(user_input)

        bot_resp = ""

        cm = CountVectorizer().fit_transform(sentence_list)
        similarity_scores = cosine_similarity(cm[-1], cm)
        similarity_scores_list = similarity_scores.flatten()
        index = index_sort(similarity_scores_list)
        index = index[1:]

        response_flag = 0
        j = 0

        for i in range(len(index)):
            if similarity_scores_list[index[i]] > 0.0:
                bot_resp = bot_resp + " " + sentence_list[index[i]]
                
                response_flag = 1
                j += 1

            if j > 2:
                break

        if response_flag == 0:
            bot_resp = bot_resp + " I apologize, I didn't understand your request."

        sentence_list.remove(user_input)

        return bot_resp


    print("Bot: I am samantha. If you want to quit, type 'bye'.")

    exit_list = ["bye", "goodbye", "exit", "see you", "see you later", "quit", "break"]

    # user_input = input()

    if user_input.lower() in exit_list:
       return("KidneyBot: See you later!")

    

    else:
        if greeting_response(user_input) != None:
            return("KidneyBot: " + greeting_response(user_input))

        else:
            return("KidneyBot: " + bot_response(user_input))
    









def hell(request):
    return 5

# neutral wink abstract sign hungry wrong shove ranch trade include tide clerk