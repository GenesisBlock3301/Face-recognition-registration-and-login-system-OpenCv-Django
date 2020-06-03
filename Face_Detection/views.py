from django.shortcuts import render,redirect
from Face_Detection.detection import FaceRecognition
from .forms import *
from django.contrib import messages

faceRecognition = FaceRecognition()

def home(request):
    return render(request,'faceDetection/home.html')


def register(request):
    if request.method == "POST":
        form = ResgistrationForm(request.POST or None)
        if form.is_valid():
            form.save()
            print("IN HERE")
            messages.success(request,"SuceessFully registered")
            addFace(request.POST['face_id'])
            redirect('home')
        else:
            messages.error(request,"Account registered failed")
    else:
        form = ResgistrationForm()

    return render(request, 'faceDetection/register.html', {'form':form})

def addFace(face_id):
    face_id = face_id
    faceRecognition.faceDetect(face_id)
    faceRecognition.trainFace()
    return redirect('/')

def login(request):
    face_id = faceRecognition.recognizeFace()
    print(face_id)
    return redirect('greeting' ,str(face_id))

def Greeting(request,face_id):
    face_id = int(face_id)
    context ={
        'user' : UserProfile.objects.get(face_id = face_id)
    }
    return render(request,'faceDetection/greeting.html',context=context)

