from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
import requests
from transformers import BertTokenizer, BertForSequenceClassification


app = Flask(__name__)

input_dir = './saved_models/intent_detection_healthcare_bert/'

loaded_model = BertForSequenceClassification.from_pretrained(input_dir)
loaded_model.eval()
loaded_tokenizer = BertTokenizer.from_pretrained(input_dir)
loaded_df_label = pd.read_pickle('./saved_models/intent_detection_healthcare_bert/df_label.pkl')

doctor_answers = {
    "Acne": "Benzoyl peroxide works as an antiseptic to reduce the number of bacteria on the surface of the skin. It also helps to reduce the number of whiteheads and blackheads, and has an anti-inflammatory effect. Benzoyl peroxide is usually available as a cream or gel. It's used either once or twice a day.\nYou may want to see a dermatologist for more treatment options.",
    "Shoulder pain": "applying an ice pack, covered in a damp towel, to your shoulder for about 20 minutes every few hours, to reduce pain and inflammation. using a covered hot water bottle or heat pack on your shoulder, for around 20 minutes several times a day, to relieve tight or sore muscles. \n You should consult an orthopedic specialist to assess the cause of your shoulder pain.",
    "Joint pain": "Joint pain is common, especially as you get older. There are things you can do to ease the pain but get medical help if it's very painful or it does not get better.\nThere are many possible causes of joint pain. It might be caused by an injury or a longer-lasting problem such as arthritis.try to rest the affected joint if you can,put an ice pack (or bag of frozen peas) wrapped in a towelon the painful area for up to 20 minutes every 2 to 3 hours , take painkillers, such as ibuprofen or paracetamol, but do not take ibuprofen in the first 48 hours after an injury. try to lose weight if you're overweight.make sure to do not carry anything heavy. Also do not completely stop moving the affected joint.",
    "Infected wound": "It's important to see a doctor promptly to prevent the infection from worsening. A primary care physician or urgent care provider can help.",
    "Knee pain": "You may benefit from a consultation with an orthopedic surgeon or sports medicine specialist for your knee pain.",
    "Cough": "A primary care physician can evaluate your cough and recommend appropriate treatment.",
    "Feeling dizzy": "Consulting a neurologist or an ear, nose, and throat (ENT) specialist may help identify the cause of your dizziness.",
    "Muscle pain": "Consider seeing a physical therapist or sports medicine specialist to address your muscle pain.",
    "Heart hurts": "Seek immediate medical attention by visiting the emergency room or calling emergency services if you experience chest pain or discomfort.",
    "Ear ache": "You should see an otolaryngologist (ENT doctor) for evaluation and treatment of your earache.",
    "Hair falling out": "Consulting a dermatologist can help identify potential causes and treatment options for hair loss.",
    "Head ache": "See a primary care physician to determine the cause of your headaches and discuss treatment options.",
    "Feeling cold": "If you're experiencing persistent cold sensations, it's important to see a doctor to rule out underlying medical conditions.",
    "Skin issue": "Consider consulting a dermatologist for evaluation and management of your skin issues.",
    "Stomach ache": "Depending on the severity and duration of your stomach ache, you may need to see a gastroenterologist or a primary care physician.",
    "Back pain": "A visit to a spine specialist or physical therapist can help address your back pain.",
    "Neck pain": "Consulting an orthopedic specialist or physical therapist may help alleviate your neck pain.",
    "Internal pain": "It's important to see a doctor for further evaluation if you're experiencing internal pain, as it could be a sign of an underlying condition.",
    "Blurry vision": "You should see an optometrist or ophthalmologist for an eye examination to determine the cause of your blurry vision.",
    "Body feels weak": "A visit to a primary care physician or neurologist may help identify the cause of your weakness.",
    "Hard to breath": "Seek immediate medical attention if you're having difficulty breathing. Visit the emergency room or call emergency services.",
    "Emotional pain": "Consider seeing a mental health professional such as a psychiatrist or therapist to address your emotional pain.",
    "Injury from sports": "You may benefit from evaluation and treatment by a sports medicine specialist or orthopedic surgeon.",
    "Foot ache": "Consulting a podiatrist can help identify the cause of your foot pain and recommend appropriate treatment.",
    "Open wound": "It's important to clean and dress open wounds properly. If the wound is severe, seek medical attention from a healthcare professional."
}

def medical_symptom_detector(intent):
    pt_batch = loaded_tokenizer(
        intent,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    pt_outputs = loaded_model(**pt_batch)
    _, id = torch.max(pt_outputs[0], dim=1)
    prediction = loaded_df_label.iloc[[id.item()]]['intent'].item()
    return prediction

def getResponse(intent):
    if intent in doctor_answers:
        answer = doctor_answers[intent]
        endpoint = 'https://api.together.xyz/v1/chat/completions'
        res = requests.post(endpoint, json={
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": [
                "[/INST]",
                "</s>"
            ],
            "messages": [
                {
                    "content": f"Reformulate while talking as a doctor, be brief, do not add any additional information, also write the treatments in the form of a list {answer}.",
                    "role": "user"
                }
            ]
        }, headers={
            "Authorization": "Bearer 2a61183859d8200f64194ef275194f62a6f97ed4e644d3354d07c7657cad8718",
        })

        return res.json()['choices'][0]['message']['content']
    else:
        print("I'm sorry, I don't have information about that.")


@app.route('/chat', methods=['POST'])
def chat():
    request_data = request.get_json()
    user_message = request_data['message']
    intent = medical_symptom_detector(user_message)
    response = getResponse(intent)
    return jsonify({'message': response})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
