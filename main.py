from flask import Flask, render_template, request, Response
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
dataset = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        uploaded_file.save("static/projectvideo.mp4")
        return render_template('predict.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    capture = cv2.VideoCapture("static/projectvideo.mp4")
    while True:
        success, frame = capture.read()
        if not success:
            break
        else:
            plates = dataset.detectMultiScale(frame, 1.2)
            for x, y, w, h in plates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 4)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(debug=True)
