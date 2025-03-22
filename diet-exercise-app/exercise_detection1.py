import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from diet_recommendation import get_diet_recommendation


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_calories_burned(reps, weight, exercise_type):
    if exercise_type == 'curl':
        calories_per_rep = 0.25
    elif exercise_type == 'situp':
        calories_per_rep = 0.20
    elif exercise_type == 'squat':
        calories_per_rep = 0.30
    elif exercise_type == 'lunge':
        calories_per_rep = 0.22
    else:
        calories_per_rep = 0

    return reps * calories_per_rep * (weight / 200)


def start_counter(exercise, weight):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                if exercise == 'curl':
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage == 'down':
                        stage = "up"
                        counter += 1
                        calories_burned = calculate_calories_burned(counter, weight, 'curl')
                        print(f"Curl count: {counter}, Calories burned: {calories_burned:.2f}")

                elif exercise == 'situp':
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    angle = calculate_angle(shoulder, hip, knee)

                    if angle > 160:
                        stage = "down"
                    if angle < 100 and stage == 'down':
                        stage = "up"
                        counter += 1
                        calories_burned = calculate_calories_burned(counter, weight, 'situp')
                        print(f"Sit-up count: {counter}, Calories burned: {calories_burned:.2f}")

                elif exercise == 'squat':
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle = calculate_angle(hip, knee, ankle)

                    if angle > 160:
                        stage = "down"
                    if angle < 90 and stage == 'down':
                        stage = "up"
                        counter += 1
                        calories_burned = calculate_calories_burned(counter, weight, 'squat')
                        print(f"Squat count: {counter}, Calories burned: {calories_burned:.2f}")

                elif exercise == 'lunge':
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle = calculate_angle(hip, knee, ankle)

                    if angle > 160:
                        stage = "down"
                    if angle < 90 and stage == 'down':
                        stage = "up"
                        counter += 1
                        calories_burned = calculate_calories_burned(counter, weight, 'lunge')
                        print(f"Lunge count: {counter}, Calories burned: {calories_burned:.2f}")

            except Exception as e:
                print(f"Error: {e}")

            cv2.putText(image, f"{exercise.capitalize()} Counter: " + str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow(f'{exercise.capitalize()} Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def diet_recommendation():
    # Here you can add your logic for diet recommendation
    # For demonstration, we'll use a simple message box
    messagebox.showinfo("Diet Recommendation", "For your workout routine, consider a balanced diet with proteins, carbs, and healthy fats.")


def main():
    root = tk.Tk()
    root.title("Exercise Detection App")

    weight = simpledialog.askfloat("Input", "Enter your weight in kg:", minvalue=1)

    start_curl_button = tk.Button(root, text="Start Bicep Curl Counter", command=lambda: start_counter('curl', weight))
    start_curl_button.pack(pady=10)

    start_sit_up_button = tk.Button(root, text="Start Sit-Up Counter", command=lambda: start_counter('situp', weight))
    start_sit_up_button.pack(pady=10)

    start_squat_button = tk.Button(root, text="Start Squat Counter", command=lambda: start_counter('squat', weight))
    start_squat_button.pack(pady=10)

    start_lunge_button = tk.Button(root, text="Start Lunge Counter", command=lambda: start_counter('lunge', weight))
    start_lunge_button.pack(pady=10)

    # Add a button for diet recommendation
    diet_button = tk.Button(root, text="Get Diet Recommendation", command=diet_recommendation)
    diet_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

    root.mainloop()

def recommend_diet():
    recommended_diet = get_diet_recommendation()
    messagebox.showinfo("Diet Recommendation", recommended_diet)


if __name__ == "__main__":
    main()
