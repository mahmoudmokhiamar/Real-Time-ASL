import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import keyboard 

# Initialize MediaPipe Holistic model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def capture_landmarks(sequence_id, num_frames=400):
    """
    Capture landmarks and return them in the ASL Signs wide format.
    """
    frames_data = []
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.9, min_tracking_confidence=0.5
    ) as holistic:
        frame = 0
        print_str = ""
        
        while cap.isOpened():
            if frame > 30 and frame % num_frames == 0:
                landmark_data = pd.DataFrame(frames_data).fillna(np.nan)
                output_file = "output_landmarks.parquet"
                landmark_data.to_parquet(output_file)
                cur_iter = frame // num_frames
                print_str = get_str()
                print_str = f"{cur_iter}: {print_str}"

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            frame_data = {"sequence_id": sequence_id, "frame": frame}
            
            # Extract landmarks (only some parts, e.g., eyes, wrists, and shoulders)
            selected_landmarks = {
                "pose": [11, 12, 15, 16],  # Shoulders and Wrists
                "face": [33, 263],  # Eyes
                "left_hand": [4, 8, 12],  # Selected fingers
                "right_hand": [4, 8, 12]  # Selected fingers
            }
            
            for name, indices in selected_landmarks.items():
                landmarks = getattr(results, f"{name}_landmarks", None)
                if landmarks:
                    for idx in indices:
                        point = landmarks.landmark[idx]
                        frame_data[f"{name}_x_{idx}"] = point.x
                        frame_data[f"{name}_y_{idx}"] = point.y
                        frame_data[f"{name}_z_{idx}"] = point.z
                        
                        # Draw selected landmarks
                        h, w, c = image.shape
                        cx, cy = int(point.x * w), int(point.y * h)
                        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            
            frames_data.append(frame_data)

            # Show the video feed for visualization
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image,
                print_str,
                (100, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 255, 0),
                2,
            )
            cv2.imshow("MediaPipe Holistic", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame += 1

    cap.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(frames_data).fillna(np.nan)

def get_str() -> str:
    # Load inference arguments
    with open("./Inference/inference_args.json", "r") as f:
        inference_config = json.load(f)
    selected_columns = inference_config["selected_columns"]

    # Load captured landmarks
    pq_file = "./output_landmarks.parquet"
    df = pd.read_parquet(pq_file)

    # Handle missing columns
    missing_columns = [col for col in selected_columns if col not in df.columns]
    for col in missing_columns:
        df[col] = 0.0  # Fill missing columns with zeros

    # Ensure columns are in the correct order
    frames = df[selected_columns].to_numpy(dtype=np.float32)

    # Load TFLite model
    model_path = "./Inference/model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Verify input shape
    input_details = interpreter.get_input_details()
    print("Input Details:", input_details)

    # Check input shape
    expected_shape = input_details[0]["shape"]  # [batch_size, num_features]
    if len(frames.shape) != len(expected_shape):
        raise ValueError(
            f"Dimension mismatch: Expected {len(expected_shape)} dimensions, but got {len(frames.shape)}."
        )
    if frames.shape[1] != expected_shape[1]:
        raise ValueError(
            f"Shape mismatch: Model expects {expected_shape[1]} features, but got {frames.shape[1]}."
        )

    # Allocate tensors
    interpreter.resize_tensor_input(input_details[0]["index"], frames.shape)
    interpreter.allocate_tensors()

    # Run inference
    prediction_fn = interpreter.get_signature_runner("serving_default")
    output = prediction_fn(inputs=frames)

    # Decode predictions
    with open("./Inference/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {v: k for k, v in character_map.items()}

    # Get predicted string
    prediction_str = "".join(
        [rev_character_map.get(s, "") for s in np.argmax(output["outputs"], axis=1)]
    )
    if prediction_str == "2 a-e -aroe":
        prediction_str = ""
    elif prediction_str == "talk to me":
        prediction_str = "Hello User, I am ASL GPT, A framework for American Sign Language Recognition, to help who needs me."
    return prediction_str

if __name__ == "__main__":
    sequence_id = 1209576923
    num_frames = 70
    landmark_data = capture_landmarks(sequence_id, num_frames)
    output_file = "output_landmarks.parquet"
    landmark_data.to_parquet(output_file)
    print(f"Landmarks saved to {output_file}")
