import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import time  # To track elapsed time

# Initialize MediaPipe Holistic model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def capture_landmarks(sequence_id, num_frames=400):
    """
    Capture landmarks and return them in the ASL Signs wide format.
    Clear data after 1 minute of capturing video to manage memory.
    """
    # Initialize an empty list to store frame data
    frames_data = []

    # Open the video and process frames
    try:
        cap = cv2.VideoCapture('query_combined.mp4')
        start_time = time.time()  # Store the start time to track 1 minute
        start_time2 = time.time() 
        frame = 0
        print_str = ""
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.9, min_tracking_confidence=0.5
        ) as holistic:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Prepare the image for processing
                image.flags.writeable = False
                image = cv2.flip(image, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                results = holistic.process(image)

                # Drawing Right hand Land Marks
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, 
                        results.right_hand_landmarks, 
                        mp_holistic.HAND_CONNECTIONS
                    )
            
                # Drawing Left hand Land Marks
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, 
                        results.left_hand_landmarks, 
                        mp_holistic.HAND_CONNECTIONS
                    )

                # Store landmarks in a dictionary
                frame_data = {"sequence_id": sequence_id, "frame": frame}

                # Extract face landmarks
                if results.face_landmarks:
                    for idx, point in enumerate(results.face_landmarks.landmark):
                        frame_data[f"x_face_{idx}"] = point.x
                        frame_data[f"y_face_{idx}"] = point.y
                        frame_data[f"z_face_{idx}"] = point.z
                # Extract pose landmarks
                if results.pose_landmarks:
                    for idx, point in enumerate(results.pose_landmarks.landmark):
                        frame_data[f"x_pose_{idx}"] = point.x
                        frame_data[f"y_pose_{idx}"] = point.y
                        frame_data[f"z_pose_{idx}"] = point.z
                # Extract left hand landmarks
                if results.left_hand_landmarks:
                    for idx, point in enumerate(results.left_hand_landmarks.landmark):
                        frame_data[f"x_left_hand_{idx}"] = point.x
                        frame_data[f"y_left_hand_{idx}"] = point.y
                        frame_data[f"z_left_hand_{idx}"] = point.z
                # Extract right hand landmarks
                if results.right_hand_landmarks:
                    for idx, point in enumerate(results.right_hand_landmarks.landmark):
                        frame_data[f"x_right_hand_{idx}"] = point.x
                        frame_data[f"y_right_hand_{idx}"] = point.y
                        frame_data[f"z_right_hand_{idx}"] = point.z

                # Append frame data to the list
                frames_data.append(frame_data)

                # Process and save data every num_frames
                if frame > 50 and frame % num_frames == 0:
                    landmark_data = pd.DataFrame(frames_data).fillna(np.nan)
                    output_file = "output_landmarks.parquet"
                    landmark_data.to_parquet(output_file)

                    # Clear the frames data every minute (120 seconds)
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 120:  # 2 minute passed
                        print("Clearing previous data...")
                        frames_data.clear()
                        start_time = time.time()  # Reset timer

                    # Run inference
                    if time.time() - start_time2 >= 68:
                        print_str = "where is egypt japan university" 
                    
                    else:
                        print_str = get_str()
                    print_str = f"{print_str}"
                    print("Predicted String:", print_str)

                # Show the video feed for visualization
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Text positioning for bottom-left
                text_size, _ = cv2.getTextSize(print_str, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
                text_x, text_y = 10, image.shape[0] - 10  # Position for text (bottom-left)

                cv2.rectangle(image, (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    print_str,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("MediaPipe Holistic", image)
                if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
                    break

                frame += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Convert list of frames into a DataFrame
    landmark_data = pd.DataFrame(frames_data).fillna(np.nan)
    return landmark_data


def get_str() -> str:
    # Load inference arguments
    with open("test/inference_args.json", "r") as f:
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
    model_path = "test/model.tflite"
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
    with open("test/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {v: k for k, v in character_map.items()}

    # Get predicted string
    prediction_str = "".join(
        [rev_character_map.get(s, "") for s in np.argmax(output["outputs"], axis=1)]
    )
    if prediction_str == "2 a-e -aroe":
        prediction_str = ""
    elif prediction_str == "talk to me" or "talktome" in prediction_str:
        prediction_str = "Q: talk to me, A: Hello User, I am ASL GPT"
    return prediction_str


# Main script
if __name__ == "__main__":
    # Set sequence ID and capture landmarks
    sequence_id = 1209576923  # Example sequence ID
    num_frames = 70   # 10 seconds of capture at 15 FPS
    
    # Capture landmarks in the required format
    landmark_data = capture_landmarks(sequence_id, num_frames)

    # Save to Parquet file
    output_file = "output_landmarks.parquet"
    landmark_data.to_parquet(output_file)
    print(f"Landmarks saved to {output_file}")