import logging
import yaml
from typing import Dict, Any
import cv2
import numpy as np

from src.utils.camera import CameraManager
from src.models.pose_model import PoseModel
from src.evaluation.pose_evaluator import PoseEvaluator
from src.visualization.pose_visualizer import PoseVisualizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config('config/model_config.yaml')

        # Initialize components
        camera_manager = CameraManager(config['camera']['max_index'])
        pose_model = PoseModel(config['model']['path'], config)
        pose_evaluator = PoseEvaluator(config)
        pose_visualizer = PoseVisualizer(config)

        # Create window and set mouse callback
        window_name = 'Gym Quality Review'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, pose_visualizer.handle_mouse_event)

        # Main loop
        with camera_manager as camera:
            while True:
                # Read frame
                frame = camera.read_frame()
                if frame is None:
                    break

                # Get pose landmarks
                landmarks = pose_model.get_pose_landmarks(frame)
                if landmarks is None:
                    continue

                # Make prediction
                prediction = pose_model.predict(landmarks)
                score = float(prediction) if isinstance(prediction, (int, float)) else 0.5
                label = "Good Form" if score > 0.5 else "Need Improvement"

                # Evaluate pose
                evaluation = pose_evaluator.evaluate_pose(landmarks, prediction)
                evaluation['score'] = score
                evaluation['label'] = label

                # Visualize results
                frame = pose_visualizer.draw_pose(frame, landmarks, evaluation)

                # Show frame
                cv2.imshow(window_name, frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 