import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mediapipe as mp

# MediaPipe landmark connection constants
mp_holistic = mp.solutions.holistic

def view_interactive_animation_with_slider(data_path, sequence_index):
    """
    Generates an interactive 3D plot with a slider to control the animation frame
    from a specific sequence of preprocessed MediaPipe landmarks. This version
    includes auto-zoom and hand emphasis for better visibility.

    Args:
        data_path (str): Path to the .npz file containing the landmark data.
        sequence_index (int): The index of the sequence to animate.
    """
    # 1. Load and prepare the data
    try:
        data = np.load(data_path)
        sequences = data['X']
        labels = data['y']
        action_map = {0: "Fire", 1: "Toilet", 2: "None"}
        action = action_map.get(np.argmax(labels[sequence_index]), "Unknown")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading data: {e}")
        return

    position_data = sequences[sequence_index][:, :225].reshape((-1, 75, 3))
    position_data[:, :, 1] = -position_data[:, :, 1]  # Y-axis inversion for correct orientation

    # 2. Set up the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0, 0.1, 1, 0.9], projection='3d')

    # Calculate dynamic plot limits for auto-zoom
    x_min, x_max = position_data[:, :, 0].min(), position_data[:, :, 0].max()
    y_min, y_max = position_data[:, :, 1].min(), position_data[:, :, 1].max()
    z_min, z_max = position_data[:, :, 2].min(), position_data[:, :, 2].max()
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x, mid_y, mid_z = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0

    # Define landmark connections
    connections = []
    if mp_holistic.POSE_CONNECTIONS:
        connections.extend(list(mp_holistic.POSE_CONNECTIONS))
    if mp_holistic.HAND_CONNECTIONS:
        connections.extend([[c[0] + 33, c[1] + 33] for c in mp_holistic.HAND_CONNECTIONS])
        connections.extend([[c[0] + 54, c[1] + 54] for c in mp_holistic.HAND_CONNECTIONS])

    # 3. Create the Slider
    ax_slider = fig.add_axes([0.2, 0.02, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider, label='Frame', valmin=0, valmax=len(position_data) - 1,
        valinit=0, valstep=1
    )

    # 4. Update function for the slider
    def update(frame_idx):
        frame_idx = int(frame_idx)
        ax.clear()
        landmarks = position_data[frame_idx]

        for start_idx, end_idx in connections:
            if start_idx < 75 and end_idx < 75:
                ax.plot([landmarks[start_idx, 0], landmarks[end_idx, 0]],
                        [landmarks[start_idx, 1], landmarks[end_idx, 1]],
                        [landmarks[start_idx, 2], landmarks[end_idx, 2]],
                        color='blue', linewidth=1.5)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title(f'Action: {action} | Frame: {frame_idx + 1}/{len(position_data)}')
        ax.view_init(elev=10, azim=-90)
        fig.canvas.draw_idle()

    # 5. Connect slider and draw initial frame
    frame_slider.on_changed(update)
    update(0)

    print("Interactive plot ready. Rotate the plot with your mouse and use the slider to change frames.")
    plt.show()

if __name__ == '__main__':
    # You can change the sequence index to view a different animation.
    # e.g., 0 for 'Fire', 70 for 'Toilet'.
    view_interactive_animation_with_slider('improved_preprocessed_data.npz', 0) 