import io
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml
from PIL import Image

from .._util import transform_to_tuple


class Recorder:
    """Records data from a simulation run to an archive."""

    # To use this, initialize it in the agent setup, and then call it every run_step
    # When the agent is done, call stop to save the data
    agent: None  # AutonomousAgent
    max_size: float  # Maximum size of the archive in GB
    tar_path: Path  # Output archive path
    tar_file: tarfile.TarFile  # Output archive

    # Recording state
    done: bool = False  # Whether the recording is done
    paused: bool = False  # Whether the recording is paused

    # Data buffers
    metadata: dict  # Metadata
    initial: dict  # Initial data
    frames: list  # Frames
    camera_frames: dict  # Camera frames

    # Custom recorders
    custom_records: dict = {}  # Custom records

    def __init__(self, agent, output="", max_size: float = 10):
        """Initialize the recorder.

        Args:
            agent: The agent to record data from
            output: Output directory or file name
            max_size: Maximum size of the archive in GB
        """
        self.agent = agent
        self.max_size = max_size

        # Create the archive file
        self.tar_path = self._parse_file_name(output)
        self.tar_file = tarfile.open(self.tar_path, "w:gz")

        # Create numerical data buffers
        self.metadata = {
            "data_spec": "0.1",
            "description": "",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": "",
            "frames": 0,
        }
        self.initial = {
            "fiducials": None,
            "lander": None,
            "rover": None,
            "cameras": {},
        }
        self.frames = []
        self.camera_frames = {}

        # Record agent configuration and simulation start parameters
        self.initial["fiducials"] = self.agent.use_fiducials()  # bool
        self.initial["lander"] = transform_to_tuple(
            self.agent.get_initial_lander_position()
        )  # [x, y, z, roll, pitch, yaw]
        self.initial["rover"] = transform_to_tuple(
            self.agent.get_initial_position()
        )  # [x, y, z, roll, pitch, yaw]

        # Record initial sensor configuration
        for camera, config in self.agent.sensors().items():
            self.initial["cameras"][str(camera)] = config

            # Initialize camera frame buffers
            self.camera_frames[str(camera)] = []

        # Write initial data to the archive as a toml file
        self._add_file(
            "initial.toml", io.BytesIO(toml.dumps(self.initial).encode("utf-8"))
        )

    def __call__(self, frame: int, input_data: dict):
        """Record a frame of data from the simulation."""
        self.record_all(frame, input_data)

    def record_all(self, frame: int, input_data: dict):
        """Record a frame of data from the simulation.

        Args:
            frame: The frame number
            input_data: The input data from the simulation
        """

        # The simulation may keep running after we are done recording so do nothing
        if self.done or self.paused:
            return

        self.record_sensors(frame)
        self.record_cameras(frame, input_data)

    def record_sensors(self, frame: int):
        """Record state and sensor data.

        Args:
            frame: The frame number
        """

        if self.done or self.paused:
            return

        # Get agent data
        pose = transform_to_tuple(self.agent.get_transform())
        imu_data = self.agent.get_imu_data()  # [ax, ay, az, gx, gy, gz]
        mission_time = self.agent.get_mission_time()  # float [s]
        power = self.agent.get_current_power()  # float [Wh]
        linear_speed = self.agent.get_linear_speed()  # float [m/s]
        angular_speed = self.agent.get_angular_speed()  # float [rad/s]
        cover_angle = self.agent.get_radiator_cover_angle()  # float [rad]

        self.frames.append(
            {
                "frame": frame,
                "x": pose[0],
                "y": pose[1],
                "z": pose[2],
                "roll": pose[3],
                "pitch": pose[4],
                "yaw": pose[5],
                "accel_x": imu_data[0],
                "accel_y": imu_data[1],
                "accel_z": imu_data[2],
                "gyro_x": imu_data[3],
                "gyro_y": imu_data[4],
                "gyro_z": imu_data[5],
                "mission_time": mission_time,
                "power": power,
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
                "cover_angle": cover_angle,
            }
        )

    def record_cameras(self, frame: int, input_data: dict):
        """Record camera sensor data."""

        if self.done or self.paused:
            return

        # Iterate over items for each configured camera
        for camera, config in self.agent.sensors().items():
            enabled = self.agent.get_camera_state(camera)  # bool

            # Skip if the camera is not enabled
            if not enabled:
                continue

            # Skip if the camera is enabled, but has no image
            if input_data["Grayscale"][camera] is None:
                continue

            # This means that the light state may not be accurate for frames without images, but we rarely change the lights

            position = transform_to_tuple(self.agent.get_camera_position(camera))
            light_intensity = self.agent.get_light_state(camera)

            # This is always a fixed offset from the camera position and we dont really care about it anyways
            # light_position = transform_to_tuple(self.agent.get_light_position(camera))

            # Get the grayscale image if the camera is active
            # This should never raise a KeyError since we check enabled first
            image = input_data["Grayscale"][camera]
            grayscale = self._add_image(image, camera, "grayscale", frame).split("/")[
                -1
            ]

            # Only attempt this if the camera has semantics enabled
            if config["use_semantic"]:
                # This should never raise a KeyError since we check enabled first
                # This should always have an image since we check grayscale earlier
                image = input_data["Semantic"][camera]
                semantic = self._add_image(image, camera, "semantic", frame).split("/")[
                    -1
                ]
            else:
                semantic = ""

            # Add the frame data to the camera buffer
            self.camera_frames[str(camera)].append(
                {
                    "frame": frame,
                    "enable": enabled,
                    "camera_x": position[0],
                    "camera_y": position[1],
                    "camera_z": position[2],
                    "camera_roll": position[3],
                    "camera_pitch": position[4],
                    "camera_yaw": position[5],
                    "light_intensity": light_intensity,
                    # "light_x": light_position[0],
                    # "light_y": light_position[1],
                    # "light_z": light_position[2],
                    # "light_roll": light_position[3],
                    # "light_pitch": light_position[4],
                    # "light_yaw": light_position[5],
                    "grayscale": grayscale,
                    "semantic": semantic,
                }
            )

        self._check_size()

        """
        input_data is a dictionary that contains the sensors data:
        - Active sensors will have their data represented as a numpy array 
        - Active sensors without any data in this tick will instead contain 'None' > ???
        - Inactive sensors will not be present in the dictionary. > KeyError

        Example:

        input_data = {
            'Grayscale': {
                carla.SensorPosition.FrontLeft:  np.array(...),
                carla.SensorPosition.FrontRight:  np.array(...),
            },
            'Semantic':{
                carla.SensorPosition.FrontLeft:  np.array(...),
            }
        }
        """

    def record_custom(self, frame: int, record_name: str, data: dict):
        """Record custom data."""
        data["frame"] = frame
        try:
            self.custom_records[record_name].append(data)
        except KeyError:
            self.custom_records[record_name] = [data]

    def _add_image(self, image, camera, type: str, frame: int) -> str:
        """Add an image in the archive."""

        # Determine the filepath of the image
        filepath = f"cameras/{str(camera)}/{type}/{str(camera)}_{type}_{str(frame)}.png"

        # Convert the image to a PIL image and save it to a buffer
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        self._add_file(filepath, buffer)

        return filepath

    def _check_size(self):
        """Check if the archive is over the size limit."""

        self.tar_file.flush()
        if self.tar_path.stat().st_size > self.max_size * 1024 * 1024 * 1024:
            self.stop()

    def _parse_file_name(self, file_path: str) -> Path:
        """Parse the output file name."""

        # Expand the user path and make it absolute
        file_path = Path(file_path).expanduser().resolve()

        # Check if the output is a directory
        if file_path.is_dir():
            # Create a new file name with the current timestamp
            file_path = (
                file_path / f"sim-{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.lac"
            )

        if file_path.exists():
            # Append a timestamp to the file name
            file_path = (
                file_path.stem
                / f"-{datetime.now().strftime('%H.%M.%S')}"
                / file_path.suffix
            )

        # Return the file path
        return file_path

    def description(self, description: str):
        """Set a description for the run if desired."""

        if self.done:
            raise RuntimeError("Cannot set description after recording is done.")

        self.metadata["description"] = description

    def _add_file(self, name: str, data: io.BytesIO):
        """Add a file to the archive."""
        data.seek(0)

        # Build a tarinfo object
        tar_info = tarfile.TarInfo(name=name)
        tar_info.size = len(data.getvalue())
        tar_info.mtime = int(datetime.now().timestamp())
        tar_info.mode = 0o666  # Any user can read and write

        self.tar_file.addfile(tar_info, data)

    def pause(self):
        """Pause the recording."""
        self.paused = True

    def resume(self):
        """Resume the recording."""

        if self.done:
            raise RuntimeError("Cannot resume recording after it has been stopped.")

        self.paused = False

    def is_done(self):
        """Check if the recording is done."""
        return self.done

    def stop(self):
        """Stop recording and save the archive."""

        # Cannot save if the recording as already been saved
        if self.done:
            return

        # Write metadata to the archive
        self.metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["frames"] = len(self.frames)
        self._add_file(
            "metadata.toml", io.BytesIO(toml.dumps(self.metadata).encode("utf-8"))
        )

        # Write frame data
        csv_buffer = io.StringIO()
        pd.DataFrame(self.frames).to_csv(csv_buffer, index=False)
        self._add_file("frames.csv", io.BytesIO(csv_buffer.getvalue().encode("utf-8")))

        # Write camera frame data
        for camera in self.camera_frames.keys():
            # Check if the camera has any frames
            if len(self.camera_frames[camera]) == 0:
                continue

            # Determine the filepath
            filepath = f"cameras/{str(camera)}/{str(camera)}_frames.csv"

            # Create a dataframe and save it to a buffer
            csv_buffer = io.StringIO()
            pd.DataFrame(self.camera_frames[camera]).to_csv(csv_buffer, index=False)
            self._add_file(filepath, io.BytesIO(csv_buffer.getvalue().encode("utf-8")))

        # Write custom data
        for name, record in self.custom_records.items():
            # Create a dataframe and save it to a buffer
            csv_buffer = io.StringIO()
            pd.DataFrame(record).to_csv(csv_buffer, index=False)
            self._add_file(
                f"custom/{str(name)}.csv",
                io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
            )

        self.tar_file.close()  # Flush the buffer to the file
        self.done = True  # Set the done flag
