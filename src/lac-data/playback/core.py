import os
import tarfile
import toml
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict


class FrameDataReader:
    """Reads frame data from a LAC simulator recording."""

    _initial: dict
    _frames: pd.DataFrame
    _camera_frames: dict[str, pd.DataFrame] = {}
    _custom_records: dict[str, pd.DataFrame] = {}

    def __init__(self, path: str):
        """Read LAC simulator data from a file.

        Args:
            path: The path to the data file.
        """

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        tar_file = tarfile.open(path, "r:gz")

        # Read the initialization data
        self._initial = toml.loads(
            tar_file.extractfile("initial.toml").read().decode("utf-8")
        )
        try:
            [self._initial[key] for key in ["fiducials", "lander", "rover", "cameras"]]
        except KeyError:
            raise ValueError("initial.toml is missing required keys")

        # Read the frame sensor data
        self._frames = pd.read_csv(tar_file.extractfile("frames.csv"))

        # Read frame data for each camera
        for camera in self._initial["cameras"].keys():
            try:
                self._camera_frames[camera] = pd.read_csv(
                    tar_file.extractfile(f"cameras/{camera}/{camera}_frames.csv")
                )
            except (pd.errors.EmptyDataError, KeyError):
                # Some cameras may not have any frames
                pass

        # Read any custom records
        for record in tar_file.getnames():
            if record.startswith("custom/"):
                self._custom_records[record.split("/")[-1].split(".")[0]] = pd.read_csv(
                    tar_file.extractfile(record)
                )

        tar_file.close()

    def __getitem__(self, frame: int) -> dict:
        """Convenience function to get a row from the frame data."""
        return self._frames[self._frames["frame"] == frame][0].to_dict()

    @property
    def initial(self) -> dict:
        return self._initial

    @property
    def frames(self) -> pd.DataFrame:
        return self._frames

    @property
    def camera_frames(self) -> dict[str, pd.DataFrame]:
        return self._camera_frames

    @property
    def custom_records(self) -> dict[str, pd.DataFrame]:
        return self._custom_records


class CameraDataReader:
    """Read image data from a LAC simulator recording."""

    _tar_file: tarfile.TarFile
    _frame_data: FrameDataReader

    def __init__(self, path: str):
        """Read image data from a LAC simulator recording.

        Args:
            path: The path to the data file.
        """

        # Get the tabular data
        self._frame_data = FrameDataReader(path)

        # Open the tar file
        self._tar_file = tarfile.open(path, "r:gz")

    def __del__(self):
        try:
            self._tar_file.close()
        except AttributeError:
            # There is no tar file to close
            pass

    def get_cameras(self) -> list[str]:
        """Get the list of cameras in the recording."""
        return list(self._frame_data.camera_frames.keys())

    def get_frame(self, camera: str, frame: int, use_previous_frame=False) -> dict:
        """Get the camera data for a given frame number.

        Args:
            camera: The camera to get the data for.
            frame: The frame number to get the data for.
            use_previous_frame: If True, use the previous frame if the frame number is not found.

        Returns:
            A dictionary containing the camera data.
        """

        # Get the frame data
        try:
            camera_frame = self._frame_data.camera_frames[camera]
        except KeyError:
            raise ValueError(f"Camera {camera} not found")

        # Find the row for the frame number
        try:
            if use_previous_frame:
                # Elements are ordered by frame number, so we can just take the last one
                row = camera_frame[camera_frame["frame"] <= frame].iloc[-1]
            else:
                row = camera_frame[camera_frame["frame"] == frame].iloc[0]

            return row.to_dict()
        except IndexError:
            return None

    def get_image(self, camera: str, frame: int, image_type="grayscale") -> np.ndarray:
        """Get an image from a camera for a given frame number.

        Args:
            camera: The camera to get the image from.
            frame: The frame number to get the image for.
            image_type: The type of image to get ("grayscale" or "semantic")
        Returns:
            A numpy array image from the camera.
        """

        # If semantic, check if the camera had it enabled
        if (
            image_type == "semantic"
            and not self._frame_data.initial["cameras"][camera]["use_semantic"]
        ):
            raise ValueError(f"Camera {camera} does not have semantic images enabled.")

        # Find the file name for the image at this frame
        frame_data = self.get_frame(camera, frame)

        # If the frame data is not found, return None
        if frame_data is None:
            return None

        try:
            file_name = frame_data[image_type]
        except KeyError:
            raise ValueError(
                f"image_type '{image_type}' must be either 'grayscale' or 'semantic'"
            )

        # Extract the image from the tar file
        try:
            image_file = self._tar_file.extractfile(
                f"images/{camera}/{image_type}/{file_name}"
            )
        except KeyError:
            raise RuntimeError(
                f"Image {file_name} not found. Record is likely malformed."
            )

        # Read the image as a numpy array
        # TODO: Will this work for RGB semantic images?
        return np.array(Image.open(image_file))

    def input_data(self, frame: int) -> dict:
        """Get a LAC style input data dictionary for a given frame number.

        Args:
            frame: The frame number to get the input data for.

        Returns:
            A LAC style input data dictionary.
        """

        input_data = defaultdict(dict)

        # Iterate over each camera and get grayscale images
        for camera, config in self._frame_data.initial["cameras"].items():
            # Get the grayscale image
            input_data["Grayscale"][camera] = self.get_image(camera, frame, "grayscale")

            # If semantic is enabled, get the semantic image
            if config["use_semantic"]:
                input_data["Semantic"][camera] = self.get_image(
                    camera, frame, "semantic"
                )

        return input_data

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
