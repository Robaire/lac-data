class _Location:
    x: float
    y: float
    z: float

    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]


class _Rotation:
    roll: float
    pitch: float
    yaw: float

    def __init__(self, e):
        self.roll = e[0]
        self.pitch = e[1]
        self.yaw = e[2]


class Transform:
    """Mock carla Transform class."""

    location: _Location
    rotation: _Rotation

    def __init__(
        self,
        p: tuple[float, float, float] = (0.0, 0.0, 0.0),
        e: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Create a mock carla Transform.

        Args:
            p: position in (x, y, z) [m]
            e: euler angles (roll, pitch, yaw) [rad]
        """
        self.location = _Location(p)
        self.rotation = _Rotation(e)
