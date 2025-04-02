class G1_CONSTS:
    def __init__(self):
        self.FEET_SITES = [
            "left_foot",
            "right_foot",
        ]

        self.HAND_SITES = [
            "left_palm",
            "right_palm",
        ]

        self.LEFT_FEET_GEOMS = ["left_foot"]
        self.RIGHT_FEET_GEOMS = ["right_foot"]
        self.FEET_GEOMS = self.LEFT_FEET_GEOMS + self.RIGHT_FEET_GEOMS

        self.ROOT_BODY = "torso_link"

        self.GRAVITY_SENSOR = "upvector"
        self.GLOBAL_LINVEL_SENSOR = "global_linvel"
        self.GLOBAL_ANGVEL_SENSOR = "global_angvel"
        self.LOCAL_LINVEL_SENSOR = "local_linvel"
        self.ACCELEROMETER_SENSOR = "accelerometer"
        self.GYRO_SENSOR = "gyro"

        self.RESTRICTED_JOINT_RANGE = (
            # Left leg.
            (-1.57, 1.57),
            (-0.5, 0.5),
            (-0.7, 0.7),
            (0, 1.57),
            (-0.4, 0.4),
            (-0.2, 0.2),
            # Right leg.
            (-1.57, 1.57),
            (-0.5, 0.5),
            (-0.7, 0.7),
            (0, 1.57),
            (-0.4, 0.4),
            (-0.2, 0.2),
            # Waist.
            (-2.618, 2.618),
            (-0.52, 0.52),
            (-0.52, 0.52),
            # Left shoulder.
            (-3.0892, 2.6704),
            (-1.5882, 2.2515),
            (-2.618, 2.618),
            (-1.0472, 2.0944),
            (-1.97222, 1.97222),
            (-1.61443, 1.61443),
            (-1.61443, 1.61443),
            # Right shoulder.
            (-3.0892, 2.6704),
            (-2.2515, 1.5882),
            (-2.618, 2.618),
            (-1.0472, 2.0944),
            (-1.97222, 1.97222),
            (-1.61443, 1.61443),
            (-1.61443, 1.61443),
        )