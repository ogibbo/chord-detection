class ManualCalibration:
    def __init__(self):
        # Initialize the calibration coordinates
        self.x0_fret = None
        self.x3_fret = None
        self.y1_string = None
        self.y6_string = None
        self.count = 0
        self.complete = False

    def check_if_set(self):
        if all(
            coord is not None
            for coord in [self.x0_fret, self.x3_fret, self.y1_string, self.y6_string]
        ):
            self.complete = True

    def set_fretboard_coord(self, point):
        """Set the coordinates for calibration."""

        if self.count == 0:
            self.x0_fret = point.x
            self.count += 1
            print("Calibration coordinate set for 0th fret.")
        elif self.count == 1:
            self.x3_fret = point.x
            self.count += 1
            print("Calibration coordinate set for 3rd fret.")
        elif self.count == 2:
            self.y1_string = point.y
            self.count += 1
            print("Calibration coordinate set for 1st string.")
        elif self.count == 3:
            self.y6_string = point.y
            self.count = 0
            print("Calibration coordinate set for 6th string.")
        else:
            print("Calibration coordinates are already set.")
