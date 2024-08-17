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
            self.x0_fret = point[0]
            self.count += 1
            print("Calibration coordinate set for 0th fret.")
        elif self.count == 1:
            self.x3_fret = point[0]
            self.count += 1
            print("Calibration coordinate set for 3rd fret.")
        elif self.count == 2:
            self.y1_string = point[1]
            self.count += 1
            print("Calibration coordinate set for 1st string.")
        elif self.count == 3:
            self.y6_string = point[1]
            self.count = 0
            print("Calibration coordinate set for 6th string.")
        else:
            print("Calibration coordinates are already set.")

    def get_fret_and_string(self, x, y):
        """Determine the fret and string number for a given (x, y) coordinate."""

        # Check if calibration values are set
        if None in [self.x0_fret, self.x3_fret, self.y1_string, self.y6_string]:
            raise ValueError("Calibration coordinates are not fully set.")

        # Normalize the x and y coordinates
        fret_ratio = (x - self.x0_fret) / (self.x3_fret - self.x0_fret)
        string_ratio = (self.y1_string - y) / (self.y1_string - self.y6_string)

        # Clamp the ratios to be within [0, 1] range
        fret_ratio = min(max(fret_ratio, 0), 1)
        string_ratio = min(max(string_ratio, 0), 1)

        # Calculate fret and string numbers
        # Fret numbers should be in the range 1 to 3
        fret_number = round(fret_ratio * 2) + 1  # 0 to 2 range + 1 gives 1 to 3 range

        # String numbers should be in the range 1 to 6
        string_number = (
            round(string_ratio * 5) + 1
        )  # 0 to 5 range + 1 gives 1 to 6 range

        # Ensure string number is within bounds
        string_number = max(min(string_number, 6), 1)

        return fret_number, string_number
