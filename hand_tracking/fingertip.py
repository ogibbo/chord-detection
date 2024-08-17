class FingerTip:
    def __init__(self):
        self.x = None
        self.y = None

    def set_coords(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

    def get_fret_and_string(self, ref_frame):
        """Determine the fret and string number for a given (x, y) coordinate."""

        # Normalize the x and y coordinates
        fret_ratio = (self.x - ref_frame.x0_fret) / (
            ref_frame.x3_fret - ref_frame.x0_fret
        )
        string_ratio = (ref_frame.y1_string - self.y) / (
            ref_frame.y1_string - ref_frame.y6_string
        )

        # Clamp the ratios to be within [0, 1] range
        fret_ratio = min(max(fret_ratio, 0), 1)
        string_ratio = min(max(string_ratio, 0), 1)

        # Calculate fret and string numbers
        # Fret numbers should be in the range 1 to 3
        fret_number = round(fret_ratio * 2) + 1

        # String numbers should be in the range 1 to 6
        string_number = round(string_ratio * 5) + 1

        # Ensure string number is within bounds
        string_number = max(min(string_number, 6), 1)

        return (fret_number, string_number)
