def get_fret_string_dict(ref_frame, f1, f2, f3):
    """Return the chord dictionary for the given fingertip coordinates."""
    return {
        f1.get_fret_and_string(ref_frame),
        f2.get_fret_and_string(ref_frame),
        f3.get_fret_and_string(ref_frame),
    }
