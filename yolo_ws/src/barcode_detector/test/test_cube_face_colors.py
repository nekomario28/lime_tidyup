"""Tests for face colour observation and the abstain discipline."""

import unittest

import cv2
import numpy as np

from barcode_detector.cube_face_colors import (
    REFERENCE_LAB,
    REFERENCE_RGB,
    classify_lab,
    face_color_labels,
    observe_face_colors,
    project_point,
    rgb_to_lab,
)
from barcode_detector.cube_pose_hypotheses import FACE_NAMES
from barcode_detector.cube_pose_hypotheses import visible_face_names
from barcode_detector.cube_visibility import (
    CANONICAL_VERTEX_SIGNS,
    visible_vertex_mask,
)
from test_cube_visibility import CAMERA_MATRIX


SIDE_M = 0.057
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def _rotation_from_axis_angle(axis, angle):
    import numpy as np

    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def _quaternion_from_matrix(m):
    import math

    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s, 0.25 * s)
    index = int(max(range(3), key=lambda i: m[i][i]))
    if index == 0:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if index == 1:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s,
                (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s,
            0.25 * s, (m[1][0] - m[0][1]) / s)


def three_face_pose(distance_m=0.45):
    """A centred pose that shows the +x, -y and -z faces."""
    import numpy as np

    direction = np.array([-1.0, 1.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(direction, target)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(np.clip(direction @ target, -1.0, 1.0)))
    rotation = _rotation_from_axis_angle(axis, angle)
    position = (0.02, 0.01, distance_m)
    return position, _quaternion_from_matrix(rotation), rotation


THREE_FACE_POSITION_M, THREE_FACE_ORIENTATION, _R = three_face_pose()
VISIBLE_SLOTS = visible_face_names(
    THREE_FACE_POSITION_M, THREE_FACE_ORIENTATION, side_length_m=SIDE_M
)
FACE_TO_LABEL = {
    '+x': 'red', '-x': 'orange',
    '+y': 'white', '-y': 'yellow',
    '+z': 'blue', '-z': 'green',
}
PAINTED = {
    slot: REFERENCE_RGB[FACE_TO_LABEL[slot]] for slot in VISIBLE_SLOTS
}


def project_with_rotation(vertex, position, rotation):
    camera = [
        sum(rotation[i][j] * vertex[j] for j in range(3)) + position[i]
        for i in range(3)
    ]
    return (
        CAMERA_MATRIX[0] * camera[0] / camera[2] + CAMERA_MATRIX[2],
        CAMERA_MATRIX[4] * camera[1] / camera[2] + CAMERA_MATRIX[5],
    )


def keypoints_for_view():
    half = SIDE_M / 2.0
    return [
        project_with_rotation(
            tuple(sign * half for sign in signs),
            THREE_FACE_POSITION_M,
            _R,
        )
        for signs in CANONICAL_VERTEX_SIGNS
    ]


def _face_corners(slot, side_m=SIDE_M):
    """Object-frame corners of a face, in a ring."""
    from barcode_detector.cube_pose_hypotheses import FACE_NORMALS

    half = side_m / 2.0
    normal = FACE_NORMALS[slot]
    axis = [index for index in range(3) if abs(normal[index]) > 0.5][0]
    others = [index for index in range(3) if index != axis]
    corners = []
    for first, second in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        point = [0.0, 0.0, 0.0]
        point[axis] = normal[axis] * half
        point[others[0]] = first * half
        point[others[1]] = second * half
        corners.append(tuple(point))
    return corners


def render_faces(painted):
    """Render the visible faces as filled quads in the given colours."""
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, dtype=np.uint8)
    for slot, rgb in painted.items():
        points = []
        for corner in _face_corners(slot):
            projected = project_point(
                corner,
                THREE_FACE_POSITION_M,
                THREE_FACE_ORIENTATION,
                CAMERA_MATRIX,
            )
            points.append(projected)
        polygon = np.array(points, dtype=np.int32)
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.fillConvexPoly(image, polygon, color)
    return image


class ReferenceClassificationTest(unittest.TestCase):
    """Every anchor classifies as itself; anything unclear abstains."""

    def test_reference_colours_round_trip(self):
        for label, lab in REFERENCE_LAB.items():
            self.assertEqual(classify_lab(lab), label)

    def test_labels_are_deterministic(self):
        self.assertEqual(face_color_labels(), tuple(sorted(REFERENCE_LAB)))

    def test_shading_never_produces_a_wrong_label(self):
        # A shaded face may abstain; it must never be mislabelled. Deep shading
        # of a warm colour can genuinely approach its neighbour, so "abstain"
        # is the correct answer there, not a guess.
        for label in ('red', 'orange', 'yellow', 'green', 'blue'):
            rgb = REFERENCE_RGB[label]
            for factor in (0.4, 0.5, 0.7):
                shaded = tuple(value * factor for value in rgb)
                result = classify_lab(rgb_to_lab(shaded))
                self.assertIn(result, (label, None),
                              f'{label} at {factor} shading -> {result}')

    def test_well_separated_colours_survive_moderate_shading(self):
        for label in ('red', 'green', 'blue', 'yellow'):
            rgb = REFERENCE_RGB[label]
            shaded = tuple(value * 0.7 for value in rgb)
            self.assertEqual(classify_lab(rgb_to_lab(shaded)), label)

    def test_shaded_white_is_indistinguishable_from_grey(self):
        # Half-brightness white and mid grey are the same colour: abstain.
        self.assertIsNone(
            classify_lab(rgb_to_lab((122, 122, 122)))
        )

    def test_grey_is_not_forced_into_a_hue(self):
        self.assertIsNone(classify_lab(rgb_to_lab((120, 120, 120))))

    def test_white_is_recognised_by_lightness(self):
        self.assertEqual(classify_lab(rgb_to_lab((245, 245, 245))), 'white')

    def test_between_two_anchors_abstains(self):
        red = REFERENCE_LAB['red']
        orange = REFERENCE_LAB['orange']
        middle = tuple((a + b) / 2.0 for a, b in zip(red, orange))
        self.assertIsNone(classify_lab(middle))

    def test_non_finite_input_is_not_a_colour(self):
        self.assertIsNone(classify_lab((float('nan'), 0.0, 0.0)))


class FaceObservationTest(unittest.TestCase):
    """Observed colours come from sticker patches, not the whole quad."""

    def test_painted_faces_are_read_back(self):
        image = render_faces(PAINTED)
        observed, evidence = observe_face_colors(
            image,
            THREE_FACE_POSITION_M,
            THREE_FACE_ORIENTATION,
            CAMERA_MATRIX,
            VISIBLE_SLOTS,
            side_length_m=SIDE_M,
        )
        self.assertEqual(observed['+x'], 'red')
        self.assertEqual(observed['-y'], 'yellow')
        self.assertEqual(observed['-z'], 'green')
        for slot in VISIBLE_SLOTS:
            self.assertEqual(evidence[slot]['status'], 'observed')
            self.assertEqual(evidence[slot]['sample_count'], 9)

    def test_an_unpainted_face_abstains(self):
        image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, dtype=np.uint8)
        observed, evidence = observe_face_colors(
            image,
            THREE_FACE_POSITION_M,
            THREE_FACE_ORIENTATION,
            CAMERA_MATRIX,
            VISIBLE_SLOTS,
            side_length_m=SIDE_M,
        )
        for slot in VISIBLE_SLOTS:
            self.assertIsNone(observed[slot])
            self.assertEqual(evidence[slot]['status'], 'unknown')

    def test_black_separators_do_not_dominate_the_sticker_patch(self):
        image = render_faces(PAINTED)
        # Draw a black grid over the +x face to imitate separators.
        for slot in ('+x',):
            points = [
                project_point(
                    corner,
                    THREE_FACE_POSITION_M,
                    THREE_FACE_ORIENTATION,
                    CAMERA_MATRIX,
                )
                for corner in _face_corners(slot)
            ]
            polygon = np.array(points, dtype=np.int32)
            cv2.polylines(image, [polygon], True, (0, 0, 0), 3)
        observed, _ = observe_face_colors(
            image,
            THREE_FACE_POSITION_M,
            THREE_FACE_ORIENTATION,
            CAMERA_MATRIX,
            VISIBLE_SLOTS,
            side_length_m=SIDE_M,
        )
        self.assertEqual(observed['+x'], 'red')

    def test_faces_outside_the_image_abstain(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observed, evidence = observe_face_colors(
            image,
            THREE_FACE_POSITION_M,
            THREE_FACE_ORIENTATION,
            CAMERA_MATRIX,
            VISIBLE_SLOTS,
            side_length_m=SIDE_M,
        )
        for slot in VISIBLE_SLOTS:
            self.assertIsNone(observed[slot])
            self.assertEqual(evidence[slot]['status'],
                             'insufficient_samples')


class SlotCoverageTest(unittest.TestCase):
    """The sampler must be able to name every face it is asked about."""

    def test_all_face_slots_have_geometry(self):
        for slot in FACE_NAMES:
            self.assertEqual(len(_face_corners(slot)), 4)

    def test_keypoints_are_not_needed_for_sampling(self):
        # Colour observation depends on the pose, not on the detector output.
        self.assertEqual(len(keypoints_for_view()), 8)

    def test_the_canonical_view_has_three_visible_faces(self):
        self.assertEqual(len(VISIBLE_SLOTS), 3)
        mask = visible_vertex_mask(
            THREE_FACE_POSITION_M, THREE_FACE_ORIENTATION,
            side_length_m=SIDE_M,
        )
        self.assertEqual(sum(1 for visible in mask if visible), 7)


if __name__ == '__main__':
    unittest.main()
