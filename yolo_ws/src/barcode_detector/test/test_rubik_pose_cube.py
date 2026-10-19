"""Tests for the generated 57 mm Rubik-style pose cube asset."""

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

from barcode_detector.cube_pose_hypotheses import FACE_NAMES
from barcode_detector.rubik_pose_cube import (
    FACELET_COUNT,
    FACELET_GAP_M,
    FACELET_GRID_SIZE,
    FACELET_SIZE_M,
    FACELET_THICKNESS_M,
    FACE_SPECS,
    MASS_KG,
    MODEL_NAME,
    SIDE_LENGTH_M,
    RubikPoseCubeError,
    build_model_sdf,
    check_geometry,
    facelet_count_in,
    facelet_extent_m,
    facelet_pose_and_size,
    write_model,
)


def _model(sdf_text=None):
    root = ET.fromstring(sdf_text or build_model_sdf())
    model = root.find('model')
    assert model is not None
    return model


class GeometryTest(unittest.TestCase):
    """The constants must describe a 57 mm cube that can actually be built."""

    def test_side_length_is_the_measured_cube(self):
        self.assertAlmostEqual(SIDE_LENGTH_M, 0.057, places=9)

    def test_facelet_grid_fits_the_face(self):
        self.assertAlmostEqual(facelet_extent_m(), 0.0510, places=9)
        self.assertLess(facelet_extent_m(), SIDE_LENGTH_M)

    def test_check_geometry_passes(self):
        self.assertTrue(check_geometry())

    def test_facelet_grid_is_three_by_three_per_face(self):
        cells = facelet_pose_and_size('pos_x')
        self.assertEqual(len(cells), FACELET_GRID_SIZE ** 2)
        self.assertEqual(FACELET_COUNT, 54)

    def test_plates_sit_proud_of_the_core(self):
        _, _, pose, box = facelet_pose_and_size('pos_x')[0]
        self.assertAlmostEqual(
            pose[0], SIDE_LENGTH_M / 2.0 + FACELET_THICKNESS_M / 2.0, places=9
        )
        self.assertEqual(box[0], FACELET_THICKNESS_M)

    def test_neighbour_gap_is_exact(self):
        cells = facelet_pose_and_size('pos_x')
        centres = sorted({round(cell[2][1], 9) for cell in cells})
        self.assertEqual(len(centres), 3)
        step = centres[1] - centres[0]
        self.assertAlmostEqual(step, FACELET_SIZE_M + FACELET_GAP_M, places=9)
        self.assertAlmostEqual(step - FACELET_SIZE_M, FACELET_GAP_M, places=9)

    def test_every_face_has_its_own_colour(self):
        colors = [color for _, _, color in FACE_SPECS]
        self.assertEqual(len({tuple(color) for color in colors}), 6)
        names = sorted(name for _, name, _ in FACE_SPECS)
        self.assertEqual(names, sorted(
            ['red', 'orange', 'blue', 'green', 'white', 'yellow']
        ))

    def test_inconsistent_constants_raise(self):
        import barcode_detector.rubik_pose_cube as module

        original = module.FACELET_GAP_M
        module.FACELET_GAP_M = 0.05
        try:
            with self.assertRaises(RubikPoseCubeError):
                module.check_geometry()
        finally:
            module.FACELET_GAP_M = original


class GeneratedSdfTest(unittest.TestCase):
    """The generated document must be a usable, mesh-free model."""

    def test_model_identity_and_body_count(self):
        model = _model()
        self.assertEqual(model.attrib['name'], MODEL_NAME)
        self.assertEqual(len(model.findall('./link')), 1)
        self.assertEqual(model.find('static').text, 'false')

    def test_static_variant_only_changes_the_flag(self):
        dynamic = build_model_sdf()
        frozen = build_model_sdf(static=True)
        self.assertIn('<static>true</static>', frozen)
        self.assertIn('<static>false</static>', dynamic)
        self.assertEqual(
            frozen.replace('<static>true</static>',
                           '<static>false</static>'),
            dynamic,
        )

    def test_collision_is_one_box_of_the_nominal_size(self):
        model = _model()
        size = model.findtext('./link/collision/geometry/box/size')
        self.assertEqual(size, '0.057 0.057 0.057')

    def test_inertia_matches_a_solid_cube(self):
        model = _model()
        expected = MASS_KG * SIDE_LENGTH_M ** 2 / 6.0
        self.assertAlmostEqual(
            float(model.findtext('./link/inertial/inertia/ixx')),
            expected, places=8,
        )
        self.assertAlmostEqual(
            float(model.findtext('./link/inertial/mass')), MASS_KG, places=9
        )

    def test_fifty_four_facelet_visuals_plus_a_core(self):
        model = _model()
        visuals = model.findall('./link/visual')
        self.assertEqual(len(visuals), FACELET_COUNT + 1)
        self.assertEqual(facelet_count_in(build_model_sdf()), FACELET_COUNT)
        names = {visual.attrib['name'] for visual in visuals}
        self.assertIn('grooved_core', names)

    def test_every_face_name_appears_in_the_visual_names(self):
        names = ' '.join(
            visual.attrib['name']
            for visual in _model().findall('./link/visual')
        )
        for prefix, face in (
            ('pos_x', '+x'), ('neg_x', '-x'),
            ('pos_y', '+y'), ('neg_y', '-y'),
            ('pos_z', '+z'), ('neg_z', '-z'),
        ):
            self.assertIn(f'face_{prefix}_', names)
            self.assertIn(face, FACE_NAMES)

    def test_no_mesh_plugin_or_sensor_is_declared(self):
        text = build_model_sdf()
        for forbidden in ('<mesh', '<plugin', '<sensor', 'model://', '.stl'):
            self.assertNotIn(forbidden, text)

    def test_generation_is_deterministic(self):
        self.assertEqual(build_model_sdf(), build_model_sdf())

    def test_sdf_parses_as_xml(self):
        self.assertEqual(_model().tag, 'model')

    def test_facelet_colours_are_distinct_in_the_document(self):
        model = _model()
        colors = set()
        for visual in model.findall('./link/visual'):
            if not visual.attrib['name'].startswith('face_'):
                continue
            colors.add(visual.findtext('./material/diffuse'))
        self.assertEqual(len(colors), 6)


class PaletteAgreementTest(unittest.TestCase):
    """The generated cube and the hypothesis scorer must agree on colours.

    They are separate modules on purpose (an asset generator and a geometry
    scorer), so the shared convention needs an explicit pin: a silent
    disagreement makes every identity hypothesis score as a mismatch, which is
    exactly how a resolved pose turns into an unresolved one.
    """

    def test_face_to_colour_matches_the_hypothesis_palette(self):
        from barcode_detector.cube_face_colors import REFERENCE_RGB
        from barcode_detector.cube_pose_hypotheses import STANDARD_FACE_COLORS

        expected = {
            'pos_x': '+x', 'neg_x': '-x',
            'pos_y': '+y', 'neg_y': '-y',
            'pos_z': '+z', 'neg_z': '-z',
        }
        self.assertEqual(len(FACE_SPECS), 6)
        for face, label, _ in FACE_SPECS:
            self.assertEqual(
                STANDARD_FACE_COLORS[expected[face]], label,
                f'{face} colour disagrees with the scorer palette',
            )
            self.assertIn(label, REFERENCE_RGB)

    def test_opposite_faces_never_share_a_colour(self):
        pairs = (('pos_x', 'neg_x'), ('pos_y', 'neg_y'), ('pos_z', 'neg_z'))
        lookup = {face: label for face, label, _ in FACE_SPECS}
        for first, second in pairs:
            self.assertNotEqual(lookup[first], lookup[second])


class WriteModelTest(unittest.TestCase):
    """Regeneration must leave the provenance next to the artefact."""

    def test_writes_three_files_with_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdf, config, attribution = write_model(temporary)
            self.assertEqual(sdf.name, 'model.sdf')
            self.assertEqual(config.name, 'model.config')
            self.assertTrue(sdf.is_file() and config.is_file())
            text = attribution.read_text(encoding='utf-8')
            self.assertIn('osrf/gazebo_models', text)
            self.assertIn('cube_20k', text)
            self.assertIn('Creative Commons', text)
            self.assertIn('54 facelet visuals', text)
            self.assertIn('No upstream file is copied', text)

    def test_regeneration_overwrites_in_place(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, _, _ = write_model(temporary)
            before = first.read_text(encoding='utf-8')
            second, _, _ = write_model(temporary)
            self.assertEqual(first, second)
            self.assertEqual(before, second.read_text(encoding='utf-8'))

    def test_written_model_parses(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdf, _, _ = write_model(temporary)
            root = ET.parse(sdf).getroot()
            self.assertEqual(root.find('model').attrib['name'], MODEL_NAME)
            self.assertTrue(math.isclose(
                float(root.findtext(
                    './model/link/inertial/inertia/izz')),
                MASS_KG * SIDE_LENGTH_M ** 2 / 6.0,
                rel_tol=1e-9,
            ))


if __name__ == '__main__':
    unittest.main()
