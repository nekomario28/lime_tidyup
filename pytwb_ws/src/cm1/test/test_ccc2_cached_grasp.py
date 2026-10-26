from pathlib import Path
import xml.etree.ElementTree as ET


def test_ccc2_grasps_from_the_pre_approach_observation():
    tree_path = (
        Path(__file__).resolve().parents[1]
        / 'cm1'
        / 'trees'
        / 'ccc2.xml'
    )
    root = ET.parse(tree_path).getroot()
    sequence = root.find('.//Sequence')
    children = list(sequence)
    tags = [child.tag for child in children]

    approach_index = tags.index('GoFrontCube')
    first_grasp_index = tags.index('ArmToCube', approach_index)
    assert not any(
        child.tag == 'TakeAim'
        for child in children[approach_index + 1:first_grasp_index]
    )
    assert children[approach_index].attrib['cache_grasp_observation'] == '1'
    assert all(
        child.attrib['use_cached_observation'] == '1'
        for child in children
        if child.tag == 'ArmToCube'
    )
