"""Every attribute a behavior tree sets must be accepted by its adapter.

A tree attribute the adapter does not declare is dropped without any error, so
the actor runs with its default and the tree silently does nothing. This walks
each tree in ``cm1/trees`` and checks the attribute names against the matching
adapter's ``__init__`` parameters in ``cm1/behavior``.
"""

import ast
from pathlib import Path

import xml.etree.ElementTree as ET


TREES_DIR = Path(__file__).resolve().parents[1] / 'cm1' / 'trees'
BEHAVIOR_DIR = Path(__file__).resolve().parents[1] / 'cm1' / 'behavior'

#: Attributes the behavior-tree runtime consumes itself, not the adapter.
RUNTIME_ATTRIBUTES = {'name'}


def _constructor_parameters():
    parameters = {}
    for path in sorted(BEHAVIOR_DIR.glob('*.py')):
        module = ast.parse(path.read_text())
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef) or item.name != '__init__':
                    continue
                names = [
                    argument.arg
                    for argument in item.args.args
                    if argument.arg not in ('self', 'name', 'node')
                ]
                parameters.setdefault(node.name, set()).update(names)
    return parameters


def test_tree_attributes_are_accepted_by_their_adapters():
    parameters = _constructor_parameters()
    dropped = []
    checked = 0

    for tree_path in sorted(TREES_DIR.glob('*.xml')):
        root = ET.parse(tree_path).getroot()
        for element in root.iter():
            accepted = parameters.get(element.tag)
            if accepted is None:
                continue
            for attribute in element.attrib:
                if attribute in RUNTIME_ATTRIBUTES:
                    continue
                checked += 1
                if attribute not in accepted:
                    dropped.append(
                        f'{tree_path.name}: <{element.tag} {attribute}=...> '
                        f'is not an argument of {element.tag}.__init__'
                    )

    assert checked > 0, 'no tree attribute was checked; the tree layout moved'
    assert not dropped, 'tree attributes dropped by their adapters:\n' + '\n'.join(dropped)
