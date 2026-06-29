import speech_recognition as sr


def list_microphones():
    return sr.Microphone.list_microphone_names()


def print_microphones():
    names = list_microphones()
    if not names:
        print('[voice] no microphone found')
        return
    for i, name in enumerate(names):
        print(f'[{i}] {name}')


def select_microphone_index(device_index=None):
    names = list_microphones()
    if not names:
        raise RuntimeError('No microphone found')

    if device_index is not None and device_index >= 0:
        return int(device_index)

    for i, name in enumerate(names):
        if name == 'default':
            return i

    for i, name in enumerate(names):
        lower = name.lower()
        if 'hdmi' in lower or 'monitor' in lower:
            continue
        if 'analog' in lower or 'mic' in lower or 'input' in lower:
            return i

    return None
