from threading import Thread

from lib.actor.system import MapSystem, Tb3
from ros_actor import (
    init_server,
    init_spin,
    register_subsystem,
    run_actor,
    shutdown_server,
)
from ros_actor.command import CommandInterpreter


def init_realsense_background():
    try:
        run_actor("init_realsense")
        print("[cm1] realsense initialized")
    except Exception as e:
        print(f"[cm1] realsense initialization skipped: {e}")


def cm_init(node):
    register_subsystem("robot", Tb3)
    init_spin(node)
    # register_subsystem('map', MapSystem)
    run_actor("make_symbolic_link")
    run_actor("update_bt")
    Thread(target=CommandInterpreter().do_command, daemon=True).start()
    Thread(target=init_realsense_background, daemon=True).start()


def main():
    init_server(cm_init)
    shutdown_server()


if __name__ == "__main__":
    main()
