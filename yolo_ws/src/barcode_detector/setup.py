from setuptools import setup

package_name = 'barcode_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='YOLO pose detector package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'roboflow_node = barcode_detector.roboflow_node:main',
            'yolo_pose_node = barcode_detector.yolo_pose_node:main',
            'yolo_save_node = barcode_detector.yolo_save_node:main',
        ],
    },
)
