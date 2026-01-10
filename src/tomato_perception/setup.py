from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tomato_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('tomato_perception/streamlit_ui.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotic',
    maintainer_email='chinhonglee14@gamil.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tomato_node = tomato_perception.tomato_node:main',
            'harvest_control = tomato_perception.harvest_control:main',
            'streamlit_ui = tomato_perception.streamlit_ui:main',
        ],
    },
)
