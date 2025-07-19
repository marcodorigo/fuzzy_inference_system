from setuptools import find_packages, setup

package_name = 'fuzzy_inference_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marco',
    maintainer_email='marco3.dorigo@mail.polimi.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ee_distance_metrics_node = fuzzy_inference_system.ee_distance_metrics_node:main',
            'fuzzy_safety_node = fuzzy_inference_system.fuzzy_safety_node:main'
        ],
    },
)
