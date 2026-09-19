from setuptools import setup, find_packages
from glob import glob
from pathlib import Path

package_name = "scene_manager"
setup(name=package_name, version="0.1.0", packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/" + package_name]),
                ("share/" + package_name, ["package.xml"])] +
               [("share/" + package_name + "/" + folder, [p for p in glob(folder + "/*") if Path(p).is_file()])
                for folder in ("launch", "config", "example_assets") if glob(folder + "/*")],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="AVM Developers", maintainer_email="dev@example.com",
    description="AVM simulation scene_manager", license="Apache-2.0",
    entry_points={"console_scripts": ["scene_manager_node = scene_manager.ros_node:main"]})

