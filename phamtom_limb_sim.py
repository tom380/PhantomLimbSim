import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("phantom_limb.xml")
viewer = mujoco.viewer.launch(model)

