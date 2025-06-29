# z1_simulator
The simulator is built based on isaac gym and Unitree z1 robotic arm.
## Architecture
### Descriptions
- **b1**: The robot dog that can be equipped with z1 robotic arm, used only for platform in the program.
- **z1**: Urdf of z1, the core component of the program.
### Main
- **KeyBoard**: Use keyboard to control the robotic arm.
- **Coordinate**: Input the coordinate and use IK to find out the motion.