import bpy 

# Parameters
radius = 1.0
height = 7.0
segments = 64

# Delete default cube
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


# Create cylinder
bpy.ops.mesh.primitive_cylinder_add(
    vertices=segments,
    radius=radius,
    depth=height,
    location=(0, 0, 0)
)