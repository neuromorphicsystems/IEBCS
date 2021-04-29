import bpy


def create_custom_mesh(objname,dx, dy,  px, py, pz):
    """
        Taken from https://blender.stackexchange.com/questions/67517/possible-to-add-a-plane-using-vertices-via-python
    """
    # Define arrays for holding data
    myvertex = []
    myfaces = []
    # Create all Vertices
    # vertex 0
    mypoint = [(-dx/2, -dy/2, 0.0)]
    myvertex.extend(mypoint)
    # vertex 1
    mypoint = [(-dx/2, dy/2, 0.0)]
    myvertex.extend(mypoint)
    # vertex 2
    mypoint = [(dx/2, dy/2, 0.0)]
    myvertex.extend(mypoint)
    # vertex 3
    mypoint = [(dx/2, -dy/2, 0.0)]
    myvertex.extend(mypoint)

    # -------------------------------------
    # Create all Faces
    # -------------------------------------
    myface = [(1, 2, 3, 0)]
    myfaces.extend(myface)

    mymesh = bpy.data.meshes.new(objname)

    myobject = bpy.data.objects.new(objname, mymesh)

    bpy.context.scene.objects.link(myobject)
    # Generate mesh data
    mymesh.from_pydata(myvertex, [], myfaces)
    # Calculate the edges
    mymesh.update(calc_edges=True)

    # Set Location
    myobject.location.x = px
    myobject.location.y = py
    myobject.location.z = pz

    myobject.rotation_euler.x = 3.14
    myobject.rotation_euler.y = 0
    myobject.rotation_euler.z = 0

    return myobject
