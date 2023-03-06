import madcad as cad
import pyvista as pv
import numpy as np

from madcad import dvec3


def poly2mad(poly: pv.PolyData, **options) -> cad.Mesh:
    assert poly.n_faces > 0
    tri = poly.triangulate()  # avoid mutation and make all faces tri
    faces = tri.faces.reshape((-1, 4))[:, 1:].tolist()
    return cad.Mesh(tri.points.tolist(), faces, options=options)


def mad2poly(mesh: cad.Mesh) -> pv.PolyData:
    points = np.array([tuple(v) for v in mesh.points])
    face_arr = np.array([tuple(v) for v in mesh.faces])
    faces = np.pad(
        face_arr,
        pad_width=((0, 0), (1, 0)),
        constant_values=3,
    ).ravel()
    poly = pv.PolyData(points, faces)

    return poly


if __name__ == "__main__":
    poly = pv.Cube().triangulate().subdivide(2)
    poly.plot()
    cube = poly2mad(poly)
    cube.check()

    cyl_top = dvec3(0, 0, 1)
    cyl_bot = dvec3(0, 0, 0)
    cyl = cad.cylinder(cyl_bot, cyl_top, radius=0.3)

    # cad.show([cube, cyl])
    cut1 = cad.difference(cube, cyl)

    cad.show([cut1])
    res = mad2poly(cut1)
    res.plot()
    assert res.n_open_edges == 0
    assert res.is_manifold
