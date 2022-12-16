import madcad as cad
from madcad import typedlist, dvec3, Axis

import pyvista as pv
import numpy as np


def poly2mad(poly: pv.PolyData) -> cad.Mesh:
    """
    Helper to convert pyvista.PolyData to madcad
    """
    assert poly.n_faces > 0
    tri = poly.triangulate()  # avoid mutation and make all faces tri
    faces = tri.faces.reshape((-1, 4))[:, 1:].tolist()
    points = tri.points.tolist()

    mesh = cad.Mesh(points, faces)
    mesh.check()
    assert mesh.isvalid()
    assert mesh.issurface()
    return mesh


def mad2poly(mesh: cad.Mesh) -> pv.PolyData:
    """
    helper to convert madcad.Mesh to pyvista.PolyData
    """
    face_arr = np.array([tuple(v) for v in mesh.faces])
    faces = np.pad(
        face_arr,
        pad_width=((0, 0), (1, 0)),
        constant_values=3,
    ).ravel()

    points = np.array([tuple(v) for v in mesh.points])

    poly = pv.PolyData(points, faces)
    return poly


def randsurf(res=30, seed=1, **kwargs) -> cad.Mesh:
    """
    Creates a surface with random hills
    """
    poly = pv.ParametricRandomHills(
        u_res=res, v_res=res, w_res=res, randomseed=seed, **kwargs
    )
    poly.translate((0, -10, 0))
    return poly2mad(poly)


def plot_normals(mesh: cad.Mesh):
    mad2poly(mesh).plot_normals()


def inspect_open_edges(mesh: cad.Mesh, **kwargs):
    poly = mad2poly(mesh)

    edges = poly.extract_feature_edges(
        feature_edges=False, non_manifold_edges=False, manifold_edges=False
    )
    pmesh = pv.PolyData(edges.points)
    p = pv.Plotter()
    p.add_mesh(poly)
    p.add_mesh(edges, color="red", line_width=2)
    p.add_mesh(pmesh, color="blue", point_size=9.0)
    p.show(**kwargs)
    print(poly)
    print("poly n open edges", poly.n_open_edges)
    print("poly is manifold", poly.is_manifold)
    print("diff1 is envelope?", mesh.isenvelope())


def random_block(res=20, seed=1) -> cad.Mesh:
    """A block shape that has been cut with randomhills"""
    surf1 = randsurf(res=res, seed=1)
    # axis1 = Axis(dvec3(0, 10, -1), dvec3(0, 0, 1))
    # plane1 = cad.square(axis1, 20)
    surf2 = randsurf(res=res, seed=1)
    for i, p in enumerate(surf2.points):
        surf2.points[i] = dvec3(p[0] * 1.2, p[1] * 1.2, -2.0)

    ex1 = cad.extrusion(dvec3(0, 0, -10), surf1)
    diff1 = cad.difference(ex1, surf2)
    return diff1


def random_block1(res=20, seed=1) -> cad.Mesh:

    top = randsurf(res=res, seed=1)

    skirt = cad.extrusion(dvec3(0, 0, -3), top.outlines())

    n = len(skirt.points) // 2
    skirt.points[n:] = typedlist(
        [[p[0], p[1], -2.0] for p in skirt.points[n:]],
        dtype=dvec3,
    )

    bot_boundary = skirt.outlines().islands()[1]
    cad.show([bot_boundary])
    # bot = cad.flatsurface(bot_boundary)

    # cad.show([top, skirt, bot])


def test_block(block, dbg=False):
    """
    Test boolean operations robustness with a operations on a random generated Mesh
    """
    surf1 = randsurf(res=res, seed=1)
    axis1 = Axis(dvec3(0, 0, -1), dvec3(0, 0, 1))
    plane1 = cad.square(axis1, 20)
    block1 = cad.extrusion(dvec3(0, 0, 20), plane1)
    ex1 = cad.extrusion(dvec3(0, 0, -10), surf1)

    assert ex1.isvalid()
    assert ex1.issurface()
    assert ex1.isenvelope()
    if dbg:
        cad.show([ex1, plane1])
        plot_normals(ex1)
        plot_normals(block1)

    print(ex1.precision(), plane1.precision())
    diff1 = cad.intersection(ex1, block1.flip())
    if dbg:
        cad.show([diff1])
        plot_normals(diff1)
        inspect_open_edges(diff1, title=f"Boolean op  {res} * {res}")

    assert diff1.isvalid()
    assert diff1.issurface()
    assert diff1.isenvelope()


if __name__ == "__main__":
    # booleans work on low res
    random_block()
    # test_boolean(res=100, dbg=True)  # set dbg to inspect what is going on
    for n in range(r, 55, 5):
        block = random_block(r)

        try:
            test_boolean(res=n)
            print(f"resolution: {n} worked")
        except AssertionError as e:
            test_boolean(res=n, dbg=True)  # set dbg to inspect what is going on
            print(f"resolution: {n} failed")
