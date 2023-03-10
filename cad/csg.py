import madcad as cad
from madcad import typedlist, dvec3, Axis

import pyvista as pv
import numpy as np

from typing import Tuple


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


def randsurf(res=30, seed=1, ydist=0, **kwargs) -> cad.Mesh:
    """
    Creates a surface with random hills
    """
    poly = pv.ParametricRandomHills(
        u_res=res, v_res=res, w_res=res, randomseed=seed, **kwargs
    )
    poly.translate((0, ydist, 0), inplace=True)
    return poly2mad(poly)


def plot_normals(mesh: cad.Mesh):
    mad2poly(mesh).plot_normals()


def inspect_open_edges(mesh: cad.Mesh, **kwargs):
    poly = mad2poly(mesh)

    edges = poly.extract_feature_edges(
        feature_edges=False, non_manifold_edges=False, manifold_edges=False
    )
    pmesh = pv.PolyData(edges.points)
    print("pv.poly n open edges", poly.n_open_edges)
    print("pv.poly is manifold", poly.is_manifold)
    print("cad.mesh is envelope?", mesh.isenvelope())
    if edges.n_points > 0:
        p = pv.Plotter()
        p.add_mesh(poly)
        p.add_mesh(edges, color="red", line_width=2)
        p.add_mesh(pmesh, color="blue", point_size=9.0)
        p.show(**kwargs)
        print(poly)


def random_block(res=20, seed=1, ydist=0) -> Tuple[cad.Mesh, ...]:
    """A block shape that has been cut with randomhills"""
    surf1 = randsurf(res=res, seed=1, ydist=ydist)
    center = surf1.box().center

    base = cad.extrusion(dvec3(0, 0, -10), surf1)
    axis1 = Axis(dvec3(center[0], center[1], -2), dvec3(0, 0, 1))
    tool = cad.square(axis1, 20)
    # tool = randsurf(res=res, seed=1)
    # for i, p in enumerate(tool.points):
    #     tool.points[i] = dvec3(p[0] * 1.2, p[1] * 1.2, -2.0)

    res = cad.boolean.boolean(base, tool, (False, True))
    return base, tool, res


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


def debug_boolean(base, tool, res):
    print("base_min:", base.box().min)
    print("base_max:", base.box().max)
    cad.show([base, tool])
    plot_normals(base)
    plot_normals(tool)

    cad.show([res])
    plot_normals(res)
    inspect_open_edges(res, title=f"Boolean op  {res} * {res}")


def test_boolean(base: cad.Mesh, tool: cad.Mesh, res: cad.Mesh, dbg=False):
    assert base.isvalid()
    assert base.issurface()
    assert base.isenvelope()

    assert tool.isvalid()
    assert tool.issurface()

    assert res.isvalid()
    assert res.issurface()
    assert res.isenvelope()


if __name__ == "__main__":
    # booleans work on low res
    # test_boolean(res=100, dbg=True)  # set dbg to inspect what is going on
    for n in range(10, 110, 10):
        for y in np.linspace(-1, 1, 1):
            parts = random_block(res=n, seed=1, ydist=y)

            try:
                print(f"testing res: {n}, ymin: {y}", end=", ")
                test_boolean(*parts)

                print("ok! :)")
            except AssertionError as e:
                print(f"resolution: {n} failed")
                debug_boolean(*parts)
                raise e

    debug_boolean(*parts)
