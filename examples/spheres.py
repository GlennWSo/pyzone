from time import time


from rscad.boolean.polydata import diff
import pyvista as pv
from madcad.boolean import difference

from helper import mad2poly, poly2mad

sp1 = pv.Sphere(center=(0.3, 0, 0)).clean()
sp2 = pv.Sphere().subdivide(2).clean()
print(f"sphere1 has {sp1.n_faces} faces")
print(f"sphere2 has {sp2.n_faces} faces\n")
# sp1.plot_normals(mag=0.1)
p = pv.Plotter()
p.add_mesh(sp1, color="blue")
p.add_mesh(sp2, color="yellow")
p.add_text("blue is base, yellow is the operand")
p.show()


def mad_diff(m1, m2):
    args = [poly2mad(m) for m in [m1, m2]]
    mesh = difference(*args)
    return mad2poly(mesh)


def test_diff(m1, m2, fn, name="noname"):
    t0 = time()
    res = fn(m1, m2)
    t = time() - t0
    info = f"{name} diff time: {round(t, 3)}"
    print(info)
    print(f"n open edges: {res.n_open_edges}")
    print(f"is_manifold: {res.is_manifold}")
    print()
    print()
    res.plot(text=info)


def test_poly(m1: pv.PolyData, m2: pv.PolyData):
    test_diff(m1, m2, pv.PolyData.boolean_difference, "poly")


def test_mad(m1: pv.PolyData, m2: pv.PolyData):
    test_diff(m1, m2, mad_diff, "mad")


def test_rscad(m1: pv.PolyData, m2: pv.PolyData):
    test_diff(m1, m2, diff, "rscad")


test_rscad(sp1, sp2)
test_poly(sp1, sp2)
test_mad(sp1, sp2)
