import casadi as ca
import chaospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
fsize = 10
tsize = 18
tdir = 'in'
major = 5.0
minor = 3.0
style = 'default'
plt.style.use(style)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor

def func(x:float, u):
    return (u - x**2)**3

x = ca.SX.sym('x', 1)
u = ca.SX.sym('u', 1)

def find_coefficients(x, u, phis, points):
    fphis = [ca.Function('fphi', [x], [phi]) for phi in phis]

    # build matrix
    matrix = []
    fs = []
    for xj in points:
        row = []
        for fphi in fphis:
            row.append(fphi(xj).full()[0][0])
        matrix.append(row)
        fs.append(func(xj, u))
    matrix = ca.SX(matrix)
    inv_m = ca.inv(matrix)

    # build collocation function evals
    vector = []
    for xj, _f in zip(points, fs):
        vector.append(_f)
    vector = ca.vertcat(*vector)
    coeffs = ca.mtimes(inv_m, vector)
    return coeffs

# Collocation method, 2nd order
phi_0 = ca.DM(1)
phi_1 = x 
phi_2 = x**2 - 1/3
phis = [phi_0, phi_1, phi_2]
points1 = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]

fmodels = []
models = []
for points in [points1]:
    coeffs = find_coefficients(x, u, phis, points)
    model_approx = ca.DM(0)
    for coeff, phi in zip(ca.vertsplit(coeffs), phis):
        model_approx += coeff*phi
    models.append(model_approx)
    fmodel = ca.Function('fmodel', [x, u], [model_approx])    
    fmodels.append(fmodel)
    
# Collocation method, 4th order
phi_0 = ca.DM(1)
phi_1 = x 
phi_2 = x**2 - 1/3
phi_3 = x**3 - 3/5*x 
phi_4 = x**4 - 30*(x**2)/35 + 3/35
phis = [phi_0, phi_1, phi_2, phi_3, phi_4]
coeff1 = u**3 - u**2 + 3*u/5 - 1/7
coeff2 = ca.SX(0.0)
coeff3 = -45*(21*u**2 - 18*u + 5)/315
coeff4 = ca.SX(0.0)
coeff5 = 11025*(11*u-5)/40425
coeffs = [coeff1, coeff2, coeff3, coeff4, coeff5]
points1 = [-0.90617984593866, -0.538469310105683, 0.000000000000000, 0.538469310105683, 0.90617984593866]

models = []
for points in [points1]:
    coeffs = find_coefficients(x, u, phis, points)
    model_approx = ca.DM(0)
    for coeff, phi in zip(ca.vertsplit(coeffs), phis):
        model_approx += coeff*phi
    models.append(model_approx)
    fmodel = ca.Function('fmodel', [x, u], [model_approx])    
    fmodels.append(fmodel)
    
# Collocation method, 6th order
phi_0 = ca.DM(1)
phi_1 = x 
phi_2 = x**2 - 1/3
phi_3 = x**3 - 3/5*x 
phi_4 = x**4 - 30*(x**2)/35 + 3/35
phi_5 = x**5 - 70*(x**3)/63 + 15/63
phi_6 = x**6 - 315*(x**4)/231 + 105*(x**2)/231 - 5/231
phis = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]
coeff1 = u**3 - u**2 + 3*u/5 - 1/7
coeff2 = ca.SX(0.0)
coeff3 = -45*(21*u**2 - 18*u + 5)/315
coeff4 = ca.SX(0.0)
coeff5 = 11025*(11*u-5)/40425
coeff6 = ca.SX(0.0)
coeff7 = ca.SX(-1.0)
coeffs = [coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7]
points1 = [-0.949107912342759, -0.741531185599394, -0.405845151377397, 0.000000000000000, 0.405845151377397, 0.741531185599394, 0.949107912342759]

models = []
for points in [points1]:
    coeffs = find_coefficients(x, u, phis, points)
    model_approx = ca.DM(0)
    for coeff, phi in zip(ca.vertsplit(coeffs), phis):
        model_approx += coeff*phi
    models.append(model_approx)
    fmodel = ca.Function('fmodel', [x, u], [model_approx])    
    fmodels.append(fmodel)
    
# for i, _u in enumerate(np.linspace(-1, 1, 10)):
_u = 0.5
xs = np.linspace(-1.0, 1.0, 20)
plt.figure()
plt.plot(xs, func(xs, _u), color='black', label="Original")
plt.plot(xs, fmodels[0](xs, _u), 'o-', alpha=0.5, label=f"CM - 2nd order")
plt.plot(xs, fmodels[1](xs, _u), 'o-', alpha=0.5, label=f"CM - 4th order")
plt.plot(xs, fmodels[2](xs, _u), 'o-', alpha=0.5, label=f"CM - 6th order")
plt.legend()
plt.title(f"$u = {_u}$")
plt.ylabel("$f, p_n$")
plt.xlabel("$x$")
plt.savefig(f"./Examples/comparison_col.png")

# Arbitrary PCE
np.random.seed(123)
samples = np.random.uniform(-1,1, 1000)
def raw_moment(x:np.ndarray, A:float, order:int):
    return np.sum((x-A)**order)/x.shape[0]

plt.figure()
plt.title("Input data")
plt.ylabel("frequency")
plt.xlabel("$x$")
plt.hist(samples, bins=20, edgecolor="black", color='cornflowerblue')
plt.savefig(f"./Examples/Histogram.png")

mu0 = raw_moment(samples, 0, 0)
mu1 = raw_moment(samples, 0, 1)
mu2 = raw_moment(samples, 0, 2)
mu3 = raw_moment(samples, 0, 3)
mu4 = raw_moment(samples, 0, 4)
mu5 = raw_moment(samples, 0, 5)
mu6 = raw_moment(samples, 0, 6)
mu7 = raw_moment(samples, 0, 7)
mu8 = raw_moment(samples, 0, 8)
mu9 = raw_moment(samples, 0, 9)
mu10 = raw_moment(samples, 0, 10)
mu11 = raw_moment(samples, 0, 11)

# Solve for the second orthogonal polynomial
mmatrix1 = ca.SX([[mu0, mu1], 
                  [0.0, 1.0]])
inv_mm1 = ca.inv(mmatrix1)
mvector1 = ca.SX([0.0, 1.0])
opcoeffs1 = ca.mtimes(inv_mm1, mvector1)

# Solve for the third orthogonal polynomial
mmatrix2 = ca.SX([[mu0, mu1, mu2], 
                  [mu1, mu2, mu3], 
                  [0.0, 0.0, 1.0]])
inv_mm2 = ca.inv(mmatrix2)
mvector2 = ca.SX([0.0, 0.0, 1.0])
opcoeffs2 = ca.mtimes(inv_mm2, mvector2)

# Solve for the fourth orthogonal polynomial
mmatrix2 = ca.SX([[mu0, mu1, mu2, mu3], 
                  [mu1, mu2, mu3, mu4], 
                  [mu2, mu3, mu4, mu5], 
                  [0.0, 0.0, 0.0, 1.0]])
inv_mm2 = ca.inv(mmatrix2)
mvector2 = ca.SX([0.0, 0.0, 1.0])
opcoeffs2 = ca.mtimes(inv_mm2, mvector2)

phi_0 = ca.DM(1)
phi_1 = x + opcoeffs1[0]
phi_2 = x**2 + opcoeffs2[1]*x + opcoeffs2[0]
phis = [phi_0, phi_1, phi_2]
points = [-np.sqrt(3/5), 0.0, np.sqrt(3/5)]
coeffs = find_coefficients(x, u, phis, points)
model_approx = ca.DM(0)
for coeff, phi in zip(ca.vertsplit(coeffs), phis):
    model_approx += coeff*phi
models.append(model_approx)
fmodel = ca.Function('fmodel', [x, u], [model_approx])    
fmodels.append(fmodel)


    
asdfasd

#Direct 0th order
phi_0 = ca.DM(1)
phis = [phi_0]
coeff1 = u**3 - u**2 + 3*u/5 - 1/7
coeffs = [coeff1]
for coeff, phi in zip(coeffs, phis):
    model_approx += coeff*phi
models.append(model_approx)
fmodel = ca.Function('fmodel', [x, u], [model_approx])    
fmodels.append(fmodel)

#Direct 2nd order
phi_0 = ca.DM(1)
phi_2 = x**2 - 1/3
phis = [phi_0, phi_2]
coeff1 = u**3 - u**2 + 3*u/5 - 1/7
coeff2 = ca.SX(0.0)
# coeff3 = -45*u**2/30 + 45*u/35 - 45/126
coeff3 = -45*(21*u**2 - 18*u + 5)/315
coeffs = [coeff1, coeff2, coeff3]
for coeff, phi in zip(coeffs, phis):
    model_approx += coeff*phi
models.append(model_approx)
fmodel = ca.Function('fmodel', [x, u], [model_approx])    
fmodels.append(fmodel)

#Direct 4th order
phi_0 = ca.DM(1)
phi_2 = x**2 - 1/3
phi_4 = x**4 - 30*(x**2)/35 + 3/35
phis = [phi_0, phi_2, phi_4]
coeff1 = u**3 - u**2 + 3*u/5 - 1/7
coeff3 = -45*(21*u**2 - 18*u + 5)/315
coeff5 = 11025*(11*u-5)/40425
coeffs = [coeff1, coeff3, coeff5]
for coeff, phi in zip(coeffs, phis):
    model_approx += coeff*phi
models.append(model_approx)
fmodel = ca.Function('fmodel', [x, u], [model_approx])    
fmodels.append(fmodel)

#Direct 6th order
phi_0 = ca.DM(1)
phi_2 = x**2 - 1/3
phi_4 = x**4 - 30*(x**2)/35 + 3/35
phi_6 = x**6 - 315*(x**4)/231 + 105*(x**2)/231 - 5/231
phis = [phi_0, phi_2, phi_4, phi_6]
coeff1 = u**3 - u**2 + 3*u/5 - 1/7
coeff3 = -45*(21*u**2 - 18*u + 5)/315
coeff5 = 11025*(11*u-5)/40425
coeff7 = ca.SX(-1.0)
coeffs = [coeff1, coeff3, coeff5, coeff7]
for coeff, phi in zip(coeffs, phis):
    model_approx += coeff*phi
models.append(model_approx)
fmodel = ca.Function('fmodel', [x, u], [model_approx])    
fmodels.append(fmodel)

# # for i, _u in enumerate(np.linspace(-1, 1, 10)):
# _u = 0.5
# xs = np.linspace(-1.0, 1.0, 20)
# plt.figure()
# plt.plot(xs, func(xs, _u), color='black', label="Original")
# # for j, _ in enumerate(fmodels):
# plt.plot(xs, fmodels[0](xs, _u), '--', label=f"CM - 2nd order")
# plt.plot(xs, fmodels[1](xs, _u), '8', label=f"CM - 4th order")
# # plt.plot(xs, fmodels[2](xs, _u), 'o', label=f"Moment based method")
# # plt.plot(xs, fmodels[2](xs, _u), 'x', label=f"Direct - 0th order")
# # plt.plot(xs, fmodels[3](xs, _u), 'x', label=f"Direct - 2nd order")
# # plt.plot(xs, fmodels[4](xs, _u), 'x', label=f"Direct - 4th order")
# # plt.plot(xs, fmodels[5](xs, _u), 'x', label=f"Direct - 6th order")
# plt.legend()
# plt.title(f"$u = {_u}$")
# plt.ylabel("$f, p_n$")
# plt.xlabel("$x$")
# plt.savefig(f"./Examples/comparison_col_mb.png")
    


# # example u = 1
# xdist = chaospy.Uniform(-1, 1)
# gauss_quads = chaospy.generate_quadrature(2, xdist, rule="gaussian")
# nodes, weights = gauss_quads
# expansions = chaospy.generate_expansion(2, xdist)
# gauss_evals = np.array([func(node, 3/5) for node in nodes.T])
# gauss_model_approx = chaospy.fit_quadrature(expansions, nodes, weights, gauss_evals)
# gauss_model_approx_evals = np.array([gauss_model_approx(node) for node in nodes.T])