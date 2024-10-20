#ifdef _WIN32
#define EXPORT extern "C" __declspec( dllexport )
#else
#define EXPORT extern "C" __attribute__((visibility ("default")))
#endif

#include <math.h>
#include <Eigen/Eigen>

typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;

template <typename T>
class Pair {
public:
    T* cur;
    T* nxt;
    Pair(): cur(reinterpret_cast<T*>(0)), nxt(reinterpret_cast<T*>(0)) {}
    Pair(T* _cur, T* _nxt): cur(_cur), nxt(_nxt) {}
    void swap() {
        auto tmp = this->cur;
        this->cur = this->nxt;
        this->nxt = tmp;
    }
};

template <typename T>
class Field {
public:
    T* data;
    int row;
    int col;
    Field(): data(0), row(0), col(0) {}
    Field(T* _data, int _row, int _col): data(_data), row(_row), col(_col) {}
    T* operator[](int i) const {
        return this->data + i * this->col;
    }
};

class Parameters {
public:
    double dt;
    double decay;
    int compression_iters;
    Eigen::Vector2i resolution;
    Vec2 impulse_center;
    double force_radius;
    double impulse_strength;
    // Parameters(): dt(0), decay(0), compression_iters(0), resolution(Eigen::Vector2i(0, 0)) {}
    Parameters() {}
    Parameters(double _dt, double _decay, int _compression_iters, int x, int y):
        dt(_dt),
        decay(_decay),
        compression_iters(_compression_iters),
        resolution(Eigen::Vector2i(x, y)),
        impulse_center(Vec2(0, y/2.0)),
        force_radius(resolution.minCoeff()/4.0),
        impulse_strength(10000)
    {}
};

Parameters parameters;

Vec2* _velocities_data;
Field<Vec2> velocities;
Field<Vec2> new_velocities;
Pair<Field<Vec2>> velocities_pair;

Field<double> pressures;
Field<double> new_pressures;
Pair<Field<double>> pressures_pair;
Field<double> velocity_divergences;
Field<double> velocity_curls;

EXPORT void initialize(Vec2* velocities_data, Vec2* new_velocities_data, double* velocity_curls_data,
int res_x, int res_y, double dt, double decay, int compression_iters) {
    _velocities_data = velocities_data;

    velocities = Field<Vec2>(velocities_data, res_x, res_y);
    new_velocities = Field<Vec2>(new_velocities_data, res_x, res_y);
    velocities_pair = Pair<Field<Vec2>>(&velocities, &new_velocities);

    double* pressures_data = new double[res_x*res_y];
    double* new_pressures_data = new double[res_x*res_y];
    pressures = Field<double>(pressures_data, res_x, res_y);
    new_pressures = Field<double>(new_pressures_data, res_x, res_y);
    pressures_pair = Pair<Field<double>>(&pressures, &new_pressures);

    double* velocity_divergences_data = new double[res_x*res_y];
    velocity_divergences = Field<double>(velocity_divergences_data, res_x, res_y);
    velocity_curls = Field<double>(velocity_curls_data, res_x, res_y);

    parameters = Parameters(dt, decay, compression_iters, res_x, res_y);
}

EXPORT void clear() {
    for (int i=0; i<parameters.resolution.x(); i++) {
        for (int j=0; j<parameters.resolution.y(); j++) {
            (*pressures_pair.cur)[i][j] = 0;
            (*pressures_pair.nxt)[i][j] = 0;
            velocity_divergences[i][j] = 0;
        }
    }
}

template <typename T>
T sample(const Field<T> &qf, int i, int j) {
    if (i < 0) {
        i = 0;
    } else if (i > qf.row-1) {
        i = qf.row-1;
    }
    if (j < 0) {
        j = 0;
    } else if (j > qf.row-1) {
        j = qf.col-1;
    }
    return qf[i][j];
}

template <typename T>
T interpolation(T vl, T vr, double frac) {
    return vl + frac * (vr - vl);
}

template <typename T>
T bilinear_interpolation(const Field<T> &vf, Vec2 p) {
    Vec2 v = p - Vec2(0.5, 0.5);
    Vec2 i = v.array().floor();
    Vec2 f = v - i;
    return interpolation(
        interpolation(
            sample(vf, i.x(), i.y()),
            sample(vf, i.x() + 1, i.y()),
        f.x()),
        interpolation(
            sample(vf, i.x(), i.y() + 1),
            sample(vf, i.x() + 1, i.y() + 1),
        f.x()),
    f.y());
}

Vec2 RK3(const Field<Vec2> &vf, Vec2 p, double dt) {
    Vec2 v1 = bilinear_interpolation(vf, p);
    Vec2 p1 = p - 0.5 * dt * v1;
    Vec2 v2 = bilinear_interpolation(vf, p1);
    Vec2 p2 = p - 0.75 * dt * v2;
    Vec2 v3 = bilinear_interpolation(vf, p2);
    p -= dt * ((2.0 / 9.0) * v1 + (1.0 / 3.0) * v2 + (4.0 / 9.0) * v3);
    return p;
}

// velocity -> velocity
void advection(const Parameters &params, const Field<Vec2> &vf, Field<Vec2> &new_vf/* const Field<T> &qf, Field<T> &new_qf */) {
    for (int i=0; i<vf.row; i++) {
        for (int j=0; j<vf.col; j++) {
            Vec2 p = RK3(vf, Vec2(i+0.5, j+0.5), params.dt);
            new_vf[i][j] = bilinear_interpolation(vf, p) * params.decay;
        }
    }
}

// velocity -> divergence
void divergence(Field<Vec2> &vf, Field<double> &vdf) {
    for (int i=0; i<vf.row; i++) {
        for (int j=0; j<vf.col; j++) {
            Vec2 vc = sample(vf, i, j);
            Vec2 vl = sample(vf, i - 1, j);
            Vec2 vr = sample(vf, i + 1, j);
            Vec2 vb = sample(vf, i, j - 1);
            Vec2 vt = sample(vf, i, j + 1);
            if (i == 0) {
                vl[0] = -vc.x();
            }
            if (i == vf.row - 1) {
                vr[0] = -vc.x();
            }
            if (j == 0) {
                vb[1] = -vc.y();
            }
            if (j == vf.col - 1) {
                vt[1] = -vc.y();
            }
            vdf[i][j] = (vr.x() - vl.x() + vt.y() - vb.y()) * 0.5;
        }
    }
}

// (preasure, divergence) -> preasure
void compression(int compression_iters, const Field<double> &vdf, Pair<Field<double>> &p_pair) {
    for (int iter=0; iter<compression_iters; iter++) {
        for (int i=0; i<p_pair.cur->row; i++) {
            for (int j=0; j<p_pair.cur->col; j++) {
                double pl = sample(*p_pair.cur, i - 1, j);
                double pr = sample(*p_pair.cur, i + 1, j);
                double pb = sample(*p_pair.cur, i, j - 1);
                double pt = sample(*p_pair.cur, i, j + 1);
                double div = vdf[i][j];
                (*p_pair.nxt)[i][j] = (pl + pr + pb + pt - div) * 0.25;
            }
        }
        p_pair.swap();
    }
}

// preasure -> delta_velocity
void subtract_gradient(const Field<double> &pf, Field<Vec2> &vf) {
    for (int i=0; i<pf.row; i++) {
        for (int j=0; j<pf.col; j++) {
            double pl = sample(pf, i - 1, j);
            double pr = sample(pf, i + 1, j);
            double pb = sample(pf, i, j - 1);
            double pt = sample(pf, i, j + 1);
            vf[i][j] -= 0.5 * Vec2(pr - pl, pt - pb);
        }
    }
}

void apply_impulse(const Parameters &params, Field<Vec2> &vf) {
    for (int i=0; i<vf.row; i++) {
        for (int j=0; j<vf.col; j++) {
            Vec2 p = Vec2(i + 0.5, j + 0.5);
            Vec2 d = p - params.impulse_center;
            double d2 = d.dot(d);
            double factor = exp(-d2 / params.force_radius);
            vf[i][j] += Vec2(1.0, 0.0) * params.impulse_strength * factor * params.dt;
        }
    }
}

EXPORT int step() {
    advection(parameters, *velocities_pair.cur, *velocities_pair.nxt);
    velocities_pair.swap();
    apply_impulse(parameters, *velocities_pair.cur);
    divergence(*velocities_pair.cur, velocity_divergences);
    compression(parameters.compression_iters, velocity_divergences, pressures_pair);
    subtract_gradient(*pressures_pair.cur, *velocities_pair.cur);
    return velocities_pair.cur->data != _velocities_data;
}

EXPORT void calculate_vorticity() {
    auto vf = velocities_pair.cur;
    for (int i=0; i<vf->row; i++) {
        for (int j=0; j<vf->col; j++) {
            Vec2 vl = sample(*vf, i - 1, j);
            Vec2 vr = sample(*vf, i + 1, j);
            Vec2 vb = sample(*vf, i, j - 1);
            Vec2 vt = sample(*vf, i, j + 1);
            velocity_curls[i][j] = (vr.y() - vl.y() - vt.x() + vb.x()) * 0.5;
        }
    }
}