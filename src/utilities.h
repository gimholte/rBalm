#ifndef UTILITIES_H
#define UTILITIES_H

template <typename Func>
double unimodalSliceSampler(RngStream rng, double x_init,
        double x_upper, double x_lower,
        double & w, Func & log_density);


template <typename Func>
double unimodalSliceSampler(RngStream rng, const double x_init,
        const double x_upper, const double x_lower,
        double & w, Func & log_density)
{
    /* z is the y-value for horizontal slice */
    const double z = log_density(x_init) - RngStream_GA1(1.0, rng);
    double R(fmin(x_upper, x_init + w));
    double L(fmax(x_lower, x_init - w));
    double g_R = log_density(R);
    double g_L = log_density(L);

    // grow interval endpoints until we find upper and lower endpoints
    // s.t. G(R) < z and G(L) < z. Assumes unimodal density.
    while (g_R > z) {
        R += w;
        R = fmin(R, x_upper);
        g_R = log_density(R);
    }

    while (g_L > z) {
        L -= w;
        L = fmax(L, x_lower);
        g_L = log_density(L);
    }

    // sample [L, R] interval, shrinking L,R if
    // a sample X has density G(X) < z
    double s;
    while (s = RngStream_UnifAB(L, R, rng), log_density(s) < z) {
        if (s > x_init)
            R = s;
        else
            L = s;
    }
    w = fmax((R - L) / 2.0, .1);
    return s;
}


void mvNormSim(RngStream rng, Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & solver,
        const Eigen::SparseMatrix<double> & omega, const Eigen::VectorXd & tau,
        Eigen::VectorXd & sample);

bool checkListNames(const std::vector<std::string> & names, const Rcpp::List & ll);

double rho2phi(double rho, int p);
double phi2rho(double phi, int p);
double drho_dphi(double phi, int p);
double dphi_drho(double rho, int p);
#endif
