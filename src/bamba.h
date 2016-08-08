class Bamba {
public:
    ModelData data;
    ChainParameters chain_pars;
    Latent latent;
    Hypers hypers;
    Linear linear;

    // helpers
    RngStream rng;
    Eigen::VectorXd all_fitted;
    Eigen::VectorXd y_tilde;
    bool update_beads;

    // trace
    Eigen::MatrixXd trace_b;
    Eigen::MatrixXd trace_u;
    Eigen::MatrixXd trace_p;
    Eigen::MatrixXd trace_mu_g;
    Eigen::MatrixXd trace_nu_g;
    Eigen::MatrixXd trace_hypers;
    Eigen::MatrixXd trace_bead_mfi;
    Eigen::MatrixXd trace_bead_prec;
    Eigen::MatrixXd trace_theta;
    Eigen::VectorXd ppr;

    Bamba(SEXP r_bead_list, SEXP r_chain_pars,
            SEXP r_latent_terms, SEXP r_linear_terms,
            SEXP r_hyper_list, SEXP r_at_matrix,
            SEXP r_update_beads);

    void iterate();
    void gatherTrace(const int i);
};
