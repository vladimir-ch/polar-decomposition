#pragma once

#include <boost/assert.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

namespace polar {

namespace detail {

template<class Matrix>
bool invert(Matrix const& input, Matrix& inverse)
{
    using namespace boost::numeric::ublas;

    typedef permutation_matrix<std::size_t> pmatrix;

    BOOST_ASSERT(input.size1() == input.size2());

    Matrix A(input);
    pmatrix pm(A.size1());

    if (lu_factorize(A, pm) != 0)
        return false;

    inverse.assign(identity_matrix<typename Matrix::value_type>(A.size1()));
    lu_substitute(A, pm, inverse);

    return true;
}

} // namespace detail

template<typename Matrix_in, typename Matrix_out>
void polar_decomposition(Matrix_in const& A, Matrix_out& U, Matrix_out& H,
        double rtol = 1.0e-8, std::size_t max_iter = 0)
{
    typedef typename Matrix_out::value_type value_type;

    BOOST_ASSERT(A.size1() == A.size2());

    U.resize(A.size1(), A.size2());
    U.assign(A);
    H.resize(A.size1(), A.size2());
    Matrix_out X(A.size1(), A.size2());

    std::size_t count = 0;
    bool close_to_convergence = false;

    while (true)
    {
        X.assign(U);
        detail::invert(X, H);
        value_type gamma = 1.0;

        if (!close_to_convergence)
        {
            value_type alpha = std::sqrt(norm_1(X) * norm_inf(X));
            value_type beta  = std::sqrt(norm_1(H) * norm_inf(H));
            gamma = std::sqrt(beta / alpha);
        }

        U.assign(0.5 * (gamma * X + 1.0 / gamma * trans(H)));

        if (norm_1(X - U) < rtol * norm_1(X))
            break;

        if (max_iter != 0 && ++count == max_iter)
            break;

        if (!close_to_convergence && norm_frobenius(X - U) < 1.0e-2)
            close_to_convergence = true;
    }

    H.assign(0.5 * (prod(trans(U), A) + prod(trans(A), U)));
}

} // namespace polar
