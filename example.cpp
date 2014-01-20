#include <cstddef>
#include <iostream>

#include <boost/random.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "polar_decomposition.hpp"

int main()
{
    boost::mt19937 seed(42);
    boost::uniform_real<> dist(-10.0, 10.0);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> > random(seed, dist);

    boost::numeric::ublas::matrix<double> A(3,3), U, H;

    for (std::size_t i = 0; i < 100000; ++i)
    {
        std::generate(A.data().begin(), A.data().end(), random);
        polar::polar_decomposition(A, U, H);
    }

    std::cout << A << std::endl;
    std::cout << prod(U, trans(U)) << std::endl;
}
