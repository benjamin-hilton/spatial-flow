#ifndef ERF_HPP
#define ERF_HPP
#include <cmath>

//  "A handy approximation for the error function and its inverse" by Sergei Winitzki.

namespace erf_hpp {

    template<typename T>
    T erfinv(T x){

        const T sign = (x < 0) ? -1.0 : 1.0;

        T logx = std::log((1 - x) * (1 + x));

        T term1 = 2/(M_PI * 0.147) + 0.5 * logx;
        T term2 = 1/(0.147) * logx;

        return sign * std::sqrt(-term1 + std::sqrt(term1 * term1 - term2));
    }

}

#endif // ERF_HPP
